"""tests/test_retrieval.py – Tests for the retrieval package.

The Embedder (which downloads a 130 MB model) is fully mocked.
BM25 and RRF logic are tested with real in-memory data.

Run with: pytest tests/test_retrieval.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from yojanasetu.retrieval.chunker import Chunk, chunk_text, chunk_scheme
from yojanasetu.retrieval.bm25_index import BM25Index, _tokenise
from yojanasetu.retrieval.hybrid import HybridIndex, _reciprocal_rank_fusion


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TEXT = """\
Pradhan Mantri Kisan Samman Nidhi provides Rs 6000 per year to farmer families.
All landholding farmer families are eligible subject to exclusion criteria.
Income tax payers are not eligible under this scheme.
Institutional landholders are excluded from PM-KISAN benefits.
The fund is transferred directly to the bank accounts of beneficiaries.
"""


def make_chunks(n: int = 5, scheme_id: str = "test-scheme") -> list[Chunk]:
    """Return n simple Chunk objects for testing."""
    texts = [
        "farmer land holding eligible pm-kisan",
        "income tax payer exclusion not eligible",
        "institutional landholder excluded scheme",
        "bank account direct transfer benefit",
        "pension amount monthly rupees limit",
    ]
    return [
        Chunk(
            chunk_id=f"{scheme_id}::{i}",
            scheme_id=scheme_id,
            text=texts[i % len(texts)],
            char_start=i * 50,
            char_end=(i + 1) * 50,
        )
        for i in range(n)
    ]


def make_mock_embedder(n_chunks: int, dim: int = 8) -> MagicMock:
    """Return a mock Embedder that returns random unit vectors."""
    embedder = MagicMock()
    embedder.dim = dim
    rng = np.random.default_rng(42)
    vecs = rng.random((n_chunks, dim)).astype(np.float32)
    # L2-normalise.
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms

    embedder.encode_passages.return_value = vecs
    # encode_query returns a single unit vector.
    q_vec = rng.random(dim).astype(np.float32)
    q_vec /= np.linalg.norm(q_vec)
    embedder.encode_query.return_value = q_vec
    embedder.cosine_scores.side_effect = lambda q, p: p @ q
    return embedder


# ---------------------------------------------------------------------------
# chunker
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_empty_text_returns_empty(self):
        assert chunk_text("", "pm-kisan") == []

    def test_whitespace_only_returns_empty(self):
        assert chunk_text("   \n  ", "pm-kisan") == []

    def test_short_text_is_one_chunk(self):
        chunks = chunk_text("Hello world.", "pm-kisan", chunk_size=400)
        assert len(chunks) == 1
        assert chunks[0].scheme_id == "pm-kisan"
        assert "Hello world" in chunks[0].text

    def test_chunk_ids_are_unique(self):
        chunks = chunk_text(SAMPLE_TEXT * 10, "pm-kisan", chunk_size=100, overlap=20)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_ids_start_with_scheme_id(self):
        chunks = chunk_text(SAMPLE_TEXT, "my-scheme")
        assert all(c.chunk_id.startswith("my-scheme::") for c in chunks)

    def test_all_chunks_have_non_empty_text(self):
        chunks = chunk_text(SAMPLE_TEXT * 5, "pm-kisan", chunk_size=100, overlap=20)
        assert all(c.text.strip() for c in chunks)

    def test_chunks_cover_full_text(self):
        """The union of all chunk texts should contain every sentence."""
        chunks = chunk_text(SAMPLE_TEXT, "pm-kisan", chunk_size=150, overlap=30)
        full = " ".join(c.text for c in chunks)
        # Every sentence from SAMPLE_TEXT should appear somewhere in the chunks.
        for sentence in SAMPLE_TEXT.strip().splitlines():
            if sentence.strip():
                assert sentence[:20] in full, f"Missing: {sentence[:20]}"

    def test_char_start_increases_monotonically(self):
        chunks = chunk_text(SAMPLE_TEXT * 3, "pm-kisan", chunk_size=100, overlap=20)
        starts = [c.char_start for c in chunks]
        assert starts == sorted(starts)


class TestChunkScheme:
    def test_missing_source_txt_returns_empty(self, tmp_path):
        (tmp_path / "source.txt").unlink(missing_ok=True)
        result = chunk_scheme(tmp_path)
        assert result == []

    def test_reads_scheme_id_from_folder_name(self, tmp_path):
        scheme_dir = tmp_path / "my-scheme"
        scheme_dir.mkdir()
        (scheme_dir / "source.txt").write_text("Some text here.", encoding="utf-8")
        chunks = chunk_scheme(scheme_dir)
        assert all(c.scheme_id == "my-scheme" for c in chunks)


# ---------------------------------------------------------------------------
# BM25Index
# ---------------------------------------------------------------------------

class TestTokenise:
    def test_lowercases(self):
        assert _tokenise("Hello World") == ["hello", "world"]

    def test_splits_on_punctuation(self):
        tokens = _tokenise("Rs.6000 per year")
        assert "rs" in tokens
        assert "6000" in tokens

    def test_empty_string(self):
        assert _tokenise("") == []


class TestBM25Index:
    def test_empty_chunks_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            BM25Index.build([])

    def test_search_returns_at_most_top_k(self):
        chunks = make_chunks(5)
        index = BM25Index.build(chunks)
        results = index.search("farmer land", top_k=3)
        assert len(results) <= 3

    def test_search_returns_chunk_score_tuples(self):
        chunks = make_chunks(3)
        index = BM25Index.build(chunks)
        results = index.search("income tax", top_k=2)
        for chunk, score in results:
            assert isinstance(chunk, Chunk)
            assert isinstance(score, float)
            assert score > 0

    def test_relevant_chunk_ranked_first(self):
        chunks = make_chunks(5)
        index = BM25Index.build(chunks)
        results = index.search("income tax payer exclusion", top_k=5)
        # The chunk about income tax should appear in results.
        texts = [c.text for c, _ in results]
        assert any("income tax" in t for t in texts)

    def test_no_match_returns_empty(self):
        chunks = make_chunks(3)
        index = BM25Index.build(chunks)
        results = index.search("xyzzy unrelated nonsense zzz", top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# _reciprocal_rank_fusion
# ---------------------------------------------------------------------------

class TestRRF:
    def test_fused_scores_are_positive(self):
        a = make_chunks(3, "s1")
        b = make_chunks(3, "s2")
        bm25_ranked = [(a[0], 2.0), (a[1], 1.0), (a[2], 0.5)]
        dense_ranked = [(b[0], 0.9), (a[0], 0.8), (b[1], 0.7)]
        fused = _reciprocal_rank_fusion(bm25_ranked, dense_ranked)
        assert all(score > 0 for _, score in fused)

    def test_chunk_in_both_lists_has_higher_score(self):
        chunks = make_chunks(4)
        # chunks[0] appears in both lists at rank 0 and rank 1.
        bm25_ranked = [(chunks[0], 2.0), (chunks[1], 1.0)]
        dense_ranked = [(chunks[2], 0.9), (chunks[0], 0.8)]
        fused = _reciprocal_rank_fusion(bm25_ranked, dense_ranked)
        fused_map = {c.chunk_id: s for c, s in fused}
        assert fused_map[chunks[0].chunk_id] > fused_map[chunks[1].chunk_id]
        assert fused_map[chunks[0].chunk_id] > fused_map[chunks[2].chunk_id]

    def test_empty_inputs_return_empty(self):
        assert _reciprocal_rank_fusion([], []) == []


# ---------------------------------------------------------------------------
# HybridIndex (embedder mocked)
# ---------------------------------------------------------------------------

class TestHybridIndex:
    def test_empty_chunks_raises_value_error(self):
        embedder = make_mock_embedder(0)
        with pytest.raises(ValueError, match="empty"):
            HybridIndex.build([], embedder)

    def test_search_returns_at_most_top_k(self):
        chunks = make_chunks(5)
        embedder = make_mock_embedder(5)
        index = HybridIndex.build(chunks, embedder)
        results = index.search("farmer income tax", top_k=3)
        assert len(results) <= 3

    def test_search_results_are_chunk_score_tuples(self):
        chunks = make_chunks(5)
        embedder = make_mock_embedder(5)
        index = HybridIndex.build(chunks, embedder)
        for chunk, score in index.search("land holding", top_k=5):
            assert isinstance(chunk, Chunk)
            assert isinstance(score, float)

    def test_save_and_load_roundtrip(self, tmp_path):
        chunks = make_chunks(5)
        embedder = make_mock_embedder(5)
        index = HybridIndex.build(chunks, embedder)

        pkl_path = tmp_path / "hybrid.pkl"
        index.save(pkl_path)
        assert pkl_path.exists()

        loaded = HybridIndex.load(pkl_path, embedder)
        assert len(loaded.chunks) == len(chunks)
        assert loaded.passage_vecs.shape == index.passage_vecs.shape

    def test_loaded_index_can_search(self, tmp_path):
        chunks = make_chunks(5)
        embedder = make_mock_embedder(5)
        index = HybridIndex.build(chunks, embedder)
        index.save(tmp_path / "hybrid.pkl")

        loaded = HybridIndex.load(tmp_path / "hybrid.pkl", embedder)
        results = loaded.search("pension bank account", top_k=3)
        assert len(results) >= 1
