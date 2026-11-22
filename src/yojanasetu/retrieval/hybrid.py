"""retrieval/hybrid.py – Combine BM25 and dense retrieval via Reciprocal Rank Fusion.

Why Reciprocal Rank Fusion (RRF)?
----------------------------------
RRF combines two ranked lists without needing to tune score scaling between
BM25 and cosine similarity (they live on very different scales).  It only uses
the rank positions, so it is robust and needs no hyperparameter tuning.

Formula:  RRF(d) = Σ  1 / (k + rank_i(d))
where k=60 is the standard constant that dampens the influence of very high ranks.

Usage
-----
    index = HybridIndex.build(chunks, embedder)
    results = index.search("land holding exclusion pension", top_k=5)
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from yojanasetu.retrieval.bm25_index import BM25Index
from yojanasetu.retrieval.chunker import Chunk
from yojanasetu.retrieval.embedder import Embedder

RRF_K = 60   # standard constant; higher = less weight on top ranks


def _reciprocal_rank_fusion(
    bm25_ranked: list[tuple[Chunk, float]],
    dense_ranked: list[tuple[Chunk, float]],
    k: int = RRF_K,
) -> list[tuple[Chunk, float]]:
    """Fuse two ranked result lists using RRF.

    Parameters
    ----------
    bm25_ranked, dense_ranked:
        Each is a list of (Chunk, score) sorted by descending score.
    k:
        RRF damping constant.

    Returns
    -------
    list of (Chunk, rrf_score) sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}

    for rank, (chunk, _) in enumerate(bm25_ranked):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank + 1)
        chunk_map[chunk.chunk_id] = chunk

    for rank, (chunk, _) in enumerate(dense_ranked):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank + 1)
        chunk_map[chunk.chunk_id] = chunk

    return sorted(
        [(chunk_map[cid], score) for cid, score in scores.items()],
        key=lambda x: x[1],
        reverse=True,
    )


@dataclass
class HybridIndex:
    """Combined BM25 + dense retrieval index.

    Build with :meth:`build`, query with :meth:`search`,
    persist with :meth:`save` / :meth:`load`.

    Attributes
    ----------
    chunks:
        All indexed chunks.
    bm25:
        BM25 index over the chunks.
    embedder:
        SentenceTransformer wrapper.
    passage_vecs:
        Pre-computed L2-normalised embedding matrix, shape (N, dim).
    """

    chunks: list[Chunk]
    bm25: BM25Index
    embedder: Embedder
    passage_vecs: np.ndarray

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> "HybridIndex":
        """Build the hybrid index from a list of chunks.

        Encodes all chunks using the embedder (slow first run, cached after).

        Parameters
        ----------
        chunks:
            All text chunks to index.
        embedder:
            A loaded :class:`Embedder` instance.

        Returns
        -------
        HybridIndex
        """
        if not chunks:
            raise ValueError("Cannot build a HybridIndex from an empty chunk list.")

        bm25 = BM25Index.build(chunks)
        texts = [c.text for c in chunks]
        passage_vecs = embedder.encode_passages(texts)
        return cls(chunks=chunks, bm25=bm25, embedder=embedder, passage_vecs=passage_vecs)

    def search(
        self,
        query: str,
        top_k: int = 5,
        bm25_candidates: int = 20,
        dense_candidates: int = 20,
    ) -> list[tuple[Chunk, float]]:
        """Return top-k chunks most relevant to *query* by RRF hybrid score.

        Parameters
        ----------
        query:
            Free-text user query.
        top_k:
            Final number of results to return.
        bm25_candidates, dense_candidates:
            How many candidates each retriever contributes before fusion.

        Returns
        -------
        list of (Chunk, rrf_score) sorted by descending score.
        """
        # BM25 side.
        bm25_results = self.bm25.search(query, top_k=bm25_candidates)

        # Dense side.
        query_vec = self.embedder.encode_query(query)
        cosine = self.embedder.cosine_scores(query_vec, self.passage_vecs)
        top_dense_idx = np.argsort(cosine)[::-1][:dense_candidates]
        dense_results = [(self.chunks[i], float(cosine[i])) for i in top_dense_idx]

        fused = _reciprocal_rank_fusion(bm25_results, dense_results)
        return fused[:top_k]

    def save(self, path: Path) -> None:
        """Persist the index to disk using pickle.

        Saves chunks, BM25 model, passage vectors, and embedder model name.
        The embedder itself is NOT pickled (SentenceTransformer is not
        serialisable); it is reloaded from HuggingFace cache on :meth:`load`.

        Parameters
        ----------
        path:
            File path to write (e.g. ``data/index/hybrid.pkl``).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": self.chunks,
            "bm25": self.bm25,
            "passage_vecs": self.passage_vecs,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Path, embedder: Embedder) -> "HybridIndex":
        """Load a previously saved index from disk.

        Parameters
        ----------
        path:
            Path to the pickle file written by :meth:`save`.
        embedder:
            A freshly loaded :class:`Embedder` (needed for query encoding).

        Returns
        -------
        HybridIndex
        """
        with open(path, "rb") as f:
            payload = pickle.load(f)
        return cls(
            chunks=payload["chunks"],
            bm25=payload["bm25"],
            embedder=embedder,
            passage_vecs=payload["passage_vecs"],
        )
