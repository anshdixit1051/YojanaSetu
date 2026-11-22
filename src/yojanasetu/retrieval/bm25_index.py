"""retrieval/bm25_index.py – BM25 keyword index over scheme chunks.

BM25 (Best Match 25) is a classic TF-IDF-style ranking function.  It works
well for exact keyword matches (scheme names, amounts, caste categories) and
runs entirely on CPU with no model weights.

We use the `rank-bm25` library (BM25Okapi variant).

Usage
-----
    index = BM25Index.build(chunks)
    results = index.search("income tax payer exclusion", top_k=5)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from yojanasetu.retrieval.chunker import Chunk


def _tokenise(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters.

    Simple but effective for English + Hindi mixed text.
    Hindi words are kept intact (Unicode letters pass the \\w filter).
    """
    return re.findall(r"\w+", text.lower())


@dataclass
class BM25Index:
    """BM25 index over a list of Chunk objects.

    Build with :meth:`build`, then call :meth:`search`.

    Attributes
    ----------
    chunks:
        The original chunks in index order.
    _model:
        The underlying rank_bm25.BM25Okapi instance.
    """

    chunks: list[Chunk]
    _model: object = field(repr=False, default=None)

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "BM25Index":
        """Build a BM25 index from a list of chunks.

        Parameters
        ----------
        chunks:
            All chunks to index (may span multiple schemes).

        Returns
        -------
        BM25Index
            Ready to query.

        Raises
        ------
        ImportError
            If rank-bm25 is not installed.
        ValueError
            If chunks is empty.
        """
        if not chunks:
            raise ValueError("Cannot build a BM25 index from an empty chunk list.")

        try:
            from rank_bm25 import BM25Okapi  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "rank-bm25 is not installed. Run: pip install rank-bm25"
            ) from exc

        tokenised = [_tokenise(c.text) for c in chunks]
        model = BM25Okapi(tokenised)
        return cls(chunks=chunks, _model=model)

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Return the top-k chunks most relevant to *query* by BM25 score.

        Parameters
        ----------
        query:
            Free-text query, e.g. ``"income tax payer exclusion"``.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        list of (Chunk, score) sorted by descending score.
        Chunks with score 0 are excluded.
        """
        tokens = _tokenise(query)
        import numpy as np  # noqa: PLC0415

        scores: list[float] = self._model.get_scores(tokens).tolist()
        ranked = sorted(
            zip(self.chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(chunk, score) for chunk, score in ranked[:top_k] if score > 0]
