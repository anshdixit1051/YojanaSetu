"""retrieval/embedder.py – Dense embeddings with BAAI/bge-small-en-v1.5.

bge-small-en is a 33 M-parameter sentence embedding model from BAAI.
It runs on CPU in ~100 ms per batch, needs ~130 MB RAM, and fits comfortably
within the project's 16 GB constraint.

The model is downloaded once from HuggingFace Hub on first use and cached
under ~/.cache/huggingface/.

Usage
-----
    embedder = Embedder.load()
    vecs = embedder.encode(["some text", "another text"])
    # vecs.shape == (2, 384)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


MODEL_NAME = "BAAI/bge-small-en-v1.5"

# bge-small instruction prefix that improves retrieval quality for passages.
# For queries we use a different prefix; for passages we use empty string.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass
class Embedder:
    """Wraps a SentenceTransformer model for encoding text.

    Attributes
    ----------
    _model:
        The underlying SentenceTransformer instance.
    dim:
        Embedding dimensionality (384 for bge-small-en-v1.5).
    """

    _model: object
    dim: int

    @classmethod
    def load(cls, model_name: str = MODEL_NAME, device: str = "cpu") -> "Embedder":
        """Load the sentence embedding model.

        Downloads the model on first call (~130 MB).
        Subsequent calls use the local HuggingFace cache.

        Parameters
        ----------
        model_name:
            HuggingFace model identifier.  Default: ``BAAI/bge-small-en-v1.5``.
        device:
            ``"cpu"`` (default) or ``"cuda"`` if a GPU is available.

        Returns
        -------
        Embedder

        Raises
        ------
        ImportError
            If sentence-transformers is not installed.
        """
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc

        model = SentenceTransformer(model_name, device=device)
        dim = model.get_sentence_embedding_dimension()
        return cls(_model=model, dim=dim)

    def encode_passages(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Encode a list of passage texts into L2-normalised embedding vectors.

        Parameters
        ----------
        texts:
            List of strings to embed (source text chunks).
        batch_size:
            Number of texts per GPU/CPU batch.

        Returns
        -------
        np.ndarray of shape (len(texts), dim), dtype float32, L2-normalised.
        """
        vecs = self._model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(vecs, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string with the retrieval instruction prefix.

        Parameters
        ----------
        query:
            User query, e.g. ``"farmer income tax exemption"``.

        Returns
        -------
        np.ndarray of shape (dim,), dtype float32, L2-normalised.
        """
        prefixed = QUERY_PREFIX + query
        vec = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(vec, dtype=np.float32)

    def cosine_scores(self, query_vec: np.ndarray, passage_vecs: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between a query vector and all passage vectors.

        Since both are L2-normalised, cosine similarity = dot product.

        Parameters
        ----------
        query_vec:
            Shape (dim,).
        passage_vecs:
            Shape (N, dim).

        Returns
        -------
        np.ndarray of shape (N,), values in [-1, 1].
        """
        return passage_vecs @ query_vec
