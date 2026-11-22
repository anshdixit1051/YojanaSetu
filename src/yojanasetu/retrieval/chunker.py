"""retrieval/chunker.py – Split scheme source text into overlapping chunks.

Why chunking
------------
Source documents are long (up to 10 000+ chars).  An embedding model has a
fixed token limit (~512 tokens for bge-small).  Chunking splits the text into
overlapping windows so each piece fits the model and no sentence is cut off
between adjacent chunks.

Each chunk carries:
  - its text
  - which scheme it came from
  - its character offset in source.txt (for citation)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    """One text chunk from a scheme source document.

    Attributes
    ----------
    chunk_id:
        Unique identifier, e.g. ``"pm-kisan::0"``.
    scheme_id:
        The scheme this chunk belongs to.
    text:
        The chunk text (stripped).
    char_start:
        Byte offset of the first character in source.txt.
    char_end:
        Byte offset one past the last character in source.txt.
    """

    chunk_id: str
    scheme_id: str
    text: str
    char_start: int
    char_end: int


def chunk_text(
    text: str,
    scheme_id: str,
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[Chunk]:
    """Split *text* into overlapping character-based chunks.

    Parameters
    ----------
    text:
        Full contents of source.txt (UTF-8 string, NFC-normalised).
    scheme_id:
        Slug of the owning scheme, e.g. ``"pm-kisan"``.
    chunk_size:
        Target size of each chunk in characters.  Default 400 ≈ 100 tokens,
        well within bge-small's 512-token limit even for Hindi text.
    overlap:
        Characters shared between consecutive chunks so sentences at
        boundaries are not cut off.

    Returns
    -------
    list[Chunk]
        Chunks in order.  Empty if text is blank.
    """
    text = text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Try to break at a sentence boundary (. or \n) within the last 60 chars.
        if end < len(text):
            boundary = -1
            for sep in ("\n\n", ".\n", ". ", "\n"):
                pos = text.rfind(sep, start + chunk_size - 60, end)
                if pos != -1:
                    boundary = pos + len(sep)
                    break
            if boundary != -1:
                end = boundary

        chunk_text_str = text[start:end].strip()
        if chunk_text_str:
            chunks.append(
                Chunk(
                    chunk_id=f"{scheme_id}::{idx}",
                    scheme_id=scheme_id,
                    text=chunk_text_str,
                    char_start=start,
                    char_end=end,
                )
            )
            idx += 1

        # Next window starts overlap chars before end.
        start = end - overlap if end - overlap > start else end

    return chunks


def chunk_scheme(scheme_dir: Path, **kwargs) -> list[Chunk]:
    """Chunk the source.txt file of a scheme directory.

    Parameters
    ----------
    scheme_dir:
        Path to ``data/raw/<scheme-id>/``.
    **kwargs:
        Passed through to :func:`chunk_text` (chunk_size, overlap).

    Returns
    -------
    list[Chunk]
        Empty if source.txt is missing or blank.
    """
    source_path = scheme_dir / "source.txt"
    if not source_path.exists():
        return []

    text = source_path.read_text(encoding="utf-8")
    scheme_id = scheme_dir.name
    return chunk_text(text, scheme_id, **kwargs)
