"""scripts/build_index.py – Build and save the hybrid retrieval index.

Reads every scheme folder in data/raw/ (skipping _TEMPLATE),
chunks each source.txt, encodes with bge-small-en, and saves the
combined index to data/index/hybrid.pkl.

Usage:
    python scripts/build_index.py           # build from all schemes
    python scripts/build_index.py --scheme pm-kisan   # one scheme only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR   = PROJECT_ROOT / "data" / "raw"
INDEX_DIR = PROJECT_ROOT / "data" / "index"
INDEX_PATH = INDEX_DIR / "hybrid.pkl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scheme", default=None,
        help="Build index for a single scheme only (e.g. pm-kisan).",
    )
    args = parser.parse_args()

    # Import here so the module loads fast when used as a library.
    from yojanasetu.retrieval.chunker import chunk_scheme
    from yojanasetu.retrieval.embedder import Embedder
    from yojanasetu.retrieval.hybrid import HybridIndex

    # Discover scheme folders.
    if args.scheme:
        scheme_dirs = [RAW_DIR / args.scheme]
    else:
        scheme_dirs = sorted(
            d for d in RAW_DIR.iterdir()
            if d.is_dir() and d.name != "_TEMPLATE"
        )

    if not scheme_dirs:
        print("No scheme folders found.", file=sys.stderr)
        sys.exit(1)

    # Chunk all source files.
    all_chunks = []
    for scheme_dir in scheme_dirs:
        chunks = chunk_scheme(scheme_dir)
        print(f"  {scheme_dir.name}: {len(chunks)} chunk(s)")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No chunks produced. Check that source.txt files are non-empty.", file=sys.stderr)
        sys.exit(1)

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Loading embedder (downloads model on first run ~130 MB)…")
    embedder = Embedder.load()
    print(f"Embedder ready – dim={embedder.dim}")

    print("Encoding passages and building BM25 index…")
    index = HybridIndex.build(all_chunks, embedder)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index.save(INDEX_PATH)
    print(f"\nIndex saved → {INDEX_PATH}")


if __name__ == "__main__":
    main()
