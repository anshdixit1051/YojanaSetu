"""scripts/evaluate_retrieval.py – Measure Precision@5 across real queries."""

from __future__ import annotations

from pathlib import Path
from yojanasetu.retrieval.embedder import Embedder
from yojanasetu.retrieval.hybrid import HybridIndex

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = PROJECT_ROOT / "data" / "index" / "hybrid.pkl"

EVAL_QUERIES = [
    ("pm-kisan", "farmer cultivable land income support 6000 exclusion income tax"),
    ("ayushman-bharat-pmjay", "health insurance 5 lakh hospital secondary tertiary care poor families SECC"),
    ("nsp-post-matric-sc", "scheduled caste student scholarship post-matric higher education parent income"),
    ("pm-svanidhi", "street vendor urban working capital loan identity card certificate"),
    ("up-kanya-sumangala", "girl child financial assistance higher education birth uttar pradesh resident"),
]


def run_evaluation():
    embedder = Embedder.load()
    index = HybridIndex.load(INDEX_PATH, embedder)

    print("=" * 70)
    print("HYBRID RETRIEVAL PRECISION@5 BENCHMARK")
    print("=" * 70)

    all_p5 = []
    for expected_scheme, q in EVAL_QUERIES:
        results = index.search(q, top_k=5)
        relevant_count = sum(1 for c, _ in results if c.scheme_id == expected_scheme)
        p5 = relevant_count / 5.0
        all_p5.append(p5)
        print(f"\nQuery: '{q}'")
        print(f"Target Scheme : {expected_scheme}")
        print(f"Precision@5   : {p5:.2f} ({relevant_count}/5 top chunks from target scheme)")
        for rank, (chunk, score) in enumerate(results, start=1):
            rel = "✓" if chunk.scheme_id == expected_scheme else "✗"
            preview = chunk.text.replace("\n", " ")[:75]
            print(f"  [{rank}] {rel} {chunk.chunk_id:<25} score={score:.4f} | {preview}...")

    mean_p5 = sum(all_p5) / len(all_p5)
    print("\n" + "=" * 70)
    print(f"OVERALL MEAN PRECISION@5: {mean_p5:.1%}")
    print("=" * 70)
    return mean_p5


if __name__ == "__main__":
    run_evaluation()
