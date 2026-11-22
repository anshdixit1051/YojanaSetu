"""scripts/review_rules.py – Interactive CLI to review pending rules.

Purpose
-------
LLM-extracted rules start as status="pending_review".  This script shows each
pending rule alongside the relevant source text so a human can decide:
  [a] approve  – sets status="approved"; rule is now used by the engine.
  [r] reject   – removes the rule from the file permanently.
  [s] skip     – leaves the rule as pending_review; review it another time.
  [q] quit     – stop reviewing; all remaining rules stay as pending_review.

Safety
------
Only approved rules reach the eligibility engine.  This reviewer is the human
checkpoint between LLM output and production use.  Never approve a rule without
verifying it against the source text shown on screen.

Usage
-----
  python scripts/review_rules.py pm-kisan
  python scripts/review_rules.py --all          # review all schemes with pending rules
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RULES_DIR = PROJECT_ROOT / "data" / "rules"
RAW_DIR   = PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------------------------
# Source text helpers
# ---------------------------------------------------------------------------

def _load_source(scheme_id: str) -> str:
    """Return the contents of data/raw/<scheme-id>/source.txt, or a placeholder."""
    source_path = RAW_DIR / scheme_id / "source.txt"
    if source_path.exists():
        return source_path.read_text(encoding="utf-8")
    return "[source.txt not found]"


def _find_context(source: str, description: str, window: int = 400) -> str:
    """Return up to `window` chars of source text around the first occurrence
    of any significant word from description_en.

    Falls back to the first `window` chars of source if nothing matches.
    """
    # Try to find a matching snippet by searching for key words.
    words = [w for w in description.split() if len(w) > 5]
    for word in words:
        idx = source.lower().find(word.lower())
        if idx != -1:
            start = max(0, idx - 100)
            end = min(len(source), idx + window)
            return source[start:end].strip()
    return source[:window].strip()


# ---------------------------------------------------------------------------
# Review logic
# ---------------------------------------------------------------------------

def _review_scheme(scheme_id: str) -> dict[str, int]:
    """Interactively review pending rules for one scheme.

    Returns a dict of counts: approved, rejected, skipped.
    """
    rules_file = RULES_DIR / f"{scheme_id}.json"
    if not rules_file.exists():
        print(f"  No rules file: {rules_file}")
        return {"approved": 0, "rejected": 0, "skipped": 0}

    rules: list[dict] = json.loads(rules_file.read_text(encoding="utf-8"))
    pending = [r for r in rules if r.get("status") == "pending_review"]

    if not pending:
        print(f"  No pending rules for {scheme_id}.")
        return {"approved": 0, "rejected": 0, "skipped": 0}

    source = _load_source(scheme_id)
    counts = {"approved": 0, "rejected": 0, "skipped": 0}
    rule_map = {r["rule_id"]: r for r in rules}

    print(f"\n{'='*70}")
    print(f"  Scheme: {scheme_id}  ({len(pending)} pending rule(s))")
    print(f"{'='*70}")

    for rule in pending:
        rid = rule["rule_id"]
        context = _find_context(source, rule.get("description_en", ""))

        print(f"\n--- Rule: {rid} ---")
        print(f"  Field       : {rule.get('field')}")
        print(f"  Operator    : {rule.get('operator')} {rule.get('value')}")
        print(f"  Verdict if  : {rule.get('verdict_if_true')}")
        print(f"  Description : {rule.get('description_en')}")
        print(f"  Clause      : {rule.get('source_clause')}")
        print(f"  Needs human : {rule.get('requires_human_judgement', False)}")
        print()
        print("  SOURCE CONTEXT:")
        print("  " + "\n  ".join(context.splitlines()[:12]))
        print()

        while True:
            choice = input("  [a]pprove / [r]eject / [s]kip / [q]uit: ").strip().lower()
            if choice in ("a", "r", "s", "q"):
                break
            print("  Please enter a, r, s, or q.")

        if choice == "q":
            print("\n  Quitting review. Remaining rules stay as pending_review.")
            break
        elif choice == "a":
            rule_map[rid]["status"] = "approved"
            counts["approved"] += 1
            print(f"  ✓ Approved: {rid}")
        elif choice == "r":
            # Remove from list by marking for deletion.
            rule_map[rid]["_delete"] = True
            counts["rejected"] += 1
            print(f"  ✗ Rejected: {rid}")
        elif choice == "s":
            counts["skipped"] += 1
            print(f"  → Skipped: {rid}")

    # Write back: keep all rules except rejected ones.
    updated = [r for r in rules if not r.pop("_delete", False)]
    rules_file.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n  Saved {rules_file.name}: "
          f"{counts['approved']} approved, "
          f"{counts['rejected']} rejected, "
          f"{counts['skipped']} skipped.")
    return counts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("scheme_id", nargs="?", help="Scheme to review (e.g. pm-kisan)")
    group.add_argument("--all", action="store_true", help="Review all schemes with pending rules")
    args = parser.parse_args()

    if args.all:
        schemes = sorted(p.stem for p in RULES_DIR.glob("*.json"))
        if not schemes:
            print("No rules files found in data/rules/.")
            sys.exit(0)
    else:
        schemes = [args.scheme_id]

    total = {"approved": 0, "rejected": 0, "skipped": 0}
    for scheme_id in schemes:
        counts = _review_scheme(scheme_id)
        for k in total:
            total[k] += counts[k]

    print(f"\n{'='*70}")
    print(f"  Session total: {total['approved']} approved, "
          f"{total['rejected']} rejected, {total['skipped']} skipped.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
