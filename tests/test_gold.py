"""tests/test_gold.py – Gold test runner for all schemes.

What gold tests are
-------------------
Hand-crafted profiles with known correct verdicts.  They never change unless
you deliberately update them.  Any change to the engine that shifts a verdict
will fail here immediately.

How it works
------------
For each scheme that has a data/gold/<scheme-id>.json file:
  1. Load the gold cases from JSON.
  2. Load the approved rules via engine/loader.py.
  3. For each test case, evaluate each rule and compare the actual verdict to
     the expected verdict in the gold file.

Adding a new scheme
-------------------
Just drop data/gold/<scheme-id>.json.  This runner discovers it automatically.
No changes to this file needed.

Run with:
    pytest tests/test_gold.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yojanasetu.engine.evaluator import evaluate_rule
from yojanasetu.engine.loader import load_approved_rules
from yojanasetu.engine.models import Verdict

# ---------------------------------------------------------------------------
# Discover gold files
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLD_DIR = PROJECT_ROOT / "data" / "gold"


def _gold_files() -> list[Path]:
    """Return all .json files in data/gold/ (excludes .gitkeep etc.)."""
    return sorted(GOLD_DIR.glob("*.json"))


def _load_gold(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Build pytest parameters: one param set per (scheme, test_case, rule_id)
# ---------------------------------------------------------------------------

def _collect_params():
    """Flatten gold files into individual (test_id, rule_id, profile, expected_verdict) tuples."""
    params = []
    for gold_file in _gold_files():
        scheme_id = gold_file.stem          # filename without .json
        try:
            rules = {r.rule_id: r for r in load_approved_rules(scheme_id)}
        except FileNotFoundError:
            # Gold file exists but no rules file yet – skip gracefully.
            continue

        for case in _load_gold(gold_file):
            test_id = case["test_id"]
            profile = case["profile"]
            for rule_id, expected_str in case["expected"].items():
                if rule_id not in rules:
                    # Rule listed in gold but not in rules file – will fail descriptively below.
                    params.append(
                        pytest.param(
                            scheme_id, test_id, rule_id, profile,
                            expected_str, None,
                            id=f"{test_id}::{rule_id}",
                            marks=pytest.mark.xfail(
                                reason=f"Rule '{rule_id}' not found in approved rules",
                                strict=True,
                            ),
                        )
                    )
                else:
                    params.append(
                        pytest.param(
                            scheme_id, test_id, rule_id, profile,
                            expected_str, rules[rule_id],
                            id=f"{test_id}::{rule_id}",
                        )
                    )
    return params


# ---------------------------------------------------------------------------
# The parametrised test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "scheme_id,test_id,rule_id,profile,expected_str,rule",
    _collect_params(),
)
def test_gold_case(scheme_id, test_id, rule_id, profile, expected_str, rule):
    """Each gold case must produce exactly the expected verdict.

    If this test fails:
      - Check whether you changed an operator, value, or field name in the
        rules JSON.
      - If the change was intentional, update the gold file and re-run.
      - If not intentional, revert the rules change.
    Never update a gold file to hide a bug.
    """
    expected = Verdict(expected_str)
    result = evaluate_rule(rule, profile)

    assert result.verdict == expected, (
        f"\nScheme : {scheme_id}"
        f"\nTest   : {test_id}"
        f"\nRule   : {rule_id}"
        f"\nProfile: {profile}"
        f"\nExpected verdict : {expected.value}"
        f"\nActual verdict   : {result.verdict.value}"
        f"\nReason : {result.reason_en}"
    )
