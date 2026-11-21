"""engine/loader.py – Load and validate rules from data/rules/<scheme-id>.json.

Why this exists
---------------
Rules are stored as JSON (not Python code) so non-programmers can review and
approve them without touching source code.  This module is the only place that
reads those files and converts them into typed Rule objects.

Contract
--------
load_rules(scheme_id)  → list[Rule]
    Reads data/rules/<scheme-id>.json, validates every entry, and returns a
    list of Rule objects.  Raises descriptively if anything is wrong.

load_approved_rules(scheme_id) → list[Rule]
    Same as load_rules but filters to status="approved" only.
    Use this in the evaluator so pending rules never reach users.

Design note: strict validation
-------------------------------
Every field is checked before a Rule is constructed.  An unknown operator, a
missing required key, or a bad verdict string raises ValueError with the
offending rule_id included.  This surfaces data errors at load time, not
mid-evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from yojanasetu.engine.models import Rule, Verdict

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]   # src/yojanasetu/engine → repo root
RULES_DIR = PROJECT_ROOT / "data" / "rules"

REQUIRED_KEYS = frozenset({
    "rule_id", "scheme_id", "field", "operator",
    "value", "verdict_if_true", "description_en", "source_clause",
})

VALID_OPERATORS = frozenset(
    {"eq", "neq", "lt", "lte", "gt", "gte", "in", "not_in", "exists"}
)

VALID_STATUSES = frozenset({"approved", "pending_review"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_rule(raw: dict[str, Any], source_file: Path) -> Rule:
    """Convert one raw JSON dict into a validated Rule.

    Parameters
    ----------
    raw:
        A single entry from the JSON array.
    source_file:
        Path to the JSON file (included in error messages).

    Raises
    ------
    ValueError
        If any required key is missing, or any value is invalid.
    """
    rule_id = raw.get("rule_id", "<unknown>")
    ctx = f"rule '{rule_id}' in {source_file.name}"

    # 1. Required keys must all be present.
    missing = REQUIRED_KEYS - raw.keys()
    if missing:
        raise ValueError(
            f"{ctx}: missing required fields: {sorted(missing)}"
        )

    # 2. verdict_if_true must be a valid Verdict string.
    verdict_str = raw["verdict_if_true"]
    try:
        verdict = Verdict(verdict_str)
    except ValueError:
        raise ValueError(
            f"{ctx}: verdict_if_true '{verdict_str}' is not valid. "
            f"Use one of: {[v.value for v in Verdict]}"
        )

    # 3. operator must be supported.
    operator = raw["operator"]
    if operator not in VALID_OPERATORS:
        raise ValueError(
            f"{ctx}: operator '{operator}' is not supported. "
            f"Use one of: {sorted(VALID_OPERATORS)}"
        )

    # 4. status must be a known value.
    status = raw.get("status", "pending_review")
    if status not in VALID_STATUSES:
        raise ValueError(
            f"{ctx}: status '{status}' is not valid. "
            f"Use one of: {sorted(VALID_STATUSES)}"
        )

    # 5. rule_id must match scheme_id prefix (defensive check).
    scheme_id = raw["scheme_id"]
    if not raw["rule_id"].startswith(scheme_id):
        raise ValueError(
            f"{ctx}: rule_id must start with scheme_id '{scheme_id}' "
            f"(got '{raw['rule_id']}')"
        )

    return Rule(
        rule_id=raw["rule_id"],
        scheme_id=scheme_id,
        field=raw["field"],
        operator=operator,
        value=raw["value"],
        verdict_if_true=verdict,
        description_en=raw["description_en"],
        description_hi=raw.get("description_hi", ""),
        source_clause=raw["source_clause"],
        requires_human_judgement=bool(raw.get("requires_human_judgement", False)),
        status=status,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_rules(scheme_id: str) -> list[Rule]:
    """Load and validate all rules for a scheme from data/rules/<scheme-id>.json.

    Parameters
    ----------
    scheme_id:
        Slug matching the JSON filename, e.g. ``"pm-kisan"``.

    Returns
    -------
    list[Rule]
        All rules (approved + pending) in file order.

    Raises
    ------
    FileNotFoundError
        If the rules file does not exist.
    ValueError
        If the file is not valid JSON, not a JSON array, or any rule fails
        validation.
    """
    rules_file = RULES_DIR / f"{scheme_id}.json"

    if not rules_file.exists():
        raise FileNotFoundError(
            f"No rules file found for scheme '{scheme_id}'. "
            f"Expected: {rules_file}"
        )

    try:
        raw_list = json.loads(rules_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"data/rules/{scheme_id}.json is not valid JSON: {exc}"
        ) from exc

    if not isinstance(raw_list, list):
        raise ValueError(
            f"data/rules/{scheme_id}.json must contain a JSON array of rules, "
            f"got {type(raw_list).__name__}"
        )

    rules: list[Rule] = []
    for i, raw in enumerate(raw_list):
        if not isinstance(raw, dict):
            raise ValueError(
                f"data/rules/{scheme_id}.json: item {i} is not a JSON object"
            )
        rules.append(_parse_rule(raw, rules_file))

    return rules


def load_approved_rules(scheme_id: str) -> list[Rule]:
    """Load only the approved rules for a scheme.

    Wraps load_rules and filters out pending entries.  Use this in the
    evaluator so unreviewed rules never reach users.

    Parameters
    ----------
    scheme_id:
        Slug matching the JSON filename, e.g. ``"pm-kisan"``.

    Returns
    -------
    list[Rule]
        Only rules with status="approved", in file order.
    """
    return [r for r in load_rules(scheme_id) if r.is_approved()]
