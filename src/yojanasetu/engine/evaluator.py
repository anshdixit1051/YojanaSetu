"""engine/evaluator.py – Deterministic three-valued eligibility evaluator.

Core contract
-------------
evaluate_rule(rule, profile) → EligibilityResult

  - If the required field is ABSENT from the profile  → NEEDS_MORE_INFO
  - If rule.requires_human_judgement is True          → NEEDS_MORE_INFO
  - If rule.status != "approved"                      → raises RuntimeError
      (never silently run an unreviewed rule)
  - Otherwise apply the operator and return ELIGIBLE or NOT_ELIGIBLE
    based on rule.verdict_if_true / verdict_if_false.

evaluate_scheme(rules, profile) → list[EligibilityResult]

  Evaluates every approved rule for a scheme and returns all results.
  Callers decide how to aggregate (e.g. first NOT_ELIGIBLE wins, or
  collect all NEEDS_MORE_INFO first).

The LLM never calls this module.  It is pure Python with no I/O.
"""

from __future__ import annotations

from typing import Any

from yojanasetu.engine.models import EligibilityResult, Rule, Verdict

# ---------------------------------------------------------------------------
# Supported operators
# ---------------------------------------------------------------------------

SUPPORTED_OPERATORS = frozenset(
    {"eq", "neq", "lt", "lte", "gt", "gte", "in", "not_in", "exists"}
)


def _apply_operator(operator: str, field_value: Any, rule_value: Any) -> bool:
    """Apply a comparison operator and return True/False.

    Raises
    ------
    ValueError
        If the operator is not in SUPPORTED_OPERATORS.
    TypeError
        If the operand types are incompatible (e.g. comparing str to int).
    """
    if operator not in SUPPORTED_OPERATORS:
        raise ValueError(
            f"Unknown operator '{operator}'. "
            f"Supported: {sorted(SUPPORTED_OPERATORS)}"
        )

    if operator == "exists":
        # rule_value is ignored; we only check the field was present (caller handles absence)
        return True  # field is present (absence is handled before this call)
    if operator == "eq":
        return field_value == rule_value
    if operator == "neq":
        return field_value != rule_value
    if operator == "lt":
        return field_value < rule_value
    if operator == "lte":
        return field_value <= rule_value
    if operator == "gt":
        return field_value > rule_value
    if operator == "gte":
        return field_value >= rule_value
    if operator == "in":
        return field_value in rule_value
    if operator == "not_in":
        return field_value not in rule_value

    # Should never reach here because of the guard above, but keeps mypy happy.
    raise ValueError(f"Unhandled operator '{operator}'")  # pragma: no cover


# ---------------------------------------------------------------------------
# Single-rule evaluation
# ---------------------------------------------------------------------------

def evaluate_rule(rule: Rule, profile: dict[str, Any]) -> EligibilityResult:
    """Evaluate one Rule against a user profile dict.

    Parameters
    ----------
    rule:
        A single eligibility criterion.  Must be ``status="approved"``.
    profile:
        Dict of user-supplied facts, e.g. ``{"age": 35, "state": "UP"}``.
        Keys must be plain strings.  Values may be any JSON-compatible type.

    Returns
    -------
    EligibilityResult
        verdict is always one of ELIGIBLE, NOT_ELIGIBLE, NEEDS_MORE_INFO.

    Raises
    ------
    RuntimeError
        If the rule is not approved.  The engine must never silently run a
        pending rule – that would let LLM-hallucinated rules reach users.
    """
    # Safety gate: refuse to run unapproved rules.
    if not rule.is_approved():
        raise RuntimeError(
            f"Rule '{rule.rule_id}' has status='{rule.status}'. "
            "Only 'approved' rules may be evaluated. "
            "Review and approve it in the human review queue first."
        )

    # Discretionary criteria can never be decided automatically.
    if rule.requires_human_judgement:
        return EligibilityResult(
            verdict=Verdict.NEEDS_MORE_INFO,
            rule=rule,
            reason_en=(
                f"Need human judgement for: {rule.description_en} "
                f"({rule.source_clause})"
            ),
            reason_hi=rule.description_hi,
            missing_fields=[rule.field],
        )

    # Missing field → UNKNOWN → NEEDS_MORE_INFO.
    # We use `in` not `.get()` so that an explicit None counts as present.
    if rule.field not in profile:
        return EligibilityResult(
            verdict=Verdict.NEEDS_MORE_INFO,
            rule=rule,
            reason_en=(
                f"Need to know: {rule.description_en} "
                f"(missing field: '{rule.field}')"
            ),
            reason_hi=rule.description_hi,
            missing_fields=[rule.field],
        )

    # Apply the operator.
    field_value = profile[rule.field]
    try:
        condition_holds = _apply_operator(rule.operator, field_value, rule.value)
    except TypeError as exc:
        # Type mismatch in comparison (e.g. user sent a string where int expected).
        return EligibilityResult(
            verdict=Verdict.NEEDS_MORE_INFO,
            rule=rule,
            reason_en=(
                f"Could not evaluate '{rule.field}': unexpected value type "
                f"({exc}). Please re-enter the value."
            ),
            missing_fields=[rule.field],
        )

    if condition_holds:
        verdict = rule.verdict_if_true
        # Build the reason using the project's safe wording policy.
        if verdict == Verdict.ELIGIBLE:
            prefix = "Appears eligible based on published criteria"
        elif verdict == Verdict.NOT_ELIGIBLE:
            prefix = "Not eligible"
        else:
            prefix = "More information needed"

        reason_en = f"{prefix}: {rule.description_en} ({rule.source_clause})"
        return EligibilityResult(
            verdict=verdict,
            rule=rule,
            reason_en=reason_en,
            reason_hi=rule.description_hi,
        )
    else:
        # Condition did not hold → flip the verdict.
        flipped = (
            Verdict.NOT_ELIGIBLE
            if rule.verdict_if_true == Verdict.ELIGIBLE
            else Verdict.ELIGIBLE
        )
        if flipped == Verdict.ELIGIBLE:
            prefix = "Appears eligible based on published criteria"
        else:
            prefix = "Not eligible"

        reason_en = f"{prefix}: {rule.description_en} ({rule.source_clause})"
        return EligibilityResult(
            verdict=flipped,
            rule=rule,
            reason_en=reason_en,
            reason_hi=rule.description_hi,
        )


# ---------------------------------------------------------------------------
# Whole-scheme evaluation
# ---------------------------------------------------------------------------

def evaluate_scheme(
    rules: list[Rule],
    profile: dict[str, Any],
) -> list[EligibilityResult]:
    """Evaluate all approved rules for a scheme against a profile.

    Skips rules with status != "approved" silently (they are simply not ready).
    The caller decides aggregation strategy:
      - Collect all NEEDS_MORE_INFO → ask the user for missing fields.
      - Any NOT_ELIGIBLE → scheme is excluded.
      - All ELIGIBLE → scheme appears eligible.

    Parameters
    ----------
    rules:
        List of Rule objects for one scheme (may include pending ones).
    profile:
        User profile dict.

    Returns
    -------
    List of EligibilityResult, one per approved rule.
    """
    results: list[EligibilityResult] = []
    for rule in rules:
        if not rule.is_approved():
            continue  # skip pending rules quietly
        results.append(evaluate_rule(rule, profile))
    return results
