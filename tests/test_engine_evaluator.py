"""tests/test_engine_evaluator.py – Tests for engine/evaluator.py.

Run with: pytest tests/test_engine_evaluator.py -v
"""

import pytest

from yojanasetu.engine.evaluator import evaluate_rule, evaluate_scheme
from yojanasetu.engine.models import EligibilityResult, Rule, Verdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def approved_rule(**overrides) -> Rule:
    """Return a minimal approved Rule with optional field overrides."""
    defaults = dict(
        rule_id="pm-kisan-land",
        scheme_id="pm-kisan",
        field="owns_cultivable_land",
        operator="eq",
        value=True,
        verdict_if_true=Verdict.ELIGIBLE,
        description_en="Must own cultivable land.",
        source_clause="Section 2",
        status="approved",
    )
    defaults.update(overrides)
    return Rule(**defaults)


# ---------------------------------------------------------------------------
# Safety gates
# ---------------------------------------------------------------------------

def test_pending_rule_raises_runtime_error():
    """Engine must refuse to run unapproved rules."""
    rule = approved_rule(status="pending_review")
    with pytest.raises(RuntimeError, match="approved"):
        evaluate_rule(rule, {"owns_cultivable_land": True})


def test_requires_human_judgement_always_needs_more_info():
    """Discretionary criteria must never produce ELIGIBLE or NOT_ELIGIBLE."""
    rule = approved_rule(requires_human_judgement=True)
    result = evaluate_rule(rule, {"owns_cultivable_land": True})
    assert result.verdict == Verdict.NEEDS_MORE_INFO
    assert rule.field in result.missing_fields


# ---------------------------------------------------------------------------
# Missing field → NEEDS_MORE_INFO  (the core three-valued-logic safety rule)
# ---------------------------------------------------------------------------

def test_missing_field_gives_needs_more_info():
    rule = approved_rule(field="age")
    result = evaluate_rule(rule, {})   # age not in profile
    assert result.verdict == Verdict.NEEDS_MORE_INFO
    assert "age" in result.missing_fields


def test_field_present_as_none_is_not_missing():
    """An explicit None means the user answered 'unknown' – treat as present."""
    rule = approved_rule(field="age", operator="gte", value=18)
    # None < 18 raises TypeError → NEEDS_MORE_INFO, but field IS present.
    result = evaluate_rule(rule, {"age": None})
    # Should not say "missing field: age" – instead a type-error NEEDS_MORE_INFO.
    assert result.verdict == Verdict.NEEDS_MORE_INFO
    # The reason should mention type mismatch, not "missing field"
    assert "missing" not in result.reason_en.lower() or "age" not in result.missing_fields


# ---------------------------------------------------------------------------
# Operators – condition holds → verdict_if_true
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("operator,field_value,rule_value", [
    ("eq",     True,  True),
    ("neq",    False, True),
    ("gt",     30,    18),
    ("gte",    18,    18),
    ("lt",     17,    18),
    ("lte",    18,    18),
    ("in",     "UP",  ["UP", "MH", "RJ"]),
    ("not_in", "GJ",  ["UP", "MH", "RJ"]),
    ("exists", "any", None),
])
def test_condition_holds_returns_verdict_if_true(operator, field_value, rule_value):
    rule = approved_rule(
        field="x",
        operator=operator,
        value=rule_value,
        verdict_if_true=Verdict.ELIGIBLE,
    )
    result = evaluate_rule(rule, {"x": field_value})
    assert result.verdict == Verdict.ELIGIBLE


# ---------------------------------------------------------------------------
# Operators – condition does NOT hold → verdict flipped
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("operator,field_value,rule_value", [
    ("eq",     False, True),
    ("neq",    True,  True),
    ("gt",     17,    18),
    ("gte",    17,    18),
    ("lt",     18,    18),
    ("lte",    19,    18),
    ("in",     "GJ",  ["UP", "MH"]),
    ("not_in", "UP",  ["UP", "MH"]),
])
def test_condition_fails_flips_verdict(operator, field_value, rule_value):
    rule = approved_rule(
        field="x",
        operator=operator,
        value=rule_value,
        verdict_if_true=Verdict.ELIGIBLE,
    )
    result = evaluate_rule(rule, {"x": field_value})
    assert result.verdict == Verdict.NOT_ELIGIBLE


# ---------------------------------------------------------------------------
# NOT_ELIGIBLE verdict_if_true → flips to ELIGIBLE when condition fails
# ---------------------------------------------------------------------------

def test_not_eligible_verdict_if_true_flips_to_eligible():
    """Exclusion rule: income-tax payer → NOT_ELIGIBLE.
    If user does NOT pay income tax, they should flip to ELIGIBLE."""
    rule = approved_rule(
        field="paid_income_tax",
        operator="eq",
        value=True,
        verdict_if_true=Verdict.NOT_ELIGIBLE,
        description_en="Income tax payers are excluded.",
        source_clause="Section 5(ii)(e)",
    )
    # User did NOT pay income tax → condition fails → flipped to ELIGIBLE
    result = evaluate_rule(rule, {"paid_income_tax": False})
    assert result.verdict == Verdict.ELIGIBLE


def test_not_eligible_verdict_if_true_holds():
    """Income-tax payer → NOT_ELIGIBLE (condition holds)."""
    rule = approved_rule(
        field="paid_income_tax",
        operator="eq",
        value=True,
        verdict_if_true=Verdict.NOT_ELIGIBLE,
        description_en="Income tax payers are excluded.",
        source_clause="Section 5(ii)(e)",
    )
    result = evaluate_rule(rule, {"paid_income_tax": True})
    assert result.verdict == Verdict.NOT_ELIGIBLE


# ---------------------------------------------------------------------------
# Reason wording (safe language policy)
# ---------------------------------------------------------------------------

def test_eligible_reason_uses_appears_eligible():
    rule = approved_rule(verdict_if_true=Verdict.ELIGIBLE)
    result = evaluate_rule(rule, {"owns_cultivable_land": True})
    assert "appears eligible" in result.reason_en.lower()


def test_not_eligible_reason_uses_not_eligible():
    rule = approved_rule(
        verdict_if_true=Verdict.NOT_ELIGIBLE,
        description_en="Institutional landholders excluded.",
        source_clause="Section 5(i)",
    )
    result = evaluate_rule(rule, {"owns_cultivable_land": True})
    assert "not eligible" in result.reason_en.lower()


# ---------------------------------------------------------------------------
# Unknown operator
# ---------------------------------------------------------------------------

def test_unknown_operator_raises_value_error():
    rule = approved_rule(operator="between")  # not supported
    with pytest.raises(ValueError, match="Unknown operator"):
        evaluate_rule(rule, {"owns_cultivable_land": True})


# ---------------------------------------------------------------------------
# evaluate_scheme
# ---------------------------------------------------------------------------

def test_evaluate_scheme_skips_pending_rules():
    rules = [
        approved_rule(rule_id="r1", status="approved"),
        approved_rule(rule_id="r2", status="pending_review"),
    ]
    results = evaluate_scheme(rules, {"owns_cultivable_land": True})
    assert len(results) == 1
    assert results[0].rule.rule_id == "r1"


def test_evaluate_scheme_returns_all_approved_results():
    rules = [
        approved_rule(rule_id="r1", field="a", operator="eq", value=1,
                      verdict_if_true=Verdict.ELIGIBLE),
        approved_rule(rule_id="r2", field="b", operator="eq", value=2,
                      verdict_if_true=Verdict.ELIGIBLE),
    ]
    results = evaluate_scheme(rules, {"a": 1, "b": 2})
    assert len(results) == 2
    assert all(r.verdict == Verdict.ELIGIBLE for r in results)


def test_evaluate_scheme_empty_rules_returns_empty():
    assert evaluate_scheme([], {"age": 30}) == []


def test_evaluate_scheme_missing_fields_collected():
    rules = [
        approved_rule(rule_id="r1", field="age"),
        approved_rule(rule_id="r2", field="state"),
    ]
    results = evaluate_scheme(rules, {})  # both fields missing
    assert all(r.verdict == Verdict.NEEDS_MORE_INFO for r in results)
    missing = [f for r in results for f in r.missing_fields]
    assert "age" in missing
    assert "state" in missing
