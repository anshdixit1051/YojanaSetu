"""tests/test_engine_models.py – Unit tests for engine/models.py.

Run with: pytest tests/test_engine_models.py -v
"""

import pytest

from yojanasetu.engine.models import EligibilityResult, Rule, Verdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_rule(**overrides) -> Rule:
    """Return a minimal valid approved Rule, with optional overrides."""
    defaults = dict(
        rule_id="pm-kisan-land-owner",
        scheme_id="pm-kisan",
        field="owns_cultivable_land",
        operator="eq",
        value=True,
        verdict_if_true=Verdict.ELIGIBLE,
        description_en="Applicant must own cultivable land.",
        source_clause="Section 2 – Definition of Farmer Family",
        status="approved",
    )
    defaults.update(overrides)
    return Rule(**defaults)


# ---------------------------------------------------------------------------
# Verdict enum
# ---------------------------------------------------------------------------

def test_verdict_values_are_strings():
    # Verdict inherits from str so values are JSON-serialisable without extra work.
    assert Verdict.ELIGIBLE == "ELIGIBLE"
    assert Verdict.NOT_ELIGIBLE == "NOT_ELIGIBLE"
    assert Verdict.NEEDS_MORE_INFO == "NEEDS_MORE_INFO"


def test_verdict_has_exactly_three_members():
    assert len(Verdict) == 3


# ---------------------------------------------------------------------------
# Rule
# ---------------------------------------------------------------------------

def test_approved_rule_is_approved():
    rule = make_rule(status="approved")
    assert rule.is_approved() is True


def test_pending_rule_is_not_approved():
    rule = make_rule(status="pending_review")
    assert rule.is_approved() is False


def test_requires_human_judgement_defaults_to_false():
    rule = make_rule()
    assert rule.requires_human_judgement is False


def test_rule_is_frozen():
    """Rules must be immutable so they cannot be tampered with at runtime."""
    rule = make_rule()
    with pytest.raises((AttributeError, TypeError)):
        rule.field = "something_else"  # type: ignore[misc]


def test_rule_description_hi_defaults_to_empty_string():
    rule = make_rule()
    assert rule.description_hi == ""


# ---------------------------------------------------------------------------
# EligibilityResult
# ---------------------------------------------------------------------------

def test_eligible_result_as_dict():
    rule = make_rule()
    result = EligibilityResult(
        verdict=Verdict.ELIGIBLE,
        rule=rule,
        reason_en="Applicant appears eligible: owns cultivable land.",
    )
    d = result.as_dict()
    assert d["verdict"] == "ELIGIBLE"
    assert d["rule_id"] == "pm-kisan-land-owner"
    assert d["scheme_id"] == "pm-kisan"
    assert d["missing_fields"] == []


def test_needs_more_info_result_carries_missing_fields():
    rule = make_rule(field="land_holding_hectares")
    result = EligibilityResult(
        verdict=Verdict.NEEDS_MORE_INFO,
        rule=rule,
        reason_en="Need to know: land holding area in hectares.",
        missing_fields=["land_holding_hectares"],
    )
    d = result.as_dict()
    assert d["verdict"] == "NEEDS_MORE_INFO"
    assert "land_holding_hectares" in d["missing_fields"]


def test_not_eligible_result_as_dict():
    rule = make_rule(
        verdict_if_true=Verdict.NOT_ELIGIBLE,
        description_en="Income tax payers are excluded.",
        source_clause="Section 5(ii)(e)",
    )
    result = EligibilityResult(
        verdict=Verdict.NOT_ELIGIBLE,
        rule=rule,
        reason_en="Not eligible: paid income tax in the last assessment year.",
    )
    d = result.as_dict()
    assert d["verdict"] == "NOT_ELIGIBLE"
    assert d["source_clause"] == "Section 5(ii)(e)"


def test_missing_fields_defaults_to_empty_list():
    rule = make_rule()
    result = EligibilityResult(
        verdict=Verdict.ELIGIBLE,
        rule=rule,
        reason_en="Appears eligible.",
    )
    assert result.missing_fields == []


def test_reason_hi_defaults_to_empty_string():
    rule = make_rule()
    result = EligibilityResult(
        verdict=Verdict.ELIGIBLE,
        rule=rule,
        reason_en="Appears eligible.",
    )
    assert result.reason_hi == ""
