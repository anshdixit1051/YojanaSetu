"""tests/test_aggregate.py – Scheme-level verdict (the "do you qualify?" answer).

This is the most safety-critical logic in the project: it turns many per-rule
results into one verdict.  It must be conservative and never say ELIGIBLE on
missing, unknown or empty information.

Run with: pytest tests/test_aggregate.py -v
"""

from __future__ import annotations

import json
from typing import Any

from yojanasetu.engine.aggregate import evaluate_scheme_verdict
from yojanasetu.engine.loader import load_approved_rules
from yojanasetu.engine.models import Rule, Verdict


def make_rule(
    rule_id: str = "s-r1",
    field_: str = "age",
    op: str = "gte",
    value: Any = 18,
    verdict: Verdict = Verdict.ELIGIBLE,
    human: bool = False,
    status: str = "approved",
) -> Rule:
    return Rule(
        rule_id=rule_id,
        scheme_id="s",
        field=field_,
        operator=op,
        value=value,
        verdict_if_true=verdict,
        description_en=f"desc {rule_id}",
        source_clause=f"Clause {rule_id}",
        requires_human_judgement=human,
        status=status,
    )


class TestPolicy:
    def test_all_rules_eligible_gives_eligible(self):
        rules = [make_rule("s-a", "age", "gte", 18), make_rule("s-b", "owns", "eq", True)]
        v = evaluate_scheme_verdict("s", rules, {"age": 30, "owns": True})
        assert v.verdict == Verdict.ELIGIBLE
        assert len(v.deciding_rules) == 2

    def test_one_failing_rule_gives_not_eligible_even_if_others_missing(self):
        rules = [make_rule("s-a", "age", "gte", 18), make_rule("s-b", "owns", "eq", True)]
        v = evaluate_scheme_verdict("s", rules, {"age": 10})  # 'owns' unknown
        assert v.verdict == Verdict.NOT_ELIGIBLE
        assert [r.rule.rule_id for r in v.deciding_rules] == ["s-a"]
        assert v.missing_fields == []  # no point asking more questions

    def test_exclusion_rule_triggers_not_eligible(self):
        rules = [make_rule("s-tax", "pays_tax", "eq", True, verdict=Verdict.NOT_ELIGIBLE)]
        v = evaluate_scheme_verdict("s", rules, {"pays_tax": True})
        assert v.verdict == Verdict.NOT_ELIGIBLE

    def test_exclusion_rule_not_triggered_gives_eligible(self):
        rules = [make_rule("s-tax", "pays_tax", "eq", True, verdict=Verdict.NOT_ELIGIBLE)]
        v = evaluate_scheme_verdict("s", rules, {"pays_tax": False})
        assert v.verdict == Verdict.ELIGIBLE

    def test_missing_field_gives_needs_more_info_with_fields_listed(self):
        rules = [make_rule("s-a", "age", "gte", 18), make_rule("s-b", "owns", "eq", True)]
        v = evaluate_scheme_verdict("s", rules, {"age": 30})
        assert v.verdict == Verdict.NEEDS_MORE_INFO
        assert v.missing_fields == ["owns"]

    def test_empty_profile_never_eligible(self):
        rules = [make_rule("s-a", "age", "gte", 18)]
        assert evaluate_scheme_verdict("s", rules, {}).verdict == Verdict.NEEDS_MORE_INFO

    def test_human_judgement_rule_blocks_eligible(self):
        rules = [make_rule("s-a", "age", "gte", 18), make_rule("s-h", "x", "eq", True, human=True)]
        v = evaluate_scheme_verdict("s", rules, {"age": 30, "x": True})
        assert v.verdict == Verdict.NEEDS_MORE_INFO
        assert v.needs_human_review == ["s-h"]
        assert v.missing_fields == []  # the user cannot answer a discretionary rule

    def test_no_rules_never_eligible(self):
        v = evaluate_scheme_verdict("s", [], {"age": 30})
        assert v.verdict == Verdict.NEEDS_MORE_INFO
        assert "No approved rules" in v.note

    def test_pending_rules_are_ignored(self):
        rules = [make_rule("s-a", "age", "gte", 18, status="pending_review")]
        assert evaluate_scheme_verdict("s", rules, {"age": 30}).verdict == Verdict.NEEDS_MORE_INFO

    def test_as_dict_is_json_serialisable(self):
        rules = [make_rule("s-a", "age", "gte", 18)]
        data = evaluate_scheme_verdict("s", rules, {"age": 30}).as_dict()
        assert json.loads(json.dumps(data))["verdict"] == "ELIGIBLE"


class TestRealData:
    FULL = {
        "owns_cultivable_land": True,
        "is_institutional_landholder": False,
        "paid_income_tax_last_year": False,
        "holds_constitutional_post": False,
        "monthly_pension_amount": 0,
    }

    def test_pm_kisan_fully_eligible(self):
        v = evaluate_scheme_verdict("pm-kisan", load_approved_rules("pm-kisan"), self.FULL)
        assert v.verdict == Verdict.ELIGIBLE

    def test_pm_kisan_taxpayer_is_not_eligible_and_cites_rule(self):
        profile = {**self.FULL, "paid_income_tax_last_year": True}
        v = evaluate_scheme_verdict("pm-kisan", load_approved_rules("pm-kisan"), profile)
        assert v.verdict == Verdict.NOT_ELIGIBLE
        assert [r.rule.rule_id for r in v.deciding_rules] == ["pm-kisan-no-income-tax"]

    def test_pm_kisan_empty_profile_needs_more_info(self):
        v = evaluate_scheme_verdict("pm-kisan", load_approved_rules("pm-kisan"), {})
        assert v.verdict == Verdict.NEEDS_MORE_INFO
        assert "owns_cultivable_land" in v.missing_fields
