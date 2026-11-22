"""tests/test_agent_safety.py – The LLM must not be able to change a verdict.

Every LLM call is mocked.  These tests are the proof behind the viva claim
"the LLM never decides eligibility".

Run with: pytest tests/test_agent_safety.py -v
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from yojanasetu.agent.nodes import evaluate_node, intake_node, respond_node
from yojanasetu.agent.state import AgentState
from yojanasetu.agent.summary import format_verdict_summary, is_safe_question

FULL_ELIGIBLE = {
    "owns_cultivable_land": True,
    "is_institutional_landholder": False,
    "paid_income_tax_last_year": False,
    "holds_constitutional_post": False,
    "monthly_pension_amount": 0,
}
LLM = "yojanasetu.agent.nodes._llm_call"


def make_state(**overrides) -> AgentState:
    state = AgentState(
        messages=[{"role": "user", "content": "hello"}],
        profile={},
        results=[],
        scheme_verdicts=[],
        missing_fields=[],
        scheme_ids=["pm-kisan"],
        language="en",
        done=False,
    )
    state.update(overrides)
    return state


def evaluated(**overrides) -> AgentState:
    """A state after evaluate_node ran on the given profile."""
    state = make_state(**overrides)
    state.update(evaluate_node(state))
    return state


class TestEvaluateNode:
    def test_scheme_verdict_is_eligible_for_full_profile(self):
        state = evaluated(profile=FULL_ELIGIBLE)
        assert [v["verdict"] for v in state["scheme_verdicts"]] == ["ELIGIBLE"]

    def test_scheme_verdict_is_not_eligible_for_taxpayer(self):
        state = evaluated(profile={**FULL_ELIGIBLE, "paid_income_tax_last_year": True})
        verdict = state["scheme_verdicts"][0]
        assert verdict["verdict"] == "NOT_ELIGIBLE"
        assert verdict["deciding_rules"][0]["rule_id"] == "pm-kisan-no-income-tax"


class TestLlmCannotChangeVerdicts:
    def test_eligible_reply_is_engine_text_even_if_llm_disagrees(self):
        state = evaluated(profile=FULL_ELIGIBLE)
        with patch(LLM, return_value="Sorry, you are NOT eligible.") as llm:
            reply = respond_node(state)["messages"][0]["content"]
        assert "Appears eligible based on published criteria" in reply
        assert "NOT eligible" not in reply
        llm.assert_not_called()  # nothing left to ask, so no LLM involved at all

    def test_not_eligible_reply_cites_rule_and_clause(self):
        state = evaluated(profile={**FULL_ELIGIBLE, "paid_income_tax_last_year": True})
        reply = respond_node(state)["messages"][0]["content"]
        assert "Not eligible" in reply
        assert "Section 5(ii)(e)" in reply

    def test_llm_text_claiming_a_verdict_is_replaced_by_fixed_question(self):
        state = evaluated(profile={"owns_cultivable_land": True})
        with patch(LLM, return_value="Great news, you are eligible!"):
            reply = respond_node(state)["messages"][0]["content"]
        assert "Great news" not in reply
        assert "To continue, I need this detail" in reply

    def test_safe_llm_question_is_used(self):
        state = evaluated(profile={"owns_cultivable_land": True})
        with patch(LLM, return_value="Is the land held by an institution?"):
            reply = respond_node(state)["messages"][0]["content"]
        assert "Is the land held by an institution?" in reply

    def test_llm_failure_never_leaks_error_text(self):
        state = evaluated(profile={"owns_cultivable_land": True})
        with patch(LLM, side_effect=RuntimeError("secret-key-123 exploded")):
            reply = respond_node(state)["messages"][0]["content"]
        assert "secret-key-123" not in reply
        assert "Error" not in reply
        assert "To continue, I need this detail" in reply

    def test_done_only_when_nothing_is_missing(self):
        assert respond_node(evaluated(profile=FULL_ELIGIBLE))["done"] is True
        with patch(LLM, return_value="Do you pay income tax?"):
            assert respond_node(evaluated(profile={}))["done"] is False


class TestSummary:
    def test_hindi_summary_for_eligible(self):
        state = evaluated(profile=FULL_ELIGIBLE, language="hi")
        text = format_verdict_summary(state["scheme_verdicts"], "hi")
        assert "पात्र प्रतीत होते हैं" in text

    def test_no_schemes_message(self):
        assert "could not find" in format_verdict_summary([], "en")

    def test_is_safe_question(self):
        assert is_safe_question("Do you own farmland?")
        assert not is_safe_question("You are eligible.")
        assert not is_safe_question("Do you qualify?")
        assert not is_safe_question("")
        assert not is_safe_question("x" * 400)


class TestIntakeValidation:
    def test_unknown_fields_and_wrong_types_are_dropped(self):
        facts = {"owns_cultivable_land": True, "made_up": 5, "monthly_pension_amount": "lots"}
        with patch(LLM, return_value=json.dumps(facts)):
            profile = intake_node(make_state())["profile"]
        assert profile == {"owns_cultivable_land": True}

    def test_markdown_fenced_json_is_accepted(self):
        fenced = '```json\n{"owns_cultivable_land": true}\n```'
        with patch(LLM, return_value=fenced):
            assert intake_node(make_state())["profile"] == {"owns_cultivable_land": True}

    def test_llm_failure_keeps_existing_profile(self):
        with patch(LLM, side_effect=RuntimeError("network down")):
            result = intake_node(make_state(profile={"monthly_pension_amount": 0}))
        assert result["profile"] == {"monthly_pension_amount": 0}

    def test_previous_assistant_message_is_given_as_context(self):
        state = make_state(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Do you pay income tax?"},
                {"role": "user", "content": "no"},
            ]
        )
        mock = MagicMock(return_value=json.dumps({"paid_income_tax_last_year": False}))
        with patch(LLM, mock):
            profile = intake_node(state)["profile"]
        assert "Do you pay income tax?" in mock.call_args[0][1]
        assert profile == {"paid_income_tax_last_year": False}

    def test_prompt_lists_allowed_fields(self):
        mock = MagicMock(return_value="{}")
        with patch(LLM, mock):
            intake_node(make_state())
        assert '"owns_cultivable_land"' in mock.call_args[0][0]
