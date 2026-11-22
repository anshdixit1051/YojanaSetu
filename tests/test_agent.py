"""tests/test_agent.py – Tests for agent state, nodes, and graph.

All LLM calls are mocked.  No API key required.

Run with: pytest tests/test_agent.py -v
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from yojanasetu.agent.state import AgentState
from yojanasetu.agent.nodes import evaluate_node, should_continue
from yojanasetu.agent.graph import initial_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(**overrides) -> AgentState:
    defaults = AgentState(
        messages=[{"role": "user", "content": "I am a farmer in UP"}],
        profile={},
        results=[],
        missing_fields=[],
        scheme_ids=["pm-kisan"],
        language="en",
        done=False,
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class TestAgentState:
    def test_state_has_required_keys(self):
        state = make_state()
        for key in ("messages", "profile", "results", "missing_fields",
                    "scheme_ids", "language", "done"):
            assert key in state

    def test_initial_state_helper_adds_user_message(self):
        state = initial_state("hello", ["pm-kisan"])
        assert state["messages"][-1] == {"role": "user", "content": "hello"}

    def test_initial_state_merges_history(self):
        history = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "hi"}]
        state = initial_state("second", ["pm-kisan"], history=history)
        assert len(state["messages"]) == 3

    def test_initial_state_merges_profile(self):
        state = initial_state("hello", ["pm-kisan"], existing_profile={"age": 30})
        assert state["profile"]["age"] == 30


# ---------------------------------------------------------------------------
# evaluate_node – the safety-critical node (no LLM)
# ---------------------------------------------------------------------------

class TestEvaluateNode:
    def test_fully_eligible_profile_returns_eligible_results(self):
        state = make_state(
            profile={
                "owns_cultivable_land": True,
                "is_institutional_landholder": False,
                "paid_income_tax_last_year": False,
                "holds_constitutional_post": False,
                "monthly_pension_amount": 0,
            },
            scheme_ids=["pm-kisan"],
        )
        result = evaluate_node(state)
        assert "results" in result
        verdicts = {r["verdict"] for r in result["results"]}
        assert "ELIGIBLE" in verdicts
        assert "NOT_ELIGIBLE" not in verdicts

    def test_empty_profile_returns_all_needs_more_info(self):
        state = make_state(profile={}, scheme_ids=["pm-kisan"])
        result = evaluate_node(state)
        verdicts = {r["verdict"] for r in result["results"]}
        assert verdicts == {"NEEDS_MORE_INFO"}

    def test_missing_fields_populated_on_needs_more_info(self):
        state = make_state(profile={}, scheme_ids=["pm-kisan"])
        result = evaluate_node(state)
        assert len(result["missing_fields"]) > 0

    def test_exclusion_rule_returns_not_eligible(self):
        state = make_state(
            profile={
                "owns_cultivable_land": True,
                "is_institutional_landholder": False,
                "paid_income_tax_last_year": True,   # excluded
                "holds_constitutional_post": False,
                "monthly_pension_amount": 0,
            },
            scheme_ids=["pm-kisan"],
        )
        result = evaluate_node(state)
        verdicts = [r["verdict"] for r in result["results"]]
        assert "NOT_ELIGIBLE" in verdicts

    def test_missing_scheme_is_skipped_gracefully(self):
        state = make_state(profile={"age": 30}, scheme_ids=["no-such-scheme"])
        result = evaluate_node(state)
        assert result["results"] == []
        assert result["missing_fields"] == []

    def test_results_contain_source_clause(self):
        state = make_state(
            profile={"owns_cultivable_land": True},
            scheme_ids=["pm-kisan"],
        )
        result = evaluate_node(state)
        for r in result["results"]:
            assert "source_clause" in r
            assert r["source_clause"]  # non-empty


# ---------------------------------------------------------------------------
# intake_node – LLM mocked
# ---------------------------------------------------------------------------

class TestIntakeNode:
    def test_extracts_facts_from_user_message(self):
        from yojanasetu.agent.nodes import intake_node  # noqa: PLC0415

        fake_facts = json.dumps({"owns_cultivable_land": True, "paid_income_tax_last_year": False})
        with patch("yojanasetu.agent.nodes._llm_call", return_value=fake_facts):
            state = make_state(messages=[{"role": "user", "content": "I own land and do not pay tax"}])
            result = intake_node(state)

        assert result["profile"]["owns_cultivable_land"] is True
        assert result["profile"]["paid_income_tax_last_year"] is False

    def test_merges_with_existing_profile(self):
        from yojanasetu.agent.nodes import intake_node  # noqa: PLC0415

        new_facts = json.dumps({"owns_cultivable_land": True})
        with patch("yojanasetu.agent.nodes._llm_call", return_value=new_facts):
            state = make_state(
                profile={"age": 35},
                messages=[{"role": "user", "content": "I own land"}],
            )
            result = intake_node(state)

        assert result["profile"]["age"] == 35          # preserved
        assert result["profile"]["owns_cultivable_land"] is True  # added

    def test_bad_llm_response_returns_empty_profile(self):
        from yojanasetu.agent.nodes import intake_node  # noqa: PLC0415

        with patch("yojanasetu.agent.nodes._llm_call", return_value="not json {{{}"):
            state = make_state()
            result = intake_node(state)

        # Should not crash; profile stays unchanged.
        assert isinstance(result.get("profile", {}), dict)

    def test_no_messages_returns_empty(self):
        from yojanasetu.agent.nodes import intake_node  # noqa: PLC0415

        state = make_state(messages=[])
        result = intake_node(state)
        assert result == {}


# ---------------------------------------------------------------------------
# should_continue router
# ---------------------------------------------------------------------------

class TestShouldContinue:
    def test_missing_fields_returns_ask(self):
        state = make_state(missing_fields=["age"], results=[{"verdict": "NEEDS_MORE_INFO"}])
        assert should_continue(state) == "ask"

    def test_no_missing_fields_with_results_returns_end(self):
        state = make_state(missing_fields=[], results=[{"verdict": "ELIGIBLE"}])
        assert should_continue(state) == "end"

    def test_no_results_returns_ask(self):
        state = make_state(missing_fields=[], results=[])
        assert should_continue(state) == "ask"


# ---------------------------------------------------------------------------
# build_graph (structural smoke test)
# ---------------------------------------------------------------------------

class TestBuildGraph:
    def test_graph_compiles_without_error(self):
        from yojanasetu.agent.graph import build_graph  # noqa: PLC0415
        graph = build_graph()
        assert graph is not None

    def test_graph_invoke_with_mocked_llm(self):
        from yojanasetu.agent.graph import build_graph, initial_state  # noqa: PLC0415

        fake_facts = json.dumps({"owns_cultivable_land": True})
        fake_reply = "You appear eligible based on the land ownership rule."

        with patch("yojanasetu.agent.nodes._llm_call", side_effect=[fake_facts, fake_reply]):
            graph = build_graph()
            state = initial_state(
                user_message="I own land and don't pay income tax",
                scheme_ids=["pm-kisan"],
                existing_profile={
                    "is_institutional_landholder": False,
                    "paid_income_tax_last_year": False,
                    "holds_constitutional_post": False,
                    "monthly_pension_amount": 0,
                },
            )
            final = graph.invoke(state)

        assert len(final["messages"]) >= 2
        assert any(m["role"] == "assistant" for m in final["messages"])
        assert len(final["results"]) > 0
