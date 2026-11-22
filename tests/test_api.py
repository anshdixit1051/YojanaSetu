"""tests/test_api.py – FastAPI endpoint tests.

Uses httpx.AsyncClient with TestClient — no real server needed.
LLM calls are mocked so no API key is required.

Run with: pytest tests/test_api.py -v
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from yojanasetu.api.app import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_lists_schemes(self):
        resp = client.get("/health")
        schemes = resp.json()["schemes"]
        assert isinstance(schemes, list)
        assert "pm-kisan" in schemes

    def test_health_includes_all_five_schemes(self):
        resp = client.get("/health")
        schemes = resp.json()["schemes"]
        expected = {
            "pm-kisan", "ayushman-bharat-pmjay",
            "nsp-post-matric-sc", "pm-svanidhi", "up-kanya-sumangala",
        }
        assert expected.issubset(set(schemes))


# ---------------------------------------------------------------------------
# POST /eligibility  (direct engine, no LLM)
# ---------------------------------------------------------------------------

class TestEligibility:
    def test_eligible_profile_returns_eligible(self):
        resp = client.post("/eligibility", json={
            "scheme_id": "pm-kisan",
            "profile": {
                "owns_cultivable_land": True,
                "is_institutional_landholder": False,
                "paid_income_tax_last_year": False,
                "holds_constitutional_post": False,
                "monthly_pension_amount": 0,
            },
        })
        assert resp.status_code == 200
        data = resp.json()
        verdicts = {r["verdict"] for r in data["results"]}
        assert "ELIGIBLE" in verdicts
        assert "NOT_ELIGIBLE" not in verdicts

    def test_excluded_taxpayer_returns_not_eligible(self):
        resp = client.post("/eligibility", json={
            "scheme_id": "pm-kisan",
            "profile": {
                "owns_cultivable_land": True,
                "is_institutional_landholder": False,
                "paid_income_tax_last_year": True,
                "holds_constitutional_post": False,
                "monthly_pension_amount": 0,
            },
        })
        assert resp.status_code == 200
        verdicts = {r["verdict"] for r in resp.json()["results"]}
        assert "NOT_ELIGIBLE" in verdicts

    def test_empty_profile_returns_needs_more_info(self):
        resp = client.post("/eligibility", json={
            "scheme_id": "pm-kisan",
            "profile": {},
        })
        assert resp.status_code == 200
        verdicts = {r["verdict"] for r in resp.json()["results"]}
        assert verdicts == {"NEEDS_MORE_INFO"}

    def test_unknown_scheme_returns_404(self):
        resp = client.post("/eligibility", json={
            "scheme_id": "no-such-scheme",
            "profile": {},
        })
        assert resp.status_code == 404

    def test_results_contain_source_clause(self):
        resp = client.post("/eligibility", json={
            "scheme_id": "pm-kisan",
            "profile": {"owns_cultivable_land": True},
        })
        for r in resp.json()["results"]:
            assert "source_clause" in r

    def test_results_contain_rule_id(self):
        resp = client.post("/eligibility", json={
            "scheme_id": "pm-kisan",
            "profile": {},
        })
        for r in resp.json()["results"]:
            assert "rule_id" in r
            assert r["rule_id"].startswith("pm-kisan")


# ---------------------------------------------------------------------------
# POST /chat  (agent – LLM mocked)
# ---------------------------------------------------------------------------

class TestChat:
    def _fake_llm(self, *args, **kwargs):
        """Return plausible fake LLM output alternately for intake and respond."""
        return '{"owns_cultivable_land": true}'

    def test_chat_returns_session_id(self):
        fake_facts = json.dumps({"owns_cultivable_land": True})
        fake_reply = "You appear eligible based on the land rule."
        with patch("yojanasetu.agent.nodes._llm_call", side_effect=[fake_facts, fake_reply]):
            resp = client.post("/chat", json={
                "message": "I own land in UP",
                "scheme_ids": ["pm-kisan"],
            })
        assert resp.status_code == 200
        assert resp.json()["session_id"]

    def test_chat_returns_reply(self):
        fake_facts = json.dumps({})
        fake_reply = "Could you tell me if you own cultivable land?"
        with patch("yojanasetu.agent.nodes._llm_call", side_effect=[fake_facts, fake_reply]):
            resp = client.post("/chat", json={
                "message": "hello",
                "scheme_ids": ["pm-kisan"],
            })
        assert resp.status_code == 200
        assert resp.json()["reply"]

    def test_chat_session_persists_profile(self):
        """Second turn should inherit profile from first turn."""
        fake1 = json.dumps({"owns_cultivable_land": True})
        fake_reply1 = "Got it. Do you pay income tax?"
        fake2 = json.dumps({"paid_income_tax_last_year": False})
        fake_reply2 = "You appear eligible."

        with patch("yojanasetu.agent.nodes._llm_call",
                   side_effect=[fake1, fake_reply1]):
            resp1 = client.post("/chat", json={
                "message": "I own land",
                "scheme_ids": ["pm-kisan"],
            })
        session_id = resp1.json()["session_id"]

        with patch("yojanasetu.agent.nodes._llm_call",
                   side_effect=[fake2, fake_reply2]):
            resp2 = client.post("/chat", json={
                "message": "No I don't pay tax",
                "session_id": session_id,
                "scheme_ids": ["pm-kisan"],
            })

        assert resp2.status_code == 200
        # Profile should have both keys now.
        # (We can't directly inspect session here, but results should have changed.)
        assert resp2.json()["results"]
