"""tests/test_api_hardening.py – Input validation, session limits and verdict fields.

Run with: pytest tests/test_api_hardening.py -v
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import yojanasetu.api.app as app_module
from yojanasetu.api.app import app

client = TestClient(app)

FULL_ELIGIBLE = {
    "owns_cultivable_land": True,
    "is_institutional_landholder": False,
    "paid_income_tax_last_year": False,
    "holds_constitutional_post": False,
    "monthly_pension_amount": 0,
}


class TestEligibilityVerdict:
    def test_returns_scheme_level_verdict(self):
        resp = client.post("/eligibility", json={"scheme_id": "pm-kisan", "profile": FULL_ELIGIBLE})
        assert resp.status_code == 200
        assert resp.json()["scheme_verdict"]["verdict"] == "ELIGIBLE"

    def test_taxpayer_scheme_verdict_is_not_eligible(self):
        profile = {**FULL_ELIGIBLE, "paid_income_tax_last_year": True}
        resp = client.post("/eligibility", json={"scheme_id": "pm-kisan", "profile": profile})
        body = resp.json()["scheme_verdict"]
        assert body["verdict"] == "NOT_ELIGIBLE"
        assert body["deciding_rules"][0]["rule_id"] == "pm-kisan-no-income-tax"


class TestInputValidation:
    @pytest.mark.parametrize(
        "bad_id", ["../data/gold/pm-kisan", "..\\..\\.env", "PM_KISAN", "a/b", ""]
    )
    def test_path_traversal_and_bad_slugs_rejected_on_eligibility(self, bad_id):
        resp = client.post("/eligibility", json={"scheme_id": bad_id, "profile": {}})
        assert resp.status_code == 422

    def test_bad_slug_rejected_on_chat(self):
        resp = client.post("/chat", json={"message": "hi", "scheme_ids": ["../x"]})
        assert resp.status_code == 422

    def test_unknown_scheme_on_chat_is_404(self):
        resp = client.post("/chat", json={"message": "hi", "scheme_ids": ["no-such-scheme"]})
        assert resp.status_code == 404

    def test_empty_and_oversized_messages_rejected(self):
        assert client.post("/chat", json={"message": ""}).status_code == 422
        too_long = "x" * (app_module.MAX_MESSAGE_CHARS + 1)
        assert client.post("/chat", json={"message": too_long}).status_code == 422

    def test_unsupported_language_rejected(self):
        resp = client.post("/chat", json={"message": "hi", "language": "fr"})
        assert resp.status_code == 422

    def test_bad_session_id_rejected(self):
        resp = client.post("/chat", json={"message": "hi", "session_id": "../../etc"})
        assert resp.status_code == 422


class TestSessionStore:
    def test_store_is_bounded_and_evicts_oldest(self, monkeypatch):
        monkeypatch.setattr(app_module, "MAX_SESSIONS", 3)
        app_module._sessions.clear()
        for i in range(5):
            app_module._get_session(f"session-{i}")
        assert len(app_module._sessions) == 3
        assert "session-0" not in app_module._sessions
        assert "session-4" in app_module._sessions
        app_module._sessions.clear()


class TestChatEndToEnd:
    def test_complete_profile_gives_engine_verdict_and_done(self):
        facts = json.dumps(FULL_ELIGIBLE)
        with patch("yojanasetu.agent.nodes._llm_call", return_value=facts):
            resp = client.post(
                "/chat", json={"message": "I am a farmer", "scheme_ids": ["pm-kisan"]}
            )
        body = resp.json()
        assert resp.status_code == 200
        assert body["done"] is True
        assert body["scheme_verdicts"][0]["verdict"] == "ELIGIBLE"
        assert "Appears eligible based on published criteria" in body["reply"]

    def test_hindi_reply(self):
        facts = json.dumps(FULL_ELIGIBLE)
        with patch("yojanasetu.agent.nodes._llm_call", return_value=facts):
            resp = client.post(
                "/chat",
                json={"message": "मैं किसान हूँ", "scheme_ids": ["pm-kisan"], "language": "hi"},
            )
        assert "पात्र प्रतीत होते हैं" in resp.json()["reply"]
