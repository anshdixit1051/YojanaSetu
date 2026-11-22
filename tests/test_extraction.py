"""tests/test_extraction.py – Tests for extraction/prompts.py and extractor.py.

All LLM calls are mocked.  No real API key needed to run these tests.

Run with: pytest tests/test_extraction.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from yojanasetu.extraction.prompts import SYSTEM_PROMPT, build_user_prompt
from yojanasetu.extraction.extractor import (
    _parse_json_response,
    _validate_raw_rule,
    extract_rules,
    save_pending_rules,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_RAW_RULE = {
    "rule_id": "pm-kisan-test-rule",
    "scheme_id": "pm-kisan",
    "field": "owns_land",
    "operator": "eq",
    "value": True,
    "verdict_if_true": "ELIGIBLE",
    "description_en": "Must own land.",
    "description_hi": "",
    "source_clause": "Section 2",
    "requires_human_judgement": False,
    "status": "pending_review",
}


# ---------------------------------------------------------------------------
# prompts.py
# ---------------------------------------------------------------------------

class TestBuildUserPrompt:
    def test_contains_scheme_id(self):
        p = build_user_prompt("pm-kisan", "PM-KISAN", "farmer", "some source text")
        assert "pm-kisan" in p

    def test_contains_source_text(self):
        p = build_user_prompt("pm-kisan", "PM-KISAN", "farmer", "source text here")
        assert "source text here" in p

    def test_truncates_long_source(self):
        long_text = "x" * 20_000
        p = build_user_prompt("pm-kisan", "PM-KISAN", "farmer", long_text, max_chars=100)
        assert "truncated" in p
        assert len(p) < 500   # much shorter than 20k

    def test_short_source_not_truncated(self):
        text = "short text"
        p = build_user_prompt("pm-kisan", "PM-KISAN", "farmer", text, max_chars=100)
        assert "truncated" not in p

    def test_system_prompt_mentions_pending_review(self):
        assert "pending_review" in SYSTEM_PROMPT

    def test_system_prompt_forbids_inventing_rules(self):
        assert "NOT" in SYSTEM_PROMPT or "Only" in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# _parse_json_response
# ---------------------------------------------------------------------------

class TestParseJsonResponse:
    def test_plain_json_array(self):
        result = _parse_json_response('[{"a": 1}]')
        assert result == [{"a": 1}]

    def test_strips_markdown_fences(self):
        text = "```json\n[{\"a\": 1}]\n```"
        assert _parse_json_response(text) == [{"a": 1}]

    def test_strips_plain_fences(self):
        text = "```\n[{\"a\": 1}]\n```"
        assert _parse_json_response(text) == [{"a": 1}]

    def test_invalid_json_raises_value_error(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            _parse_json_response("not json at all")

    def test_json_object_not_array_raises_value_error(self):
        with pytest.raises(ValueError, match="JSON array"):
            _parse_json_response('{"key": "value"}')

    def test_empty_array_ok(self):
        assert _parse_json_response("[]") == []


# ---------------------------------------------------------------------------
# _validate_raw_rule
# ---------------------------------------------------------------------------

class TestValidateRawRule:
    def test_valid_rule_has_no_errors(self):
        assert _validate_raw_rule(VALID_RAW_RULE.copy(), "pm-kisan") == []

    def test_missing_field_key_reported(self):
        bad = {k: v for k, v in VALID_RAW_RULE.items() if k != "field"}
        errors = _validate_raw_rule(bad, "pm-kisan")
        assert any("field" in e for e in errors)

    def test_wrong_scheme_id_reported(self):
        bad = {**VALID_RAW_RULE, "scheme_id": "other-scheme"}
        errors = _validate_raw_rule(bad, "pm-kisan")
        assert any("scheme_id" in e for e in errors)

    def test_rule_id_not_starting_with_scheme_reported(self):
        bad = {**VALID_RAW_RULE, "rule_id": "other-scheme-rule"}
        errors = _validate_raw_rule(bad, "pm-kisan")
        assert any("rule_id" in e for e in errors)

    def test_bad_operator_reported(self):
        bad = {**VALID_RAW_RULE, "operator": "between"}
        errors = _validate_raw_rule(bad, "pm-kisan")
        assert any("operator" in e for e in errors)

    def test_bad_verdict_reported(self):
        bad = {**VALID_RAW_RULE, "verdict_if_true": "MAYBE"}
        errors = _validate_raw_rule(bad, "pm-kisan")
        assert any("verdict" in e for e in errors)

    def test_non_pending_status_reported(self):
        bad = {**VALID_RAW_RULE, "status": "approved"}
        errors = _validate_raw_rule(bad, "pm-kisan")
        assert any("pending_review" in e for e in errors)


# ---------------------------------------------------------------------------
# extract_rules (mocked LLM)
# ---------------------------------------------------------------------------

class TestExtractRules:
    def test_missing_scheme_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="no-such-scheme"):
            extract_rules("no-such-scheme")

    def test_valid_llm_response_returns_rules(self):
        good_response = json.dumps([VALID_RAW_RULE])
        with patch(
            "yojanasetu.extraction.extractor._call_llm",
            return_value=good_response,
        ):
            rules, warnings = extract_rules("pm-kisan")
        assert len(rules) == 1
        assert rules[0]["status"] == "pending_review"
        assert warnings == []

    def test_llm_response_with_approved_status_is_forced_to_pending(self):
        """Even if the LLM returns approved, we override it to pending_review."""
        rule_with_approved = {**VALID_RAW_RULE, "status": "approved"}
        with patch(
            "yojanasetu.extraction.extractor._call_llm",
            return_value=json.dumps([rule_with_approved]),
        ):
            # Note: status is forced to pending_review before validation,
            # so the validation step won't flag it.
            rules, warnings = extract_rules("pm-kisan")
        assert all(r["status"] == "pending_review" for r in rules)

    def test_invalid_rule_in_response_produces_warning(self):
        bad_rule = {**VALID_RAW_RULE, "operator": "between"}  # invalid
        with patch(
            "yojanasetu.extraction.extractor._call_llm",
            return_value=json.dumps([bad_rule]),
        ):
            rules, warnings = extract_rules("pm-kisan")
        assert rules == []
        assert len(warnings) > 0

    def test_non_dict_item_in_response_produces_warning(self):
        with patch(
            "yojanasetu.extraction.extractor._call_llm",
            return_value=json.dumps(["not a dict"]),
        ):
            rules, warnings = extract_rules("pm-kisan")
        assert rules == []
        assert any("not a JSON object" in w for w in warnings)

    def test_no_api_key_raises_runtime_error(self):
        """If no key is configured, extract_rules must fail early with a clear message."""
        with patch(
            "yojanasetu.extraction.extractor.get_settings",
            return_value=MagicMock(gemini_api_key=None, groq_api_key=None),
        ):
            with pytest.raises(RuntimeError, match="API key"):
                extract_rules("pm-kisan")


# ---------------------------------------------------------------------------
# save_pending_rules
# ---------------------------------------------------------------------------

class TestSavePendingRules:
    def test_saves_new_rules_to_file(self, tmp_path, monkeypatch):
        import yojanasetu.extraction.extractor as ext_mod
        monkeypatch.setattr(ext_mod, "RULES_DIR", tmp_path)

        rules, _ = [VALID_RAW_RULE.copy()], []
        path = save_pending_rules("pm-kisan", rules)

        saved = json.loads(path.read_text(encoding="utf-8"))
        assert len(saved) == 1
        assert saved[0]["rule_id"] == "pm-kisan-test-rule"

    def test_does_not_overwrite_existing_rule(self, tmp_path, monkeypatch):
        import yojanasetu.extraction.extractor as ext_mod
        monkeypatch.setattr(ext_mod, "RULES_DIR", tmp_path)

        # Pre-existing approved rule with same ID.
        existing = [{**VALID_RAW_RULE, "status": "approved"}]
        (tmp_path / "pm-kisan.json").write_text(json.dumps(existing), encoding="utf-8")

        new_rules = [VALID_RAW_RULE.copy()]
        save_pending_rules("pm-kisan", new_rules)

        saved = json.loads((tmp_path / "pm-kisan.json").read_text(encoding="utf-8"))
        # Still only 1 rule, and it's still approved (not overwritten).
        assert len(saved) == 1
        assert saved[0]["status"] == "approved"

    def test_adds_new_rule_alongside_existing(self, tmp_path, monkeypatch):
        import yojanasetu.extraction.extractor as ext_mod
        monkeypatch.setattr(ext_mod, "RULES_DIR", tmp_path)

        existing = [{**VALID_RAW_RULE, "rule_id": "pm-kisan-rule-a", "status": "approved"}]
        (tmp_path / "pm-kisan.json").write_text(json.dumps(existing), encoding="utf-8")

        new_rule = {**VALID_RAW_RULE, "rule_id": "pm-kisan-rule-b"}
        save_pending_rules("pm-kisan", [new_rule])

        saved = json.loads((tmp_path / "pm-kisan.json").read_text(encoding="utf-8"))
        assert len(saved) == 2
        ids = {r["rule_id"] for r in saved}
        assert "pm-kisan-rule-a" in ids
        assert "pm-kisan-rule-b" in ids
