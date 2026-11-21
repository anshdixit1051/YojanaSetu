"""tests/test_engine_loader.py – Tests for engine/loader.py.

Run with: pytest tests/test_engine_loader.py -v
"""

import json
import pytest
from pathlib import Path

from yojanasetu.engine.loader import load_rules, load_approved_rules
from yojanasetu.engine.models import Verdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_rules(tmp_path: Path, scheme_id: str, rules: list) -> Path:
    """Write a rules JSON file into a temp directory and patch RULES_DIR."""
    f = tmp_path / f"{scheme_id}.json"
    f.write_text(json.dumps(rules), encoding="utf-8")
    return f


VALID_RULE = {
    "rule_id": "test-scheme-field-a",
    "scheme_id": "test-scheme",
    "field": "field_a",
    "operator": "eq",
    "value": True,
    "verdict_if_true": "ELIGIBLE",
    "description_en": "Field A must be true.",
    "source_clause": "Section 1",
    "status": "approved",
}


# ---------------------------------------------------------------------------
# Fixture: patch RULES_DIR to use tmp_path
# ---------------------------------------------------------------------------

@pytest.fixture()
def rules_dir(tmp_path, monkeypatch):
    """Point the loader at a temporary directory for the duration of a test."""
    import yojanasetu.engine.loader as loader_mod
    monkeypatch.setattr(loader_mod, "RULES_DIR", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Happy-path: load the real pm-kisan rules file
# ---------------------------------------------------------------------------

def test_load_pm_kisan_rules_returns_list():
    rules = load_rules("pm-kisan")
    assert isinstance(rules, list)
    assert len(rules) > 0


def test_pm_kisan_rules_have_correct_scheme_id():
    rules = load_rules("pm-kisan")
    assert all(r.scheme_id == "pm-kisan" for r in rules)


def test_pm_kisan_approved_rules_all_approved():
    rules = load_approved_rules("pm-kisan")
    assert all(r.is_approved() for r in rules)


def test_pm_kisan_rule_ids_start_with_scheme_id():
    rules = load_rules("pm-kisan")
    assert all(r.rule_id.startswith("pm-kisan") for r in rules)


def test_pm_kisan_verdicts_are_valid_enum_members():
    rules = load_rules("pm-kisan")
    valid = {v for v in Verdict}
    assert all(r.verdict_if_true in valid for r in rules)


# ---------------------------------------------------------------------------
# Error: file not found
# ---------------------------------------------------------------------------

def test_missing_scheme_raises_file_not_found(rules_dir):
    with pytest.raises(FileNotFoundError, match="no-such-scheme"):
        load_rules("no-such-scheme")


# ---------------------------------------------------------------------------
# Error: bad JSON
# ---------------------------------------------------------------------------

def test_invalid_json_raises_value_error(rules_dir):
    (rules_dir / "bad-scheme.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_rules("bad-scheme")


# ---------------------------------------------------------------------------
# Error: JSON is not an array
# ---------------------------------------------------------------------------

def test_json_object_not_array_raises_value_error(rules_dir):
    (rules_dir / "bad-scheme.json").write_text('{"key": "value"}', encoding="utf-8")
    with pytest.raises(ValueError, match="JSON array"):
        load_rules("bad-scheme")


# ---------------------------------------------------------------------------
# Error: missing required keys
# ---------------------------------------------------------------------------

def test_missing_required_key_raises_value_error(rules_dir):
    bad = {k: v for k, v in VALID_RULE.items() if k != "field"}  # drop "field"
    (rules_dir / "test-scheme.json").write_text(json.dumps([bad]), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required fields"):
        load_rules("test-scheme")


# ---------------------------------------------------------------------------
# Error: invalid verdict_if_true
# ---------------------------------------------------------------------------

def test_invalid_verdict_raises_value_error(rules_dir):
    bad = {**VALID_RULE, "verdict_if_true": "MAYBE"}
    (rules_dir / "test-scheme.json").write_text(json.dumps([bad]), encoding="utf-8")
    with pytest.raises(ValueError, match="verdict_if_true"):
        load_rules("test-scheme")


# ---------------------------------------------------------------------------
# Error: unsupported operator
# ---------------------------------------------------------------------------

def test_unsupported_operator_raises_value_error(rules_dir):
    bad = {**VALID_RULE, "operator": "between"}
    (rules_dir / "test-scheme.json").write_text(json.dumps([bad]), encoding="utf-8")
    with pytest.raises(ValueError, match="operator"):
        load_rules("test-scheme")


# ---------------------------------------------------------------------------
# Error: invalid status
# ---------------------------------------------------------------------------

def test_invalid_status_raises_value_error(rules_dir):
    bad = {**VALID_RULE, "status": "rejected"}
    (rules_dir / "test-scheme.json").write_text(json.dumps([bad]), encoding="utf-8")
    with pytest.raises(ValueError, match="status"):
        load_rules("test-scheme")


# ---------------------------------------------------------------------------
# Error: rule_id does not start with scheme_id
# ---------------------------------------------------------------------------

def test_rule_id_mismatch_raises_value_error(rules_dir):
    bad = {**VALID_RULE, "rule_id": "other-scheme-something"}
    (rules_dir / "test-scheme.json").write_text(json.dumps([bad]), encoding="utf-8")
    with pytest.raises(ValueError, match="rule_id must start with scheme_id"):
        load_rules("test-scheme")


# ---------------------------------------------------------------------------
# load_approved_rules filters pending
# ---------------------------------------------------------------------------

def test_load_approved_rules_excludes_pending(rules_dir):
    rules_data = [
        {**VALID_RULE, "rule_id": "test-scheme-r1", "status": "approved"},
        {**VALID_RULE, "rule_id": "test-scheme-r2", "status": "pending_review"},
    ]
    (rules_dir / "test-scheme.json").write_text(
        json.dumps(rules_data), encoding="utf-8"
    )
    rules = load_approved_rules("test-scheme")
    assert len(rules) == 1
    assert rules[0].rule_id == "test-scheme-r1"


def test_load_approved_rules_empty_file_returns_empty(rules_dir):
    (rules_dir / "test-scheme.json").write_text("[]", encoding="utf-8")
    assert load_approved_rules("test-scheme") == []
