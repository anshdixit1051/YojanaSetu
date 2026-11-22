"""tests/test_mcp.py – Unit tests for MCP server tools.

Tests the standalone tool_* functions directly without needing a client-server
transport, plus verifies that create_mcp_server builds the FastMCP instance.

Run with:
    pytest tests/test_mcp.py -v
"""

from __future__ import annotations

import json
import pytest

from yojanasetu.mcp_tools.server import (
    create_mcp_server,
    tool_check_eligibility,
    tool_get_missing_fields,
    tool_list_schemes,
)


class TestCheckEligibilityTool:
    def test_eligible_profile_returns_eligible_verdict(self):
        profile = {
            "owns_cultivable_land": True,
            "is_institutional_landholder": False,
            "paid_income_tax_last_year": False,
            "holds_constitutional_post": False,
            "monthly_pension_amount": 0,
        }
        res = tool_check_eligibility("pm-kisan", json.dumps(profile))
        assert "error" not in res
        assert res["scheme_verdict"]["verdict"] == "ELIGIBLE"
        assert len(res["results"]) > 0
        for r in res["results"]:
            assert "source_clause" in r

    def test_taxpayer_returns_not_eligible(self):
        profile = {
            "owns_cultivable_land": True,
            "is_institutional_landholder": False,
            "paid_income_tax_last_year": True,
            "holds_constitutional_post": False,
            "monthly_pension_amount": 0,
        }
        res = tool_check_eligibility("pm-kisan", json.dumps(profile))
        assert "error" not in res
        assert res["scheme_verdict"]["verdict"] == "NOT_ELIGIBLE"

    def test_empty_profile_returns_needs_more_info(self):
        res = tool_check_eligibility("pm-kisan", json.dumps({}))
        assert "error" not in res
        assert res["scheme_verdict"]["verdict"] == "NEEDS_MORE_INFO"

    def test_invalid_json_profile_returns_error(self):
        res = tool_check_eligibility("pm-kisan", "not-json")
        assert "error" in res
        assert "not valid JSON" in res["error"]

    def test_non_dict_json_profile_returns_error(self):
        res = tool_check_eligibility("pm-kisan", json.dumps([1, 2, 3]))
        assert "error" in res
        assert "must be a JSON object" in res["error"]

    def test_invalid_slug_returns_error(self):
        res = tool_check_eligibility("../data/gold", json.dumps({}))
        assert "error" in res
        assert "slug" in res["error"]

    def test_unknown_scheme_returns_error(self):
        res = tool_check_eligibility("non-existent-scheme", json.dumps({}))
        assert "error" in res
        assert "No rules found" in res["error"]


class TestGetMissingFieldsTool:
    def test_empty_profile_returns_missing_fields(self):
        res = tool_get_missing_fields("pm-kisan", json.dumps({}))
        assert "error" not in res
        assert res["scheme_id"] == "pm-kisan"
        assert len(res["missing_fields"]) > 0
        assert "owns_cultivable_land" in res["missing_fields"]

    def test_fully_eligible_has_no_missing_fields(self):
        profile = {
            "owns_cultivable_land": True,
            "is_institutional_landholder": False,
            "paid_income_tax_last_year": False,
            "holds_constitutional_post": False,
            "monthly_pension_amount": 0,
        }
        res = tool_get_missing_fields("pm-kisan", json.dumps(profile))
        assert "error" not in res
        assert res["missing_fields"] == []

    def test_bad_profile_returns_error(self):
        res = tool_get_missing_fields("pm-kisan", "{bad json")
        assert "error" in res


class TestListSchemesTool:
    def test_list_schemes_includes_all_configured(self):
        schemes = tool_list_schemes()
        ids = {s["scheme_id"] for s in schemes}
        expected = {
            "pm-kisan",
            "ayushman-bharat-pmjay",
            "nsp-post-matric-sc",
            "pm-svanidhi",
            "up-kanya-sumangala",
        }
        assert expected.issubset(ids)
        pm_kisan = next(s for s in schemes if s["scheme_id"] == "pm-kisan")
        assert pm_kisan["category"] == "farmer"
        assert pm_kisan["level"] == "central"


class TestCreateMcpServer:
    def test_server_builds_and_registers_tools(self):
        server = create_mcp_server()
        assert server is not None
        assert server.name == "yojanasetu"
