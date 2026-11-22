"""mcp_tools/server.py – MCP server exposing the deterministic engine as tools.

Tools
-----
check_eligibility(scheme_id, profile_json)
    One deterministic verdict for the scheme, plus the per-rule results with
    their source clauses.  The engine, not the LLM, decides.
get_missing_fields(scheme_id, profile_json)
    Which facts are still needed to resolve a NEEDS_MORE_INFO verdict.
list_schemes()
    Available schemes with names and categories.

Run as a standalone stdio server:
    python -m yojanasetu.mcp_tools.server

Design note
-----------
The logic lives in plain functions (``tool_*``) that never import the MCP SDK,
so they are easy to unit-test.  ``create_mcp_server`` only wraps them.
Errors are returned as ``{"error": ...}`` so a calling LLM can read them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RULES_DIR = PROJECT_ROOT / "data" / "rules"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

_SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def _parse_profile(profile_json: str) -> dict[str, Any]:
    """Parse the profile argument, or raise ValueError with a readable message."""
    try:
        data = json.loads(profile_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"profile_json is not valid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError("profile_json must be a JSON object, e.g. {\"age\": 35}")
    return data


def _run_engine(scheme_id: str, profile_json: str):
    """Validate inputs and run the engine.  Raises ValueError / FileNotFoundError."""
    from yojanasetu.engine.aggregate import evaluate_scheme_verdict  # noqa: PLC0415
    from yojanasetu.engine.loader import load_approved_rules  # noqa: PLC0415

    if not _SLUG_RE.match(scheme_id):
        raise ValueError("scheme_id must be a lowercase slug like 'pm-kisan'")
    profile = _parse_profile(profile_json)
    rules = load_approved_rules(scheme_id)
    return evaluate_scheme_verdict(scheme_id, rules, profile)


def tool_check_eligibility(scheme_id: str, profile_json: str) -> dict[str, Any]:
    """Deterministic scheme verdict plus per-rule results."""
    try:
        verdict = _run_engine(scheme_id, profile_json)
    except FileNotFoundError:
        return {"error": f"No rules found for scheme '{scheme_id}'."}
    except ValueError as exc:
        return {"error": str(exc)}
    return {
        "scheme_verdict": verdict.as_dict(),
        "results": [r.as_dict() for r in verdict.rule_results],
    }


def tool_get_missing_fields(scheme_id: str, profile_json: str) -> dict[str, Any]:
    """Fields the user can still supply for a scheme."""
    try:
        verdict = _run_engine(scheme_id, profile_json)
    except FileNotFoundError:
        return {"error": f"No rules found for scheme '{scheme_id}'."}
    except ValueError as exc:
        return {"error": str(exc)}
    return {
        "scheme_id": scheme_id,
        "missing_fields": verdict.missing_fields,
        "needs_human_review": verdict.needs_human_review,
    }


def tool_list_schemes() -> list[dict[str, str]]:
    """Every scheme with a rules file, with metadata when provenance exists."""
    schemes: list[dict[str, str]] = []
    for rules_file in sorted(RULES_DIR.glob("*.json")):
        scheme_id = rules_file.stem
        entry = {"scheme_id": scheme_id, "name_en": scheme_id, "category": "", "level": ""}
        meta_path = RAW_DIR / scheme_id / "meta.yaml"
        if meta_path.exists():
            try:
                meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                meta = {}
            entry["name_en"] = str(meta.get("name_en", scheme_id))
            entry["category"] = str(meta.get("category", ""))
            entry["level"] = str(meta.get("level", ""))
        schemes.append(entry)
    return schemes


def create_mcp_server():
    """Create the MCP server (FastMCP) with the three tools registered.

    Raises
    ------
    ImportError
        If the mcp package is not installed.
    """
    try:
        from mcp.server.mcpserver import MCPServer as FastMCP  # noqa: PLC0415
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("mcp is not installed. Run: pip install mcp") from exc

    mcp = FastMCP("yojanasetu")

    @mcp.tool()
    def check_eligibility(scheme_id: str, profile_json: str) -> dict[str, Any]:
        """Run the deterministic eligibility engine for one scheme.

        Returns scheme_verdict (ELIGIBLE, NOT_ELIGIBLE or NEEDS_MORE_INFO) and
        per-rule results with source clauses. The engine decides; never
        override or reinterpret the verdict.

        Args:
            scheme_id: Scheme slug, e.g. "pm-kisan".
            profile_json: JSON object of known facts, e.g. '{"owns_cultivable_land": true}'.
        """
        return tool_check_eligibility(scheme_id, profile_json)

    @mcp.tool()
    def get_missing_fields(scheme_id: str, profile_json: str) -> dict[str, Any]:
        """List the facts still needed to fully evaluate a scheme.

        Use this to decide what to ask the user next.

        Args:
            scheme_id: Scheme slug, e.g. "pm-kisan".
            profile_json: JSON object of known facts.
        """
        return tool_get_missing_fields(scheme_id, profile_json)

    @mcp.tool()
    def list_schemes() -> list[dict[str, str]]:
        """List all available schemes with their names and categories."""
        return tool_list_schemes()

    return mcp


if __name__ == "__main__":
    create_mcp_server().run()
