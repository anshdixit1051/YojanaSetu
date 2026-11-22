"""extraction/extractor.py – LLM-powered rule extractor.

Contract
--------
extract_rules(scheme_id) → list[dict]
    Reads source.txt and meta.yaml for a scheme, calls the configured LLM,
    parses the JSON response, validates each rule, and returns a list of raw
    dicts with status = "pending_review".

    The LLM is called ONCE per scheme.  The result is NEVER automatically
    approved.  It goes to data/rules/<scheme-id>.json with status=pending_review
    for a human to review via scripts/review_rules.py.

save_pending_rules(scheme_id, rules) → Path
    Merges the new pending rules into the existing rules file (preserving any
    already-approved rules) and writes it back to disk.

Design note: LLM never decides eligibility
-------------------------------------------
This module only produces pending rules.  The evaluator in engine/evaluator.py
refuses to run any rule that is not status="approved".  So even if the LLM
hallucinates a rule, it cannot affect any eligibility verdict until a human
explicitly approves it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from yojanasetu.config import get_settings
from yojanasetu.engine.evaluator import SUPPORTED_OPERATORS
from yojanasetu.engine.loader import RULES_DIR, VALID_STATUSES
from yojanasetu.engine.models import Verdict
from yojanasetu.extraction.prompts import SYSTEM_PROMPT, build_user_prompt
from yojanasetu.schemas.source_meta import load_meta

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

VALID_VERDICTS = {v.value for v in Verdict}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

REQUIRED_RULE_KEYS = {
    "rule_id", "scheme_id", "field", "operator",
    "value", "verdict_if_true", "description_en", "source_clause",
}


def _validate_raw_rule(raw: dict[str, Any], scheme_id: str) -> list[str]:
    """Return a list of validation error strings for one raw rule dict."""
    errors: list[str] = []
    rule_id = raw.get("rule_id", "<missing>")

    missing = REQUIRED_RULE_KEYS - raw.keys()
    if missing:
        errors.append(f"[{rule_id}] missing keys: {sorted(missing)}")

    if raw.get("scheme_id") != scheme_id:
        errors.append(
            f"[{rule_id}] scheme_id mismatch: "
            f"expected '{scheme_id}', got '{raw.get('scheme_id')}'"
        )
    if not str(raw.get("rule_id", "")).startswith(scheme_id):
        errors.append(
            f"[{rule_id}] rule_id must start with '{scheme_id}'"
        )
    if raw.get("operator") not in SUPPORTED_OPERATORS:
        errors.append(
            f"[{rule_id}] unsupported operator: '{raw.get('operator')}'"
        )
    if raw.get("verdict_if_true") not in VALID_VERDICTS:
        errors.append(
            f"[{rule_id}] invalid verdict_if_true: '{raw.get('verdict_if_true')}'"
        )
    # Status must always be pending_review from the LLM.
    status = raw.get("status", "pending_review")
    if status != "pending_review":
        errors.append(
            f"[{rule_id}] LLM-generated rules must have status='pending_review', "
            f"got '{status}'"
        )
    return errors


# ---------------------------------------------------------------------------
# LLM call (Gemini via google-generativeai)
# ---------------------------------------------------------------------------

def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call the configured LLM and return the raw text response.

    Uses Google Gemini (google-generativeai) if GOOGLE_API_KEY is set,
    otherwise raises RuntimeError with a helpful message.

    Returns
    -------
    str
        Raw text from the model (expected to be a JSON array).

    Raises
    ------
    RuntimeError
        If no API key is configured.
    """
    settings = get_settings()

    if settings.gemini_api_key is not None:
        return _call_gemini(
            settings.gemini_api_key.get_secret_value(),
            settings.model_name if hasattr(settings, "model_name") else "gemini-1.5-flash",
            system_prompt,
            user_prompt,
        )

    raise RuntimeError(
        "No LLM API key found.  Set GOOGLE_API_KEY in your .env file.\n"
        "Get a free key at https://aistudio.google.com"
    )


def _call_gemini(api_key: str, model_name: str, system_prompt: str, user_prompt: str) -> str:
    """Call the Gemini API and return the response text."""
    try:
        import google.generativeai as genai  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "google-generativeai is not installed. "
            "Run: pip install google-generativeai"
        ) from exc

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_prompt,
    )
    response = model.generate_content(user_prompt)
    return response.text


# ---------------------------------------------------------------------------
# JSON parsing (strips markdown fences if model wraps response)
# ---------------------------------------------------------------------------

def _parse_json_response(text: str) -> list[dict[str, Any]]:
    """Extract a JSON array from a model response string.

    The model may wrap its output in ```json … ``` fences even when told not to.
    This strips them before parsing.

    Raises
    ------
    ValueError
        If no valid JSON array is found.
    """
    # Strip markdown fences: ```json ... ``` or ``` ... ```
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM response is not valid JSON: {exc}\n"
            f"Raw response (first 500 chars):\n{text[:500]}"
        ) from exc

    if not isinstance(result, list):
        raise ValueError(
            f"LLM response is a {type(result).__name__}, expected a JSON array."
        )
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_rules(scheme_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Extract rules from a scheme's source.txt using the configured LLM.

    Parameters
    ----------
    scheme_id:
        Slug matching a folder in data/raw/, e.g. ``"pm-kisan"``.

    Returns
    -------
    (rules, warnings)
        rules    – list of validated raw rule dicts with status=pending_review.
        warnings – list of warning strings for rules the LLM returned that
                   failed validation (they are excluded from rules).

    Raises
    ------
    FileNotFoundError
        If data/raw/<scheme-id>/ does not exist.
    RuntimeError
        If no LLM API key is configured.
    ValueError
        If the LLM response cannot be parsed as a JSON array.
    """
    scheme_dir = RAW_DIR / scheme_id
    if not scheme_dir.exists():
        raise FileNotFoundError(
            f"Scheme folder not found: {scheme_dir}\n"
            f"Create data/raw/{scheme_id}/ with meta.yaml and source.txt first."
        )

    meta = load_meta(scheme_dir / "meta.yaml")
    source_text = (scheme_dir / "source.txt").read_text(encoding="utf-8")

    user_prompt = build_user_prompt(
        scheme_id=scheme_id,
        name_en=meta.name_en,
        category=meta.category,
        source_text=source_text,
    )

    raw_response = _call_llm(SYSTEM_PROMPT, user_prompt)
    raw_rules = _parse_json_response(raw_response)

    # Validate each rule; collect warnings for bad ones.
    good: list[dict[str, Any]] = []
    warnings: list[str] = []

    for i, raw in enumerate(raw_rules):
        if not isinstance(raw, dict):
            warnings.append(f"Item {i} is not a JSON object; skipped.")
            continue

        # Force status=pending_review regardless of what the model returned.
        raw["status"] = "pending_review"

        errs = _validate_raw_rule(raw, scheme_id)
        if errs:
            warnings.extend(errs)
        else:
            good.append(raw)

    return good, warnings


def save_pending_rules(scheme_id: str, new_rules: list[dict[str, Any]]) -> Path:
    """Merge new pending rules into data/rules/<scheme-id>.json.

    Existing rules are preserved.  New rules with the same rule_id as an
    existing rule are SKIPPED (no silent overwrites of approved rules).

    Parameters
    ----------
    scheme_id:
        Slug, e.g. ``"pm-kisan"``.
    new_rules:
        List of validated raw rule dicts from extract_rules().

    Returns
    -------
    Path
        Path to the updated rules file.
    """
    rules_file = RULES_DIR / f"{scheme_id}.json"

    # Load existing rules (if any).
    existing: list[dict[str, Any]] = []
    if rules_file.exists():
        existing = json.loads(rules_file.read_text(encoding="utf-8"))

    existing_ids = {r["rule_id"] for r in existing}

    added = 0
    skipped = 0
    for rule in new_rules:
        if rule["rule_id"] in existing_ids:
            skipped += 1
        else:
            existing.append(rule)
            added += 1

    rules_file.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved to {rules_file}: {added} rule(s) added, {skipped} skipped (duplicate id).")
    return rules_file
