"""agent/fields.py – Which profile fields exist, and how to validate them.

Why this file exists
--------------------
The intake LLM turns free text into a profile.  If we let it invent keys or
loose types, two bad things happen:
  * a key the rules never read ("is_girl" instead of "applicant_is_girl")
    leaves the engine stuck on NEEDS_MORE_INFO forever;
  * a wrong type ("yes", True where a number is expected) can silently flip a
    comparison.  For example ``True < 10000`` is True in Python.

So the set of allowed fields and their types is derived from the approved
rules themselves, and every extracted value is checked against it.
Anything that does not fit is dropped, and the user is asked again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from yojanasetu.engine.loader import RULES_DIR, load_approved_rules

MAX_STRING_LENGTH = 100


@dataclass(frozen=True)
class FieldSpec:
    """What the rules expect for one profile field."""

    name: str
    kind: str  # "boolean" | "number" | "string" | "any"
    description_en: str
    description_hi: str = ""
    known_values: tuple[str, ...] = ()


def _kind_of(value: Any) -> str:
    """Map a rule's comparison value to a field kind."""
    if isinstance(value, bool):  # bool before int: bool is a subclass of int
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list) and value:
        return _kind_of(value[0])
    return "any"


def _strings_in(value: Any) -> list[str]:
    """Collect string values a rule compares against (for snapping)."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    return []


def all_scheme_ids() -> list[str]:
    """Every scheme that has a rules file."""
    return sorted(p.stem for p in RULES_DIR.glob("*.json"))


def build_field_catalog(scheme_ids: list[str] | None = None) -> dict[str, FieldSpec]:
    """Build the allowed-field catalog from approved rules.

    Parameters
    ----------
    scheme_ids:
        Schemes to include.  ``None`` or empty means every scheme with rules.
        Unknown schemes are skipped.
    """
    ids = scheme_ids or all_scheme_ids()
    kinds: dict[str, str] = {}
    first_rule: dict[str, Any] = {}
    values: dict[str, list[str]] = {}

    for scheme_id in ids:
        try:
            rules = load_approved_rules(scheme_id)
        except FileNotFoundError:
            continue
        for rule in rules:
            kind = _kind_of(rule.value)
            if rule.field not in kinds:
                kinds[rule.field] = kind
                first_rule[rule.field] = rule
            elif kinds[rule.field] != kind:
                kinds[rule.field] = "any"  # schemes disagree -> accept any simple type
            values.setdefault(rule.field, []).extend(_strings_in(rule.value))

    return {
        name: FieldSpec(
            name=name,
            kind=kinds[name],
            description_en=first_rule[name].description_en,
            description_hi=first_rule[name].description_hi,
            known_values=tuple(dict.fromkeys(values.get(name, []))),
        )
        for name in kinds
    }


def coerce_value(spec: FieldSpec, value: Any) -> tuple[bool, Any]:
    """Validate one extracted value against its spec.

    Returns ``(True, cleaned_value)`` or ``(False, None)`` if it must be dropped.
    Deliberately strict: "yes", "5 lakh", NaN and negative numbers are rejected,
    so the user is asked again instead of the engine guessing.
    """
    if spec.kind == "boolean":
        return (True, value) if isinstance(value, bool) else (False, None)

    if spec.kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False, None
        if not math.isfinite(value) or value < 0:
            return False, None
        return True, value

    if spec.kind == "string":
        return _clean_string(spec, value)

    # kind == "any": accept simple JSON scalars only.
    if isinstance(value, bool):
        return True, value
    if isinstance(value, (int, float)):
        return (True, value) if math.isfinite(value) and value >= 0 else (False, None)
    return _clean_string(spec, value)


def _clean_string(spec: FieldSpec, value: Any) -> tuple[bool, Any]:
    if not isinstance(value, str):
        return False, None
    text = " ".join(value.split())
    if not text or len(text) > MAX_STRING_LENGTH:
        return False, None
    # Snap "uttar pradesh" -> "Uttar Pradesh" so a casing difference alone
    # can never produce a wrong NOT_ELIGIBLE.
    for known in spec.known_values:
        if text.casefold() == known.casefold():
            return True, known
    return True, text


def clean_extracted_facts(
    raw: Any, catalog: dict[str, FieldSpec]
) -> dict[str, Any]:
    """Keep only known fields whose values pass validation."""
    if not isinstance(raw, dict):
        return {}
    cleaned: dict[str, Any] = {}
    for key, value in raw.items():
        spec = catalog.get(key) if isinstance(key, str) else None
        if spec is None:
            continue
        ok, clean = coerce_value(spec, value)
        if ok:
            cleaned[key] = clean
    return cleaned


def catalog_prompt(catalog: dict[str, FieldSpec]) -> str:
    """Render the catalog as a prompt section for the intake LLM."""
    lines = []
    for spec in catalog.values():
        line = f'- "{spec.name}" ({spec.kind}): {spec.description_en}'
        if spec.known_values:
            quoted = ", ".join(f'"{v}"' for v in spec.known_values[:6])
            line += f" Known values: {quoted}."
        lines.append(line)
    return "\n".join(lines)
