"""agent/summary.py – Deterministic, citation-bearing reply text.

Why this file exists
--------------------
The sentence "you appear eligible for X" is the most dangerous thing this
product says, so it is produced by plain Python from the engine's verdicts,
never by an LLM.  The LLM may only add a short follow-up question (and even
that is checked, see ``is_safe_question``).
"""

from __future__ import annotations

from typing import Any

from yojanasetu.agent.fields import FieldSpec

MAX_QUESTION_LENGTH = 300

# Words a follow-up question must never contain: a question is not a verdict.
_VERDICT_WORDS = ("eligible", "qualify", "ineligible", "पात्र", "योग्य")


def _clauses(rule_dicts: list[dict[str, Any]]) -> str:
    seen = dict.fromkeys(r["source_clause"] for r in rule_dicts if r.get("source_clause"))
    return "; ".join(seen)


def _one_scheme_en(v: dict[str, Any]) -> str:
    sid = v["scheme_id"]
    rules = v["deciding_rules"]
    if v["verdict"] == "ELIGIBLE":
        return (
            f"{sid}: Appears eligible based on published criteria "
            f"(checked: {_clauses(rules)}). Meeting the criteria does not "
            "guarantee the benefit; please confirm on the official portal."
        )
    if v["verdict"] == "NOT_ELIGIBLE":
        reasons = " ".join(r["reason_en"] for r in rules)
        return f"{sid}: {reasons}"
    text = f"{sid}: I need more information."
    if v["missing_fields"]:
        text += f" Still needed: {', '.join(v['missing_fields'])}."
    if v["needs_human_review"]:
        text += " Some criteria need an official's judgement and cannot be decided here."
    if v.get("note"):
        text += f" {v['note']}"
    return text


def _one_scheme_hi(v: dict[str, Any]) -> str:
    sid = v["scheme_id"]
    rules = v["deciding_rules"]
    if v["verdict"] == "ELIGIBLE":
        return (
            f"{sid}: प्रकाशित मानदंडों के आधार पर पात्र प्रतीत होते हैं "
            f"(जाँचे गए खंड: {_clauses(rules)})। मानदंड पूरे होने का अर्थ लाभ की "
            "गारंटी नहीं है; कृपया आधिकारिक पोर्टल पर पुष्टि करें।"
        )
    if v["verdict"] == "NOT_ELIGIBLE":
        reasons = " ".join(
            f"{r.get('reason_hi') or r['reason_en']} ({r['source_clause']})" for r in rules
        )
        return f"{sid}: पात्र नहीं। {reasons}"
    text = f"{sid}: अधिक जानकारी चाहिए।"
    if v["missing_fields"]:
        text += f" आवश्यक: {', '.join(v['missing_fields'])}।"
    if v["needs_human_review"]:
        text += " कुछ मानदंडों पर अधिकारी का निर्णय आवश्यक है।"
    return text


def format_verdict_summary(scheme_verdicts: list[dict[str, Any]], language: str) -> str:
    """Render engine verdicts as plain text, one block per scheme."""
    if not scheme_verdicts:
        if language == "hi":
            return "अभी जाँचने के लिए कोई योजना नियम उपलब्ध नहीं हैं।"
        return "I could not find any scheme rules to check yet."
    render = _one_scheme_hi if language == "hi" else _one_scheme_en
    return "\n".join(render(v) for v in scheme_verdicts)


def fallback_question(spec: FieldSpec | None, field_name: str, language: str) -> str:
    """A fixed follow-up question used when the LLM is unavailable or unsafe."""
    if language == "hi":
        detail = (spec.description_hi or spec.description_en) if spec else field_name
        return f"आगे बढ़ने के लिए यह जानकारी चाहिए: {detail}"
    detail = spec.description_en if spec else field_name
    return f"To continue, I need this detail: {detail}"


def is_safe_question(text: str) -> bool:
    """True if LLM text is a short question that states no verdict."""
    cleaned = text.strip()
    if not cleaned or len(cleaned) > MAX_QUESTION_LENGTH:
        return False
    lowered = cleaned.casefold()
    return not any(word in lowered for word in _VERDICT_WORDS)
