"""engine/models.py – Core data types for the deterministic rules engine.

Why these types exist
---------------------
Every eligibility decision must be traceable to an exact rule and an exact
source clause.  These dataclasses make that traceability a first-class concern
rather than an afterthought.

Design note: three-valued verdict
----------------------------------
A missing user field is UNKNOWN, never False.  UNKNOWN always produces
NEEDS_MORE_INFO so the system asks the user for the missing fact rather than
silently deciding they are ineligible.  This is the primary safety guardrail
against wrong "you don't qualify" answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Verdict – the three-valued outcome of an eligibility check
# ---------------------------------------------------------------------------

class Verdict(str, Enum):
    """The three possible outcomes of evaluating a rule against a user profile.

    Using str as a mixin makes the value JSON-serialisable without extra steps.
    """

    ELIGIBLE = "ELIGIBLE"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"
    NEEDS_MORE_INFO = "NEEDS_MORE_INFO"


# ---------------------------------------------------------------------------
# Rule – a single eligibility criterion extracted from an official document
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Rule:
    """One eligibility criterion from an official scheme document.

    Attributes
    ----------
    rule_id:
        Unique slug, e.g. ``"pm-kisan-age-min"``.  Format: ``<scheme-id>-<name>``.
    scheme_id:
        The scheme this rule belongs to, e.g. ``"pm-kisan"``.
    field:
        The user-profile key this rule inspects, e.g. ``"age"`` or
        ``"land_holding_hectares"``.
    operator:
        Comparison to perform.  Allowed values:
        ``"eq"``, ``"neq"``, ``"lt"``, ``"lte"``, ``"gt"``, ``"gte"``,
        ``"in"``, ``"not_in"``, ``"exists"``.
    value:
        The threshold or set to compare against.  Type depends on operator
        (number, string, or list).
    verdict_if_true:
        Verdict to return when the condition holds.
        Typically ``ELIGIBLE`` or ``NOT_ELIGIBLE``.
    description_en:
        Human-readable English description of this rule (shown in citations).
    description_hi:
        Hindi translation of the description (shown in Hindi-language responses).
    source_clause:
        The clause / section / page reference in the source document,
        e.g. ``"Section 5(ii)(e)"``.
    requires_human_judgement:
        When True the evaluator always returns NEEDS_MORE_INFO regardless of
        the profile value.  Use for discretionary criteria such as
        "as decided by the committee".
    status:
        ``"pending_review"`` until a human approves it; only ``"approved"``
        rules are used by the engine.  This prevents LLM-hallucinated rules
        from reaching users.
    """

    rule_id: str
    scheme_id: str
    field: str
    operator: str
    value: Any
    verdict_if_true: Verdict
    description_en: str
    source_clause: str

    # Optional with safe defaults
    description_hi: str = ""
    requires_human_judgement: bool = False
    status: str = "pending_review"   # "pending_review" | "approved"

    def is_approved(self) -> bool:
        """Return True only for human-approved rules (safe to run in engine)."""
        return self.status == "approved"


# ---------------------------------------------------------------------------
# EligibilityResult – the full output of evaluating one rule
# ---------------------------------------------------------------------------

@dataclass
class EligibilityResult:
    """The outcome of evaluating a single Rule against a user profile.

    Attributes
    ----------
    verdict:
        One of ELIGIBLE, NOT_ELIGIBLE, NEEDS_MORE_INFO.
    rule:
        The Rule that produced this verdict (for citation).
    reason_en:
        One-sentence English explanation, ready to show the user.
        Starts with "appears eligible" / "not eligible" / "need to know …"
        to match the project's safe wording policy.
    reason_hi:
        Hindi version of the reason (empty string if not yet translated).
    missing_fields:
        Fields whose absence caused NEEDS_MORE_INFO.  Empty for other verdicts.
    """

    verdict: Verdict
    rule: Rule
    reason_en: str
    reason_hi: str = ""
    missing_fields: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a plain dict suitable for JSON serialisation or logging."""
        return {
            "verdict": self.verdict.value,
            "rule_id": self.rule.rule_id,
            "scheme_id": self.rule.scheme_id,
            "source_clause": self.rule.source_clause,
            "reason_en": self.reason_en,
            "reason_hi": self.reason_hi,
            "missing_fields": self.missing_fields,
        }
