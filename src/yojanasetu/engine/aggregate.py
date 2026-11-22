"""engine/aggregate.py – Combine per-rule results into ONE verdict per scheme.

Why this file exists
--------------------
``evaluate_scheme`` returns one result per rule.  Someone has to combine them
into a single answer ("is the user eligible for this scheme?").  That step
decides whether we tell a person "you qualify", so it MUST be plain Python and
never an LLM.

Policy (conservative on purpose)
--------------------------------
1. Any rule NOT_ELIGIBLE                 -> scheme NOT_ELIGIBLE
   (the failing rules are the "deciding rules"; no need to ask for more info)
2. Otherwise, any rule NEEDS_MORE_INFO   -> scheme NEEDS_MORE_INFO
   (missing fields are listed; rules needing human judgement are listed
   separately because the user cannot answer those)
3. Otherwise, every approved rule ELIGIBLE -> scheme ELIGIBLE
4. Zero approved rules                   -> NEEDS_MORE_INFO, never ELIGIBLE
   (an empty AND is vacuously true; we refuse to say "you qualify" on no rules)

This is Kleene three-valued AND.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from yojanasetu.engine.evaluator import evaluate_scheme
from yojanasetu.engine.models import EligibilityResult, Rule, Verdict


@dataclass
class SchemeVerdict:
    """The deterministic, scheme-level outcome.

    Attributes
    ----------
    scheme_id:
        Scheme this verdict is for.
    verdict:
        ELIGIBLE, NOT_ELIGIBLE or NEEDS_MORE_INFO.
    deciding_rules:
        Results that decided the verdict: the failing rules for NOT_ELIGIBLE,
        the unresolved rules for NEEDS_MORE_INFO, all rules for ELIGIBLE.
    rule_results:
        Every per-rule result, in rule order (for citations).
    missing_fields:
        Fields the user can supply to resolve NEEDS_MORE_INFO (deduplicated).
    needs_human_review:
        rule_ids of discretionary rules the engine cannot decide.
    note:
        Short machine-written remark, e.g. "no approved rules".
    """

    scheme_id: str
    verdict: Verdict
    deciding_rules: list[EligibilityResult] = field(default_factory=list)
    rule_results: list[EligibilityResult] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    needs_human_review: list[str] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Plain-dict form for JSON responses and agent state."""
        return {
            "scheme_id": self.scheme_id,
            "verdict": self.verdict.value,
            "deciding_rules": [r.as_dict() for r in self.deciding_rules],
            "missing_fields": list(self.missing_fields),
            "needs_human_review": list(self.needs_human_review),
            "note": self.note,
        }


def evaluate_scheme_verdict(
    scheme_id: str,
    rules: list[Rule],
    profile: dict[str, Any],
) -> SchemeVerdict:
    """Evaluate all approved rules and return one deterministic scheme verdict."""
    results = evaluate_scheme(rules, profile)

    if not results:
        return SchemeVerdict(
            scheme_id=scheme_id,
            verdict=Verdict.NEEDS_MORE_INFO,
            note="No approved rules for this scheme yet.",
        )

    failed = [r for r in results if r.verdict == Verdict.NOT_ELIGIBLE]
    if failed:
        return SchemeVerdict(
            scheme_id=scheme_id,
            verdict=Verdict.NOT_ELIGIBLE,
            deciding_rules=failed,
            rule_results=results,
        )

    unresolved = [r for r in results if r.verdict == Verdict.NEEDS_MORE_INFO]
    if unresolved:
        missing: list[str] = []
        human: list[str] = []
        for r in unresolved:
            if r.rule.requires_human_judgement:
                human.append(r.rule.rule_id)
            else:
                missing.extend(r.missing_fields)
        return SchemeVerdict(
            scheme_id=scheme_id,
            verdict=Verdict.NEEDS_MORE_INFO,
            deciding_rules=unresolved,
            rule_results=results,
            missing_fields=list(dict.fromkeys(missing)),
            needs_human_review=human,
        )

    return SchemeVerdict(
        scheme_id=scheme_id,
        verdict=Verdict.ELIGIBLE,
        deciding_rules=results,
        rule_results=results,
    )
