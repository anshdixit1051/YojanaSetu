"""explanation/generator.py – Generate cited bilingual explanations for scheme verdicts.

Contract
--------
generate_explanation(verdict, meta=None) → SchemeExplanation
    Deterministically combines the SchemeVerdict, the underlying rule citations,
    and official metadata into a bilingual explanation object.
    Never changes or overrides any eligibility verdict.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any

from yojanasetu.engine.aggregate import SchemeVerdict
from yojanasetu.engine.models import EligibilityResult, Verdict
from yojanasetu.explanation.models import (
    DEFAULT_DISCLAIMER_EN,
    DEFAULT_DISCLAIMER_HI,
    Citation,
    RuleExplanation,
    SchemeExplanation,
)
from yojanasetu.schemas.source_meta import SchemeSourceMeta, load_meta

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def _get_scheme_meta(scheme_id: str) -> SchemeSourceMeta | None:
    """Load SchemeSourceMeta for a scheme if available on disk."""
    meta_path = RAW_DIR / scheme_id / "meta.yaml"
    if meta_path.exists():
        try:
            return load_meta(meta_path)
        except Exception:
            return None
    return None


def _build_citation(result: EligibilityResult, meta: SchemeSourceMeta | None) -> Citation:
    """Build official citation provenance from a rule result and scheme metadata."""
    scheme_id = result.rule.scheme_id
    scheme_name_en = meta.name_en if meta else scheme_id
    scheme_name_hi = meta.name_hi if meta else ""
    source_url = str(meta.source_url) if meta else ""
    download_date = str(meta.download_date) if meta else ""

    return Citation(
        scheme_id=scheme_id,
        scheme_name_en=scheme_name_en,
        scheme_name_hi=scheme_name_hi,
        source_url=source_url,
        source_clause=result.rule.source_clause,
        download_date=download_date,
    )


def _build_rule_explanation(
    result: EligibilityResult, meta: SchemeSourceMeta | None
) -> RuleExplanation:
    """Create a bilingual RuleExplanation with citation."""
    citation = _build_citation(result, meta)
    expl_en = result.reason_en or result.rule.description_en
    expl_hi = result.reason_hi or result.rule.description_hi or expl_en

    return RuleExplanation(
        rule_id=result.rule.rule_id,
        field=result.rule.field,
        verdict=result.verdict.value,
        explanation_en=expl_en,
        explanation_hi=expl_hi,
        citation=citation,
    )


def generate_explanation(
    verdict: SchemeVerdict,
    meta: SchemeSourceMeta | None = None,
) -> SchemeExplanation:
    """Generate a deterministic, citation-backed bilingual SchemeExplanation.

    Parameters
    ----------
    verdict:
        SchemeVerdict returned by engine.aggregate.evaluate_scheme_verdict.
    meta:
        Optional SchemeSourceMeta provenance. If None, loads from data/raw/<scheme-id>/meta.yaml.

    Returns
    -------
    SchemeExplanation
        Bilingual structured explanation citing official sources.
    """
    if meta is None:
        meta = _get_scheme_meta(verdict.scheme_id)

    scheme_id = verdict.scheme_id
    name_en = meta.name_en if meta else scheme_id
    name_hi = meta.name_hi if meta else ""
    portal_url = str(meta.source_url) if meta else ""

    all_rule_expls = [_build_rule_explanation(r, meta) for r in verdict.rule_results]
    deciding_rule_expls = [_build_rule_explanation(r, meta) for r in verdict.deciding_rules]

    if verdict.verdict == Verdict.ELIGIBLE:
        summary_en = (
            f"Appears eligible for {name_en} based on published criteria "
            "(verified against official guidelines)."
        )
        summary_hi = (
            f"प्रकाशित मानदंडों के आधार पर आप {name_hi or name_en} के लिए पात्र प्रतीत होते हैं "
            "(आधिकारिक दिशानिर्देशों के अनुसार)।"
        )
    elif verdict.verdict == Verdict.NOT_ELIGIBLE:
        failed_count = len(verdict.deciding_rules)
        summary_en = (
            f"Not eligible for {name_en}: did not satisfy {failed_count} "
            f"published criteria."
        )
        summary_hi = (
            f"{name_hi or name_en} के लिए पात्र नहीं: {failed_count} प्रकाशित "
            "मानदंड पूरे नहीं होते हैं।"
        )
    else:  # NEEDS_MORE_INFO
        if verdict.missing_fields:
            fields_str = ", ".join(verdict.missing_fields)
            summary_en = (
                f"Additional information needed to evaluate {name_en}. "
                f"Still needed details: {fields_str}."
            )
            summary_hi = (
                f"{name_hi or name_en} की पात्रता निर्धारण के लिए अतिरिक्त जानकारी चाहिए। "
                f"अपेक्षित विवरण: {fields_str}।"
            )
        elif verdict.needs_human_review:
            summary_en = (
                f"Evaluation for {name_en} requires official discretion or committee review "
                "and cannot be decided automatically."
            )
            summary_hi = (
                f"{name_hi or name_en} के लिए कुछ मानदंडों पर आधिकारिक या समिति का निर्णय "
                "आवश्यक है और स्वतः तय नहीं हो सकता।"
            )
        else:
            summary_en = verdict.note or f"No approved rules available for {name_en}."
            summary_hi = f"{name_hi or name_en} के लिए वर्तमान में कोई स्वीकृत नियम उपलब्ध नहीं हैं।"

    return SchemeExplanation(
        scheme_id=scheme_id,
        scheme_name_en=unicodedata.normalize("NFC", name_en),
        scheme_name_hi=unicodedata.normalize("NFC", name_hi),
        verdict=verdict.verdict.value,
        summary_en=unicodedata.normalize("NFC", summary_en),
        summary_hi=unicodedata.normalize("NFC", summary_hi),
        deciding_rules=deciding_rule_expls,
        all_rules=all_rule_expls,
        missing_fields=verdict.missing_fields,
        needs_human_review=verdict.needs_human_review,
        portal_url=portal_url,
        disclaimer_en=DEFAULT_DISCLAIMER_EN,
        disclaimer_hi=DEFAULT_DISCLAIMER_HI,
    )
