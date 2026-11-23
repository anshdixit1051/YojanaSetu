"""explanation/models.py – Data models for bilingual cited explanations.

Why these models exist
----------------------
Every eligibility answer shown to a citizen must be transparent, verifiable,
and bilingual (English and Hindi). These models define the structured representation
of an explanation, binding the deterministic verdict directly to official source
citations (URL, clause, download date) and standardized safe disclaimers.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any


DEFAULT_DISCLAIMER_EN = (
    "Appears eligible based on published criteria. Meeting the criteria "
    "does not guarantee the benefit; please confirm on the official portal."
)

DEFAULT_DISCLAIMER_HI = (
    "प्रकाशित मानदंडों के आधार पर पात्र प्रतीत होते हैं। मानदंड पूरे होने का अर्थ "
    "लाभ की गारंटी नहीं है; कृपया आधिकारिक पोर्टल पर पुष्टि करें।"
)


@dataclass(frozen=True)
class Citation:
    """Official source provenance linking a rule to its authoritative publication."""

    scheme_id: str
    scheme_name_en: str
    scheme_name_hi: str
    source_url: str
    source_clause: str
    download_date: str

    def as_dict(self) -> dict[str, str]:
        """Convert citation to dictionary."""
        return {
            "scheme_id": self.scheme_id,
            "scheme_name_en": self.scheme_name_en,
            "scheme_name_hi": self.scheme_name_hi,
            "source_url": self.source_url,
            "source_clause": self.source_clause,
            "download_date": self.download_date,
        }

    def format_citation(self, language: str = "en") -> str:
        """Format a human-readable citation string."""
        if language == "hi":
            text = (
                f"{self.scheme_name_hi or self.scheme_name_en} "
                f"({self.source_clause}) | स्रोत: {self.source_url}"
            )
        else:
            text = (
                f"{self.scheme_name_en} "
                f"({self.source_clause}) | Source: {self.source_url}"
            )
        return unicodedata.normalize("NFC", text)


@dataclass(frozen=True)
class RuleExplanation:
    """Bilingual explanation for a single rule's evaluation outcome."""

    rule_id: str
    field: str
    verdict: str  # "ELIGIBLE" | "NOT_ELIGIBLE" | "NEEDS_MORE_INFO"
    explanation_en: str
    explanation_hi: str
    citation: Citation

    def as_dict(self) -> dict[str, Any]:
        """Convert rule explanation to dictionary."""
        return {
            "rule_id": self.rule_id,
            "field": self.field,
            "verdict": self.verdict,
            "explanation_en": self.explanation_en,
            "explanation_hi": self.explanation_hi,
            "citation": self.citation.as_dict(),
        }


@dataclass
class SchemeExplanation:
    """Complete bilingual cited explanation for a scheme."""

    scheme_id: str
    scheme_name_en: str
    scheme_name_hi: str
    verdict: str  # "ELIGIBLE" | "NOT_ELIGIBLE" | "NEEDS_MORE_INFO"
    summary_en: str
    summary_hi: str
    deciding_rules: list[RuleExplanation] = field(default_factory=list)
    all_rules: list[RuleExplanation] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    needs_human_review: list[str] = field(default_factory=list)
    portal_url: str = ""
    disclaimer_en: str = DEFAULT_DISCLAIMER_EN
    disclaimer_hi: str = DEFAULT_DISCLAIMER_HI

    def as_dict(self) -> dict[str, Any]:
        """Convert complete scheme explanation to dictionary."""
        return {
            "scheme_id": self.scheme_id,
            "scheme_name_en": self.scheme_name_en,
            "scheme_name_hi": self.scheme_name_hi,
            "verdict": self.verdict,
            "summary_en": self.summary_en,
            "summary_hi": self.summary_hi,
            "deciding_rules": [r.as_dict() for r in self.deciding_rules],
            "all_rules": [r.as_dict() for r in self.all_rules],
            "missing_fields": list(self.missing_fields),
            "needs_human_review": list(self.needs_human_review),
            "portal_url": self.portal_url,
            "disclaimer_en": self.disclaimer_en,
            "disclaimer_hi": self.disclaimer_hi,
        }

    def render_markdown(self, language: str = "en") -> str:
        """Render a formatted markdown block ready for chat or UI presentation."""
        is_hi = language == "hi"
        name = (self.scheme_name_hi if is_hi and self.scheme_name_hi else self.scheme_name_en) or self.scheme_id
        summary = self.summary_hi if is_hi else self.summary_en
        disclaimer = self.disclaimer_hi if is_hi else self.disclaimer_en

        lines = [f"### {name}"]
        lines.append(f"**{'निष्कर्ष' if is_hi else 'Verdict'}:** {self.verdict}\n")
        lines.append(summary)

        if self.deciding_rules:
            header = "निर्णायक मानदंड एवं संदर्भ:" if is_hi else "Deciding Criteria & Citations:"
            lines.append(f"\n**{header}**")
            for r in self.deciding_rules:
                expl = r.explanation_hi if is_hi and r.explanation_hi else r.explanation_en
                lines.append(f"- {expl}")
                lines.append(f"  *({'खंड' if is_hi else 'Clause'}: {r.citation.source_clause})*")

        if self.missing_fields:
            header = "अपेक्षित अतिरिक्त जानकारी:" if is_hi else "Still Needed Details:"
            lines.append(f"\n**{header}** {', '.join(self.missing_fields)}")

        if self.portal_url:
            portal_label = "आधिकारिक पोर्टल" if is_hi else "Official Portal"
            lines.append(f"\n[{portal_label}]({self.portal_url})")

        if self.verdict == "ELIGIBLE":
            lines.append(f"\n> ⚠️ *{disclaimer}*")

        return unicodedata.normalize("NFC", "\n".join(lines))
