"""explanation – Cited explanation generator for eligibility outcomes (Hindi + English)."""

from yojanasetu.explanation.generator import generate_explanation
from yojanasetu.explanation.models import (
    DEFAULT_DISCLAIMER_EN,
    DEFAULT_DISCLAIMER_HI,
    Citation,
    RuleExplanation,
    SchemeExplanation,
)

__all__ = [
    "Citation",
    "RuleExplanation",
    "SchemeExplanation",
    "DEFAULT_DISCLAIMER_EN",
    "DEFAULT_DISCLAIMER_HI",
    "generate_explanation",
]
