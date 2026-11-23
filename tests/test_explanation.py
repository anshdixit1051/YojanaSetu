"""tests/test_explanation.py – Unit and integration tests for cited explanations.

Verifies:
1. Citation data model, dictionary serialization, and bilingual formatting.
2. RuleExplanation and SchemeExplanation structures and dictionary serialization.
3. Markdown rendering for both English and Hindi with safe disclaimers.
4. Deterministic explanation generation across ELIGIBLE, NOT_ELIGIBLE, and NEEDS_MORE_INFO.
5. Verbatim clause citations and source URL preservation.
6. Fallback handling when meta.yaml is absent.
7. Safe language policy enforcement (never promises benefits, uses 'appears eligible').
8. Unicode NFC normalization.
"""

from __future__ import annotations

import unicodedata
from datetime import date

import pytest
from pydantic import HttpUrl

from yojanasetu.engine.aggregate import evaluate_scheme_verdict
from yojanasetu.engine.loader import load_approved_rules
from yojanasetu.engine.models import (
    EligibilityResult,
    Rule,
    Verdict,
)
from yojanasetu.explanation.generator import (
    _build_citation,
    _build_rule_explanation,
    generate_explanation,
)
from yojanasetu.explanation.models import (
    DEFAULT_DISCLAIMER_EN,
    DEFAULT_DISCLAIMER_HI,
    Citation,
    RuleExplanation,
    SchemeExplanation,
)
from yojanasetu.schemas.source_meta import SchemeSourceMeta


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_meta() -> SchemeSourceMeta:
    return SchemeSourceMeta(
        scheme_id="test-scheme",
        name_en="Test Scheme Name",
        name_hi="परीक्षण योजना नाम",
        category="farmer",
        level="central",
        state="",
        source_url=HttpUrl("https://agricoop.gov.in/guidelines.pdf"),
        source_type="guidelines-pdf",
        source_language="en",
        download_date=date(2026, 3, 1),
    )


@pytest.fixture
def sample_citation(sample_meta: SchemeSourceMeta) -> Citation:
    return Citation(
        scheme_id=sample_meta.scheme_id,
        scheme_name_en=sample_meta.name_en,
        scheme_name_hi=sample_meta.name_hi,
        source_url=str(sample_meta.source_url),
        source_clause="Section 4(a)",
        download_date=str(sample_meta.download_date),
    )


# ---------------------------------------------------------------------------
# Citation Model Tests
# ---------------------------------------------------------------------------

class TestCitationModel:
    def test_as_dict(self, sample_citation: Citation):
        data = sample_citation.as_dict()
        assert data["scheme_id"] == "test-scheme"
        assert data["scheme_name_en"] == "Test Scheme Name"
        assert data["scheme_name_hi"] == "परीक्षण योजना नाम"
        assert data["source_clause"] == "Section 4(a)"
        assert data["source_url"] == "https://agricoop.gov.in/guidelines.pdf"

    def test_format_citation_en(self, sample_citation: Citation):
        formatted = sample_citation.format_citation(language="en")
        assert "Test Scheme Name (Section 4(a)) | Source: https://agricoop.gov.in/guidelines.pdf" == formatted
        assert unicodedata.is_normalized("NFC", formatted)

    def test_format_citation_hi(self, sample_citation: Citation):
        formatted = sample_citation.format_citation(language="hi")
        assert "परीक्षण योजना नाम (Section 4(a)) | स्रोत: https://agricoop.gov.in/guidelines.pdf" == formatted
        assert unicodedata.is_normalized("NFC", formatted)

    def test_format_citation_hi_fallback_to_en_name(self):
        c = Citation(
            scheme_id="s",
            scheme_name_en="English Name",
            scheme_name_hi="",
            source_url="https://gov.in",
            source_clause="Cl. 1",
            download_date="2026-01-01",
        )
        assert "English Name (Cl. 1) | स्रोत: https://gov.in" == c.format_citation("hi")


# ---------------------------------------------------------------------------
# RuleExplanation Model Tests
# ---------------------------------------------------------------------------

class TestRuleExplanationModel:
    def test_as_dict(self, sample_citation: Citation):
        rule_expl = RuleExplanation(
            rule_id="r1",
            field="age",
            verdict="ELIGIBLE",
            explanation_en="Age criteria satisfied",
            explanation_hi="आयु मानदंड पूरा हुआ",
            citation=sample_citation,
        )
        d = rule_expl.as_dict()
        assert d["rule_id"] == "r1"
        assert d["field"] == "age"
        assert d["verdict"] == "ELIGIBLE"
        assert d["explanation_en"] == "Age criteria satisfied"
        assert d["citation"]["source_clause"] == "Section 4(a)"


# ---------------------------------------------------------------------------
# SchemeExplanation Model & Markdown Rendering Tests
# ---------------------------------------------------------------------------

class TestSchemeExplanationRendering:
    def test_render_markdown_eligible_en(self, sample_citation: Citation):
        rule_expl = RuleExplanation(
            rule_id="r1",
            field="land",
            verdict="ELIGIBLE",
            explanation_en="Owns cultivable land",
            explanation_hi="कृषि भूमि है",
            citation=sample_citation,
        )
        expl = SchemeExplanation(
            scheme_id="test-scheme",
            scheme_name_en="Test Scheme Name",
            scheme_name_hi="परीक्षण योजना नाम",
            verdict="ELIGIBLE",
            summary_en="Appears eligible based on published criteria.",
            summary_hi="प्रकाशित मानदंडों के आधार पर पात्र प्रतीत होते हैं।",
            deciding_rules=[rule_expl],
            all_rules=[rule_expl],
            portal_url="https://agricoop.gov.in",
        )
        md = expl.render_markdown(language="en")
        assert "### Test Scheme Name" in md
        assert "**Verdict:** ELIGIBLE" in md
        assert "Appears eligible based on published criteria." in md
        assert "Deciding Criteria & Citations:" in md
        assert "- Owns cultivable land" in md
        assert "*(Clause: Section 4(a))*" in md
        assert "[Official Portal](https://agricoop.gov.in)" in md
        assert f"> ⚠️ *{DEFAULT_DISCLAIMER_EN}*" in md
        assert unicodedata.is_normalized("NFC", md)

    def test_render_markdown_eligible_hi(self, sample_citation: Citation):
        rule_expl = RuleExplanation(
            rule_id="r1",
            field="land",
            verdict="ELIGIBLE",
            explanation_en="Owns cultivable land",
            explanation_hi="कृषि भूमि उपलब्ध है",
            citation=sample_citation,
        )
        expl = SchemeExplanation(
            scheme_id="test-scheme",
            scheme_name_en="Test Scheme Name",
            scheme_name_hi="परीक्षण योजना नाम",
            verdict="ELIGIBLE",
            summary_en="Appears eligible based on published criteria.",
            summary_hi="प्रकाशित मानदंडों के आधार पर पात्र प्रतीत होते हैं।",
            deciding_rules=[rule_expl],
            all_rules=[rule_expl],
            portal_url="https://agricoop.gov.in",
        )
        md = expl.render_markdown(language="hi")
        assert "### परीक्षण योजना नाम" in md
        assert "**निष्कर्ष:** ELIGIBLE" in md
        assert "प्रकाशित मानदंडों के आधार पर पात्र प्रतीत होते हैं।" in md
        assert "निर्णायक मानदंड एवं संदर्भ:" in md
        assert "- कृषि भूमि उपलब्ध है" in md
        assert "*(खंड: Section 4(a))*" in md
        assert "[आधिकारिक पोर्टल](https://agricoop.gov.in)" in md
        assert f"> ⚠️ *{DEFAULT_DISCLAIMER_HI}*" in md
        assert unicodedata.is_normalized("NFC", md)

    def test_render_markdown_not_eligible(self, sample_citation: Citation):
        rule_expl = RuleExplanation(
            rule_id="r2",
            field="taxpayer",
            verdict="NOT_ELIGIBLE",
            explanation_en="Paid income tax last year (excluded)",
            explanation_hi="गत वर्ष आयकर दिया (अपवर्जित)",
            citation=sample_citation,
        )
        expl = SchemeExplanation(
            scheme_id="test-scheme",
            scheme_name_en="Test Scheme Name",
            scheme_name_hi="परीक्षण योजना नाम",
            verdict="NOT_ELIGIBLE",
            summary_en="Not eligible for Test Scheme Name.",
            summary_hi="परीक्षण योजना नाम के लिए पात्र नहीं।",
            deciding_rules=[rule_expl],
        )
        md = expl.render_markdown(language="en")
        assert "**Verdict:** NOT_ELIGIBLE" in md
        assert "- Paid income tax last year (excluded)" in md
        # Disclaimers for "appears eligible" must NOT be appended when NOT_ELIGIBLE
        assert "⚠️" not in md

    def test_render_markdown_needs_more_info(self):
        expl = SchemeExplanation(
            scheme_id="test-scheme",
            scheme_name_en="Test Scheme Name",
            scheme_name_hi="परीक्षण योजना नाम",
            verdict="NEEDS_MORE_INFO",
            summary_en="Additional information needed.",
            summary_hi="अतिरिक्त जानकारी चाहिए।",
            missing_fields=["owns_cultivable_land", "paid_income_tax_last_year"],
        )
        md_en = expl.render_markdown(language="en")
        assert "**Verdict:** NEEDS_MORE_INFO" in md_en
        assert "**Still Needed Details:** owns_cultivable_land, paid_income_tax_last_year" in md_en

        md_hi = expl.render_markdown(language="hi")
        assert "**अपेक्षित अतिरिक्त जानकारी:** owns_cultivable_land, paid_income_tax_last_year" in md_hi


# ---------------------------------------------------------------------------
# Generator Integration Tests (Real Rules & Metadata)
# ---------------------------------------------------------------------------

class TestExplanationGeneratorRealData:
    def test_pm_kisan_eligible_explanation(self):
        rules = load_approved_rules("pm-kisan")
        profile = {
            "owns_cultivable_land": True,
            "is_institutional_landholder": False,
            "paid_income_tax_last_year": False,
            "holds_constitutional_post": False,
            "monthly_pension_amount": 0,
        }
        verdict = evaluate_scheme_verdict("pm-kisan", rules, profile)
        assert verdict.verdict == Verdict.ELIGIBLE

        expl = generate_explanation(verdict)

        assert expl.scheme_id == "pm-kisan"
        assert expl.verdict == "ELIGIBLE"
        assert "Pradhan Mantri Kisan Samman Nidhi" in expl.scheme_name_en
        assert "प्रधानमंत्री किसान सम्मान निधि" in expl.scheme_name_hi
        assert expl.portal_url.startswith("https://pmkisan.gov.in/")
        assert "Appears eligible for Pradhan Mantri Kisan Samman Nidhi" in expl.summary_en
        assert "based on published criteria" in expl.summary_en
        assert "प्रकाशित मानदंडों के आधार पर आप प्रधानमंत्री किसान सम्मान निधि" in expl.summary_hi
        assert len(expl.all_rules) == 5
        # Deciding rules for ELIGIBLE lists all 5 satisfying criteria
        assert len(expl.deciding_rules) == 5
        for r in expl.deciding_rules:
            assert r.citation.source_url.startswith("https://pmkisan.gov.in/")
            assert len(r.citation.source_clause) > 0

    def test_pm_kisan_not_eligible_explanation(self):
        rules = load_approved_rules("pm-kisan")
        profile = {
            "owns_cultivable_land": True,
            "is_institutional_landholder": False,
            "paid_income_tax_last_year": True,  # Disqualifying exclusion
            "holds_constitutional_post": False,
            "monthly_pension_amount": 0,
        }
        verdict = evaluate_scheme_verdict("pm-kisan", rules, profile)
        assert verdict.verdict == Verdict.NOT_ELIGIBLE

        expl = generate_explanation(verdict)

        assert expl.verdict == "NOT_ELIGIBLE"
        assert "Not eligible for Pradhan Mantri Kisan Samman Nidhi (PM-KISAN): did not satisfy 1 published criteria." == expl.summary_en
        assert "प्रधानमंत्री किसान सम्मान निधि के लिए पात्र नहीं: 1 प्रकाशित मानदंड पूरे नहीं होते हैं।" == expl.summary_hi
        assert len(expl.deciding_rules) == 1
        deciding = expl.deciding_rules[0]
        assert deciding.rule_id == "pm-kisan-no-income-tax"
        assert deciding.field == "paid_income_tax_last_year"
        assert deciding.verdict == "NOT_ELIGIBLE"
        assert deciding.citation.source_clause == "Section 5(ii)(e) – Exclusion Categories"

    def test_pm_kisan_missing_fields_explanation(self):
        rules = load_approved_rules("pm-kisan")
        verdict = evaluate_scheme_verdict("pm-kisan", rules, {})
        assert verdict.verdict == Verdict.NEEDS_MORE_INFO

        expl = generate_explanation(verdict)

        assert expl.verdict == "NEEDS_MORE_INFO"
        assert "Additional information needed to evaluate Pradhan Mantri Kisan Samman Nidhi" in expl.summary_en
        assert "Still needed details:" in expl.summary_en
        assert "owns_cultivable_land" in expl.missing_fields
        assert "प्रधानमंत्री किसान सम्मान निधि की पात्रता निर्धारण के लिए अतिरिक्त जानकारी चाहिए।" in expl.summary_hi

    def test_human_review_rule_explanation(self, sample_meta: SchemeSourceMeta):
        rule = Rule(
            rule_id="r-judgement",
            scheme_id="test-scheme",
            field="committee_decision",
            operator="eq",
            value=True,
            verdict_if_true=Verdict.ELIGIBLE,
            source_clause="Clause 9",
            description_en="Approved by district committee",
            requires_human_judgement=True,
            status="approved",
            source_quote="Approved by committee",
        )
        res = EligibilityResult(
            verdict=Verdict.NEEDS_MORE_INFO,
            rule=rule,
            reason_en="Requires human judgement: committee decision cannot be decided automatically.",
        )
        from yojanasetu.engine.aggregate import SchemeVerdict
        sv = SchemeVerdict(
            scheme_id="test-scheme",
            verdict=Verdict.NEEDS_MORE_INFO,
            rule_results=[res],
            deciding_rules=[],
            missing_fields=[],
            needs_human_review=["r-judgement"],
        )
        expl = generate_explanation(sv, meta=sample_meta)
        assert expl.verdict == "NEEDS_MORE_INFO"
        assert "requires official discretion or committee review" in expl.summary_en
        assert "आधिकारिक या समिति का निर्णय आवश्यक है" in expl.summary_hi

    def test_fallback_when_meta_yaml_absent(self):
        from yojanasetu.engine.aggregate import SchemeVerdict
        sv = SchemeVerdict(
            scheme_id="non-existent-scheme",
            verdict=Verdict.NEEDS_MORE_INFO,
            rule_results=[],
            deciding_rules=[],
            missing_fields=[],
            needs_human_review=[],
            note="No rules on disk.",
        )
        expl = generate_explanation(sv, meta=None)
        assert expl.scheme_id == "non-existent-scheme"
        assert expl.scheme_name_en == "non-existent-scheme"
        assert expl.portal_url == ""
        assert expl.summary_en == "No rules on disk."

    def test_as_dict_json_serializable(self):
        rules = load_approved_rules("pm-kisan")
        profile = {
            "owns_cultivable_land": True,
            "is_institutional_landholder": False,
            "paid_income_tax_last_year": False,
            "holds_constitutional_post": False,
            "monthly_pension_amount": 0,
        }
        verdict = evaluate_scheme_verdict("pm-kisan", rules, profile)
        expl = generate_explanation(verdict)
        d = expl.as_dict()

        import json
        dumped = json.dumps(d)
        assert "Pradhan Mantri Kisan Samman Nidhi" in dumped
        assert d["verdict"] == "ELIGIBLE"
        assert d["disclaimer_en"] == DEFAULT_DISCLAIMER_EN


# ---------------------------------------------------------------------------
# API Integration Test for Explanations
# ---------------------------------------------------------------------------

class TestApiEligibilityExplanation:
    def test_api_eligibility_returns_explanation(self):
        from fastapi.testclient import TestClient
        from yojanasetu.api.app import app

        client = TestClient(app)
        resp = client.post("/eligibility", json={
            "scheme_id": "pm-kisan",
            "profile": {
                "owns_cultivable_land": True,
                "is_institutional_landholder": False,
                "paid_income_tax_last_year": False,
                "holds_constitutional_post": False,
                "monthly_pension_amount": 0,
            },
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "explanation" in data
        expl = data["explanation"]
        assert expl is not None
        assert expl["scheme_id"] == "pm-kisan"
        assert expl["verdict"] == "ELIGIBLE"
        assert "Appears eligible based on published criteria" in expl["disclaimer_en"]
        assert expl["portal_url"].startswith("https://pmkisan.gov.in/")
        assert len(expl["deciding_rules"]) == 5
