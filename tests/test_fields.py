"""tests/test_fields.py – Field catalog and validation of LLM-extracted facts.

Run with: pytest tests/test_fields.py -v
"""

from __future__ import annotations

import math

from yojanasetu.agent.fields import (
    FieldSpec,
    build_field_catalog,
    catalog_prompt,
    clean_extracted_facts,
    coerce_value,
)

BOOL = FieldSpec("b", "boolean", "a boolean field")
NUM = FieldSpec("n", "number", "a number field")
STR = FieldSpec("s", "string", "a string field", known_values=("Uttar Pradesh",))


class TestCatalog:
    def test_catalog_built_from_approved_rules(self):
        catalog = build_field_catalog(["pm-kisan"])
        assert catalog["owns_cultivable_land"].kind == "boolean"
        assert catalog["monthly_pension_amount"].kind == "number"
        assert catalog["owns_cultivable_land"].description_en

    def test_unknown_scheme_is_skipped(self):
        assert build_field_catalog(["no-such-scheme"]) == {}

    def test_empty_scheme_list_means_all_schemes(self):
        catalog = build_field_catalog([])
        assert "owns_cultivable_land" in catalog  # pm-kisan
        assert "state_domicile" in catalog  # up-kanya-sumangala

    def test_known_string_values_collected(self):
        catalog = build_field_catalog(["up-kanya-sumangala"])
        assert "Uttar Pradesh" in catalog["state_domicile"].known_values

    def test_prompt_lists_every_field(self):
        catalog = build_field_catalog(["pm-kisan"])
        prompt = catalog_prompt(catalog)
        for name in catalog:
            assert f'"{name}"' in prompt


class TestCoerce:
    def test_boolean_accepts_only_real_booleans(self):
        assert coerce_value(BOOL, True) == (True, True)
        assert coerce_value(BOOL, False) == (True, False)
        for bad in ("yes", "true", 1, 0, None):
            assert coerce_value(BOOL, bad)[0] is False

    def test_number_rejects_bool_text_negative_and_non_finite(self):
        assert coerce_value(NUM, 0) == (True, 0)
        assert coerce_value(NUM, 5000.5) == (True, 5000.5)
        for bad in (True, "5000", -1, math.nan, math.inf, None):
            assert coerce_value(NUM, bad)[0] is False

    def test_string_snaps_to_known_casing(self):
        assert coerce_value(STR, "uttar   pradesh") == (True, "Uttar Pradesh")

    def test_string_keeps_unknown_value_but_normalises_spaces(self):
        assert coerce_value(STR, "  Bihar ") == (True, "Bihar")

    def test_string_rejects_empty_long_and_non_string(self):
        for bad in ("", "   ", "x" * 101, 5, True):
            assert coerce_value(STR, bad)[0] is False


class TestCleanExtractedFacts:
    def test_unknown_keys_and_bad_types_are_dropped(self):
        catalog = build_field_catalog(["pm-kisan"])
        raw = {
            "owns_cultivable_land": True,
            "monthly_pension_amount": True,  # bool where a number is expected
            "made_up_field": 1,
            "paid_income_tax_last_year": "no",  # text where a boolean is expected
        }
        assert clean_extracted_facts(raw, catalog) == {"owns_cultivable_land": True}

    def test_non_dict_gives_empty(self):
        catalog = build_field_catalog(["pm-kisan"])
        for bad in (None, [], "text", 5):
            assert clean_extracted_facts(bad, catalog) == {}
