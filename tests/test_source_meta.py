"""Tests for the scheme provenance model."""

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from yojanasetu.schemas.source_meta import SchemeSourceMeta, load_meta


def valid_data() -> dict[str, Any]:
    """A minimal valid record. Tests copy it and break one field."""
    return {
        "scheme_id": "test-scheme",
        "name_en": "Test Scheme",
        "category": "pension",
        "level": "central",
        "source_url": "https://www.india.gov.in/test-scheme",
        "source_type": "portal-page",
        "source_language": "en",
        "download_date": date(2026, 11, 21),
    }


def test_valid_record_passes() -> None:
    meta = SchemeSourceMeta.model_validate(valid_data())
    assert meta.scheme_id == "test-scheme"
    assert meta.name_hi == ""


@pytest.mark.parametrize("missing", ["source_url", "download_date", "name_en"])
def test_required_fields_are_enforced(missing: str) -> None:
    data = valid_data()
    del data[missing]
    with pytest.raises(ValidationError):
        SchemeSourceMeta.model_validate(data)


@pytest.mark.parametrize("bad_id", ["Test Scheme", "test_scheme", "-test", "TEST"])
def test_scheme_id_must_be_a_slug(bad_id: str) -> None:
    data = valid_data() | {"scheme_id": bad_id}
    with pytest.raises(ValidationError):
        SchemeSourceMeta.model_validate(data)


def test_state_scheme_needs_a_state() -> None:
    data = valid_data() | {"level": "state", "state": ""}
    with pytest.raises(ValidationError, match="state is required"):
        SchemeSourceMeta.model_validate(data)


def test_placeholder_url_is_rejected() -> None:
    data = valid_data() | {"source_url": "https://example.gov.in/..."}
    with pytest.raises(ValidationError, match="placeholder"):
        SchemeSourceMeta.model_validate(data)


def test_future_download_date_is_rejected() -> None:
    data = valid_data() | {"download_date": date.today() + timedelta(days=1)}
    with pytest.raises(ValidationError, match="future"):
        SchemeSourceMeta.model_validate(data)


def test_unknown_field_is_rejected() -> None:
    data = valid_data() | {"downlod_date": "2026-11-21"}
    with pytest.raises(ValidationError):
        SchemeSourceMeta.model_validate(data)


def test_load_meta_reads_utf8_hindi(tmp_path: Path) -> None:
    data = valid_data() | {"name_hi": "परीक्षण योजना"}
    path = tmp_path / "meta.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    assert load_meta(path).name_hi == "परीक्षण योजना"


def test_shipped_template_is_not_a_valid_scheme() -> None:
    """The template must fail validation so it can never be mistaken for real data."""
    template = Path(__file__).resolve().parents[1] / "data/raw/_TEMPLATE/meta.yaml"
    with pytest.raises(ValueError):
        load_meta(template)
