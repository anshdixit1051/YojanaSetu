"""Provenance metadata for each scheme folder in data/raw/.

Why this exists: every rule we later extract must be traceable to an official
source and a download date. If the metadata is incomplete, the scheme is
rejected here, long before any rule is written.
"""

import re
from datetime import date
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

SLUG_PATTERN = r"^[a-z0-9]+(-[a-z0-9]+)*$"
SHA256_OR_EMPTY = r"^([0-9a-f]{64})?$"


class SchemeSourceMeta(BaseModel):
    """Where a scheme's source text came from, and when we copied it."""

    # extra="forbid": a typo like "downlod_date" is an error, not silently ignored.
    model_config = ConfigDict(extra="forbid")

    scheme_id: str = Field(pattern=SLUG_PATTERN)
    name_en: str = Field(min_length=1)
    name_hi: str = ""
    category: Literal[
        "scholarship", "farmer", "health", "pension", "business-credit", "women-girl"
    ]
    level: Literal["central", "state"]
    state: str = ""
    source_url: HttpUrl
    source_type: Literal["guidelines-pdf", "gazette", "portal-page", "other"]
    source_language: Literal["en", "hi", "bilingual"]
    download_date: date
    source_sha256: str = Field(default="", pattern=SHA256_OR_EMPTY)
    notes: str = ""

    @field_validator("source_url")
    @classmethod
    def reject_placeholder_url(cls, url: HttpUrl) -> HttpUrl:
        """The template URL must never pass as a real source."""
        if "example" in (url.host or ""):
            raise ValueError("source_url is still the template placeholder")
        return url

    @field_validator("download_date")
    @classmethod
    def reject_future_date(cls, value: date) -> date:
        """A download date in the future is always a typo."""
        if value > date.today():
            raise ValueError("download_date cannot be in the future")
        return value

    @model_validator(mode="after")
    def state_required_for_state_schemes(self) -> Self:
        """A state-level scheme must say which state it belongs to."""
        if self.level == "state" and not self.state.strip():
            raise ValueError("state is required when level is 'state'")
        return self


def load_meta(path: Path) -> SchemeSourceMeta:
    """Read and validate a meta.yaml file.

    Raises:
        ValueError: if the file is not a YAML mapping or fails validation.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return SchemeSourceMeta.model_validate(data)
