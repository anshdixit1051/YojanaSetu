"""Tests for configuration loading and secret handling."""

import pytest

from yojanasetu.config import Settings


def test_defaults_without_env_file() -> None:
    settings = Settings(_env_file=None)
    assert settings.app_env == "dev"
    assert settings.groq_api_key is None


def test_env_var_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STATE_NAME", "TestState")
    settings = Settings(_env_file=None)
    assert settings.state_name == "TestState"


def test_secret_not_leaked_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "super-secret-value")
    settings = Settings(_env_file=None)
    assert "super-secret-value" not in repr(settings)
    assert settings.require_groq_key() == "super-secret-value"


def test_missing_key_raises_clear_error() -> None:
    settings = Settings(_env_file=None, groq_api_key=None)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        settings.require_groq_key()
