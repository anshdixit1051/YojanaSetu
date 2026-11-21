"""Central configuration loaded from environment variables / .env file.

Why pydantic-settings: values are type-checked at startup, and SecretStr
stops API keys from appearing in logs or tracebacks.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings. Every field can be overridden by an env var."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["dev", "test", "prod"] = "dev"
    state_name: str = "UNSET"
    log_level: str = "INFO"
    database_url: str = "sqlite:///./data/yojanasetu.db"
    data_dir: Path = PROJECT_ROOT / "data"

    groq_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None

    def require_groq_key(self) -> str:
        """Return the Groq key or fail with a clear message.

        Keys are optional in Phase 0 (no LLM calls yet), but any code that
        needs one must call this instead of reading the field directly.
        """
        if self.groq_api_key is None:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
        return self.groq_api_key.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (loaded once per process)."""
    return Settings()
