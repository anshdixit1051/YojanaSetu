"""llm.py – The ONE place YojanaSetu calls the Gemini API.

Why a single module
-------------------
Both rule extraction (Phase 3) and the intake agent (Phase 5) need an LLM.
Keeping the SDK call here means the model name, timeout and error handling are
fixed in one spot, and tests mock a single function.

SDK note
--------
Uses the current ``google-genai`` package.  The older ``google-generativeai``
package is deprecated and no longer maintained.

Model note
----------
Gemini model names are retired regularly (``gemini-1.5-flash`` and
``gemini-2.0-flash`` now return 404).  The model comes from ``MODEL_NAME`` in
``.env``; check https://ai.google.dev/gemini-api/docs/models for what your key
can use.
"""

from __future__ import annotations

from yojanasetu.config import get_settings

REQUEST_TIMEOUT_MS = 30_000


def generate_text(
    system: str,
    user: str,
    *,
    api_key: str | None = None,
    model_name: str | None = None,
) -> str:
    """Send one prompt to Gemini and return the text reply.

    Parameters
    ----------
    system:
        System instruction (the model's role and rules).
    user:
        The user content for this call.
    api_key, model_name:
        Optional overrides; default to the values in settings / ``.env``.

    Raises
    ------
    RuntimeError
        If no API key is configured or the SDK is not installed.
    """
    settings = get_settings()

    key = api_key
    if key is None and settings.gemini_api_key is not None:
        key = settings.gemini_api_key.get_secret_value()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")

    try:
        from google import genai  # noqa: PLC0415
        from google.genai import types  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is not installed. Run: pip install google-genai"
        ) from exc

    client = genai.Client(
        api_key=key,
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )
    response = client.models.generate_content(
        model=model_name or settings.model_name,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
        ),
    )
    return (response.text or "").strip()
