"""api/app.py – FastAPI application: /health, /chat and /eligibility.

Run locally:
    uvicorn yojanasetu.api.app:app --reload --port 8000

Endpoints
---------
GET  /health        -> {"status": "ok", "schemes": [...]}
POST /chat          -> one turn of the intake agent (LLM-assisted)
POST /eligibility   -> direct deterministic check (no LLM at all)

Input hardening
---------------
scheme ids are used to build file paths, so they must match a strict slug
pattern (blocks "../" tricks).  Message length, language and session ids are
validated too, and the in-memory session store is bounded so one client cannot
fill the server's RAM.
"""

from __future__ import annotations

import uuid
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, StringConstraints

from yojanasetu.engine.aggregate import evaluate_scheme_verdict
from yojanasetu.engine.loader import load_approved_rules

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RULES_DIR = PROJECT_ROOT / "data" / "rules"

SLUG_PATTERN = r"^[a-z0-9]+(-[a-z0-9]+)*$"
SESSION_ID_PATTERN = r"^([A-Za-z0-9-]{8,64})?$"
MAX_MESSAGE_CHARS = 2000
MAX_SCHEMES_PER_REQUEST = 40
MAX_SESSIONS = 500
MAX_HISTORY_MESSAGES = 20

SchemeId = Annotated[str, StringConstraints(pattern=SLUG_PATTERN, max_length=80)]

app = FastAPI(
    title="YojanaSetu API",
    description="Agentic government-scheme eligibility navigator for India.",
    version="0.1.0",
)

# NOTE: "*" is fine for local development.  Before public deployment, replace
# it with the exact origin of your frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Bounded in-memory session store (swap for Redis/DB in production)
# ---------------------------------------------------------------------------

_sessions: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _get_session(session_id: str) -> dict[str, Any]:
    """Return the session, creating it (and evicting the oldest) if needed."""
    if session_id in _sessions:
        _sessions.move_to_end(session_id)
        return _sessions[session_id]
    _sessions[session_id] = {"profile": {}, "history": [], "scheme_ids": []}
    while len(_sessions) > MAX_SESSIONS:
        _sessions.popitem(last=False)
    return _sessions[session_id]


def _available_schemes() -> list[str]:
    return sorted(p.stem for p in RULES_DIR.glob("*.json"))


@lru_cache
def _graph():
    """Compile the LangGraph graph once per process."""
    from yojanasetu.agent.graph import build_graph  # noqa: PLC0415

    return build_graph()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    session_id: str = Field(default="", pattern=SESSION_ID_PATTERN)
    scheme_ids: list[SchemeId] = Field(default_factory=list, max_length=MAX_SCHEMES_PER_REQUEST)
    language: Literal["en", "hi"] = "en"


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    results: list[dict[str, Any]]
    scheme_verdicts: list[dict[str, Any]]
    missing_fields: list[str]
    done: bool


class EligibilityRequest(BaseModel):
    scheme_id: SchemeId
    profile: dict[str, Any]


class EligibilityResponse(BaseModel):
    scheme_id: str
    scheme_verdict: dict[str, Any]
    results: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, Any]:
    """Return API health and the schemes that have rules."""
    return {"status": "ok", "schemes": _available_schemes()}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Run one turn of the intake agent.

    Facts are extracted by the LLM but validated against the rules' fields;
    the verdict per scheme is computed by the deterministic engine; the
    eligibility wording in the reply is written by Python.
    """
    available = _available_schemes()
    unknown = [s for s in req.scheme_ids if s not in available]
    if unknown:
        raise HTTPException(status_code=404, detail=f"No rules found for: {unknown}")

    try:
        graph = _graph()
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"Agent unavailable: {exc}") from exc

    from yojanasetu.agent.graph import initial_state  # noqa: PLC0415

    session_id = req.session_id or str(uuid.uuid4())
    session = _get_session(session_id)
    if req.scheme_ids:
        session["scheme_ids"] = req.scheme_ids
    scheme_ids = session["scheme_ids"] or available

    state = initial_state(
        user_message=req.message,
        scheme_ids=scheme_ids,
        language=req.language,
        existing_profile=session["profile"],
        history=session["history"],
    )
    final_state = graph.invoke(state)

    session["profile"] = final_state["profile"]
    session["history"] = final_state["messages"][-MAX_HISTORY_MESSAGES:]

    assistant_msgs = [m for m in final_state["messages"] if m["role"] == "assistant"]
    return ChatResponse(
        session_id=session_id,
        reply=assistant_msgs[-1]["content"] if assistant_msgs else "",
        results=final_state["results"],
        scheme_verdicts=final_state["scheme_verdicts"],
        missing_fields=final_state["missing_fields"],
        done=final_state["done"],
    )


@app.post("/eligibility", response_model=EligibilityResponse)
def check_eligibility(req: EligibilityRequest) -> EligibilityResponse:
    """Direct deterministic check: no LLM involved."""
    try:
        rules = load_approved_rules(req.scheme_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No rules found for scheme '{req.scheme_id}'.",
        ) from None

    verdict = evaluate_scheme_verdict(req.scheme_id, rules, req.profile)
    return EligibilityResponse(
        scheme_id=req.scheme_id,
        scheme_verdict=verdict.as_dict(),
        results=[r.as_dict() for r in verdict.rule_results],
    )
