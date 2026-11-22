"""agent/state.py – Typed state shared across all LangGraph nodes.

Why a typed state
-----------------
LangGraph passes one state dict between nodes.  Typing it here makes every
field's purpose explicit and prevents silent key typos.

Safety note
-----------
`profile` holds user-supplied facts that the deterministic engine reads.
`results` and `scheme_verdicts` come from the engine, never from the LLM.
The LLM may phrase a follow-up question, but it cannot change a verdict.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State passed between every node in the intake graph.

    Fields
    ------
    messages:
        Full conversation history: ``{"role": "user"|"assistant", "content": str}``.
        Annotated with ``operator.add`` so nodes can append.
    profile:
        User facts collected so far, e.g. ``{"owns_cultivable_land": True}``.
        Only fields that appear in approved rules are ever stored here.
    results:
        Per-rule results (dicts) from the engine.  Used for citations.
    scheme_verdicts:
        One deterministic verdict dict per scheme (from engine.aggregate).
        This is the answer to "does the user appear eligible?".
    missing_fields:
        Fields the user can still supply; drives the next question.
    scheme_ids:
        Which schemes to evaluate.
    language:
        ``"en"`` or ``"hi"``.
    done:
        True when no more questions are needed.
    """

    messages: Annotated[list[dict[str, str]], operator.add]

    profile: dict[str, Any]
    results: list[dict[str, Any]]
    scheme_verdicts: list[dict[str, Any]]
    missing_fields: list[str]
    scheme_ids: list[str]
    language: str
    done: bool
