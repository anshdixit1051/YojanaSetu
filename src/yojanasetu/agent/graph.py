"""agent/graph.py – Assemble and compile the LangGraph intake graph.

Graph topology (one conversation turn)
--------------------------------------
    START -> intake -> evaluate -> respond -> END

    intake    LLM extracts facts; Python validates them against the rules' fields
    evaluate  engine decides per scheme (NO LLM)
    respond   Python writes the verdicts; LLM may add one follow-up question

Each HTTP request runs one turn.  When information is missing, the reply ends
with a question and the next request (same session) continues from the stored
profile.  The loop between turns lives in the API, not in the graph.

Usage
-----
    graph = build_graph()
    result = graph.invoke(initial_state("I own land", ["pm-kisan"]))
"""

from __future__ import annotations

from yojanasetu.agent.nodes import evaluate_node, intake_node, respond_node
from yojanasetu.agent.state import AgentState


def build_graph():
    """Build and compile the intake LangGraph StateGraph.

    Raises
    ------
    ImportError
        If langgraph is not installed.
    """
    try:
        from langgraph.graph import END, START, StateGraph  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("langgraph is not installed. Run: pip install langgraph") from exc

    builder = StateGraph(AgentState)
    builder.add_node("intake", intake_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("respond", respond_node)

    builder.add_edge(START, "intake")
    builder.add_edge("intake", "evaluate")
    builder.add_edge("evaluate", "respond")
    builder.add_edge("respond", END)

    return builder.compile()


def initial_state(
    user_message: str,
    scheme_ids: list[str],
    language: str = "en",
    existing_profile: dict | None = None,
    history: list[dict] | None = None,
) -> AgentState:
    """Build the AgentState for one conversation turn.

    Parameters
    ----------
    user_message:
        The user's latest message.
    scheme_ids:
        Schemes to evaluate, e.g. ``["pm-kisan"]``.
    language:
        ``"en"`` or ``"hi"``.
    existing_profile:
        Facts collected in earlier turns (None on the first turn).
    history:
        Earlier messages (None on the first turn).
    """
    return AgentState(
        messages=(history or []) + [{"role": "user", "content": user_message}],
        profile=dict(existing_profile or {}),
        results=[],
        scheme_verdicts=[],
        missing_fields=[],
        scheme_ids=scheme_ids,
        language=language,
        done=False,
    )
