# YojanaSetu

An agentic government-scheme eligibility navigator for India (Hindi + English).

> **Core design rule:** a deterministic Python rules engine decides eligibility
> (Eligible / Not Eligible / Needs More Info, plus the exact deciding rule).
> The LLM never decides eligibility. It only (1) extracts rules from official
> documents into validated JSON, (2) runs the intake conversation, and
> (3) explains results with citations.

**Status:** Phase 0 (setup) in progress.

## Why this project

TODO (Phase 8): 3-4 lines on the problem: people miss schemes they qualify for
because criteria are scattered across long official documents.

## Architecture

TODO (Phase 8): add the diagram once the engine, extraction, retrieval and agent exist.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -e ".[dev]"
copy .env.example .env          # macOS/Linux: cp .env.example .env
pytest
```

Never commit `.env`. API keys live only there.

## Data layout

```
data/raw/<scheme-id>/meta.yaml    # provenance: source URL, download date, language
data/raw/<scheme-id>/source.txt   # official text, pasted verbatim
data/rules/                       # validated rule JSON (Phase 2+)
data/gold/                        # hand-written test cases (Phase 2+)
```

Every scheme must record where its text came from and when it was downloaded.

## Phases

| Phase | Focus | Status |
|---|---|---|
| 0 | Repo, config, data layout | in progress |
| 1 | Deterministic rules engine | not started |
| 2 | Hand-written rules + gold tests (5 schemes) | not started |
| 3 | LLM rule extraction + human review | not started |
| 4 | Hybrid retrieval (BM25 + bge-small) | not started |
| 5 | LangGraph intake agent, FastAPI, MCP | not started |
| 6 | Cited explanations (Hindi + English) | not started |
| 7 | Frontend, Docker, deployment | not started |
| 8 | Evaluation, README, viva prep | not started |

## Limitations

Eligibility shown here is based only on published criteria in the stored source
documents. It is not a guarantee of benefit. Always confirm on the official portal.

## License

TODO
