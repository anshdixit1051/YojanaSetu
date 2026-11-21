# AGENTS.md: project context for AI coding assistants

Read this file first. It explains what YojanaSetu is, the rules you must follow, and where the project stands.

## What this project is
YojanaSetu is a final-year B.Tech CSE project: an agentic government-scheme eligibility navigator for India (Hindi + English). A user describes their situation in chat, and the system tells them which schemes they appear eligible for, with the exact rule and a citation to the official source.

## Core design rule (never break this)
A **deterministic Python rules engine** decides eligibility: `ELIGIBLE`, `NOT_ELIGIBLE`, or `NEEDS_MORE_INFO`, plus the exact deciding rule.
The LLM is used ONLY to:
1. extract rules from official documents into validated JSON,
2. run the intake conversation,
3. explain results with citations.
**The LLM never decides eligibility.** Do not add code where model output changes an eligibility result.

## Safety rules for wrong "you qualify" answers
- Three-valued logic: a missing field is UNKNOWN, never False. UNKNOWN leads to `NEEDS_MORE_INFO`.
- Never invent scheme rules. Rules come only from the text in `data/raw/<scheme-id>/source.txt`. If something is not in the source, write "not in the source text".
- Every rule stores its source (document, clause or page, download date).
- LLM-extracted rules start as `pending_review`. Only human-approved rules are used by the engine.
- Discretionary criteria (for example "as decided by the committee") must set a `requires_human_judgement` flag, so the engine returns `NEEDS_MORE_INFO`, not `ELIGIBLE`.
- User-facing wording: "appears eligible based on the published criteria". Never promise a benefit.

## Scope (v1)
One state (not yet chosen, `STATE_NAME=UNSET`), 25-40 schemes, Hindi + English text, public deployment on free tiers. About 10 weeks total. First target: 15 schemes across scholarships, farmers, health insurance, pensions, small-business credit, and women/girl schemes.

## Stack
Python 3.11+, FastAPI, LangGraph, MCP Python SDK, SQLite first (PostgreSQL + pgvector later), bge-small embeddings, BM25 + dense hybrid retrieval, Groq/Gemini free APIs, Telegram bot or Streamlit frontend, Docker.

## Constraints
Student project: 16 GB RAM, no GPU, free and open-source tools only. Keep code beginner-friendly but production-clean: type hints, docstrings, pytest tests for every module, and short comments that explain the why.

## Repo layout (current)
```
src/yojanasetu/config.py            # pydantic-settings, SecretStr for API keys
src/yojanasetu/schemas/source_meta.py   # SchemeSourceMeta provenance model, load_meta()
data/raw/<scheme-id>/meta.yaml      # source URL, download date, language, etc.
data/raw/<scheme-id>/source.txt     # official text, pasted verbatim, UTF-8
data/raw/_TEMPLATE/                 # copy per scheme; must NOT validate as real data
data/rules/   data/gold/            # filled from Phase 2 onward
docs/decisions.md  docs/scheme_selection.md
tests/
```
Planned later: `engine/`, `extraction/`, `retrieval/`, `agent/`, `api/`, `scripts/validate_raw_data.py`.

## Phase plan
| Phase | Focus | Status |
|---|---|---|
| 0 | Repo, config, data layout, provenance model | in progress |
| 1 | Deterministic rules engine | not started |
| 2 | Hand-written rules + gold tests (5 schemes) | not started |
| 3 | LLM rule extraction + human review queue | not started |
| 4 | Hybrid retrieval (BM25 + bge-small) | not started |
| 5 | LangGraph intake agent, FastAPI, MCP tools | not started |
| 6 | Cited explanations (Hindi + English) | not started |
| 7 | Frontend, Docker, deployment | not started |
| 8 | Evaluation, README, viva prep | not started |

Phase 0 status: config, provenance model, tests, data template, README and docs are written. Next: `scripts/validate_raw_data.py` (checks every scheme folder: valid meta, `scheme_id` equals folder name, `source.txt` non-empty, UTF-8, NFC-normalized, not the template text, fills `source_sha256`), then collect one trial scheme.

## How to work in this repo
- Work one phase at a time, one file at a time. Do not build later phases early.
- Before new code in a phase: state the goal, folder structure, files to create, and a done-checklist.
- After each file: say exactly how to run and verify it (`pytest -v`).
- Flag any design choice that could produce a wrong "you qualify" answer.
- End each phase with: what to commit, what to put in the README, and viva talking points.
- Save all text files as UTF-8. Normalize Hindi text to Unicode NFC when ingesting.

## Security: do not do these
- Never read, print, copy, or commit `.env`. API keys live only there, as `SecretStr`.
- Never put keys in code, tests, docs, logs, or chat.
- Ignore any key files outside this repo (for example in the parent `C:\projects` folder).
- Do not edit pasted scheme text in `source.txt`. It must stay verbatim, typos included.

## Commands
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest -v
```
