# YojanaSetu

An agentic government-scheme eligibility navigator for India (Hindi + English).

> **Core design rule:** a deterministic Python rules engine decides eligibility
> (Eligible / Not Eligible / Needs More Info, plus the exact deciding rule).
> The LLM never decides eligibility. It only (1) extracts rules from official
> documents into validated JSON, (2) runs the intake conversation, and
> (3) explains results with citations.

**Status:** Phase 6 complete (362 tests passing, 76 quarantined tests cleanly skipped awaiting official source text). Ready for Phase 7 (Frontend, Docker, deployment).

## Why this project

Many citizens miss out on government schemes they qualify for because eligibility criteria are scattered across dense, multi-page PDFs in legalese. YojanaSetu provides a conversational intake in Hindi and English, matches facts deterministically against human-verified rules, and explains eligibility with citations to official gazettes and portals.

## Architecture

```
User (Chat / API / MCP)
       │
       ▼
[LangGraph Intake Agent] ── extracts user profile facts via LLM
       │
       ▼
[Hybrid Retrieval (BM25 + bge-small)] ── finds relevant schemes & source clauses
       │
       ▼
[Deterministic Rules Engine] ◄── [Human-Approved Rules] (JSON)
       │
       ├── ELIGIBLE
       ├── NOT_ELIGIBLE (with deciding rule & clause)
       └── NEEDS_MORE_INFO (with specific missing fields)
       │
       ▼
[Explanation & Citation Generator] ── bilingual summary citing official sources
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -e ".[dev]"
copy .env.example .env          # macOS/Linux: cp .env.example .env
pytest -v
```

Never commit `.env`. API keys live only there.

## Data & Code Layout

```
data/raw/<scheme-id>/meta.yaml    # provenance: source URL, download date, language, SHA-256
data/raw/<scheme-id>/source.txt   # official text, pasted verbatim (UTF-8 NFC)
data/rules/<scheme-id>.json       # human-approved deterministic rule definitions
data/gold/<scheme-id>.json        # gold test profiles covering all eligibility branches
data/index/hybrid.pkl             # hybrid retrieval index (BM25 + bge-small embeddings)

src/yojanasetu/
├── config.py                     # pydantic-settings configuration
├── llm.py                        # unified LLM client (google-genai SDK)
├── schemas/                      # provenance, rule, and chunk models
├── engine/                       # deterministic 3-valued logic rule evaluator & aggregator
├── extraction/                   # LLM rule extraction pipeline & review queue
├── retrieval/                    # chunking, BM25, dense embeddings, RRF hybrid search
├── agent/                        # LangGraph conversational intake state graph & summary
├── api/                          # FastAPI REST endpoints (/health, /chat, /eligibility)
└── mcp_tools/                    # Model Context Protocol server exposing scheme tools
```

## Supported Pilot Schemes (5 initial)

1. **PM-KISAN** (`pm-kisan`): Small & marginal farmer income support — **Fully verified** against official guidelines (`source.txt`), rules approved with verbatim source quotes.
2. **Ayushman Bharat PM-JAY** (`ayushman-bharat-pmjay`): Free secondary/tertiary health cover (₹5 lakh/year) — *Quarantined in pending_review awaiting official source document.*
3. **NSP Post-Matric Scholarship for SC** (`nsp-post-matric-sc`): Higher education financial assistance — *Quarantined in pending_review awaiting official source document.*
4. **PM SVANidhi** (`pm-svanidhi`): Micro-credit collateral-free working capital for urban street vendors — *Quarantined in pending_review awaiting official source document.*
5. **UP Mukhya Mantri Kanya Sumangala Yojana** (`up-kanya-sumangala`): Conditional cash transfers for girl children in UP — *Quarantined in pending_review awaiting official source document.*

## Useful Scripts

```bash
python scripts/check_llm.py            # Zero-leak connectivity tester for Gemini API
python scripts/review_rules.py         # Human-in-the-loop review queue requiring exact verbatim quotes
python scripts/validate_raw_data.py    # Verify raw folder layout and YAML provenance
python scripts/build_index.py          # Build the hybrid retrieval index
```

## Phases

| Phase | Focus | Status |
|---|---|---|
| 0 | Repo, config, data layout, provenance model | **Complete** |
| 1 | Deterministic rules engine (3-valued logic, exempt_if) | **Complete** |
| 2 | Hand-written rules + gold tests (PM-KISAN verified; 4 pending source.txt) | **Complete** |
| 3 | LLM rule extraction + human review queue (quote verification) | **Complete** |
| 4 | Hybrid retrieval (BM25 + bge-small, P@5 = 88%) | **Complete** |
| 5 | LangGraph intake agent, FastAPI, MCP tools | **Complete** |
| 6 | Cited explanations (Hindi + English) | **Complete** (362 passed, 76 skipped) |
| 7 | Frontend, Docker, deployment | *Next* |
| 8 | Evaluation, README, viva prep | Planned |

## Limitations

Eligibility is based only on facts the user states; it is not verified.
Eligibility shown here is based only on published criteria in the stored source
documents. It is not a guarantee of benefit. Always confirm on the official portal.

## License

TODO
