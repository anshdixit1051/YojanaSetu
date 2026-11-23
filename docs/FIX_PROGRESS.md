# YojanaSetu Defect Fix Progress

Tracking the independent audit remediation stages.

| Stage | Name | Status | Notes |
|---|---|---|---|
| Stage 0 | Baseline & Environment Verification | **Complete** | `pip install -e ".[dev]"` completed (`mcp==1.30.0`). Full `pytest -v` passed: 344/344. |
| Stage 1 | Engine Hardening (P0) | **Complete** | 11 edge cases + table-driven tests in `tests/test_engine_edge_cases.py` (40 tests, all passing; 384 total). |
| Stage 2 | Aggregation & Profile Validation Review | **Complete** | Unified validation via `validate_profile()`, integrated into `/chat`, `POST /eligibility` (returns 422 for bad fields), and MCP tools (389 tests passing). |
| Stage 3 | Data Integrity (P0) & Source Traceability | **Complete** | PM-KISAN audit matched verbatim quotes; 4 unverified schemes set to `pending_review` with 76 gold tests skipped ("awaiting source.txt from user"); traceability tests added (320 passed, 76 skipped). |
| Stage 4 | Phase 5 Hardening | **Complete** | Path safety in `loader.py` (slug regex + directory traversal check), LLM intake failure warning & profile conflict overwrite notification, retry with exponential backoff on transient errors in `llm.py`, `scripts/check_llm.py` connectivity tester, FastMCP stdio subprocess test in `tests/test_mcp.py`, unverified facts limitation note in README (334 passed, 76 skipped). |
| Stage 5 | Extraction Safety (Phase 3) | **Complete** | Prompt injection mitigation (data isolation in `<source_document>` tags + system prompt defense), no silent truncation (raises `ValueError` on oversized docs), verbatim `source_quote` substring validation against `source.txt` (Unicode NFC), and enforced `pending_review` status on disk (341 passed, 76 skipped). |
| Stage 6 | Test Quality | **Complete** | Strengthened `/chat` session persistence tests in `test_api.py` (direct `_sessions` inspection, accumulated profile assertions, history preservation); added 5 property/invariant tests and fuzz testing with 100 randomized profiles in `test_aggregate.py` (346 passed, 76 skipped). |
| Stage 7 | Documentation & Acceptance | **Complete** | Updated `README.md`, `AGENTS.md`, `docs/decisions.md` (ADRs D7–D12), and `docs/scheme_selection.md` with complete audit remediation state, pilot scheme quarantine status, test totals, and architecture notes. Full suite verified (346 passed, 76 skipped). |
| Phase 6 | Cited Explanations (Hindi + English) | **Complete** | Implemented `explanation/models.py` (`Citation`, `RuleExplanation`, `SchemeExplanation`, safe disclaimers) and `generator.py` (`generate_explanation`). Verified bilingual markdown rendering, Unicode NFC, and `/eligibility` response integration with 16 new unit/integration tests (362 passed, 76 skipped). |
