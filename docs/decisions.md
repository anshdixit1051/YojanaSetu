# Design decisions (viva material)

Each entry: decision, why, what we gave up.

## D1. Deterministic engine decides eligibility; LLM never does
- **Why:** LLMs can state a wrong "you qualify" with full confidence. A rules engine is testable, repeatable and explainable (it returns the exact deciding rule).
- **Trade-off:** rules must be extracted and reviewed before a scheme works.

## D2. Three-valued logic (Eligible / Not Eligible / Needs More Info)
- **Why:** a missing answer must never silently become "no" or "yes".

## D3. Provenance is mandatory for every scheme
- **Why:** a rule with no source URL or download date cannot be defended; schemes change.
- Enforced by a required-field metadata model (Phase 0, File 2).

## D4. Extracted rules are never auto-trusted
- **Why:** LLM-extracted rules go to a human review state first; only approved rules go live.

## D5. Secrets only in .env, stored as SecretStr
- **Why:** free-tier keys leak easily from public repos and logs.

## D6. State-agnostic setup; state is one config value
- **Why:** the state was not yet chosen when Phase 0 started.
