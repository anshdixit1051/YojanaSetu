"""extraction/prompts.py – Prompt templates for LLM rule extraction.

Why a separate module
---------------------
Keeping prompts here (not inside extractor.py) makes them easy to review,
update, and test independently.  Any change to a prompt is a deliberate edit
to this file, not buried inside logic code.

Safety constraint embedded in the prompt
-----------------------------------------
The prompt instructs the model to:
  - copy criteria verbatim from the source text (never invent)
  - set status = "pending_review" always
  - flag discretionary criteria with requires_human_judgement = true
  - quote the exact source clause for every rule
These are the same safety rules as AGENTS.md, expressed as LLM instructions.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompt – sent once per session, sets model behaviour
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a government-scheme eligibility rule extractor.
Your only job is to read official scheme documents and output structured JSON rules.

STRICT RULES you must never break:
1. Only extract criteria that are EXPLICITLY stated in the provided source text.
   Do NOT invent, infer, or guess any rule.  If something is not in the text, do not include it.
2. Set "status": "pending_review" on EVERY rule, without exception.
   A human reviewer will decide whether to approve each rule before it is used.
3. For every rule, copy the EXACT clause, section, or page reference from the source into "source_clause".
4. If a criterion uses words like "as decided by the committee", "at the discretion of", or similar,
   set "requires_human_judgement": true.
5. Use only these operators: eq, neq, lt, lte, gt, gte, in, not_in, exists.
6. Use only these categories: scholarship, farmer, health, pension, business-credit, women-girl.
7. Use only these verdicts: ELIGIBLE, NOT_ELIGIBLE.
8. Output ONLY a valid JSON array.  No commentary, no markdown fences, no extra text.

Output format – a JSON array where each element has exactly these fields:
{
  "rule_id":                "<scheme_id>-<short-name>",
  "scheme_id":              "<scheme_id>",
  "field":                  "<snake_case_profile_key>",
  "operator":               "<operator>",
  "value":                  <threshold or list>,
  "verdict_if_true":        "ELIGIBLE" or "NOT_ELIGIBLE",
  "description_en":         "<one sentence, copied from source>",
  "description_hi":         "",
  "source_clause":          "<exact section/clause/page from document>",
  "requires_human_judgement": false,
  "status":                 "pending_review"
}
"""

# ---------------------------------------------------------------------------
# User prompt template – filled per scheme
# ---------------------------------------------------------------------------

USER_PROMPT_TEMPLATE = """\
Scheme ID  : {scheme_id}
Scheme name: {name_en}
Category   : {category}

Extract all eligibility rules from the official source text below.
Return a JSON array of rule objects.  Follow the system instructions exactly.

--- SOURCE TEXT BEGIN ---
{source_text}
--- SOURCE TEXT END ---
"""


def build_user_prompt(
    scheme_id: str,
    name_en: str,
    category: str,
    source_text: str,
    max_chars: int = 12_000,
) -> str:
    """Return the user-turn prompt for rule extraction.

    Parameters
    ----------
    scheme_id:
        Slug, e.g. ``"pm-kisan"``.
    name_en:
        Official English name, e.g. ``"Pradhan Mantri Kisan Samman Nidhi"``.
    category:
        One of the six allowed category strings.
    source_text:
        Contents of ``data/raw/<scheme-id>/source.txt``.
    max_chars:
        Truncate source text to this many characters to stay within context
        limits on free-tier APIs.  12 000 chars ≈ 3 000 tokens.

    Returns
    -------
    str
        The filled prompt string ready to send as the user message.
    """
    truncated = source_text[:max_chars]
    if len(source_text) > max_chars:
        truncated += "\n\n[... source text truncated for length ...]"

    return USER_PROMPT_TEMPLATE.format(
        scheme_id=scheme_id,
        name_en=name_en,
        category=category,
        source_text=truncated,
    )
