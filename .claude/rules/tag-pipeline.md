---
paths:
  - "tag_pipeline.py"
---

# `tag_pipeline.py` — sibling, not part of the pipeline

Turns the golden set's evidence rows into the tagged `.xlsx` the Phase 3
eval reads. Not wired into `extract`/`build`. Two commands with a human
review between them (design decision 18):

1. **`draft [memo_id ...]`** — one LLM call per claim over all its found
   chunks (a "bundle") drafts `stated directly` / `needs combining` /
   `not supported` plus the chunks needed, into `review/<memo_id>.xlsx`.
   Every check runs before credentials or any LLM call. Never overwrites a
   sheet.
2. The user reviews in Excel or Numbers and sets CHECKED on every claim.
3. **`finalize <memo_id>`** — no LLM. Either logs every problem with its
   sheet row (exit 1, nothing written) or writes `reviewed/<memo_id>.xlsx`
   in the golden-set schema (see `.claude/rules/golden-set-pipeline.md`'s
   DataFrame schema), `tag` derived from the user's answers.

- **Imports (never edits) from `golden_set_pipeline`:** `LLMClient`,
  `_call_llm_with_json_retry`, `export_for_review`,
  `_is_deterministic_claim_id`, `build_chunk_index`, `parse_claims_file`,
  `read_filing_entity`,
  `_load_source_documents`, `_derive_claim_id`, `_claims_with_occurrence`,
  `_chunk_index_of`, `_valid_memo_id`, `_HUMAN_ADDED_PREFIX`, `_QUOTE_SEPARATOR`.
  If you rename any of these, `tag_pipeline.py` is a
  second consumer —
  `tests/test_bundle_tagging.py::test_imports_from_golden_set_pipeline_resolve`
  fails loudly on a rename (mirrors `retrieval_pipeline.py`'s own guard).
- **The filing entity comes from the claims file** (`filing_entity:`
  frontmatter, optional in the grammar, read by `read_filing_entity`), not
  from code, so no real company name is committed. `prepare_draft` checks it
  for every memo it will draft before reading any memo's PDFs.
- **Model pinned in code, not `.env`** (`_BUNDLE_MODEL`,
  `_BUNDLE_MAX_TOKENS`, frozen `_BUNDLE_PROMPT`) — changing any voids the
  MEMO-004 acceptance result. `draft` uses `_bundle_client()`, so it needs
  only `LLM_BASE_URL` and `LLM_API_KEY`; `LLM_MODEL` drives `extract`/`build`.
- **Order of work per memo:** audit or reword claims → `build` → `draft` →
  review → `finalize`. Rewording a claim after drafting changes its
  `claim_id`; `finalize` refuses and names the recovery (delete the sheet,
  `build`, `draft`, review again — answers are not carried over; EV-9).
- **Sheets come back through Apple Numbers** with hidden columns made
  visible, so `finalize` maps by tab name and header and verifies hidden
  cells against the claims file and the PDFs (always reads them).
- **Constants** (edit-here literals): `_CONTEXT_CHARS` 500,
  `_MAX_CHUNKS_PER_CLAIM` 40 (a tripwire — never truncates),
  `_MIN_QUOTE_CHARS` 25.
- The MEMO-004 acceptance scripts, answers and logs behind design decision
  18, with `HANDOFF.md` (read first when resuming tagging work), live in
  the maintainer's private notes repo under `tag-pipeline-acceptance/`
  (`CONTRIBUTING.md`, section 7); absent on a fresh clone of this repo alone.
- Tests: `tests/test_bundle_tagging.py` (draft) and `tests/test_finalize.py`
  (finalize and the eval ground-truth contract), both mocked. Design:
  `docs/superpowers/specs/2026-09-13-tag-pipeline-v2-bundle-design.md`
  (gitignored; the §12–§13 addenda override earlier sections). Acceptance
  record: `docs/prompt-verification-log.md` → "tag_pipeline v2 — bundle mode".

## Non-obvious design decisions

18. **Evidence tagging drafts one verdict per claim, and `tag` is written
    only from a human-CHECKED review sheet** (`tag_pipeline.py`). `build`
    decides *whether* a chunk is evidence (`found`); tagging decides *how*.
    `finalize` derives row tags mechanically from the user's checked answers
    (needed + stated directly → `extractive`, needed + needs combining →
    `synthesized`, else `unverifiable`), so the eval's claim bucket (any
    extractive → EXTRACTIVE, else any synthesized → SYNTHESIZED, else
    UNVERIFIABLE) reproduces the verdict exactly. Per claim, not per row,
    because `synthesized` ("supports the claim only together with other
    chunks") can't be judged from one row — the row-level v1 never cleared
    its bars and was removed.

    **Residual: chunk recall 73.6% against a ≥80% bar, accepted explicitly
    by the user** (the other 7 of 8 pre-registered MEMO-004 bars passed).
    Misses sit on claims computed from a table; the cost is review time, not
    a wrong tag. Changing `_BUNDLE_MODEL`, `_BUNDLE_PROMPT` or
    `_BUNDLE_MAX_TOKENS` voids this result.

    **The review sheet is verified, not trusted.** `finalize` re-checks
    every hidden id, chunk text, chunk-row count and PDF fingerprint against
    the claims file and the rebuilt chunk index, lists every problem with
    sheet row numbers, and writes nothing until there are none. Residuals:
    a row re-pointed in every visible and hidden cell to another chunk
    passes; a sheet lacking `chunk_count`/`source_docs` (MEMO-004's) lets a
    deleted last chunk row or changed PDFs through with only a warning.

    **Method lessons from v1, binding on any prompt change here:**
    pre-register model, prompt and bars before looking at data; label blind
    first; never tune against a reference re-adjudicated each round; row
    identity is `(claim_id, chunk_id)`, never `chunk_id` alone.

    Full narrative — v1's probe, pilot and confirmation round, the bundle
    acceptance test, every `finalize` check with its reason, and the exact
    acceptance dates: `docs/design-decisions/18-evidence-tagging.md`.
