---
paths:
  - "golden_set_pipeline.py"
---

# `golden_set_pipeline.py` — function map and design decisions

The core two-stage pipeline (`extract` / `build`) lives entirely in this one
file; see `CLAUDE.md` for the Jupytext format, the pipeline shape diagram and
the commands. This rule file holds the function map, the DataFrame schema,
and design decisions 1–17.

## Current function map (`grep -n "^def \|^class "`)

One line per role, plus the cross-function facts no single docstring can
hold — the docstrings are authoritative for everything else. Detail that
used to live here (`write_claims_file`'s atomic write, `run_extract`'s
unreadable-file handling, `_error_row`'s overwritten `uuid4`) is in
`docs/design-decisions/17-two-stage-pipeline.md` → "Implemented behaviour".

- `load_pdf_text` / `load_pdf_text_pdfplumber` — PDF extraction (pypdf vs.
  layout-aware pdfplumber, decision 10). No OCR; scanned PDFs aren't handled.
- `warn_if_text_suspiciously_short` — call on every extracted doc; logs,
  doesn't diagnose a cause.
- `LLMClient` (dataclass, `from_env()`) / `call_llm` — the sole HTTP contact
  point with the LLM provider. Swap providers by editing only this function.
- `_strip_to_json` / `_call_llm_with_json_retry` — JSON robustness (fence
  stripping, retry on parse failure).
- `chunk_document` / `build_chunk_index` — character-based sliding-window
  chunking (decision 6).
- `_tokenize` / `bm25_shortlist` — fixed-`top_n` BM25 shortlist (legacy,
  still usable directly).
- `bm25_threshold_shortlist` — **default** shortlist: every chunk scoring
  `>= relative_threshold * top_score`, no upper cap, topped up to
  `min_candidates` (decision 7).
- `extract_atomic_claims` / `_STRANDED_POINTING_WORD_RE` — LLM splits a
  section into atomic claims (decisions 5, 15, 16); the regex is the
  non-LLM backstop that logs a claim still starting with a bare pronoun or
  pointing word.
- `propose_evidence_from_chunks` — asks the LLM which candidate chunks
  support one claim; returns every match; same-measure rule (decision 9);
  recovers a hallucinated `chunk_id` via a unique `evidence_span` substring
  match before dropping it. **Residual:** a short/generic span (a bare
  number or percentage) has no specificity floor, so recovery can land on
  the wrong candidate and misattribute evidence — decision 9's Austria/5%
  shape, mitigated only by the prompt's quote-enough-context instruction.
- `propose_evidence_from_chunks_batched` — **default** wrapper: batches the
  candidates and isolates batch failures (decision 8), returns `(matches,
  counts)` (decision 11), escalates its batch-count log line (decision 13).
- `_error_row` / `_normalize_span` / `_span_matches_chunk_text` /
  `_chunk_index_of` / `_UnionFind` / `_same_evidence` /
  `_group_equivalent_chunks` / `_rows_for_claim` — per-claim row
  construction and ambiguity grouping (decisions 1, 2).
  `_span_matches_chunk_text` drives both `verbatim_match` and
  hallucinated-`chunk_id` recovery. `_same_evidence` is decision 1's rule,
  defined once and shared with `eval_pipeline.py`; `_group_equivalent_chunks`
  is the union-find partition over it. `_HUMAN_ADDED_PREFIX` /
  `_QUOTE_SEPARATOR` are how `tag_pipeline` marks a human-added chunk and
  joins its quotes, defined here so the eval reads them back the same way.
- `_valid_section_name` / `_section_name_issue` / `_valid_memo_id` /
  `_claims_with_occurrence` / `_derive_claim_id` / `_is_deterministic_claim_id` —
  claim-id support (decision 17). Each rule is defined once and shared by
  `_read_memo_config`, `write_claims_file` and `parse_claims_file`: if the
  config check drifted looser than the writer's, `extract` would pay for
  every section, then fail at write time. `_ATX_CLOSE_RE` is likewise the
  one heading-close regex for validator, parser and drift scanner, or a
  name could pass validation yet parse back differently.
  `_is_deterministic_claim_id` lives here, not in `tag_pipeline.py`,
  because `eval_pipeline.py` needs it too.
- `build_golden_set_draft` / `build_golden_set_batch` — per-section and
  multi-section entry points. The claims slot is a **claims list**, not
  section text (decision 17); the batch's tuning knobs are defaults
  overridable per memo (decision 12).
- `_OVERRIDE_FIELDS` / `_MEMO_FIELDS` / `_parse_memo_overrides` /
  `_validate_memo_sections_shape` — override key sets, value validation,
  and the pre-run shape check (decision 12).
- `_read_memo_config` / `_load_source_documents` /
  `load_memo_sections_from_config` — pure YAML validation (no PDF I/O, so a
  config typo costs no spend; what `run_extract` calls), the shared
  PDF-folder scan, and the older combined loader. That loader's slot 3 is
  section text, so its output no longer feeds `build_golden_set_batch`; its
  only production caller is `preview_claim_splits`. It also got stricter
  id/section-name checks with decision 17 (CHANGELOG `### Changed`).
- `parse_claims_file` / `read_filing_entity` / `_parse_frontmatter` /
  `_classify_claims_body` / `_trips_comment_guard` /
  `_scan_claims_file_sections` — the claims-file
  grammar: a hand-rolled parser, not a Markdown library (frontmatter goes
  through `yaml.safe_load`). Every ambiguous shape raises naming the file
  and line. `_trips_comment_guard` is *the* unterminated-`<!-- -->` rule,
  shared with `write_claims_file`, so the writer's accept set matches the
  parser's by construction. `_scan_claims_file_sections` is a best-effort
  scan for `extract`'s drift check only — never raises.
- `write_claims_file` — the inverse of `parse_claims_file`; refuses anything
  the parser would reject; writes atomically.
- `load_memo_sections_from_claims` — Stage 2's loader: validates every
  claims file before reading any PDF.
- `preview_claim_splits` — interactive, opt-in split check (decision 14).
- `run_extract` — Stage 1 orchestrator: `memos.yaml` → `claims/*.md`,
  complete-or-absent per memo, never overwrites (reports `skipped`, warns
  of drift).
- `run_build` / `_main` — Stage 2 orchestrator and the CLI dispatch.
  `_main` rejects an unknown command before `LLMClient.from_env()` (a typo
  needs no credentials to diagnose) and exits 1 if `extract` reports a
  failed memo.
- `export_for_review` / `_widen_review_columns` / `import_reviewed` —
  Excel/CSV round-trip for manual review (decision 3).

## DataFrame schema

`claim_id, memo_id, section, claim_text, doc_id, chunk_id, chunk_text, bm25_score, evidence_span, found, confidence, ambiguous_match, verbatim_match, human_reviewed, tag, tag_draft, tag_rationale`

This schema is also what `tag_pipeline.py` writes into and `eval_pipeline.py`
reads from — a session touching either sibling needs it too.

- `claim_id` is stable per claim (UUID), shared across every row that claim
  produces.
- `chunk_text`/`bm25_score` are pipeline-sourced (that row's own chunk's
  text, and that claim's own BM25 score for it) — `None` on not-found/error
  rows, and intentionally excluded from `import_reviewed`'s overwritable
  columns, so they're read-only from a reviewer's perspective.
- Zero matches → one row (`found=False`). One match → one row. Multiple
  matches → one row per match.
- `confidence="error"` is reserved for pipeline failures, never a
  legitimate "not found."
- `ambiguous_match=True` only when evidence genuinely appears in different
  documents/locations — not when it's the same passage duplicated across
  overlapping chunks (see design decision 1 below; this distinction exists
  specifically to protect a later retrieval-recall metric from penalizing a
  system that surfaces a different-but-equally-valid overlapping chunk).
- `verbatim_match` — literal substring match vs. paraphrase. Not dropped
  when `False`, just flagged for review priority.
- `tag_draft` / `tag_rationale` — written by `tag_pipeline.py finalize`
  (not `build`), together with `tag`. `tag_draft` ∈ {extractive,
  synthesized, unverifiable, ""} is the AI draft's tag by the same rule as
  `tag` (blank where the AI never judged the row: a failed draft, a
  human-added chunk). `tag_rationale` records the review — `bundle review
  <date>: …`, `human-added <date>: …`, or `auto: found=False, nothing was
  found by search` — plus the user's notes. The Phase 3 eval reads `tag`;
  it reads `tag_rationale` only for the prefix `human-added`
  (`_HUMAN_ADDED_PREFIX`), to know which spans join several quotes.

## Non-obvious design decisions (read before changing anything)

1. **Ambiguity vs. chunk-overlap duplication** (`_rows_for_claim`,
   `_UnionFind`). Overlapping chunks can echo the same passage, truncated
   differently at the boundary. Matches are grouped via union-find: same
   group (→ `ambiguous_match=False`) if they share `doc_id` AND either
   normalized text is identical, or they're adjacent by chunk index
   (`_chunk_index_of`) with a substring relationship. `ambiguous_match=True`
   only when >1 group remains.

   The rule is `_same_evidence` (pairwise) and `_group_equivalent_chunks`
   (partition), shared with `eval_pipeline.py` so the golden set and the eval
   cannot disagree about what "the same evidence" means. It never groups a
   chunk with itself (index delta 0) — callers wanting exact-chunk identity
   compare `chunk_id` first.

   **Addendum — an identical `chunk_id` returned twice for one claim is
   collapsed before grouping** (`_CONFIDENCE_RANK` + the dedupe in
   `_rows_for_claim`). This is a stricter case than the overlapping-chunk
   duplication above (which is about *different* `chunk_id`s truncated at a
   shared boundary): here it is the same chunk, from a chunk appearing in
   two BM25 batches, a repeat in one evidence-LLM response, or a
   hallucinated-`chunk_id` recovery landing on an already-matched chunk.
   The strongest-by-`confidence` match is kept (tie → first seen; a missing
   or non-string confidence — a malformed answer's list or dict — ranks
   lowest rather than raising) and each collapse is logged WARNING. Without it, two matches for one chunk with
   slightly different quoted spans set `ambiguous_match=True` spuriously.

2. **`_normalize_span` collapses dot/dash/underscore leader runs** (2+ →
   single `.`), not just whitespace — PDF table extraction produces
   dot-leader artifacts whose length isn't stable across chunk boundaries.
   Single periods (decimals, sentence-ends) are untouched.

3. **`import_reviewed` treats a deleted row as a rejection**, not a
   silent no-op. Matches on `(claim_id, chunk_id)`. A row present in the
   original draft but missing from the reviewed file gets `tag="rejected"`,
   `human_reviewed=True` automatically.

4. **`call_llm`'s empty-content handling distinguishes two unrelated
   causes.** `max_tokens=4096` is set explicitly because an unset/low
   budget lets a reasoning model's hidden reasoning tokens exhaust it,
   producing empty `content` — that's `finish_reason="length"` with no
   provider `error` object, and raising `max_tokens` is the real fix.
   Separately, OpenRouter uses `finish_reason="error"` (with a
   `choices[0].error` object containing `code`/`message`/
   `metadata.error_type`) to mean the *upstream provider* failed — rate
   limit, disconnect, etc. — which is transient/provider-side, not a
   token-budget problem. `call_llm` surfaces the real `error` detail when
   present instead of defaulting to the max_tokens explanation; don't
   collapse these back into one generic message.

5. **`extract_atomic_claims`'s prompt states a general principle, not a
   fixed word list** — a claim is atomic only if it can't decompose into
   two independently verifiable facts, "regardless of what word or
   punctuation joins them" (connector words are examples, not an
   exhaustive trigger list). It also covers a fact *attached inside a
   phrase* ("X is a listed operator with a market cap of €5bn" → two
   claims), counterweighted against shredding a description into separate
   adjectives, plus a self-test ("could a source confirm one part while
   saying nothing about another?"). **Standing convention:** the worked
   examples embedded in this prompt are the regression baseline for any
   future edit — re-verify all of them, plus out-of-sample cases in both
   directions (canonical list, 14 examples: `docs/prompt-verification-log.md`).

   Three product decisions layered onto this rule, each with its own
   observed failure case and fix: garbled memo text (a stray typo can tip
   the model into literal-copy mode — the fix permits light connective
   rewording, forbids restating the whole input); a comparator entity
   inside a ranking/comparison clause is exempt from the attached-attribute
   rule (so a competitor named only for context doesn't spin off its own
   guaranteed-`found=False` claims); and a split that strands a pronoun or
   bare pointing phrase resolves it to the plainest naming the text already
   uses, including one named only in an own-line heading/label above the
   paragraph (ticket 007) — never inventing a referent.

   **Residual — the motivating case is not fixed.** An inline "Label:" at
   the head of a multi-claim paragraph (not an own-line heading) still
   doesn't propagate its name to that paragraph's sibling claims — a
   mid-claim "The clause ..." with no name passes silently and can produce
   confidently-wrong evidence downstream (five unrelated metrics matched
   at medium/high confidence in the case that motivated this decision). No
   deterministic backstop for this (unlike a claim-initial pronoun).
   EV-6 is the designed net; `preview_claim_splits` (decision
   14) the opt-in human catch. `_STRANDED_POINTING_WORD_RE` also still
   false-positives on an expletive "it" ("It is the policy of Borealis to
   offer...").

   Full narrative — every observed case, the exact fix wording, the
   over-resolution counterweights, and the identical-text-after-resolution
   WARNING behaviour: `docs/design-decisions/05-atomic-claim-decomposition.md`.

6. **Chunking is character-based** (`chunk_document`, default
   `chunk_size=1000, overlap=200`), no tokenizer dependency.

7. **`bm25_threshold_shortlist` has no upper bound on candidate count by
   design** — a fixed cap silently drops real evidence (the reason it
   replaced `bm25_shortlist`), so with generic claim text one claim can
   select hundreds of chunks (observed max: 420 on the Paxton corpus, 10+ LLM
   batches). That makes `relative_threshold`/`min_candidates` (with
   `batch_size`) real cost/latency knobs — passed through
   `build_golden_set_draft`, never hardcoded, tuned per corpus and per memo
   (decision 12).

   **Residual — BM25 is lexical.** A claim worded differently from its
   source ("headcount" vs. "employees") can leave the right chunk below any
   cutoff; lowering `relative_threshold` cannot recover a zero-overlap
   miss. EV-5 tracks a keyword-expansion mitigation.

   Full text: `docs/design-decisions/07-bm25-no-candidate-cap.md`.

8. **`propose_evidence_from_chunks_batched` isolates batch failures.** Each
   batch's LLM call has its own try/except; a failed batch is logged (claim,
   batch position, candidates never evaluated) and skipped, not allowed to
   discard evidence already confirmed by other batches. Raises only if
   every batch for a claim fails. `batch_size < 1` and
   `bm25_threshold_shortlist`'s `min_candidates < 0` both raise `ValueError`
   immediately rather than silently degrading (a negative `batch_size`
   used to silently return zero evidence with no error — fixed).

9. **`propose_evidence_from_chunks`'s prompt requires same-measure
   support, not keyword overlap — and treats "says less than the claim" as
   a separate, allowed case.** Evidence must be about the same subject and
   the same quantity measured; a shared entity name and a similar-looking
   number are not enough ("The balance in Austria (5%)." matched to an
   unrelated yield row). The prompt names **exactly three** disqualifiers:
   different measure/period/population; you can't tell what the figure
   refers to; it contradicts the claim. Keeping that list closed is
   load-bearing — an earlier, overlapping "same scope" rule contradicted
   the less-detail allowance and made verdicts random across identical
   runs. Don't reintroduce a second, overlapping rule here.

   **Product decision the user made explicitly** ("mark as found, low
   confidence"): a passage that states the claim's fact but leaves one
   qualifier unsaid is returned `found=True` at `confidence="low"`, not
   dropped. `low` is reserved for that case; the prompt explicitly forbids
   using it for a different-measure lookalike. `confidence` is a
   review-priority hint, not a stable score.

   **Residual:** a legitimate partial match is still dropped on roughly 1
   run in 5. Re-check BOTH directions if you touch this prompt, with a
   partial match, a contradiction and a lookalike distractor in one batch —
   single-candidate tests hide these interactions.

   Full narrative — the three-run evidence and the batch-contamination
   test: `docs/design-decisions/09-same-measure-evidence.md`.

10. **`load_pdf_text` (pypdf) vs. `load_pdf_text_pdfplumber`** — pypdf is
   faster/lighter but can garble table-heavy layouts; pdfplumber is slower
   but layout-aware. Both are plain extraction, no OCR.

11. **Dropped/recovered evidence counts are logged only, never added to the
   DataFrame** — the schema stays claim/match-shaped. "Dropped" is
   deliberately undifferentiated (malformed entry, unresolvable
   out-of-batch `chunk_id`, missing `evidence_span`); the per-entry
   WARNINGs still distinguish them. Counts thread through an optional
   `counts` accumulator on `propose_evidence_from_chunks` (its return type
   stays a match list); `propose_evidence_from_chunks_batched` returns
   `(matches, counts)`; `build_golden_set_draft` sums them per section. A
   fully-failed claim contributes no counts and never logs "0 dropped, 0
   recovered". **Product decision the user made explicitly:** the
   per-claim line also reports batches succeeded ("(2/3 batch(es)
   succeeded)") so a partial failure can't read as "nothing lost"; the
   section-wide line has no ratio.

   Full text: `docs/design-decisions/11-evidence-counts-logged.md`.

12. **Per-memo overrides ride along as an optional 5th tuple element, and
   the loader stores only the keys actually set.** `relative_threshold`/
   `min_candidates`/`batch_size` are batch-wide defaults on
   `build_golden_set_batch`, each overridable per memo, because corpora
   range from ~1,000 to ~6-7,000 chunks per memo (decision 7). Plain
   4-tuples stay valid for a hand-built list. No filled-in defaults in the
   dict, so each default lives only in the signatures; keys are flat on the
   memo mapping, not under a `tuning:` block.

   `_validate_memo_sections_shape` is a **pre-pass over every entry before
   the first section runs**, not a per-entry check in the loop — a bad
   entry at position 40 would otherwise throw away 39 sections of spend. It
   refuses four silent-degrade shapes: (a) a wrong tuple length, (b) a
   non-dict 5th element, (c) a misspelled key in a hand-built override
   dict, (d) slot 3 holding section text instead of a claims list.

   **Unrecognized keys raise** (`_MEMO_FIELDS`) — a **product decision the
   user made explicitly**: a typo'd `batchsize: 25` must not silently yield
   the default. No `notes:` escape hatch; YAML `#` comments cover
   annotation. Value ranges are validated at config-load, not left to
   runtime checks 30+ minutes into a run (YAML traps handled:
   `min_candidates: yes` is a `bool`, i.e. an `int`; a valueless key parses
   as `None` → malformed). **Residual:** a *repeated* key is silently
   last-wins in both `memos.yaml` and claims-file frontmatter; EV-8
   tracks the fix.

   Full narrative — why a 5th element beat a `memo_overrides=` parameter,
   the measured checkpoint loss, `key=repr` sorting:
   `docs/design-decisions/12-per-memo-overrides.md`.

13. **`propose_evidence_from_chunks_batched`'s pre-loop
   candidate/batch-count line escalates from INFO to WARNING once a claim
   splits into more than `_BATCH_COUNT_WARNING_THRESHOLD` (10) batches** —
   the same single line, not a second one. Batch count, not candidate
   count, is the trigger: hundreds of candidates are routine (decision 7),
   while batch count tracks a claim's LLM cost and its exposure to the
   out-of-batch `chunk_id` failure (decision 8). Decision 7's 420-candidate
   claim (11 batches at `batch_size=40`) tripping it is the threshold
   working, not an oversight. If the WARNING fires routinely, revisit the
   number 10, not the batch-count-over-candidate-count choice.

   Full text: `docs/design-decisions/13-batch-count-warning.md`.

14. **`preview_claim_splits` is a separate, opt-in pass, never wired into
   the real evidence-matching path.** An `input()` inside an hours-long
   batch run would break it unattended, so the preview stays outside
   `build_golden_set_draft`/`build_golden_set_batch` and re-runs
   `extract_atomic_claims` itself. It checks *the prompt against this
   section's text*, not the next run's exact output: a systematically bad
   split recurs and is caught; a one-run wording fluke is not (so ticket
   005's "the same wording the real run would produce" is approximate by
   design).

   **The `RUN_CLAIM_SPLIT_PREVIEW` cell is guarded by a plain module
   flag** (`False`, flipped by hand in a notebook) — not `if __name__ ==
   "__main__"`, which the script runs, and never a bare call, which fires
   on `import` and hangs the test suite. A toggle, not config surface.
   `tests/test_claim_split_preview.py` checks the guard structurally by
   parsing the source, since a live "does it prompt" test passes for the
   wrong reason. Contrast EV-6: an always-on check *inside*
   `build_golden_set_draft`.

   Full narrative: `docs/design-decisions/14-claim-split-preview.md`.

15. **A sentence that closes over a list stays one claim; a list it does
   not close over still splits; and a partial qualifier survives the
   split.** A sentence closes over a list when it asserts something true
   only of the whole set — a role/right/restriction defined over a named
   set, or an outcome contingent on several conditions holding together
   (illustrations: "limited to", "only", "solely", "jointly", "provided
   that" — not an exhaustive trigger list, same convention as decision 5).
   A plain enumeration ("Acme's largest customers are Globex, Initech and
   Umbrella.") is explicitly carved out as NOT a closed list and still
   splits one-per-member. Separately, when a list *does* split, a
   leading-member qualifier ("mainly"/"primarily") and a trailing-member
   qualifier ("but to a lesser extent") each ride onto the claim(s) they
   modify, never the other's.

   **Residual.** Two out-of-sample closed lists still misfire: an "only …
   and …" permission reads as two permissions 2 runs in 3, and a "limited
   to hedging FX ($2bn notional) and …" mandate splits the dollar figure
   off as its own attached-attribute claim 1 run in 3 (closure rule vs.
   the attached-measured-attribute rule, decision 5) — logged, not chased.
   EV-7 separately owns a long-section granularity ceiling
   that degrades this fix in-context (real sections dilute the signal a
   short isolated fixture doesn't).

   Full narrative — the observed Gantry/ratings-forecast/fleet-supplier cases,
   why this is an inline `Exception:` clause and not a gate, the
   granularity-ceiling diagnosis, and the embedded-example regression set:
   `docs/design-decisions/15-list-closure-and-modifier-survival.md`. Live
   verification run-by-run: `docs/prompt-verification-log.md` → "Decision
   15 / ticket 008".

16. **A split piece must keep the predicate its source sentence gave it —
   never decay to a bare "X has Y."** Fires on a shape (a split reduced a
   claim to "X has Y."/"X is Y." while the source sentence asserted more),
   not a checkability judgment — it never asks "is this claim vague," so a
   genuinely bare-but-asserted sentence ("Acme is well run.") is left
   unchanged. Boundary against decision 5's attached-attribute rule: this
   restores a *severed relationship* (reason/basis/driver), never a
   *separately measured attribute* (amount/date/holding/ranking), which
   still splits off as before ("Acme is well positioned, with $2bn of
   committed liquidity." → the $2bn still splits). Boundary against
   decision 15's closure rule, which pulls the *opposite* way: conditions
   that must hold jointly stay one claim, while drivers attributed to a
   single outcome each split, keeping the attribution ("margins improved
   on lower costs and better pricing" → two claims, each keeping "margins
   improved on ...").

   **Residual.** On an isolated many-item run-on, drivers are still
   severed into thin bare-predicate claims roughly 2/3 of the time (down
   from a worse baseline, not eliminated) — the fix's wording doesn't
   reliably reach every sibling of a long list read out of context. A
   genuine "X has Y" claim ("Acme has a revolving credit facility.") is
   also nudged toward a redundant pair with a fuller version of itself
   ("… a EUR 500m revolving credit facility.") 2/3 runs — silent
   containment the ticket-007 identical-text WARNING doesn't catch (it
   only fires on byte-identical text), a candidate for EV-6's
   always-on check. In-context reliability depends on the granularity
   ceiling (decision 15, EV-7).

   Full narrative — the observed Borealis case, why this folds into
   self-test 2 rather than a third test, the self-test-1 tension, and the
   embedded-example regression set:
   `docs/design-decisions/16-predicate-retention.md`. Live verification
   run-by-run: `docs/prompt-verification-log.md` → "Decision 16 / ticket
   009".

17. **Claim extraction and evidence matching are two stages joined only by
    a reviewable claims file.** `python golden_set_pipeline.py extract`
    reads `memos.yaml` and writes one Markdown file per memo under
    `claims/` (gitignored: `---`-fenced YAML frontmatter, `## ` sections,
    `N.` one-line claims); a human reviews, edits, or hand-authors those;
    `python golden_set_pipeline.py build` (or no argument) reads only
    `claims/*.md` and produces the golden set. An unrecognised argument
    errors. The stages share nothing but the file and the source PDFs — a
    claims file written entirely by hand, never having run `extract`, is a
    first-class input. **Why:** claim decomposition is open-ended work
    worth a stronger model or human review before any evidence-matching
    API spend; evidence matching is a constrained judgment call the
    self-hosted model handles well. One strict hand-rolled parser
    (`parse_claims_file`) reads the claims file — every ambiguous shape is
    an error naming the file and line, since a silently mis-parsed file
    poisons every downstream metric. `claim_id` is now a deterministic
    `uuid5` (namespace + memo_id + section + canonical claim text +
    occurrence index), not `uuid4` — a crashed `build` re-run is
    idempotent and its checkpoint mergeable.

    **Sharpest landmine in the implementation, not just the spec — guarded:**
    `_read_memo_config` compares memo ids **case-insensitively** and raises
    on a reused id. The id doubles as the claims-file name, and on a
    case-insensitive filesystem (APFS/HFS+, this repo's own platform, and
    NTFS) `Memo` and `memo` are one file. Without that check, two entries
    differing only in case would pass `run_extract`'s `os.path.exists`
    guard as **skipped**, not failed (exit 0), and `build` would emit a
    golden set with that memo silently, completely absent — no error, no
    `found=False` rows, nothing. Found by the final-verification audit, not
    by design; the refusal is covered in `tests/test_two_stage_pipeline.py`.
    Don't relax it to an exact-match comparison.

    Full narrative — the claims-file grammar in full, why `extract` is
    complete-or-absent, `preview_claim_splits`'s relationship to `extract`,
    every other "implementation went further than the spec text"
    correction, and the atomic-write residual:
    `docs/design-decisions/17-two-stage-pipeline.md`.

Decisions 18 (`tag_pipeline.py`) and 19 (`eval_pipeline.py`) live in their
own rule files: `.claude/rules/tag-pipeline.md` and
`.claude/rules/eval-pipeline.md`.
