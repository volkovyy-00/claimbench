# Design decision 18 — full narrative

Split out of `CLAUDE.md` to keep that file's per-session load down; the
condensed rule and current residual live there under design decision 18.
Design: `docs/superpowers/specs/2026-09-13-tag-pipeline-v2-bundle-design.md`
(gitignored; its §12–§13 addenda override earlier sections). Live results:
`docs/prompt-verification-log.md` → "tag_pipeline v2 — bundle mode" (and
"`tag_pipeline` prompt" for v1).

## What the eval needs, and what never changes

The Phase 3 eval reads, per memo, `claims/<memo_id>.md` plus a reviewed
row-format `.xlsx`, and from that file only `tag` (closed set `extractive`
/ `synthesized` / `unverifiable`; a blank halts the run as a worklist) and
the `tag_rationale` prefix `"auto: found=False"`. `tag` stays 100%
human-approved: no command writes it except `finalize`, and `finalize` only
from claims the user marked CHECKED.

## Bundle mode (current)

**The unit of judgement is a claim with all its found chunks.** `draft`
asks once per claim: is it *stated directly* (one chunk, on its own, says
the claim's fact — different wording, units or rounding still count),
*needs combining* (true only by putting chunks together, calculating, or
describing numbers with a word the text doesn't use), or *not supported*
— and which chunks are needed. Each chunk is shown with 500 characters of
source text before and after it, because pypdf strips table column headings
from the chunk itself and the context usually carries them. Chunks are
labelled A, B, C… (never shown by `chunk_id`, so there is nothing long for
the model to echo back wrongly). An answer that fails to parse or validate
(unknown label; `not supported` with chunks; another verdict with none), or
never arrives, is retried once, then shown as `draft failed` in the sheet.

**Sheet writes.** `draft` never overwrites a `review/<memo_id>.xlsx`; a memo
whose LLM calls all fail writes no sheet, so a re-run retries it. `finalize`
may overwrite `reviewed/<memo_id>.xlsx` (regenerable); `review/` is never
overwritten by any command.

**Order of work per memo:** audit or reword claims → `build` → `draft` →
review → `finalize`.

**Row tags follow mechanically:** a needed chunk of a stated-directly claim
is `extractive`; of a needs-combining claim `synthesized`; every other row
`unverifiable`. The eval's claim bucket rule (any `extractive` row →
EXTRACTIVE, else any `synthesized` → SYNTHESIZED, else UNVERIFIABLE) then
reproduces the verdict exactly, so a
claim-level human judgement becomes row-level ground truth without asking
anyone the row-level question v1 could not answer.

**Checks before spending.** `draft` runs every check for every requested
memo before it builds an LLM client: the memo is in the checkpoint and in
`_FILING_ENTITY`; every `claim_id` is a deterministic uuid5; no claim has
more than 40 found chunks (a tripwire, never truncated); no build-error
rows (a build failure must not be dressed up with the eval's "search found
nothing" prefix); the claims file and the checkpoint hold the same claims;
every found chunk rebuilt from `source_folder` carries the checkpoint's
document and text. A typo costs nothing and needs no credentials.

**Acceptance (pre-registered, MEMO-004, 2026-09-13).** Two bundle versions
× 3 runs × 2 models, scored against answer keys fixed before the run.
`gemini-3.1-pro-preview` cleared 7 of 8 bars: never `stated directly` on an
unsupported claim; 32/32 claim-versions stable across runs; precision
95.5%. It failed chunk recall — 73.6% against ≥80% — on claims computed
from a table (a region's valuation divided by the total): three such claims
came back `not supported` in every run, and their key chunks were never
marked. `gpt-5.6-luna` cleared 4 of 8. **The user accepted gemini on
2026-09-14 despite the recall miss**, on the grounds that every draft is
reviewed and a miss costs review time rather than a wrong tag; the measured
FAIL stays on record. The first live `draft` of MEMO-004 matched the
acceptance test's majority verdict on 16 of 16 claims.

**Model budget.** `_BUNDLE_MAX_TOKENS = 16000`. Hidden reasoning tokens
count against `max_tokens` and `call_llm` sends no thinking budget; at 4096,
`gemini-2.5-pro` truncated 31% of its JSON answers in v1's model probe
(mechanism: decision 4). The acceptance pre-flight measured gemini-3.1 using
about 12% of 16000 for reasoning.

**Pinned in code, not `.env`.** `_BUNDLE_MODEL`, `_BUNDLE_MAX_TOKENS` and the
frozen `_BUNDLE_PROMPT` (its SHA-256 is recorded in
`docs/prompt-verification-log.md`) live in `tag_pipeline.py`; changing any
of them voids the acceptance result above. `.env`'s `LLM_MODEL` drives
`extract`/`build` only: `draft` builds its client with `_bundle_client()`,
not `LLMClient.from_env()`, so it needs only `LLM_BASE_URL` and
`LLM_API_KEY`.

## The review sheet and `finalize`

The sheet has a "how to" tab and a "bundles" tab: a blue claim row (VERDICT
dropdown pre-filled with the draft, the AI's reason, CHECKED, a note), one
row per chunk (text, document, grey before/after context, NEEDED?
pre-filled), and two `+` rows for pasted quotes. Hidden columns carry
`row_kind`, the claim and checkpoint fields and the AI's original answer, so
`finalize` never reads the checkpoint. Dropdowns suggest but do not enforce:
`finalize` is the single enforcement point and lists every problem at once,
which beats an Excel pop-up mid-edit — and a `draft failed` claim is fixed
by typing over it.

**Apple Numbers, measured on the real MEMO-004 sheet:** it inserted an
"Export Summary" worksheet first, made 15 of 16 hidden columns visible
(only `row_kind` stayed hidden) and added an empty-header column; all 55
chunk rows' values still equalled the checkpoint. Consequences:
- the tab is opened by name and columns mapped by header — a reader that
  took the first sheet would have found no claims and reported nothing;
- hidden columns are editable in practice, so they are **verified**:
  every row's `claim_id` must equal its claim row's and every claim's must
  be in the claims file; `memo_id` must be the command's; every `chunk_id`
  must be in the chunk index rebuilt from the PDFs with the same document
  and text; the booleans and the score must still be booleans and a number.
  That is why `finalize` always reads the PDFs (MEMO-004: ~26 s). Each
  chunk row's visible passage and document must also be that chunk's, so a
  mark applies to what the reviewer read. **Residual:** a row re-pointed in
  every visible *and* hidden cell to another chunk still passes — only the
  checkpoint knows `draft`'s bundle, and `finalize` does not read it.

**Checks, all reported together with sheet row numbers:** CHECKED is `yes`
on every claim; the verdict is one of the three words (not `draft failed`);
`not supported` has nothing marked and the other verdicts have something
marked (a quoted `+` row counts); NEEDED? is `yes` or blank; a `+` row has
both a quote and `yes`, or neither — never an implied mark (decision 3's
rule: an ambiguous state is refused, not guessed toward more evidence);
the claim census matches the claims file **in both directions** — a claim
reworded after drafting, or added after drafting, is refused with the
recovery spelled out; a claim row's text and section must equal the claims
file's for its `claim_id` (the census alone would pass ids swapped between
two whole claim blocks, putting each verdict on the other claim); a deleted chunk row is refused (a claim with no chunk
rows whose AI reason is not `auto: found=False, nothing was found by
search`, chunk rows not lettered A, B, C… without a gap, or fewer chunk rows
than the hidden `chunk_count` `draft` writes on the claim row — the only
check that sees a deleted *last* row) — **residual:** a sheet drafted before
`chunk_count` was added (MEMO-004's) has no such column and is still
accepted, so deleting a claim's last chunk row (B of A, B) passes there;
`finalize` logs a warning naming this. `yes` and verdict words are compared
stripped and case-folded (Excel autocorrect types `Yes`).

**Source PDFs are pinned the same way.** `draft` writes the memo's PDF names,
each with a fingerprint of its extracted text, in a hidden `source_docs`
column, and `finalize` refuses when `source_folder` now gives a different
set — a PDF added (even one with no text), removed, or edited in place (a
claim drafted with nothing found was never searched in the new text).
**Residual:** a sheet without the column (MEMO-004's) is accepted with a
warning.

**Pasted quotes** are matched with all whitespace removed (pypdf drops the
space between table cells, which collapsing could not equal), at least 25
characters. One chunk → that chunk; two neighbouring chunks of one document
→ both (the quote sits in their shared overlap, decision 1); anything else,
including no match, is a problem naming the likely cause. A quote resolving
to a chunk already in the bundle marks it; otherwise it becomes a
human-added row. Ligatures and hyphenation are not folded. **Residual:** a
quote genuinely repeated in two neighbouring chunks over-marks one; the
summary labels that path.

**Output rows.** One per `(claim_id, chunk_id)`; claim text and section
from the claims file, chunk text from the rebuilt index, never from the
sheet's editable copies. `tag_draft` is blank where the AI never judged the
row (a failed draft, a human-added chunk). `tag_rationale` records `bundle
review <date>: verdict …; chunk marked | marked by quote | not marked; AI:
<reason>`, or `human-added <date>: quote …`, or `auto: found=False, nothing
was found by search`, then every note. `human_reviewed` is True on every
row. Text beginning `=` is kept as text (export_for_review would store a
formula that reads back blank) and NUL bytes from pypdf are stripped.
`evidence_span`, `confidence`, `ai_verdict` and `ai_needed` are carried
from the sheet without verification (the eval ignores them; `tag_draft` is
advisory).

**Changing a claim after drafting** loses the memo's review: answers are
keyed by `claim_id`, and carrying them across a rebuild is unsafe because
`build` re-rolls evidence for unchanged claims too (decision 9). EV-9
tracks it.

## Why not rows — v1 (2026-09-10 → removed 2026-09-14)

v1 asked, per evidence row, for `extractive` / `synthesized` /
`unverifiable` / `unsure`, one batched call per claim, into advisory
`tag_draft` + `tag_rationale` columns.

- **3-model probe** (16 claims / 60 rows): `gpt-4.1-mini` over-called
  `extractive` and violated the value-contradiction rule even after doing
  the arithmetic; `gemini-2.5-pro` failed 31% of calls on truncated JSON;
  `claude-sonnet-4` was pinned.
- **The filing-entity hint** (`_FILING_ENTITY`, kept by v2): without it,
  first-person passages ("we", "our") have no named subject, and a 53–74%
  first-person corpus collapsed toward `unsure`; naming the entity took
  that to 0%.
- **§12 pilot, 2026-09-12 (11 claims / 72 rows): NO-GO** — agreement with
  blind human labels 41.7% (bar ≥80%), `synthesized` on hand-`unverifiable`
  rows 47% (bar ≈0). The prompt's `unsure` example was recited in rationales
  and then not applied; the one-sided `unsure < 20%` bar passed trivially at
  0% and hid it.
- **Confirmation round, 2026-09-13 (30 unseen rows): still NO-GO.** Unaided
  human accuracy 80%, one production call 73.3%. An intermediate "the tagger
  beats the human" (78.8% vs ~67%) came from scoring six prompt versions
  against labels the same human re-adjudicated each round, and was
  retracted.
- **Model sweep** (24 signal rows, memos 1–3): gemini-3.1-pro 87.5% down to
  gpt-5.6-sol 70.8%; none transferred to MEMO-004's new industry.
- **Diagnosis:** the row question is under-determined. `synthesized` needs
  the other chunks to answer; all 6 of the human's own blind → considered
  changes on 30 rows involved `synthesized`, none between `extractive` and
  `unverifiable`. The models fail in the same place: across 8 prompt
  versions and a 6-model sweep, v1 never cleared the pilot's bars.
  Reviewing a claim with its whole bundle instead reproduced 14/14 of the
  human's confident row labels — the case for bundle mode.

## Method lessons (binding on any future prompt change)

1. **Pre-register** the model, prompt and bars before running; report the
   pre-registered cell even if another looks better.
2. **Label blind first.** The same human's accuracy moved 10–20 points with
   the model's reasoning visible (73.3% with, 60.0% without, on identical
   rows).
3. **Never tune against a reference re-adjudicated each round** — it becomes
   a second training signal; every ruling looks defensible while the
   aggregate drifts toward the model. A gate number needs blind labels on
   rows no tuning round touched.
4. **Row identity is `(claim_id, chunk_id)`**, never `chunk_id` alone (one
   passage is evidence for several claims) and never truncated claim text.
5. **A bar that can pass at zero needs a lower bound**, or a check that each
   worked example's label fires at least once.
6. **Cap rows per claim in any sample** — six rows of one claim measure a
   model's prior, not its judgement.
