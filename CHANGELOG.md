# Changelog

All notable user-visible changes to this pipeline are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/spec/v2.0.0.html). There is
no package to publish: every pull request with a user-visible change is a
release. It adds its own `## [x.y.z] - DATE` heading here, and merging it
tags `vx.y.z` and publishes a GitHub Release automatically. Each entry
cites its Jira ticket (`EV-N`); entries written before the move to Jira,
including most of 0.3.0, cite none. Pre-1.0 — the CLI/API may break between
minor versions. How to choose the version: `CONTRIBUTING.md`, section 3.

## [0.3.0] - 2026-09-23

### Added

- Each pull request with a user-visible change is now its own release: it
  adds its version heading to this file, and merging it tags `vx.y.z` and
  publishes a GitHub Release automatically. The `## [Unreleased]` section
  is gone. A new required `Release` check makes every PR title start with
  its Jira key and every release PR's version one step above the last; the
  `no-release` and `no-jira` labels exempt a PR with no user-visible change
  or no ticket. `CONTRIBUTING.md` describes the process and which file owns
  which kind of project knowledge. (EV-11)

- CI on every pull request and push to `main`: Ruff (lint), basedpyright
  (type check), the pytest suite, and SonarCloud analysis, each a separate
  required check. `pip install -r requirements-dev.txt` installs the same
  pinned tools locally; `ruff check .` and `basedpyright` apply exactly the
  rules CI does. The type check fails only on errors newer than
  `.basedpyright/baseline.json`. (EV-3)

- `python retrieval_pipeline.py retrieve` now searches every phrase three
  ways and writes a `method` column: `dense` (as before), `keyword` (BM25)
  and `both` (Reciprocal Rank Fusion of the two). `retrieval_results.parquet`
  is about three times longer; anything reading it should filter on
  `method`.
- `python retrieval_pipeline.py recheck` — searches with every claim's own
  text (dense) and writes the top 5 chunks per claim to
  `claim_queries.parquet`, for the eval's re-review list.
- `eval_pipeline.py` — scores retrieval against the golden set.
  `python eval_pipeline.py score [label]` checks every input (and refuses,
  listing every problem, on anything that would make a number silently
  wrong — a claims file that does not parse included), then writes
  `eval_runs/<run_id>/`: claim coverage split
  extractive/synthesized, recall, MRR and citation precision for every
  search method and k = 1–20. Phrases are compared with the results exactly
  as `retrieve` reads them, so surrounding whitespace in the phrase file is
  not mistaken for an edit. `python eval_pipeline.py report <run_id|latest>
  [baseline]` writes a self-contained HTML report with a coverage curve,
  per-memo bars, why missed claims were missed, traced examples and a
  re-review list for unverifiable claims. Its passage counts include every
  section searched, also one whose claims are all unverifiable.

### Changed

- `python tag_pipeline.py draft` now reads the company a memo is about from
  a new optional `filing_entity:` key in the claims file's frontmatter,
  instead of a list of memo ids written into `tag_pipeline.py`. A claims
  file without it still works for `extract`, `build` and `finalize`;
  `draft` refuses that memo, naming the file, before reading any PDF. Add
  `filing_entity: <company name>` to each claims file you draft.
- `memos.yaml.example`, `claims.example.md` and `retrieval.example.yaml` now
  describe a made-up company (Acme, PDFs expected in `sources/acme/`) instead
  of a real one, and the docs and tests use fictional names throughout. The
  real-PDF test in `tests/test_finalize.py` now reads `sources/sample.pdf`
  (any small filing; skipped when absent).
- `python retrieval_pipeline.py retrieve` (and the new `recheck`) now refuse
  an index embedded with a different model than `EMBED_MODEL`, as `embed`
  already did. A same-dimension model swap used to produce results from
  mixed vector spaces without any error.
- `retrieval_results.parquet` and `claim_queries.parquet` now record what
  produced them (model, depth, and a fingerprint of each memo's index) in
  the file's metadata, and `eval_pipeline.py score` refuses a file whose
  record no longer matches. Files written before this change are refused
  too: re-run `retrieve` and `recheck` once. `claim_queries.parquet` is
  refused as well when it was rechecked at another depth than `_TOP_K`.
- A run's golden hash now covers which evidence rows a person added in
  review (they are scored quote by quote), so runs scored before this
  change cannot be used as a baseline for runs scored after it.

### Fixed

- `retrieval_results.parquet` and `claim_queries.parquet` are written
  atomically: an interrupted `retrieve` or `recheck` keeps the last good
  file instead of leaving a truncated one.
- `eval_pipeline.py score` skips a temporary `reviewed/*.tmp.xlsx` left by
  an interrupted `finalize` (with a warning) instead of refusing the run,
  and warns when verified quotes are not in their own chunk's text.
- `tag_pipeline.py finalize` refuses a pasted quote containing " | ", the
  separator it joins one chunk's quotes with — also when a line break or
  tab surrounds the bar, or the quote starts or ends with it.
- `eval_pipeline.py score` lists an unreadable (e.g. truncated)
  `retrieval_results.parquet` or `claim_queries.parquet` with the other
  problems instead of stopping on a traceback, and names a memo the file
  never searched instead of blaming a changed index.
- The report's phrase-count warning also names a section only the
  baseline searched, and a traced example found through identical text in
  a non-adjacent passage no longer calls it "neighbouring".
- A phrase file under `retrieval/` that is not valid YAML is now reported
  like any other phrase-file problem (naming the file), instead of a raw
  traceback; `eval_pipeline.py score` lists it with the other problems.
- `python retrieval_pipeline.py retrieve` no longer crashes writing
  `retrieval_results.xlsx` when a chunk's text contains control characters
  pypdf extracted from a PDF's embedded fonts. The `.xlsx` copy has them
  removed; the `.parquet` keeps the raw text.

## [0.2.0] - 2026-09-22

### Added

- `python tag_pipeline.py draft [memo_id ...]` — v2 bundle mode: one AI
  verdict per claim over all of its found chunks (`stated directly` /
  `needs combining` / `not supported`, plus the chunks needed), written to a
  self-contained review sheet `review/<memo_id>.xlsx` for a human to check.
  Model pinned in code to `google/gemini-3.1-pro-preview` (`_BUNDLE_MODEL`),
  not `.env`, so `draft` needs only `LLM_BASE_URL` and `LLM_API_KEY`. Never
  writes `tag`. Every check (memo ids, uuid5 claim ids, claims file vs
  checkpoint — ids, and each id's section and claim text — an unreadable
  checkpoint or one missing a column, rebuilt chunk
  text, duplicate found rows for one chunk, a
  `review` path or sheet path that is not a folder / file) runs
  before credentials or any LLM call; a sheet is never overwritten; a memo whose LLM calls all fail
  writes no sheet, so a re-run retries. The MEMO-004 acceptance test missed
  only its chunk-recall bar (73.6% vs 80%) and was accepted explicitly on
  2026-09-14 — see `docs/prompt-verification-log.md` → "tag_pipeline v2 —
  bundle mode". The "how to" tab explains fixing a `draft failed` claim by
  hand.
- `python tag_pipeline.py finalize <memo_id>` — turns a checked review sheet
  into `reviewed/<memo_id>.xlsx`, the golden-set-schema file the Phase 3
  eval reads. No LLM call. `tag` is derived from the reviewer's answers
  (needed + stated directly → `extractive`, needed + needs combining →
  `synthesized`, else `unverifiable`) and `tag_draft` from the AI's; notes
  are kept in `tag_rationale`. A pasted quote (25+ characters) marks or adds
  a chunk. Refuses — listing every problem with its sheet row and writing
  nothing — on a memo id that is not a plain filename, a `reviewed` path
  that is not a folder or a `reviewed/<memo_id>.xlsx` that is not a file,
  an unreadable claims file, an unchecked claim,
  an invalid verdict or marks, a claim that differs from the claims file in
  either direction, a claim row whose text does not match the claims file
  for its id, an edited identity cell (`claim_id`, `memo_id`, `chunk_id`,
  `doc_id`, `chunk_text`), a chunk row whose shown passage or document is
  not its chunk's, a `found` that is not TRUE (`evidence_span`,
  `confidence`, `ai_verdict` and `ai_needed` are carried over unchecked),
  a deleted chunk row, or a quote that matches nowhere or in several
  unrelated places (a quote found only in the stretch two neighbouring
  chunks of one PDF share is accepted and marks both). Works on sheets
  saved by Excel or Apple Numbers. `reviewed/` may be overwritten;
  `review/` never is. `draft` records each claim's chunk-row count in a
  hidden `chunk_count` column so a deleted *last* chunk row is refused too;
  a hidden `source_docs` column (each PDF's name and a fingerprint of its
  extracted text) lets `finalize` refuse a sheet whose memo's PDFs were
  added, removed or edited since drafting. A sheet drafted before either column existed
  is still accepted, with a warning naming the check that could not run. An
  unreadable or corrupt source PDF is a refusal in both commands, not a
  traceback.
- `golden_set_pipeline._is_deterministic_claim_id` — shared uuid5 check
  used by `tag_pipeline.py` and (later) `eval_pipeline.py`.

### Changed

- Chunking defaults are now `chunk_size=1000, overlap=200` (were `500`/`100`)
  — changed in `chunk_document`, `build_chunk_index` and
  `build_golden_set_draft`, so both `build` and `retrieval_pipeline.py`
  (which pins to `build_chunk_index`'s defaults) pick it up. Snippets are
  now ~1,000 characters (roughly two paragraphs) instead of ~500. This
  changes every `chunk_id` and `chunk_text`: regenerate any existing
  `golden_set_checkpoint.parquet` and `retrieval_index/*.parquet` — a
  golden set built at the old defaults will not join to a retrieval index
  built at the new ones. No config change needed; `chunk_size`/`overlap`
  remain function-signature defaults, not a `memos.yaml` key.
- `_widen_review_columns` also widens a `tag_rationale` column when present
  (written by the new `tag_pipeline.py`). No effect on a `build`-only
  `review.xlsx`, which has no such column.

### Removed

- v1 row tagging (`python tag_pipeline.py` with no command, `run_tag`, the
  `unsure` draft label and its `review.xlsx` columns). It never cleared its
  pilot (41.7% agreement with blind human labels against an 80% bar) and
  was never run on the full golden set; `draft` + `finalize` replace it.
  History: `CLAUDE.md` design decision 18.
- `docs/copilot-memo-to-yaml-instructions.md` (added in 0.1.0) — the
  Microsoft Copilot instructions for turning a credit memo into a
  `memos.yaml` entry. Nothing in the pipeline used it; `memos.yaml.example`
  and `claims.example.md` remain the input templates.

### Fixed

- `_rows_for_claim` now collapses multiple matches that share a `chunk_id`
  into one row (strongest `confidence` kept; a missing or non-string
  confidence ranks lowest, so a malformed answer cannot turn the claim into
  an error row), before the ambiguity grouping. `propose_evidence_from_chunks`
  also stores a non-string confidence as `None` (the match is kept), so a
  malformed answer can no longer make the checkpoint parquet write fail. Removes the duplicate `(claim_id, chunk_id)` rows a golden set
  could carry, and a latent spurious `ambiguous_match=True` when one chunk
  was matched twice with different quoted spans. **Regenerate any existing
  `golden_set_checkpoint.parquet`** to drop rows already written with the
  old behaviour.

## [0.1.0] - 2026-09-08

### Added

- `docs/copilot-memo-to-yaml-instructions.md` — custom instructions for a
  Microsoft Copilot agent that converts one credit memo (Word or PDF) into a
  `memos.yaml` entry holding its Business Profile, Ownership and Industry
  Overview sections as verbatim text. Documentation only: it sits entirely
  outside `golden_set_pipeline.py` and produces pipeline *input*, so nothing in
  the pipeline imports or depends on it. Validate whatever it returns with
  `_read_memo_config` before running `extract` — that reads no PDF and makes no
  API call, so a bad block costs nothing.

- `retrieval_pipeline.py` — a standalone dense-retrieval prototype beside
  the golden-set pipeline. `python retrieval_pipeline.py embed` chunks each
  memo's source PDFs (chunking imported unchanged from
  `golden_set_pipeline`, so `chunk_id`s match the golden set) and caches one
  embedding vector per chunk to `retrieval_index/<memo_id>.parquet`;
  `python retrieval_pipeline.py retrieve` runs each section's hand-authored
  phrases (`retrieval/<memo_id>.yaml`) as independent top-k cosine queries
  and writes `retrieval_results.parquet` + `.xlsx`, one row per
  `(memo_id, section, phrase, chunk_id)` with `rank` and `score`. New
  `EMBED_BASE_URL` / `EMBED_API_KEY` / `EMBED_MODEL` / `EMBED_ASYMMETRIC`
  env vars (see `.env.example`). New dependency: `numpy`. Decoupled from the
  golden set and pre-eval — no recall/precision/MRR yet; the results table
  is shaped so a later harness can compute them. Template:
  `retrieval.example.yaml`. Design:
  `docs/superpowers/specs/2026-09-07-local-retrieval-design.md` — in the
  maintainer's private notes repo (`CONTRIBUTING.md`, section 7); not present
  on a fresh clone. Live-verified against OpenRouter's `baai/bge-m3`
  (dim=1024, standard OpenAI-compatible `{"data": [{"embedding": [...],
  "index": ...}]}` response, no `input_type` body key needed) on a real
  822-chunk corpus: 9 phrases retrieved 20 rows each with
  topically correct top matches.
- A claims file warns about every non-blank line it ignores, wherever it sits —
  including *above* the first `## ` heading, which was previously the one place
  a line could be dropped with no output at all. A claim that lost its `N. `
  (forgotten, or deleted during review) reads as ordinary prose, so there is no
  syntactic way to tell it from a deliberate note; it stays ignored, but it no
  longer disappears in silence. For a note you want ignored *silently*, wrap it
  in a single-line `<!-- ... -->` comment.
- A `memo_sections` entry whose 3rd slot still holds raw section text instead
  of a claims list is now rejected by the pre-run shape check, rather than
  partway through the run — so a hand-built list with one stale entry fails
  before spending on the entries ahead of it.
- `extract` / `build` split: `python golden_set_pipeline.py extract` writes a
  reviewable per-memo claims file (`claims/<memo_id>.md`); `build` reads only
  those. A claims file can be hand-authored without running `extract`. See
  design decision 17 in CLAUDE.md.
- Deterministic `claim_id`, derived from the claim's memo, section, text, and
  in-section occurrence — stable across re-runs, so a re-run `build` is
  idempotent.
- `docs/prompt-verification-log.md` — the run-by-run live-run results
  behind the `extract_atomic_claims` prompt edits (design decisions 5,
  15, 16), moved out of `CLAUDE.md`. `CLAUDE.md` keeps each edit's rule,
  rationale and current known residual; the log holds the `0/3 → 3/3`
  evidence and the canonical list of embedded worked examples. Cuts
  `CLAUDE.md` by roughly a third with no loss of information.

- `memos.yaml.example`, a committed template. `memos.yaml` is now
  gitignored — it holds client memo section text (the deliverable), same
  treatment as `.env` and `sources/`. Copy the example to `memos.yaml`
  and edit there.

- `preview_claim_splits`, plus an opt-in `RUN_CLAIM_SPLIT_PREVIEW` cell
  (section 6c) — an interactive pass that runs `extract_atomic_claims`
  for each section and prints the numbered claim split one section at a
  time, pausing for a yes/no between sections, so a bad split can be
  caught and the extraction prompt fixed before the slow evidence loop
  runs. Any answer other than yes, or a claim-extraction failure, stops
  it. It re-runs extraction rather than reusing the real run's output,
  so `build_golden_set_draft` / `build_golden_set_batch` and `python
  golden_set_pipeline.py` are untouched and never pause for it; the cell
  is left disabled so the plain script and the test suite don't either.
- Optional per-memo `relative_threshold`, `min_candidates` and
  `batch_size` settings in `memos.yaml`, applied to that memo's sections
  only. Source corpora seen so far run from about 1,000 to 6-7,000
  chunks per memo, too wide a spread for one setting to suit every memo
  in a batch run. Omit a setting and the memo uses the run's default,
  exactly as before. `build_golden_set_batch` gains a matching
  `batch_size` parameter, which it previously had no way to set at all.
  A malformed value — or a key the loader doesn't recognize, such as a
  misspelled `batchsize` — is reported as an error naming the memo,
  rather than silently falling back to the default.
- `load_memo_sections_from_config`, plus a `memos.yaml` config file at
  the repo root — running a new memo (or adding a section to an existing
  one) is now an edit to `memos.yaml`, not a code change. Each memo's
  `source_folder` is scanned for PDFs, read once, and shared across all
  of that memo's sections. Adds `PyYAML` to `requirements.txt`.
- `chunk_text` and `bm25_score` columns on the golden-set DataFrame, so
  reviewers can see the actual source chunk and BM25 confidence behind
  each match, not just the LLM's extracted `evidence_span`. Sourced from
  the pipeline, not reviewer-editable on re-import.
- A diagnostic warning in `extract_atomic_claims` when a returned claim
  starts with an unresolved pronoun or pointing word ("it", "the
  latter", etc.) — a tripwire for the fix below, in case a case slips
  past the prompt rule. Logged only, claim is not dropped or altered.
- `extract_atomic_claims` now logs a WARNING when it returns two or more
  claims with identical text (after stripping whitespace), naming the
  repeated text and the count. All copies are kept — never collapsed —
  because identical text can also mean the split failed to separate two
  distinct facts, and dropping one would lose a fact. The reviewer
  resolves a genuine duplicate by deleting the row (design decision 3).
- Per-claim and per-section INFO log lines summarizing how many evidence
  entries were dropped vs. recovered during evidence lookup —
  `propose_evidence_from_chunks_batched` logs a claim's totals (plus how
  many of its batches succeeded) once all its batches complete, and
  `build_golden_set_draft` logs the section-wide sum across all its
  claims. Counts are logged only, not added to the returned DataFrame;
  the existing per-entry WARNING line for each drop/recovery is
  unchanged. "Dropped" counts every entry the model returned that didn't
  become a match (malformed entry, unrecoverable out-of-batch chunk_id,
  or missing evidence_span) as one undifferentiated total.

### Changed

- Running `python golden_set_pipeline.py` with no argument now runs `build`
  (previously it ran the whole pipeline end to end).
- Demo artifacts renamed: `golden_set_checkpoint.parquet`, `review.xlsx`
  (were `pca_golden_set_checkpoint.parquet`, `review_2.xlsx`).
- `claim_id` values change with this release. An in-progress golden-set review
  must re-run `build` and re-review — a review spreadsheet from before cannot be
  re-imported against the new draft. (Keep a Changelog has no Migration heading;
  this rides under Changed.)
- **Breaking:** `memos.yaml` validation is stricter — a `memos.yaml` that was
  valid before this change can now fail to load. Check yours against these two
  new rules if `extract` reports an error where it previously ran clean:
  - A section name is now rejected unless it is already trimmed (e.g.
    `"  Business Profile  "` — leading/trailing whitespace — now errors;
    `"Business Profile"` is fine).
  - A memo `id` is now rejected unless it matches `[A-Za-z0-9._-]+` (no spaces,
    slashes, or other punctuation) — it is now also used as the claims file's
    name (`claims/<id>.md`).
  - Two memo `id`s that differ only in capitalisation (`Memo` and `memo`) are
    now rejected as duplicates. Because the id is also the claims-file name,
    on macOS or Windows — where the filesystem is case-insensitive — they are
    one file: `extract` would write the first and report the second as merely
    *skipped* (exit status 0), and `build` would then produce a golden set
    silently missing that whole memo. Rename one of them.
- **Breaking:** `build_golden_set_draft`'s 3rd positional argument is now
  `claims: list[str]` (already-split claim strings), not the raw section
  text it used to take — it no longer calls `extract_atomic_claims` itself
  (that moved to `run_extract`). A direct caller passing section text now
  hits `TypeError`; pass a list of claim strings instead (e.g. via
  `extract_atomic_claims` or `parse_claims_file`).
  `build_golden_set_batch`'s `memo_sections` tuples' 3rd slot changes the
  same way.
- `propose_evidence_from_chunks_batched` now returns a `(matches, counts)`
  tuple instead of a plain match list, so its caller gets that claim's
  dropped/recovered totals alongside the matches.
  `propose_evidence_from_chunks`'s return type is unchanged (still a
  plain match list) — it instead takes a new optional `counts`
  accumulator parameter, so a direct/manual call (see CLAUDE.md's
  function map) keeps working unmodified.
- `propose_evidence_from_chunks_batched` now logs its per-claim
  candidate/batch-count line at WARNING instead of INFO when a claim
  splits into more than 10 batches — batch count tracks that claim's
  actual LLM cost and its exposure to the out-of-batch chunk_id recovery
  path, so an unusually expensive claim is now visible while a long run is
  still in progress rather than only after the fact from batch-failure
  warnings. Same single line, no line added; 10 or fewer batches logs
  exactly as before.

### Fixed

- `_call_llm_with_json_retry` called with `max_attempts` below 1 now raises a
  `ValueError` naming the bad value, instead of `raise None`'s opaque
  `TypeError: exceptions must derive from BaseException`. No caller does this
  today; the guard was added while making the file clean under a static type
  checker, which flagged the unreachable-in-practice path.

- `extract_atomic_claims` no longer fails on a long memo section. A big
  section can yield 60+ atomic claims, and the reasoning model's hidden
  tokens plus that much JSON overran the default 4096 `max_tokens`,
  returning truncated JSON or empty content. `_call_llm_with_json_retry`
  now takes an optional `max_tokens` (still 4096 by default) and
  `extract_atomic_claims` passes 16384; evidence matching is unchanged.

- `propose_evidence_from_chunks` no longer drops a matched chunk outright
  just because the model's returned chunk_id wasn't among the batch's
  candidates. If the entry's evidence_span is an exact (normalized)
  substring of exactly one candidate's chunk_text in that batch, the match
  is now recovered under that candidate's real chunk_id instead of being
  discarded — the model's quoted evidence is kept even when it mislabels
  which chunk it came from. An evidence_span matching zero or more than one
  candidate is still dropped and logged, as before.
- `extract_atomic_claims` no longer leaves a pronoun or bare pointing
  phrase ("it", "the asset", "that figure") stranded without its
  antecedent when a split separates them. Previously this produced
  claims like "It has stabilized as of Dec'24." that carried no
  identifiable subject, which downstream evidence matching then matched
  against unrelated metrics sharing only the keyword "stable" (observed:
  net debt/EBITDA, IFRS net debt, ICR, and a portfolio yield figure, all
  wrongly matched at medium/high confidence). The claim now reads "The
  asset has stabilized as of Dec'24 versus Dec'23." `claim_text` may
  therefore substitute a referent from elsewhere in the section text, in
  addition to the existing connecting-word substitution (see design
  decision 5 in CLAUDE.md). See CLAUDE.md for the residual limitation
  this doesn't address (a resolved-but-still-generic subject like "the
  asset" can still draw an occasional false-positive match downstream).
  This now also resolves a pointing word that is not the claim's subject
  ("... uses this process ..." resolves "this process" even when the
  subject is named), and lets the prompt borrow a name from a heading or
  label line that sits on its own line above the paragraph. It does not
  propagate a name from an inline "Label:" at a paragraph head to every
  claim that paragraph yields — the shape that motivated the change — so
  a multi-claim paragraph whose subject is named only once at its head
  can still strand later claims (design decision 5 in CLAUDE.md records
  this residual, measured before and after). A section whose headings
  carry names a claim needs should use a literal YAML block (`|`) or keep
  a blank line after each heading, so a folded `>` scalar doesn't glue
  the heading onto the next sentence — see `memos.yaml.example`.

- `extract_atomic_claims` no longer splits a sentence that closes over a
  list — a role "limited to" a named set, or a forecast contingent on
  several conditions together — into separate claims that overstate the
  sentence and can contradict each other. Such a sentence stays one
  claim. A list the sentence does not close over still splits normally,
  and a partial qualifier ("but to a lesser extent") now survives onto
  the claims for the members it modifies instead of being dropped. The
  rules are verified sentence by sentence; inside a large section the
  output-volume ceiling tracked by EV-7 can still coarsen these
  shapes, so the motivating real section needs that fix as well.
  Some closed-list phrasings also still split on a minority of runs even
  in isolation, and the fix does not reach a sentence already degraded in
  the source text. Design decision 15 in CLAUDE.md records these
  residuals, measured before and after.

- `extract_atomic_claims` no longer reliably carves a driver or
  attribution list into bare-existence claims. A sentence like "earnings
  improved on pricing initiatives and cost management" used to split into
  "The company has pricing initiatives." / "…has good cost management." —
  claims no source document would confirm or deny. Each piece now keeps
  the predicate the source sentence gave it ("The company's earnings
  improved on pricing initiatives."). A whole sentence the memo asserts
  outright that happens to be unfalsifiable ("The company is well run.")
  is left unchanged — it is not this fix's concern. Design decision 16 in
  CLAUDE.md records the measured pre/post results and residuals: the
  attribution-dropping shatter is eliminated on isolated input, the
  in-context improvement is unproven (a re-run put the bare form back at
  2/3 — EV-7's granularity ceiling), and an isolated run-on still
  severs the drivers into thin verbs on a majority of runs.
