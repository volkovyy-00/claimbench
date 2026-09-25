# Golden Set Pipeline

Phase 1 of an eval framework for an internal LLM app that generates financial
memos from uploaded source documents. This builds a "golden set" of
claim → evidence pairs, extracted from human-written reference memo
sections and matched against the real source PDFs those sections were
written from. Later phases will use this golden set for retrieval-recall
and hallucination metrics — so its correctness (no silently dropped
evidence, no silently mis-tagged ambiguity) matters more than it would for
a one-off script.

The user is a non-developer who reads this code, not just runs it. Keep
functions single-purpose, keep docstrings accurate and complete (they are
the primary documentation), and don't introduce abstractions or config
surface beyond what's asked for.

## Everything lives in one file

`golden_set_pipeline.py` — Jupytext "light" format (`# %%` cell markers).
Open it in VS Code/Jupytext to work with it as a notebook, or run it as a
plain script (`python golden_set_pipeline.py [extract|build]` runs the `if
__name__ == "__main__":` block at the bottom, which dispatches to
`run_extract`/`run_build` — no argument means `build`; see "Commands" and
design decision 17). The pipeline logic itself stays in this
one file; `tests/` (see "Testing convention" below) is the one exception,
holding the committed `pytest` suite. `docs/pipeline-overview.md` is a
plain-English walkthrough (defers to this file wherever the two
disagree). `docs/prompt-verification-log.md` holds the run-by-run live
results behind the `extract_atomic_claims` prompt edits (design decisions
5, 15, 16) — this file keeps their rules and current residuals, the log
keeps the `0/3 → 3/3` evidence. `docs/design-decisions/` (committed) holds
the full observed-failure narrative for the design decisions whose entry
below is a condensed rule + current residual + pointer (5, 7,
9, 11, 12, 13, 14, 15, 16, 17, 18) — split out so this file stays loadable every session without
carrying every design decision's complete history; the numbered entry
below is authoritative on the rule itself, the linked file is the "why"
in full. Open work — including the fix for any residual named below — is
tracked in the Jira project `EV`; `CONTRIBUTING.md` says which file owns
which kind of project knowledge.
`memos.yaml.example` and `claims.example.md` (both committed, repo root)
are the templates for `extract`'s input and `build`'s input respectively
(design decision 17) — copy and edit rather than write either format from
scratch.

This is a public GitHub repository (`origin` = `volkovyy-00/claimbench`,
started 2026-09-23 from one scrubbed commit). `main` is protected: changes
go through a PR that passes the five required checks (see "Testing
convention") and follows `CONTRIBUTING.md`: the Jira key first in the PR
title, a `CHANGELOG.md` version heading in every user-visible PR, and a
tag plus GitHub Release created automatically on merge. The
pre-publication history is private (see `CLAUDE.local.md`); never push it
here. `README.md`
presents the project publicly as **ClaimBench** (MIT, `LICENSE`): install,
the six-step usage walkthrough, a command/config/layout reference, and
nothing past that. It must stay neutral (no client or company references).
This file stays the maintained reference for testing/conventions/design
decisions; don't let rationale or design detail creep into README.

## Environment

- `.env` (real credentials, never read/print its contents) holds
  `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` — set to OpenRouter +
  `openai/gpt-4.1-mini` on 2026-09-22 (deliberate). It was
  `openai/gpt-oss-120b`, a reasoning model: that's why `call_llm` sets
  `max_tokens` explicitly (design decision 4), which stays the right
  guard if a reasoning model is ever set here again. Design decisions 5,
  9, 15 and 16 record prompt behaviour measured under the *older* model,
  so re-verify before trusting a residual against a fresh run.
  `.env.example` documents the shape.
- `.venv/` has all deps installed: `source .venv/bin/activate`.
- Python 3.10+ (the code uses `X | Y` unions and `list[dict]` builtin
  generics); developed on 3.13.
- `requirements.txt`: pandas, numpy, requests, openpyxl, pyarrow, python-dotenv,
  jupytext, rank_bm25, pypdf, pdfplumber, PyYAML, pytest.
  `requirements-dev.txt` adds the CI tools, pinned exactly: ruff,
  basedpyright, pandas-stubs, pytest-cov.
- `sources/` holds the source PDFs, one folder per memo — the convention
  `load_memo_sections_from_config` expects (see "Config-file loader"
  below), so a new memo's PDFs get their own subfolder too. Gitignored,
  not committed, same as the `*.parquet`/`*.xlsx` artifacts below.
  `sources/sample.pdf` (any small real filing; a symlink is fine) lets
  `tests/test_finalize.py`'s real-PDF test run instead of skipping.
- **Committed files use fictional names** (Acme, Borealis, Gantry,
  Cobalt, Delphin, Paxton, Globex …) — the repo is public. Never commit a
  real company, person, memo id or figure from a client memo. Which
  corpora are on this machine, how to re-fetch them, and the real names
  behind the placeholders live in the gitignored `CLAUDE.local.md`
  (loaded alongside this file when present).

## Commands

```bash
source .venv/bin/activate            # deps are pre-installed in .venv/
pytest                               # full suite (mocked, a few seconds)
ruff check .                         # lint, same rules as CI (ruff.toml)
basedpyright                         # type check, same as CI (pyrightconfig.json + baseline)
python golden_set_pipeline.py extract   # memos.yaml -> claims/*.md   (review these)
python golden_set_pipeline.py build      # claims/*.md -> golden_set_checkpoint.parquet + review.xlsx
python tag_pipeline.py draft [memo_id ...]   # checkpoint + claims/ + PDFs -> review/<memo_id>.xlsx (AI verdict per claim, human checks)
python tag_pipeline.py finalize <memo_id>     # checked review/<memo_id>.xlsx + claims/ + PDFs -> reviewed/<memo_id>.xlsx (the eval's input; no LLM)
python retrieval_pipeline.py embed [memo_id ...]   # PDFs -> retrieval_index/<memo_id>.parquet (needs EMBED_* in .env)
python retrieval_pipeline.py retrieve              # retrieval/*.yaml + index -> retrieval_results.parquet/.xlsx (no args)
python retrieval_pipeline.py recheck               # claims/*.md + index -> claim_queries.parquet (each claim's text as a query)
python eval_pipeline.py score [label]              # golden set + retrieval results -> eval_runs/<run_id>/ (no API calls)
python eval_pipeline.py report <run_id|latest> [baseline_run_id] [--method=dense] [--k=5]   # -> eval_runs/<run_id>/report_dense_k5.html
python golden_set_pipeline.py            # same as build
```

Live prompt-verification run (real API, minutes each, repeat 3x — see
"Testing convention" for why and the non-determinism it guards against):
launch with `Bash` `run_in_background` or `nohup python3 ... &`, then watch
with `Monitor` (`persistent: true`, since a full corpus run exceeds its
300s default).

## Pipeline shape

Two stages, joined only by the claims file (design decision 17):

```
Stage 1 — extract (run_extract)              [human authors from scratch,
                                                skipping this stage, instead]
  memos.yaml section text                                │
        │                                                │
        ▼                                                │
  extract_atomic_claims                                  │
        │                                                │
        ▼                                                │
  claims/<memo_id>.md  ◄────── [human reviews / edits] ◄─┘
        │
        ▼
Stage 2 — build (run_build)
        │
        ▼
  parse_claims_file ──► [claim, claim, ...]
                                  │
source_documents (list of PDFs) ──► build_chunk_index ──► chunk_index
                                  │                            │
                                  ▼                            ▼
              for each claim: bm25_threshold_shortlist(claim, chunk_index)
                                  │
                                  ▼
           propose_evidence_from_chunks_batched(claim, shortlist)
                                  │
                                  ▼
                    _rows_for_claim (ambiguity grouping)
                                  │
                                  ▼
                         one section's DataFrame
```

A claims file written entirely by hand — never having run `extract` — feeds
Stage 2 exactly the same way; `parse_claims_file` doesn't know or care which
origin produced the file.

`build_golden_set_draft` runs Stage 2's chunk-index-through-DataFrame part for
one section (its 3rd argument is a `claims: list[str]`, already split — it no
longer calls `extract_atomic_claims` itself); `build_golden_set_batch` loops
it over many sections with parquet checkpointing every N sections.

`bm25_threshold_shortlist`/`propose_evidence_from_chunks_batched` are the
**default** path (no fixed candidate cap — see below). The older fixed-count
`bm25_shortlist`/`propose_evidence_from_chunks` still exist, unused by
`build_golden_set_draft`, kept available for a cheaper/faster manual pass.

## Retrieval prototype (`retrieval_pipeline.py`) — sibling, not part of the pipeline

A standalone retrieval experiment (dense, keyword and combined search),
landed to have a retrieval system to measure once the golden set exists.
Not wired into `extract`/`build`.

- **Imports (never edits) from `golden_set_pipeline`:** `build_chunk_index`,
  `chunk_document`, `_load_source_documents`, `_scan_claims_file_sections`,
  `_valid_memo_id`, `_section_name_issue`, `parse_claims_file`,
  `_derive_claim_id`, `_claims_with_occurrence`, `_tokenize`. If you rename
  any of these, `retrieval_pipeline.py` is a second consumer —
  `tests/test_retrieval_pipeline.py::test_imports_from_golden_set_pipeline_resolve`
  fails loudly on a rename.
- **Chunking is pinned** to `build_chunk_index`'s defaults (1000/200) and
  uses `_load_source_documents` (so PDF text comes from `load_pdf_text` /
  pypdf, same as `build`). `chunk_id` parity with the golden set therefore
  holds **only if that memo's golden set was built with those same defaults
  and pypdf** — `build_golden_set_draft` allows overriding chunk size, and
  `load_pdf_text_pdfplumber` is a documented swap. It is not a guarantee "by
  construction"; if a memo's golden set used non-defaults, its retrieval
  index must match — `eval_pipeline.py` refuses to score a memo whose
  golden chunks are missing from or differ from its index.
- **Own config / env:** `retrieval/<memo_id>.yaml` (phrases per section,
  gitignored, template `retrieval.example.yaml`) and
  `EMBED_BASE_URL` / `EMBED_API_KEY` / `EMBED_MODEL` / `EMBED_ASYMMETRIC`
  (`.env.example`). `EmbeddingClient` mirrors `LLMClient`; `call_embeddings`
  is the sole HTTP contact point, like `call_llm`. `embed`, `retrieve` and
  `recheck` all refuse an index whose recorded model differs from
  `EMBED_MODEL` (`_check_index_model`; a same-dimension model swap would otherwise mix vector
  spaces silently); an index with no recorded model only warns.
  `retrieve` and `recheck` write a **provenance record** into their
  parquet's file metadata (`_write_with_provenance`: command, model, depth,
  each memo's `_index_fingerprint`); `read_provenance` reads it back, and
  `eval_pipeline` refuses a file whose record doesn't match the current
  index, model and `_RETRIEVE_DEPTH`. pyarrow schema metadata, not pandas
  `attrs` (documented as experimental).
- **`embed [memo_id ...]`** (optional memo filter) →
  `retrieval_index/<memo_id>.parquet` (chunk_id → vector cache, incremental,
  atomic write). **`retrieve`** (no arguments — always whole-corpus, every
  `retrieval/*.yaml`) → `retrieval_results.parquet` + `.xlsx`, one row per
  `(memo_id, section, phrase, method, chunk_id)` with `rank`/`score`. Every
  phrase is searched three ways (`_RETRIEVAL_METHODS`): `dense` (cosine),
  `keyword` (BM25 built once per memo with `golden_set_pipeline._tokenize`;
  chunks scoring 0 are omitted, so a phrase sharing no word with the corpus
  returns no keyword rows) and `both` (Reciprocal Rank Fusion, `_RRF_K` 60;
  exact ties go to the dense ranking).
  No method flag — keyword is free and fusion is a merge.
  `dedupe_by_section()` collapses that to the modeled top-`_TOP_K` union per
  method.
- **`recheck`** (no arguments) → `claim_queries.parquet`: every claim's own
  text from `claims/<memo_id>.md` as a dense query, top `_TOP_K` chunks per
  claim, `claim_id` recomputed with `_derive_claim_id`. Reads claims files
  only — never tags, never `reviewed/` — so retrieval stays blind to the
  ground truth. It exists for `eval_pipeline.py`'s re-review candidates.
- **Constants** (`_EMBED_BATCH_SIZE`, `_TOP_K`, `_RETRIEVE_DEPTH`) are
  edit-here literals, matching golden_set_pipeline's single-site-literal
  convention. The artifact stores `_RETRIEVE_DEPTH` (20) rows per phrase;
  `_TOP_K` (5) is applied only downstream, by `dedupe_by_section` (and the
  CLI's own top-`_TOP_K` summary print) — `_RETRIEVE_DEPTH` > `_TOP_K` so
  the artifact records where a golden chunk landed even outside the modeled
  top 5, which MRR / recall@k need.
- **Measured by `eval_pipeline.py`** (section below). Keep `_RESULTS_COLUMNS`
  and `_CLAIM_QUERY_COLUMNS` stable — the eval imports and checks them.
- Tests: `tests/test_retrieval_pipeline.py` (mocked). Design:
  `docs/superpowers/specs/2026-09-07-local-retrieval-design.md`, in the
  maintainer's private notes repo cloned at `docs/superpowers/` (see
  `CONTRIBUTING.md`, section 7) — absent on a fresh clone of this repo alone.

## Evidence tagging (`tag_pipeline.py`) — sibling, not part of the pipeline

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
   in the golden-set schema, `tag` derived from the user's answers.

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
- Tests: `tests/test_bundle_tagging.py` (draft) and `tests/test_finalize.py`
  (finalize and the eval ground-truth contract), both mocked. Design:
  `docs/superpowers/specs/2026-09-13-tag-pipeline-v2-bundle-design.md`
  (gitignored; the §12–§13 addenda override earlier sections). Acceptance
  record: `docs/prompt-verification-log.md` → "tag_pipeline v2 — bundle mode".

## Eval harness (`eval_pipeline.py`) — sibling, measures only

Scores `retrieval_pipeline.py`'s results against the golden set and writes
a self-contained HTML report (design decision 19). **It never searches**:
it reads `claims/<memo_id>.md` (the claim census), `reviewed/<memo_id>.xlsx`
(evidence and tags), `retrieval_results.parquet`, the index (parity check
only) and `claim_queries.parquet`. `tests/test_eval_pipeline.py::test_the_scorer_never_retrieves`
enforces that structurally.

- **`score [label]`** → `eval_runs/<YYYY-MM-DD_HHMMSS>[_label]/`: six
  parquet tables (`claims`, `evidence`, `claim_hits`, `chunk_hits`,
  `metrics`, `candidates`) + `meta.json` (golden, phrase and results
  fingerprints, model). Never overwrites. One run holds every method and
  every k = 1…20; only a phrase change makes a new run.
- **`report <run_id|latest> [baseline_run_id] [--method=…] [--k=…]`** →
  `eval_runs/<run_id>/report_<method>_k<k>[_vs_<baseline>].html` (default
  dense, k=5). In a notebook, `show_report(run, method=, k=, baseline=)` or
  the `RUN_DEMO_REPORT` cell (kept `False`). k is **per search phrase**; the
  report also shows the distinct passages each section was handed.
- **Imports (never edits)** from `golden_set_pipeline`: `_same_evidence`,
  `_group_equivalent_chunks`, `_UnionFind`, `_HUMAN_ADDED_PREFIX`,
  `_QUOTE_SEPARATOR`, `_chunk_index_of`, `_derive_claim_id`,
  `_claims_with_occurrence`, `_is_deterministic_claim_id`, `_normalize_span`,
  `_valid_memo_id`, `parse_claims_file`; from `retrieval_pipeline`: `_RESULTS_COLUMNS`,
  `_CLAIM_QUERY_COLUMNS`, `_RETRIEVAL_METHODS`, `_RETRIEVE_DEPTH`, `_TOP_K`,
  `_excel_safe`, `_index_fingerprint`, `_load_index_model`,
  `_read_phrase_config` (so phrases compare exactly as `retrieve` read
  them), `read_provenance`, `dedupe_by_section`. It is the third consumer of
  `golden_set_pipeline` internals; `test_imports_from_sibling_modules_resolve`
  fails loudly on a rename.
- **Every input problem halts, all listed together** (`EvalInputError`):
  blank/unknown/`rejected` tags, census drift, pre-decision-17 ids, a
  claims file that does not parse, phrase-file content
  `_read_phrase_config` rejects, a section with claims but no dense
  results, a phrase with dense but no `both` rows or a memo with no
  keyword rows, a golden chunk missing from or differing from the index
  (text, or an evidence row's `doc_id`), a verified row with no `doc_id`,
  results not retrieved with the phrases on disk, results or
  `claim_queries.parquet` with no provenance or one naming another index,
  model or depth — `_RETRIEVE_DEPTH` for the results, `_TOP_K` for the
  claim queries (`check_provenance`, which also catches mixed models, and
  a memo the file never searched), stale `claim_queries.parquet`, a
  results or claim-queries file that cannot be read. Each check runs even when another
  failed; only index parity and the sections check wait for a memo's
  ground truth to load.
- Design: `docs/superpowers/specs/2026-09-10-retrieval-eval-harness-design.md`
  (gitignored, refreshed 2026-09-22).

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
    its bars and was removed 2026-09-14.

    **Residual: chunk recall 73.6% against a ≥80% bar, accepted explicitly
    by the user on 2026-09-14** (the other 7 of 8 pre-registered MEMO-004
    bars passed). Misses sit on claims computed from a table; the cost is
    review time, not a wrong tag. Changing `_BUNDLE_MODEL`, `_BUNDLE_PROMPT`
    or `_BUNDLE_MAX_TOKENS` voids this result.

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
    acceptance test, every `finalize` check with its reason:
    `docs/design-decisions/18-evidence-tagging.md`.

19. **The eval scores claims against human-tagged evidence, and refuses
    rather than scoring around a problem** (`eval_pipeline.py`). The census
    comes from the claims file, the evidence from `reviewed/`; the relevant
    set is `found` **and** a tag of `extractive`/`synthesized` — `found` is
    the pipeline's guess, the tag the human verdict (defining it by `found`
    alone inflated coverage and deflated recall on the first reviewed file).
    Unverifiable claims leave the coverage denominator and are reported
    beside a re-review worklist (`claim_queries.parquet`, chunks meaning
    search offers that no human judged), never as "absent from the sources":
    the golden set was built with lexical BM25 (decision 7). A hit is the
    same `chunk_id` **or** decision 1's rule (`_same_evidence`, golden quote
    vs retrieved text) — the rule never matches a chunk with itself, so the
    identity test is required. An adjacent chunk must also hold the whole
    golden quote inside the text the two chunks share (`_shared_edge`) —
    the only place the same passage can sit in both — so a short quote
    repeated elsewhere in the neighbour is not a hit. One direction only:
    feeding the shared text to the two-way `_same_evidence` would count a
    long quote that merely contains it (measured: +12 false hits on the real
    set). A human-added row's quotes, joined with `_QUOTE_SEPARATOR` by
    `tag_pipeline`, are tested one by one, and grouped by those same quotes;
    only such a row (its `tag_rationale` starts `human-added`) is split — any
    other span is chunk text and may contain the separator itself. Evidence takes its section from the census
    (the claim_id encodes it), never from the sheet cell. Hits are computed once at depth 20 and every
    k read off by best rank. Each hit records the golden row it matched, so
    the report quotes the evidence actually found. k is per search phrase
    (18–43 distinct passages per section at k=5 on the real set); runs record
    phrase counts and the report warns when a baseline's differ. A run is
    refused unless the results were retrieved with the phrases on disk. MRR
    (query = claim) is stored, not shown — a best rank over several phrase
    rankings rises with phrase count. Precision is citation precision, a
    floor, pooled per memo. Fusion ties go to the dense rank; with the app's
    single-word phrases, fusion scores below dense at k=5 (a chunk in both
    lists outranks dense's own #1). **Measured 2026-09-22:** at 1000/200
    chunking the overlap rule added no claim over exact matching (the "up to
    2×" first measured at 500/100 does not reproduce) — kept for
    correctness. Text is compared after removing the control characters the
    reviewed sheet cannot hold (`_excel_safe`). **Residual — duplicated
    documents:** MEMO-001's folder holds two documents that repeat
    the same passages; decision 1 keeps different documents as separate
    evidence groups (correct). When both copies were cited, coverage is
    unaffected but a claim found in only one copy can reach at most 50%
    macro recall; when the reviewer cited only one copy, a retrieved chunk
    from the other is never credited (a miss, and against precision) — 0
    such claims on the 2026-09-23 sheets. Explained in the report's Details
    note, deliberately not fixed. **Residual — quotes not verbatim in their
    chunk:** 20 of 169 golden quotes (2026-09-23) are not in their own
    chunk's text after `_normalize_span` (split words, curly apostrophes,
    quotes running past the chunk; finalize matches ignoring whitespace).
    Only their own chunk can hit them, never a neighbour; `score` logs a
    WARNING with the count. Matching the finalize way changed no coverage
    and one group. **Residual — overlapping quotes in adjacent chunks:**
    two golden rows whose quotes both span the shared overlap, neither
    containing the other, stay two groups under decision 1 (3 pairs on the
    real set; coverage unchanged, macro recall ±0.02). Not widened in the
    eval alone: the golden set and the eval must group alike.

## Testing convention

**CI (EV-3, EV-11):** five required checks on every PR to `main`, one
workflow each in `.github/workflows/` — Ruff, Pyright, Pytest, SonarCloud
Scan, and Release (`release-check.yml`: the PR title's Jira key and the
`CHANGELOG.md` version rules in `CONTRIBUTING.md`, run by
`.github/scripts/release_check.py`, which Ruff lints but Sonar and
basedpyright do not cover). A sixth workflow, `release.yml`, is not a
check: on every push to `main` it tags and publishes every changelog
version from 0.3.0 on that is not done yet.
Ruff runs only the E4/E7/E9/F rules and `ruff format` is not enforced
(the broader set had a 91-finding backlog). The type check is
**basedpyright** (a pyright fork) at `typeCheckingMode: standard` on the
four modules, with `.basedpyright/baseline.json` holding 1 error (EV-15
cleared the other 22 left once `pandas-stubs` was added in EV-13):
`tag_pipeline._plain`'s `value.item()` under `hasattr(value, "item")`, a
guard pyright cannot narrow on, kept as written by the user's choice. Only
new errors fail, and a local run that fixes old ones rewrites the file —
commit it. CI runs
`--baselinemode=lock` (reads, never writes). `pandas-stubs` is pinned
like the checker: a new stubs release rewords errors, which then read as
new, and a plain run refuses to rewrite the baseline — `basedpyright
--writebaseline` does. The installed `pandas` version never changes the
check (the stubs are not partial, so pyright reads no types from pandas
itself; CI ran pandas 3.0.6 against the 3.0.5 stubs clean), so `pandas`
stays unpinned. Bump the stubs when the code starts using pandas API
newer than they describe. An error that depends on what is installed
can't be baselined (it vanishes locally): e.g. the optional `IPython`
import in `show_report` carries a
`# pyright: ignore[reportMissingImports]`.
`pyrightconfig.json` takes `//` comments, not a `"//"` key (that key is a
config error, exit 3, while the output still reads "0 errors"). The
SonarCloud job runs `pytest --cov` first (`.coveragerc`:
`relative_files`, or Sonar sees 0% coverage) and waits for the quality
gate. Widening Ruff, enforcing `ruff format` and type-checking `tests/`
are follow-ups.

`tests/` holds a committed `pytest` suite (run with `pytest` or `python -m
pytest tests/` from the repo root; `pytest.ini` sets `pythonpath = .
.github/scripts` so `import golden_set_pipeline as gsp` and `import
release_check` work without path hacks). This
replaces the earlier practice of writing disposable, uncommitted mocked
smoke-test scripts to a scratch directory outside the repo — those scripts
still exist as the model for how to test a change here (mock
`gsp.call_llm`/`gsp.extract_atomic_claims` via `monkeypatch`, capture logs
via `caplog`, assert on returned DataFrames/lists/log messages), but new
verification should land as a committed test in `tests/`, not a throwaway
script. Coverage is one file per feature, not a broader pass over the
pipeline (extending coverage to older, untested functions is a separate
task); keep new tests in their own file, not a catch-all module:

- decision 1 addendum (same-`chunk_id` dedupe) — `test_rows_for_claim_dedupe.py`
- decision 1 (the shared evidence rule) — `test_evidence_grouping.py`
- decision 5 / ticket 007 (identical-claim-text warning) — `test_duplicate_claim_warning.py`
- decision 11 (dropped/recovered counts) — `test_evidence_counts.py`
- decision 12 (per-memo overrides) — `test_memo_overrides.py`
- decision 13 (batch-count log level) — `test_batch_count_log_level.py`
- decision 14 (claim-split preview) — `test_claim_split_preview.py`
- decision 17 — claims-file grammar `test_claims_file_format.py`; `claim_id`
  derivation `test_claim_id.py`; orchestrators + `__main__` `test_two_stage_pipeline.py`
- decision 18 — `draft` `test_bundle_tagging.py`; `finalize` + eval contract `test_finalize.py`
- `_widen_review_columns` — `test_review_columns.py`
- `_strip_to_json`'s fence regex and `_ATX_CLOSE_RE` (same matches as before
  EV-16, no super-linear backtracking) — `test_regex_backtracking.py`
- `CONTRIBUTING.md`'s release rules (`.github/scripts/release_check.py`) — `test_release_check.py`
- `retrieval_pipeline.py` — `test_retrieval_pipeline.py`
- `eval_pipeline.py` — `test_eval_pipeline.py`

Real live runs against
OpenRouter are slow (multi-minute) and cost real API credits — prefer a
mocked test first, only run live to confirm something a mock can't capture
(actual model behavior, actual provider errors).

Exception: prompt-text changes. Mocking `call_llm` verifies plumbing, not
whether new wording actually changes model behavior — those edits need
real live calls, each case repeated 3x to catch run-to-run
non-determinism (see design decisions 5 and 9, both caught this way).
Record the run-by-run results in `docs/prompt-verification-log.md` (one
section per design decision), and keep only the rule and the current
residual in the design decision itself. For live calls: `nohup python3
... &` (or `Bash` `run_in_background`) plus the `Monitor` tool avoids the
~120s default tool timeout. A full live
pipeline run against the real corpus takes 30+ minutes — longer than
`Monitor`'s default 300s — so pass `persistent: true` or a longer
`timeout_ms` instead of repeatedly re-arming it.

Pandas dtype gotchas when asserting on mocked DataFrames (pandas 3.0.5):
`None` in a list-of-dicts → `DataFrame` build comes back as `NaN` on read
even for string columns, not just numeric — use `pd.isna()`, not `is None`.
An all-`None` column also infers as `float64`; simulating a reviewer
writing a string into it (e.g. `tag`) needs `.astype(object)` first or the
assignment raises `TypeError`.

A bool column round-tripped through parquet reads back as pandas `bool`
dtype whose scalars are `numpy.bool`; `numpy.bool(True) is True` is
`False`. Never branch on `is True` / `is False` / `isinstance` against a
value pulled from a DataFrame — use a vectorised mask (`.isna()`,
`.eq()`, `.isin()`). `tag_pipeline.prepare_draft`'s `found` masks (`.eq(True)`, `.isna()`) are the
worked example.

## Current repo state / cleanup notes

- Checkpoint/review artifacts (`*_checkpoint.parquet`, `review*.xlsx`) are
  gitignored, regenerated demo output, not the deliverable — running the
  `if __name__ == "__main__":` block below (re)creates them.
- `memos.yaml` (repo root) is gitignored — it holds client memo section
  text, which is the deliverable, same treatment as `.env` and
  `sources/`. `memos.yaml.example` is the committed template; copy it to
  `memos.yaml` and edit there. The example is a made-up memo (a
  fictional Acme, 2 sections, PDFs expected in `sources/acme/`).
  `claims/` (repo root) is likewise gitignored — it holds per-memo claims
  files, which carry the same client claim text, either `extract`-written
  from `memos.yaml` or hand-authored (design decision 17).
  `claims.example.md` (repo root, committed) is that format's template —
  see docs/pipeline-overview.md's "How to write a claims file". To build
  the actual golden set: edit `memos.yaml` (add memos / sections / point
  `source_folder` at real PDFs), run `extract`, review the files under
  `claims/`, then run `build` — no code changes needed. `doc_id` is now derived from each PDF's filename
  (`os.path.basename`), not a hand-typed dict key, so the old trap where
  a demo doc_id didn't match its actual file content can't recur.
- `retrieval/` and `retrieval_index/` (repo root) are gitignored — per-memo
  phrase configs and the embedding cache for `retrieval_pipeline.py`, same
  treatment as `claims/`. `retrieval.example.yaml` is the committed
  template. `retrieval_results.parquet` / `.xlsx` are regenerated output,
  already covered by the `*.parquet` / `*.xlsx` ignore globs.
- `review/` (draft sheets the user edits — never overwritten by any
  command) and `reviewed/` (finalize output, regenerable) are gitignored:
  both hold client claim and source text.
- The MEMO-004 acceptance scripts, answers and logs behind design decision
  18, with `HANDOFF.md` (read first when resuming tagging work), live in
  the maintainer's private notes repo under `tag-pipeline-acceptance/`
  (`CONTRIBUTING.md`, section 7); absent on a fresh clone of this repo alone.
- `.superpowers/` is the superpowers skills' local working state, ignored
  per subfolder: `sdd/` has its own `*` `.gitignore`, so a new subfolder
  needs one too. Not committed.
- This CLAUDE.md is the maintained reference for setup/testing/design
  decisions; update it (not a separate handoff doc) as the pipeline
  evolves further. `CHANGELOG.md` is a separate history of user-visible
  changes (entries are never removed; fixing a reference in one is fine)
  — don't let the two merge purposes.
