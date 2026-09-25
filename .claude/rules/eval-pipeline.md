---
paths:
  - "eval_pipeline.py"
---

# `eval_pipeline.py` — sibling, measures only

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
  (gitignored).

## Non-obvious design decisions (decision 19)

19. **The eval scores claims against human-tagged evidence, and refuses
    rather than scoring around a problem** (`eval_pipeline.py`). The relevant
    set is `found` **and** a tag of `extractive`/`synthesized` (evidence
    schema: `.claude/rules/golden-set-pipeline.md`'s DataFrame schema) —
    `found` is the pipeline's guess, the tag the human verdict. Unverifiable
    claims leave the coverage denominator and are reported as a re-review
    worklist (`claim_queries.parquet`) instead of "absent from the sources":
    the golden set was built with lexical BM25 (decision 7).

    A hit is the same `chunk_id` **or** decision 1's rule (`_same_evidence`
    — full rule in `.claude/rules/golden-set-pipeline.md`), checked one
    direction only against an adjacent chunk's shared-overlap text. A
    human-added row's joined quotes are tested and grouped one by one. Hits
    are computed once at depth 20; k is read off per search phrase, and a
    run is refused unless the results were retrieved with the phrases on
    disk.

    **Residuals:** duplicated source documents cap macro recall at roughly
    50% for a claim cited from only one copy; a handful of golden quotes
    aren't verbatim in their own chunk after normalization, so only that
    chunk (never a neighbour) can hit them; and two golden rows whose quotes
    both span a chunk-overlap boundary stay two separate evidence groups
    under decision 1. None of these are fixed — the report explains each.

    Full narrative — the ground-truth and hit-matching rules in full, the
    measured false-hit rate, and the exact residual counts and dates:
    `docs/design-decisions/19-eval-harness-scoring.md`.
