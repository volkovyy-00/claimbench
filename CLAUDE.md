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
`run_extract`/`run_build` — no argument means `build`; see "Commands"). The
pipeline logic itself stays in this one file; `tests/` is the one
exception, holding the committed `pytest` suite. `docs/pipeline-overview.md`
is a plain-English walkthrough (defers to this file wherever the two
disagree). `docs/prompt-verification-log.md` holds the run-by-run live
results behind prompt edits (design decisions 5, 15, 16) — that log keeps
the `0/3 → 3/3` evidence, `.claude/rules/golden-set-pipeline.md` keeps the
current residual.
`memos.yaml.example` and `claims.example.md` (both committed, repo root)
are the templates for `extract`'s input and `build`'s input respectively
(design decision 17) — copy and edit rather than write either format from
scratch. To build the actual golden set: edit `memos.yaml` (add memos /
sections, point `source_folder` at real PDFs), run `extract`, review the
files under `claims/`, then run `build` — no code changes needed.

### Module-specific rules

The function map, the non-obvious design decisions, and each sibling
module's architecture live in `.claude/rules/*.md`, which Claude Code loads
only when it reads the matching file — this file stays under 200 lines and
holds only what every session needs regardless of which file it touches.

| Rule file | Loads when reading | Holds |
|---|---|---|
| `golden-set-pipeline.md` | `golden_set_pipeline.py` | function map, DataFrame schema, design decisions 1–17 |
| `retrieval-pipeline.md` | `retrieval_pipeline.py` | imports/config/commands for the retrieval prototype |
| `tag-pipeline.md` | `tag_pipeline.py` | imports/config/commands for evidence tagging, design decision 18 |
| `eval-pipeline.md` | `eval_pipeline.py` | imports/config/commands for the eval harness, design decision 19 |
| `testing.md` | `tests/**`, CI config files | CI checks, per-decision test map, pandas/parquet gotchas |

`docs/design-decisions/` (committed) holds the full observed-failure
narrative for the decisions whose rule-file entry is a condensed rule +
current residual + pointer (5, 7, 9, 11–19) — split out so the rule files
stay loadable every session without carrying every design decision's
complete history; the rule-file entry is authoritative on the rule
itself, the linked file is the "why" in full. Decisions 1–4, 6, 8, 10 have
no separate file (short, no residual worth narrating) and stay fully
inline in `golden-set-pipeline.md`. Open work — including the fix for any
residual — is tracked in Jira project `EV`; `CONTRIBUTING.md` says which
file owns which kind of project knowledge.

### Repository and release process

This is a public GitHub repository (`origin` = `volkovyy-00/claimbench`,
started 2026-09-23 from one scrubbed commit). `main` is protected: changes
go through a PR that passes the five required checks (see
`.claude/rules/testing.md`) and follows `CONTRIBUTING.md`'s release
process. The pre-publication history is private (see `CLAUDE.local.md`);
never push it here. `README.md` presents the project publicly as
**ClaimBench** (MIT, `LICENSE`): install, the six-step usage walkthrough, a
command/config/layout reference, and nothing past that. It must stay
neutral (no client or company references) — design and testing detail
lives in this file's `.claude/rules/*.md` companions, not README.

## Environment

- `.env` (real credentials, never read/print its contents) holds
  `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` — deliberately switched on
  2026-09-22 from a reasoning model (`openai/gpt-oss-120b`) to OpenRouter +
  `openai/gpt-4.1-mini`. That's why `call_llm` sets `max_tokens` explicitly
  (design decision 4), which stays the right guard if a reasoning model is
  ever set here again. This date is the anchor for design decisions 5, 9,
  15 and 16's residuals: each was measured under the *older* model, so
  re-verify a dated residual against it before trusting a fresh run.
  `.env.example` documents the shape.
- `.venv/` has all deps installed: `source .venv/bin/activate`.
- Python 3.10+ (the code uses `X | Y` unions and `list[dict]` builtin
  generics); developed on 3.13.
- `requirements.txt`: pandas, numpy, requests, openpyxl, pyarrow, python-dotenv,
  jupytext, rank_bm25, pypdf, pdfplumber, PyYAML, pytest.
  `requirements-dev.txt` adds the CI tools, pinned exactly: ruff,
  basedpyright, pandas-stubs, pytest-cov.
- `sources/` holds the source PDFs, one folder per memo — the convention
  `load_memo_sections_from_config` expects, so a new memo's PDFs get their
  own subfolder too. Gitignored, not committed, same as the
  `*.parquet`/`*.xlsx` artifacts. `sources/sample.pdf` (any small real
  filing; a symlink is fine) lets `tests/test_finalize.py`'s real-PDF test
  run instead of skipping.
- `.superpowers/` is the superpowers skills' local working state, ignored
  per subfolder: `sdd/` has its own `*` `.gitignore`, so a new subfolder
  needs one too. Not committed.
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
python eval_pipeline.py report <run_id|latest> [baseline_run_id] [--method=dense] [--k=5]   # -> eval_runs/<run_id>/report_dense_k5.html + _claims.html, report_rereview.html
python golden_set_pipeline.py            # same as build
```

Live prompt-verification run (real API, minutes each, repeat 3x — see
`.claude/rules/testing.md` for why and the non-determinism it guards
against): launch with `Bash` `run_in_background` or `nohup python3 ... &`,
then watch with `Monitor` (`persistent: true`, since a full corpus run
exceeds its 300s default).

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
**default** path (no fixed candidate cap — see
`.claude/rules/golden-set-pipeline.md`). The older fixed-count
`bm25_shortlist`/`propose_evidence_from_chunks` still exist, unused by
`build_golden_set_draft`, kept available for a cheaper/faster manual pass.
