---
paths:
  - "tests/**"
  - ".github/workflows/**"
  - ".github/scripts/**"
  - "pytest.ini"
  - "ruff.toml"
  - "pyrightconfig.json"
  - ".coveragerc"
  - ".basedpyright/**"
---

# Testing convention

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
