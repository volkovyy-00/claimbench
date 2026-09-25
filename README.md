# ClaimBench

> A benchmark of claims, each traced to its evidence.

Build a human-verified **golden set** of claim → evidence pairs from written
documents and the source PDFs they were based on. The pipeline splits each
document section into single-fact claims, finds the source passages that
support each claim (or records that none do), and tags how each claim is
supported, with a person reviewing the result at every stage. The output is
a ground-truth dataset for measuring retrieval recall and hallucination in
document-grounded AI systems.

## Features

- **Claim extraction:** an LLM splits section text into atomic, independently
  checkable claims, written to a plain Markdown file you can review or write
  by hand.
- **Evidence matching:** BM25 shortlisting plus LLM verification finds every
  supporting passage, not just one, and flags genuine ambiguity.
- **Evidence tagging:** an LLM drafts, per claim, whether it is stated
  directly, needs combining several passages, or is not supported; you check
  every answer in a spreadsheet before anything is final.
- **Strict by design:** malformed input, edited review sheets, and changed
  source files are refused with a clear message rather than guessed around.

## Requirements

- Python 3.10+
- An API key for an OpenAI-compatible chat completions endpoint (for example
  [OpenRouter](https://openrouter.ai))
- Source PDFs with a text layer (scanned PDFs are not OCR'd)

## Installation

```bash
git clone https://github.com/volkovyy-00/evals.git
cd evals
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then set LLM_API_KEY
```

## Usage

Run these steps in order for each document. Every command reads the output of
the step before it.

**1. Describe your documents.** Put each document's PDFs in their own folder
under `sources/`, then copy the template and add the section text:

```bash
cp memos.yaml.example memos.yaml
```

**2. Extract claims**, then read and correct the files it writes:

```bash
python golden_set_pipeline.py extract    # memos.yaml -> claims/<id>.md
```

You can also write a claims file by hand and skip this step; see
[`claims.example.md`](claims.example.md).

**3. Find the evidence** for every claim:

```bash
python golden_set_pipeline.py build      # claims/*.md -> golden_set_checkpoint.parquet
```

**4. Draft the tags.** First add `filing_entity: <company name>` to the
claims file's top block, so the model knows who "we" and "our" refer to:

```bash
python tag_pipeline.py draft <id>        # -> review/<id>.xlsx
```

**5. Review the sheet** in Excel or Apple Numbers: correct any verdict or
marked passage, paste a quote (25+ characters) for evidence that was missed,
and set **CHECKED** to `yes` on every claim. Save it under the same name.

**6. Finalize:**

```bash
python tag_pipeline.py finalize <id>     # -> reviewed/<id>.xlsx
```

`reviewed/<id>.xlsx` is the finished golden set for that document. If
`finalize` lists problems, fix those rows in the sheet and run it again.

> [!IMPORTANT]
> Editing a claim or the source PDFs after step 4 invalidates the review
> sheet. Delete `review/<id>.xlsx`, then repeat steps 3–6.

### Command reference

| Command | Reads | Writes |
|---|---|---|
| `python golden_set_pipeline.py extract` | `memos.yaml` | `claims/<id>.md` (never overwrites) |
| `python golden_set_pipeline.py build` | `claims/*.md`, PDFs | `golden_set_checkpoint.parquet`, `review.xlsx` |
| `python tag_pipeline.py draft [id ...]` | checkpoint, claims, PDFs | `review/<id>.xlsx` (never overwrites) |
| `python tag_pipeline.py finalize <id>` | checked sheet, claims, PDFs | `reviewed/<id>.xlsx` (no LLM calls) |
| `python retrieval_pipeline.py embed [id ...]` | PDFs | `retrieval_index/<id>.parquet` |
| `python retrieval_pipeline.py retrieve` | `retrieval/*.yaml`, index | `retrieval_results.parquet`, `.xlsx` |
| `python retrieval_pipeline.py recheck` | claims files, index | `claim_queries.parquet` |
| `python eval_pipeline.py score [label]` | reviewed sheets, claims, retrieval results | `eval_runs/<run_id>/` (no API calls) |
| `python eval_pipeline.py report <run_id\|latest> [baseline] [--method=…] [--k=…]` | a run | `eval_runs/<run_id>/report_*.html` (summary, claims and re-review pages, linked) |

`draft` with no ids drafts every document in the checkpoint. The three
`retrieval_pipeline.py` commands are an optional retrieval experiment to
measure against the golden set: `retrieve` searches three ways (meaning,
keyword, and both combined) and `recheck` searches with each claim's own
text; copy [`retrieval.example.yaml`](retrieval.example.yaml) to start.

## Configuration

Settings live in `.env` (see [`.env.example`](.env.example)):

| Variable | Used by | Purpose |
|---|---|---|
| `LLM_BASE_URL`, `LLM_API_KEY` | all LLM steps | Chat completions endpoint and key |
| `LLM_MODEL` | `extract`, `build` | Model for claims and evidence |
| `EMBED_BASE_URL`, `EMBED_API_KEY`, `EMBED_MODEL` | `retrieval_pipeline.py` | Embeddings endpoint |

The tagging model and prompt are pinned in `tag_pipeline.py`, so `draft`
ignores `LLM_MODEL`. Per-document tuning (how many passages are checked, and
in what batch size) is optional and set in `memos.yaml` or the claims file.

Your document text is sent only to the endpoints you configure above, and
never committed: `memos.yaml`, `claims/`, `sources/`, `review/` and
`reviewed/` are gitignored.

## Project layout

| Path | Contents |
|---|---|
| `golden_set_pipeline.py` | Claim extraction and evidence matching (runs as a script or a Jupytext notebook) |
| `tag_pipeline.py` | Evidence tagging: `draft` and `finalize` |
| `retrieval_pipeline.py` | Optional retrieval experiment (meaning, keyword and combined search) |
| `eval_pipeline.py` | Scores retrieval against the golden set; HTML report pages |
| `tests/` | `pytest` suite (mocked, no API calls) |
| `.github/workflows/` | CI: lint, type check, tests, SonarCloud |
| `docs/` | Walkthrough and design notes |
| `*.example.*` | Templates for your own config and claims files |

## Testing

```bash
pytest
```

The suite mocks every LLM call, so it runs in seconds without an API key.

Every pull request must also pass a lint check, a type check and a
SonarCloud analysis. The first three run locally exactly as in CI:

```bash
pip install -r requirements-dev.txt
ruff check .            # lint (ruff.toml)
basedpyright            # type check (pyrightconfig.json)
pytest                  # tests
```

The type check fails only on new errors: errors that predate it are listed
in `.basedpyright/baseline.json`, which shrinks as they are fixed. Commit the
updated file when a local run rewrites it.

## Documentation

- [Pipeline overview](docs/pipeline-overview.md): a plain-English walkthrough
  of every step, including how to write a claims file
- [Design decisions](docs/design-decisions/): why the trickier parts work the
  way they do
- [Changelog](CHANGELOG.md)

## License

[MIT](LICENSE) © volkovyy-00
