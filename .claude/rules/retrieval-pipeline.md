---
paths:
  - "retrieval_pipeline.py"
---

# `retrieval_pipeline.py` — sibling, not part of the pipeline

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
- **Measured by `eval_pipeline.py`** (its own rule file). Keep
  `_RESULTS_COLUMNS` and `_CLAIM_QUERY_COLUMNS` stable — the eval imports
  and checks them.
- Tests: `tests/test_retrieval_pipeline.py` (mocked). Design:
  `docs/superpowers/specs/2026-09-07-local-retrieval-design.md`, in the
  maintainer's private notes repo cloned at `docs/superpowers/` (see
  `CONTRIBUTING.md`, section 7) — absent on a fresh clone of this repo alone.
