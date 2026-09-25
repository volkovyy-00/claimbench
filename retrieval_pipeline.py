# %% [markdown]
# # Local Retrieval Prototype
#
# A standalone retrieval experiment (dense, keyword and combined search)
# beside `golden_set_pipeline.py`. It embeds the SAME PDF chunks the golden
# set uses (chunking is imported unchanged from `golden_set_pipeline`, so
# `chunk_id`s line up), then retrieves chunks by hand-authored per-section
# phrases.
#
# Three commands:
#
# - `python retrieval_pipeline.py embed [memo_id ...]` — chunk each memo's
#   PDFs, embed every chunk, cache to `retrieval_index/<memo_id>.parquet`.
#   Re-run only re-embeds chunks whose text changed.
# - `python retrieval_pipeline.py retrieve` — embed each section's phrases and
#   search three ways per phrase (dense cosine, keyword BM25, and both fused
#   by Reciprocal Rank Fusion); write `retrieval_results.parquet` + `.xlsx`
#   (one row per (memo_id, section, phrase, method, chunk_id)).
# - `python retrieval_pipeline.py recheck` — embed every claim's own text from
#   claims/<memo_id>.md as a query; write the top chunks per claim to
#   `claim_queries.parquet`, for the eval's re-review list.
#
# This is a SIBLING experiment, not part of the two-stage golden-set
# pipeline. It reads its own `retrieval/<memo_id>.yaml` config and its own
# `EMBED_*` env vars. It does NOT compute recall/precision/MRR —
# eval_pipeline.py scores its results. See
# docs/superpowers/specs/2026-09-07-local-retrieval-design.md.

# %%
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import yaml
from dotenv import load_dotenv
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from rank_bm25 import BM25Okapi

from golden_set_pipeline import (
    build_chunk_index,
    chunk_document,  # noqa: F401  (re-exported for symmetry / direct notebook use)
    _load_source_documents,
    _scan_claims_file_sections,
    _section_name_issue,
    _valid_memo_id,
    parse_claims_file,
    _claims_with_occurrence,
    _derive_claim_id,
    _tokenize,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("retrieval")


# %% [markdown]
# ## 1. Tuning constants
#
# Not CLI flags, not YAML keys — tune by editing here, matching
# golden_set_pipeline's convention for single-site documented literals
# (design decisions 7/12/13).

# %%
_EMBED_BATCH_SIZE = 64   # texts per /embeddings request
_TOP_K = 5               # modeled RAG depth per phrase (dedupe_by_section, __main__ demo)
_RETRIEVE_DEPTH = 20     # rows written per phrase to the artifact — must be >= _TOP_K,
#                          so a later eval can see where a golden chunk landed if not in the top 5
_RETRIEVAL_METHODS = ("dense", "keyword", "both")  # every retrieve run writes all three
_RRF_K = 60  # Reciprocal Rank Fusion constant for "both" — the conventional value


def _validate_depth_config(retrieve_depth: int = _RETRIEVE_DEPTH, top_k: int = _TOP_K) -> None:
    """Raise if the artifact records fewer rows per phrase than the modeled system returns."""
    if retrieve_depth < top_k:
        raise ValueError(
            f"_RETRIEVE_DEPTH ({retrieve_depth}) must be >= _TOP_K ({top_k}) — "
            "the artifact cannot be shallower than the modeled retrieval depth"
        )


_validate_depth_config()


# %% [markdown]
# ## 2. Embedding client — isolated behind one function
#
# `EmbeddingClient` holds config (mirrors `golden_set_pipeline.LLMClient`).
# `call_embeddings` is the ONLY function that knows the HTTP shape of the
# embeddings API. Swap providers (OpenRouter -> on-prem) by changing the
# EMBED_* env vars; only edit `call_embeddings` if the wire shape differs.

# %%
_TRUE_TOKENS = {"true", "1", "yes", "on"}


def _parse_bool_env(raw: str | None) -> bool:
    """
    Defensive bool parse for EMBED_ASYMMETRIC: true/1/yes/on (any case) -> True;
    everything else — including None, "", and "false" — -> False. A bare
    `EMBED_ASYMMETRIC=` in .env must read as off.
    """
    return raw is not None and raw.strip().lower() in _TRUE_TOKENS


@dataclass
class EmbeddingClient:
    """Connection config for the embeddings endpoint. Build with EmbeddingClient.from_env()."""

    base_url: str
    api_key: str
    model: str
    asymmetric: bool  # EMBED_ASYMMETRIC — send input_type ("query"/"passage") on each request

    @classmethod
    def from_env(cls) -> "EmbeddingClient":
        """
        Reads EMBED_BASE_URL, EMBED_API_KEY, EMBED_MODEL (all required) and
        EMBED_ASYMMETRIC (optional, default False). Raises EnvironmentError
        listing any missing required var — same shape as LLMClient.from_env.
        """
        base_url = os.environ.get("EMBED_BASE_URL")
        api_key = os.environ.get("EMBED_API_KEY")
        model = os.environ.get("EMBED_MODEL")
        if not base_url or not api_key or not model:
            missing = [
                name
                for name, val in [
                    ("EMBED_BASE_URL", base_url),
                    ("EMBED_API_KEY", api_key),
                    ("EMBED_MODEL", model),
                ]
                if not val
            ]
            raise EnvironmentError(f"Missing required environment variable(s): {', '.join(missing)}")
        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            asymmetric=_parse_bool_env(os.environ.get("EMBED_ASYMMETRIC")),
        )


# %% [markdown]
# ## 3. HTTP contact point — embeddings
#
# `call_embeddings` is the sole function that knows the wire shape of the
# /embeddings endpoint (mirrors `golden_set_pipeline.call_llm`). Error handling
# is deliberately NOT a mirror: the `/embeddings` endpoint signals failure via
# HTTP status code and a `{"error": {...}}` body (even on 200), unlike chat
# completions. Hence retrying semantics and error extraction differ (see
# docstring).

# %%
# HTTP status codes worth a single retry (transient / server-side). A 4xx
# other than 429 is the caller's mistake — a bad model name, a bad key, an
# oversized batch — and retrying just wastes time and (for 429) makes it
# worse. NOT a mirror of call_llm's finish_reason="error": that is a 200-body
# streaming-chat case; the /embeddings endpoint signals failure with the HTTP
# status and puts {"error": {...}} in the NON-200 body.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_RETRY_SLEEP_SECONDS = 2.0


def _provider_error_detail(response) -> str | None:
    """Best-effort: pull the {"error": {...}} object out of a response body, as a string."""
    try:
        err = response.json().get("error")
    except Exception:
        return None
    if not err:
        return None
    return (
        f"{err.get('message')!r} (code={err.get('code')!r}, "
        f"metadata={err.get('metadata')!r})"
    )


def call_embeddings(
    texts: list[str], client: EmbeddingClient, *, input_type: str | None = None
) -> list[list[float]]:
    """
    Embeds `texts` in ONE request against the configured /embeddings endpoint
    and returns the vectors in the same order as `texts`.

    Single point of HTTP contact with the embedding provider (the analogue of
    golden_set_pipeline.call_llm). OpenAI-compatible dialect: POST
    {base_url}/embeddings, Bearer auth, body {"model", "input",
    "encoding_format": "float"}; success body {"data": [{"embedding": [...],
    "index": n}, ...]}.

    input_type is added ONLY when non-None — callers pass it only when
    client.asymmetric is set, so a default OpenRouter run never sends a key
    the provider might 400 on. "passage" for documents, "query" for phrases.

    Error handling: an HTTP 429/5xx is retried ONCE after a short sleep; any
    other 4xx is not retried (a bad model/key/batch — retrying wastes spend).
    A response body's {"error": {message, code, metadata}} object is pulled
    into the raised message on either path. A 200 body that is unparseable or
    lacks a usable "data" array is also retried once. After the retry (or
    immediately, for a non-retryable 4xx), raises RuntimeError; the caller
    adds memo/batch context.

    All batching is the caller's responsibility (see _embed_texts).
    """
    url = f"{client.base_url.rstrip('/')}/embeddings"
    headers = {
        "Authorization": f"Bearer {client.api_key}",
        "Content-Type": "application/json",
    }
    payload: dict = {"model": client.model, "input": list(texts), "encoding_format": "float"}
    if input_type is not None:
        payload["input_type"] = input_type

    last_error: str = "unknown error"
    for attempt in (1, 2):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
        except requests.RequestException as exc:  # connection error, timeout, DNS
            last_error = f"transport error: {exc}"
            logger.warning("[call_embeddings] attempt %d/2: %s", attempt, last_error)
            if attempt == 1:
                time.sleep(_RETRY_SLEEP_SECONDS)
            continue

        if response.status_code >= 400:
            detail = _provider_error_detail(response)
            last_error = (
                f"HTTP {response.status_code}"
                + (f" — provider error: {detail}" if detail else "")
            )
            if response.status_code in _RETRYABLE_STATUS and attempt == 1:
                logger.warning("[call_embeddings] attempt 1/2: %s — retrying", last_error)
                time.sleep(_RETRY_SLEEP_SECONDS)
                continue
            raise RuntimeError(f"embedding request failed ({last_error})")

        try:
            data = response.json()
            rows = data["data"]
            if not rows:
                raise ValueError("empty 'data' array")
            rows_sorted = sorted(rows, key=lambda r: r.get("index", 0))
            vectors = [list(r["embedding"]) for r in rows_sorted]
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            # OpenRouter has been seen returning a 200 with {"error": {...}} and
            # no "data" — surface that detail rather than a bare "'data'".
            detail = _provider_error_detail(response)
            last_error = (
                f"unparseable 200 response: {exc}"
                + (f" — provider error: {detail}" if detail else "")
            )
            logger.warning("[call_embeddings] attempt %d/2: %s", attempt, last_error)
            if attempt == 1:
                time.sleep(_RETRY_SLEEP_SECONDS)
            continue

        if len(vectors) != len(texts):
            raise RuntimeError(
                f"embedding response returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        if any(len(v) == 0 for v in vectors):
            raise RuntimeError("embedding response contained an empty vector")
        return vectors

    raise RuntimeError(f"embedding request failed after 2 attempts ({last_error})")


def _embed_texts(
    texts: list[str],
    client: EmbeddingClient,
    *,
    input_type: str | None,
    batch_size: int = _EMBED_BATCH_SIZE,
    label: str = "",
) -> list[list[float]]:
    """
    Splits `texts` into `batch_size` groups and calls call_embeddings on each,
    concatenating the results in input order. A failing batch's RuntimeError
    is re-raised with `label` and the batch's position prepended, so a failure
    names which memo / which slice never got embedded.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    out: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            out.extend(call_embeddings(batch, client, input_type=input_type))
        except Exception as exc:
            raise RuntimeError(
                f"{label}: embedding batch [{start}:{start + len(batch)}] failed: {exc}"
            ) from exc
    return out


# %% [markdown]
# ## 4. Phrase config — retrieval/<memo_id>.yaml
#
# Pure YAML validation, no PDF I/O (the parallel to
# golden_set_pipeline._read_memo_config). Standalone by design — section
# names are NOT cross-checked against memos.yaml or claims/ here; that's a
# soft, best-effort WARNING in `retrieve` only.

# %%
_PHRASE_CONFIG_FIELDS = {"memo_id", "source_folder", "sections"}
_MIN_PHRASES = 1
_MAX_PHRASES = 20  # working range is 5-10; this is a runaway-list guard


def _read_phrase_config(path: str) -> dict:
    """
    Reads and validates one retrieval/<memo_id>.yaml. Returns
    {"memo_id": str, "source_folder": str,
     "sections": dict[str, list[str]]} — sections insertion-ordered as
    written, phrases .strip()ed.

    Raises ValueError naming the file (and the section or key at fault) on:
    text that is not valid YAML, missing/blank memo_id, a memo_id not
    matching [A-Za-z0-9._-]+ or not equal to the filename stem,
    missing/empty/non-mapping sections, a section name _section_name_issue
    rejects, a section value that isn't a list of 1-20 non-blank strings, or
    an unrecognized top-level key. Never a silent skip.
    """
    name = os.path.basename(path)
    with open(path, encoding="utf-8") as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as e:   # not a ValueError: callers catching ValueError would miss it
            raise ValueError(f"{name}: not valid YAML — {e}") from e
    if not isinstance(cfg, dict):
        raise ValueError(f"{name}: top level must be a mapping, got {type(cfg).__name__}")

    unrecognized = set(cfg) - _PHRASE_CONFIG_FIELDS
    if unrecognized:
        raise ValueError(
            f"{name}: unrecognized top-level key(s) {sorted(unrecognized, key=repr)} — "
            f"expected {sorted(_PHRASE_CONFIG_FIELDS)}"
        )

    memo_id = cfg.get("memo_id")
    stem = os.path.splitext(name)[0]
    if not isinstance(memo_id, str) or not memo_id.strip():
        raise ValueError(f"{name}: 'memo_id' must be a non-blank string (got {memo_id!r})")
    if not _valid_memo_id(memo_id):
        raise ValueError(f"{name}: 'memo_id' {memo_id!r} must match [A-Za-z0-9._-]+")
    if memo_id != stem:
        raise ValueError(f"{name}: 'memo_id' {memo_id!r} must equal the filename stem {stem!r}")

    source_folder = cfg.get("source_folder")
    if not isinstance(source_folder, str) or not source_folder.strip():
        raise ValueError(f"{name}: 'source_folder' must be a non-blank string (got {source_folder!r})")

    sections = cfg.get("sections")
    if not sections:
        raise ValueError(f"{name}: missing required field 'sections' (or it's empty)")
    if not isinstance(sections, dict):
        raise ValueError(f"{name}: 'sections' must be a mapping of section name to a list of phrases")

    clean: dict[str, list[str]] = {}
    for section_name, phrases in sections.items():
        issue = _section_name_issue(section_name)
        if issue:
            raise ValueError(f"{name} / section {section_name!r}: invalid section name — {issue}")
        if not isinstance(phrases, list):
            raise ValueError(
                f"{name} / section {section_name!r}: value must be a list of phrases "
                f"(got {type(phrases).__name__})"
            )
        if not (_MIN_PHRASES <= len(phrases) <= _MAX_PHRASES):
            raise ValueError(
                f"{name} / section {section_name!r}: expected {_MIN_PHRASES} to {_MAX_PHRASES} "
                f"phrases (working range 5-10), got {len(phrases)}"
            )
        cleaned: list[str] = []
        for phrase in phrases:
            if not isinstance(phrase, str) or not phrase.strip():
                raise ValueError(
                    f"{name} / section {section_name!r}: every phrase must be a non-blank "
                    f"string (got {phrase!r})"
                )
            cleaned.append(phrase.strip())
        clean[section_name] = cleaned

    return {"memo_id": memo_id, "source_folder": source_folder, "sections": clean}


# %% [markdown]
# ## 5. embed — build the per-memo embedding index
#
# Chunk each memo's PDFs with the SAME chunk_size/overlap the golden set
# uses (imported build_chunk_index), embed every chunk, cache to
# retrieval_index/<memo_id>.parquet. A re-run only re-embeds chunks whose
# text changed and drops chunk_ids the PDFs no longer produce. Written
# atomically (.tmp then os.replace), like write_claims_file.

# %%
_INDEX_COLUMNS = ["chunk_id", "doc_id", "chunk_text", "start_offset", "embedding", "model"]


def _build_chunk_records(source_folder: str, memo_id: str) -> list[dict]:
    """
    Loads `source_folder`'s PDFs (via the imported _load_source_documents:
    case-insensitive *.pdf, deterministic sort, load_pdf_text +
    warn_if_text_suspiciously_short per file) and chunks them with the
    golden-set defaults (chunk_size=1000, overlap=200 — pinned so chunk_ids
    line up with golden_set_checkpoint.parquet). Returns the flat chunk list:
    {chunk_id, doc_id, chunk_text, start_offset}.
    """
    documents = _load_source_documents(source_folder, memo_id)
    # NOTE: chunk_size/overlap intentionally left at build_chunk_index's
    # defaults. Changing them here would desync chunk_ids from the golden set,
    # and eval_pipeline.py's index parity check would refuse to score. Not a
    # knob on purpose.
    return build_chunk_index(documents)


def _load_index(index_path: str) -> dict[str, tuple[str, list[float]]]:
    """Reads an existing index parquet into {chunk_id: (chunk_text, embedding)}; {} if absent."""
    if not os.path.exists(index_path):
        return {}
    df = pd.read_parquet(index_path, columns=["chunk_id", "chunk_text", "embedding"])
    return {
        chunk_id: (chunk_text, list(embedding))
        for chunk_id, chunk_text, embedding in zip(
            df["chunk_id"], df["chunk_text"], df["embedding"]
        )
    }


def _load_index_model(index_path: str) -> str | None:
    """
    The embedding model name stored in an existing index parquet's constant
    `model` column, or None if the file is absent, empty, has no `model`
    column at all (an index written before model provenance was tracked), or
    the value is blank — all three "unknown" cases are treated alike by the
    caller (a WARNING, never a hard failure, since there's no way to know
    what actually wrote an old-format index).
    """
    if not os.path.exists(index_path) or "model" not in pq.read_schema(index_path).names:
        return None
    df = pd.read_parquet(index_path, columns=["model"])   # not the vectors: the name is all that's needed
    if df.empty:
        return None
    val = df["model"].iloc[0]
    if pd.isna(val) or not str(val).strip():
        return None
    return str(val)


def _check_index_model(index_path: str, model: str) -> None:
    """
    Refuses an index embedded with a model other than `model` (EMBED_MODEL);
    embed, retrieve and recheck all check this before any API call. Swapping
    EMBED_MODEL for a different model that happens to embed at the SAME
    dimension is the likely case (dims cluster hard on 768/1024/1536), not
    the exotic one — and no dimension check sees it: embed would silently mix
    two vector spaces in one index, retrieve compare queries against another
    model's vectors. An index with no recorded model predates this check;
    there's no way to know what wrote it, so that's a WARNING, not a raise.
    """
    stored_model = _load_index_model(index_path)
    if stored_model is None:
        logger.warning(
            "%s: has no recorded embedding model (written before model "
            "provenance was tracked) — cannot verify it matches the current "
            "EMBED_MODEL (%r). If in doubt, delete %s and re-run embed.",
            index_path, model, index_path,
        )
    elif stored_model != model:
        raise ValueError(
            f"{index_path}: was embedded with model {stored_model!r} but EMBED_MODEL "
            f"is now {model!r} — the index and EMBED_MODEL disagree. Delete "
            f"{index_path} and re-run embed, or restore EMBED_MODEL."
        )


def _embedding_dim(vectors: list[list[float]]) -> int | None:
    """The single shared length of `vectors`, or None if empty. Raises ValueError if they disagree."""
    lengths = {len(v) for v in vectors}
    if not lengths:
        return None
    if len(lengths) > 1:
        raise ValueError(f"embedding vectors have inconsistent dimensions: {sorted(lengths)}")
    return lengths.pop()


def _discover_phrase_configs(retrieval_dir: str, memo_ids: list[str] | None) -> list[str]:
    """
    Sorted list of retrieval/*.yaml paths, optionally filtered to memo_ids
    (error if a name is absent). Any OTHER file directly in `retrieval_dir`
    (e.g. a `.yml` instead of `.yaml`) is excluded but logs a WARNING naming
    it — `retrieve` is whole-corpus with no memo filter specifically so a
    skipped memo can't hide inside an otherwise-complete-looking results
    file; a silently-excluded config file is the same failure mode one step
    earlier. `.yml` is deliberately not also accepted: one canonical
    extension keeps the filename-stem-equals-memo_id rule unambiguous.
    """
    if not os.path.isdir(retrieval_dir):
        raise ValueError(f"{retrieval_dir}: not a directory — create it and add <memo_id>.yaml files")
    paths = []
    for n in sorted(os.listdir(retrieval_dir)):
        full = os.path.join(retrieval_dir, n)
        if not os.path.isfile(full):
            continue
        if n.lower().endswith(".yaml"):
            paths.append(full)
        else:
            logger.warning(
                "%s: skipping %r — not a .yaml phrase config (only *.yaml files are read; "
                "a memo whose config uses another extension is silently excluded from a "
                "whole-corpus retrieve)",
                retrieval_dir, n,
            )
    if not paths:
        raise ValueError(f"{retrieval_dir}: no .yaml phrase configs found")
    if memo_ids is None:
        return paths
    wanted = set(memo_ids)
    by_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in paths}
    missing = wanted - set(by_stem)
    if missing:
        raise ValueError(f"{retrieval_dir}: no phrase config for memo_id(s): {sorted(missing)}")
    return [by_stem[m] for m in sorted(wanted)]


def run_embed(
    retrieval_dir: str = "retrieval",
    index_dir: str = "retrieval_index",
    memo_ids: list[str] | None = None,
    *,
    client: EmbeddingClient,
) -> None:
    """
    Builds / refreshes retrieval_index/<memo_id>.parquet for every
    retrieval/<memo_id>.yaml (or just the named memo_ids).

    Pre-pass FIRST: every phrase config is parsed and validated and every
    source_folder is checked to be a non-empty PDF folder — before ANY chunk
    is read or embedded. A malformed config or bad folder for the third memo
    therefore costs nothing on the first two (matches _read_memo_config /
    load_memo_sections_from_claims and design decision 12's validate-before-
    spend rule). Then, per memo: chunk the PDFs, reuse any cached vector
    whose chunk_id AND chunk_text still match, embed the rest (as "passage"
    when client.asymmetric), drop cached chunk_ids the PDFs no longer
    produce, write the parquet atomically.

    Raises ValueError on a malformed config, a missing/empty source_folder, a
    memo whose PDFs yield no chunks, inconsistent embedding dimensions
    (within a response, or new-vs-cached), or a stored `model` that disagrees
    with the current client.model (the message in every case says to delete
    the parquet and re-run). An index with no recorded `model` at all (older
    format) logs a WARNING instead of raising — there's no way to know what
    wrote it. Never a silent skip.
    """
    paths = _discover_phrase_configs(retrieval_dir, memo_ids)

    # --- pre-pass: validate everything cheap, before any PDF read or API call ---
    configs: list[dict] = []
    for path in paths:
        cfg = _read_phrase_config(path)
        if not os.path.isdir(cfg["source_folder"]):
            raise ValueError(
                f"{os.path.basename(path)}: source_folder {cfg['source_folder']!r} is not a directory"
            )
        configs.append(cfg)

    # --- process: chunk + embed + write, one memo at a time ---
    os.makedirs(index_dir, exist_ok=True)
    for cfg in configs:
        memo_id = cfg["memo_id"]
        index_path = os.path.join(index_dir, f"{memo_id}.parquet")

        records = _build_chunk_records(cfg["source_folder"], memo_id)
        if not records:
            raise ValueError(
                f"{memo_id}: source PDFs produced no chunks (all empty or scanned — "
                "this pipeline does no OCR)"
            )
        cached = _load_index(index_path)
        try:
            cached_dim = _embedding_dim([v for _, v in cached.values()]) if cached else None
        except ValueError as exc:
            raise ValueError(
                f"{index_path}: stored embeddings have inconsistent dimensions ({exc}). "
                f"Delete {index_path} and re-run embed."
            ) from exc

        if os.path.exists(index_path):
            _check_index_model(index_path, client.model)

        to_embed = [r for r in records if cached.get(r["chunk_id"], (None,))[0] != r["chunk_text"]]
        reused = len(records) - len(to_embed)

        new_vectors: dict[str, list[float]] = {}
        if to_embed:
            vectors = _embed_texts(
                [r["chunk_text"] for r in to_embed],
                client,
                input_type="passage" if client.asymmetric else None,
                label=memo_id,
            )
            new_dim = _embedding_dim(vectors)
            if cached_dim is not None and new_dim is not None and new_dim != cached_dim:
                raise ValueError(
                    f"{index_path}: new embeddings are {new_dim}-dim but the cache is "
                    f"{cached_dim}-dim (EMBED_MODEL changed?). Delete {index_path} and re-run embed."
                )
            new_vectors = {r["chunk_id"]: v for r, v in zip(to_embed, vectors)}

        rows = []
        for r in records:
            cid = r["chunk_id"]
            emb = new_vectors[cid] if cid in new_vectors else cached[cid][1]
            rows.append({**r, "embedding": list(emb), "model": client.model})
        # dimension sanity across the assembled set (catches a corrupt cache too)
        _embedding_dim([row["embedding"] for row in rows])

        df = pd.DataFrame(rows)[_INDEX_COLUMNS]
        tmp_path = index_path + ".tmp"
        try:
            df.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, index_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        dropped = len(set(cached) - {r["chunk_id"] for r in records})
        batch_count = -(-len(to_embed) // _EMBED_BATCH_SIZE) if to_embed else 0  # ceil div
        logger.info(
            "embed %s: %d chunks (%d embedded, %d reused, %d stale dropped) in %d "
            "batch(es), dim=%s",
            memo_id, len(records), len(to_embed), reused, dropped, batch_count,
            _embedding_dim([row["embedding"] for row in rows]),
        )


# %% [markdown]
# ## 6. retrieve — search each section's phrases three ways (dense, keyword, both)
#
# Whole-corpus only (no memo filter): a per-memo retrieve would leave
# retrieval_results.parquet holding one memo's rows while looking complete,
# and the eval consumes the whole file. Index is chunk-embeddings only; the
# phrases are the queries; each phrase's top-_RETRIEVE_DEPTH chunks are
# written (a rank <= _TOP_K filter reproduces the modeled RAG system).

# %%
_RESULTS_COLUMNS = [
    "memo_id", "section", "phrase", "phrase_index", "method",
    "doc_id", "chunk_id", "chunk_text", "rank", "score",
]

_CLAIM_QUERY_COLUMNS = [
    "memo_id", "section", "claim_id", "claim_text",
    "doc_id", "chunk_id", "chunk_text", "rank", "score",
]


def _claims_section_drift_warnings(
    memo_id: str, phrase_section_names: list[str], claims_dir: str = "claims"
) -> None:
    """
    Opportunistic sanity check, best-effort, NEVER raises. When
    claims/<memo_id>.md exists, scan its '## ' headings and WARN on either
    kind of drift that would silently score zero recall in a later eval:
      - a phrase-file section not present in the claims file (a casing /
        wording mismatch), and
      - a claims-file section with no phrases in the phrase file.
    Modeled on run_extract's drift check (design decision 17). A missing
    file, an unreadable file, or a malformed body all just mean "no check".
    """
    claims_path = os.path.join(claims_dir, f"{memo_id}.md")
    if not os.path.isfile(claims_path):
        return
    try:
        claims_sections = set(_scan_claims_file_sections(claims_path))
    except Exception as exc:  # noqa: BLE001 — a best-effort diagnostic must never break retrieve
        logger.debug("drift check skipped for %s — could not read %s: %s", memo_id, claims_path, exc)
        return

    phrase_set = set(phrase_section_names)
    for name in phrase_section_names:
        if name not in claims_sections:
            logger.warning(
                "retrieve %s: phrase-file section %r is not in %s — "
                "the eval join is on (memo_id, section); a name mismatch scores zero recall",
                memo_id, name, claims_path,
            )
    for name in sorted(claims_sections - phrase_set):
        logger.warning(
            "retrieve %s: %s has section %r but the phrase file has no phrases for it — "
            "that section's golden chunks can never be retrieved",
            memo_id, claims_path, name,
        )


def _load_search_matrix(index_df: pd.DataFrame) -> tuple[list[str], list[str], list[str], "np.ndarray"]:
    """
    (chunk_ids, doc_ids, chunk_texts, M) from an index parquet, M being the
    row-L2-normalized (n_chunks x dim) float64 embedding matrix in index_df
    order. Raises ValueError if the stored vectors have inconsistent length.
    """
    _embedding_dim([list(v) for v in index_df["embedding"]])  # raises on ragged cache
    M = np.array([np.asarray(v, dtype=np.float64) for v in index_df["embedding"]])
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (
        index_df["chunk_id"].tolist(),
        index_df["doc_id"].tolist(),
        index_df["chunk_text"].tolist(),
        M / norms,
    )


def _validate_retrieve_inputs(
    paths: list[str], index_dir: str, claims_dir: str, model: str
) -> list[tuple[dict, pd.DataFrame]]:
    """
    Pre-pass for run_retrieve and run_recheck — NO API calls. For every
    phrase config: validate it, log any soft claims/ drift, load its index
    parquet, require the index to have been embedded with `model` (the
    current EMBED_MODEL — same rule and reason as run_embed: a different
    model at the same dimension would put the queries in another vector
    space, and the dimension check cannot see it), and require the current
    chunk set to equal the indexed chunk set EXACTLY, both directions (a
    shorter PDF would otherwise leave stale, retrievable vectors in the
    matrix — a chunk_id in the results the current golden set can never
    contain). A failure on the third memo costs nothing on the first two.
    Returns [(cfg, index_df), ...] in `paths` order.
    """
    prepared: list[tuple[dict, pd.DataFrame]] = []
    for path in paths:
        cfg = _read_phrase_config(path)
        memo_id = cfg["memo_id"]
        _claims_section_drift_warnings(memo_id, list(cfg["sections"]), claims_dir)

        index_path = os.path.join(index_dir, f"{memo_id}.parquet")
        if not os.path.exists(index_path):
            raise ValueError(
                f"{index_path}: no embedding index — run "
                f"'python retrieval_pipeline.py embed {memo_id}' first"
            )
        index_df = pd.read_parquet(index_path)
        if index_df.empty:
            raise ValueError(f"{index_path}: embedding index is empty — re-run embed")
        _check_index_model(index_path, model)

        records = _build_chunk_records(cfg["source_folder"], memo_id)
        current = {r["chunk_id"]: r["chunk_text"] for r in records}
        indexed = dict(zip(index_df["chunk_id"], index_df["chunk_text"]))
        if current.keys() != indexed.keys() or any(current[k] != indexed[k] for k in current):
            raise ValueError(
                f"{index_path}: out of sync with {cfg['source_folder']} "
                f"(current {len(current)} chunks, indexed {len(indexed)}) — run "
                f"'python retrieval_pipeline.py embed {memo_id}' first"
            )
        prepared.append((cfg, index_df))
    return prepared


def _check_query_dim(memo_id: str, vectors: list[list[float]], matrix: "np.ndarray", what: str) -> None:
    """
    Raises ValueError naming the memo and both dimensions if query
    embeddings came back a different size than the index (the index and
    EMBED_MODEL disagree) — checked before the matmul, which would otherwise
    fail with a bare NumPy shape error naming nothing. `what` is "phrase" or
    "claim", for the message.
    """
    if vectors and len(vectors[0]) != matrix.shape[1]:
        raise ValueError(
            f"{memo_id}: {what} embeddings are {len(vectors[0])}-dim but the "
            f"index is {matrix.shape[1]}-dim — the index and EMBED_MODEL disagree. Delete "
            f"retrieval_index/{memo_id}.parquet and re-run embed."
        )


def _cosine_scores(matrix: "np.ndarray", vector: list[float]) -> "np.ndarray":
    """
    Cosine similarity of one query vector against matrix, whose rows are
    already L2-normalized (_load_search_matrix). A zero query vector scores 0
    everywhere rather than dividing by zero.
    """
    q = np.asarray(vector, dtype=np.float64)
    qn = np.linalg.norm(q)
    return matrix @ (q / (qn if qn != 0 else 1.0))


def _rank_by_score(scores, chunk_ids: list[str], k: int, *, drop_nonpositive: bool) -> list[tuple[int, float]]:
    """
    The top k chunks as (row index, score), ordered by (-score, chunk_id) —
    a full sort, so a genuine tie is broken by chunk_id and the result is
    byte-deterministic. drop_nonpositive removes chunks scoring <= 0 first:
    for BM25 a zero score means the chunk shares no word with the query, so
    ranking those would hand back arbitrary chunks. Dense passes False
    (cosine <= 0 is still an ordering, and run_retrieve warns about it).
    """
    candidates = [i for i in range(len(chunk_ids)) if not drop_nonpositive or scores[i] > 0]
    order = sorted(candidates, key=lambda i: (-scores[i], chunk_ids[i]))[:k]
    return [(i, float(scores[i])) for i in order]


def _fuse_rrf(
    dense: list[tuple[int, float]],
    keyword: list[tuple[int, float]],
    chunk_ids: list[str],
    k: int,
) -> list[tuple[int, float]]:
    """
    Reciprocal Rank Fusion of two ranked lists ("both"): a chunk's fused score
    is the sum of 1 / (_RRF_K + rank) over the lists it appears in. It uses
    ranks only, so BM25 magnitudes and cosine similarities — which are not
    comparable — never need calibrating against each other. Dense is summed
    first, always, so the float result is deterministic.

    Top k by (-fused score, dense rank, chunk_id). Exact ties are routine, not
    rare: dense #r and keyword #r always score the same 1/(_RRF_K + r). The
    dense rank decides them, so the order is principled rather than whichever
    chunk_id sorts first ("x_10" < "x_9"). Note the consequence measured on
    the real corpus: a chunk in BOTH lists (>= 2/80) outranks dense's own #1
    alone (1/61), so a weak keyword list can push good dense chunks down.
    """
    fused: dict[int, float] = {}
    for ranked in (dense, keyword):
        for rank, (i, _) in enumerate(ranked, start=1):
            fused[i] = fused.get(i, 0.0) + 1.0 / (_RRF_K + rank)
    dense_rank = {i: rank for rank, (i, _) in enumerate(dense, start=1)}
    order = sorted(fused, key=lambda i: (-fused[i], dense_rank.get(i, len(dense) + 1), chunk_ids[i]))[:k]
    return [(i, fused[i]) for i in order]


_PROVENANCE_KEY = b"retrieval_provenance"


def _index_fingerprint(index_df: pd.DataFrame) -> str:
    """
    A hash of an index's chunk ids and texts — not its vectors or row
    order. Two indexes with the same fingerprint hold the same chunks, so a
    chunk_id means the same text in both.
    """
    h = hashlib.sha256()
    for chunk_id, text in sorted(zip(index_df["chunk_id"], index_df["chunk_text"])):
        h.update(f"{len(chunk_id)}:{chunk_id}{len(text)}:{text}".encode("utf-8"))
    return h.hexdigest()


def _write_with_provenance(df: pd.DataFrame, path: str, provenance: dict) -> None:
    """
    Writes df as parquet with `provenance` (what produced it: command,
    EMBED_MODEL, depth, and each memo's _index_fingerprint) stored in the
    file's own metadata, so it can never drift from the rows it describes.
    Any parquet reader still reads the rows as usual; read_provenance gets
    the record back. Written atomically (.tmp then os.replace), like the
    index: an interrupted retrieve or recheck keeps the last good file.
    """
    table = pa.Table.from_pandas(df, preserve_index=False)
    metadata = {**(table.schema.metadata or {}), _PROVENANCE_KEY: json.dumps(provenance).encode("utf-8")}
    tmp_path = path + ".tmp"
    try:
        pq.write_table(table.replace_schema_metadata(metadata), tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def read_provenance(path: str) -> dict | None:
    """The provenance _write_with_provenance stored in a parquet file, or
    None for a file written without one (before it was recorded)."""
    raw = (pq.read_schema(path).metadata or {}).get(_PROVENANCE_KEY)
    return json.loads(raw) if raw is not None else None


def _excel_safe(value) -> str:
    """
    Text with the control characters removed that openpyxl cannot store in
    a cell (pypdf leaves NUL bytes in some PDFs' text); "" for a non-string.
    Which characters is openpyxl's ILLEGAL_CHARACTERS_RE, the same object
    tag_pipeline strips with when it writes the reviewed sheet — so the two
    cannot disagree. run_retrieve's .xlsx copy uses this, and eval_pipeline
    compares index or retrieval text with the reviewed sheet only after it.
    """
    return ILLEGAL_CHARACTERS_RE.sub("", value) if isinstance(value, str) else ""


def run_retrieve(
    retrieval_dir: str = "retrieval",
    index_dir: str = "retrieval_index",
    results_path: str = "retrieval_results.parquet",
    claims_dir: str = "claims",
    *,
    client: EmbeddingClient,
) -> pd.DataFrame:
    """
    For EVERY retrieval/<memo_id>.yaml (whole-corpus — no memo filter), run
    each section's phrases as independent top-min(_RETRIEVE_DEPTH, n_chunks)
    cosine queries over that memo's embedding index.

    Pre-pass FIRST (_validate_retrieve_inputs — no API calls): every config
    is validated, every index loaded, every chunk set checked in sync. Only
    then are any phrases embedded, so a missing index for the third memo
    costs nothing on the first two.

    Top-k is a full O(n log n) sort by (-score, chunk_id) — not
    np.argpartition — so it is fully principled at a genuine score tie (which
    a degenerate all-zero phrase vector produces) and byte-deterministic,
    while still <1 ms at <=10k chunks.

    Deliberately NOT cached: phrase embeddings are recomputed on every call
    (phrases churn during authoring while chunk text is stable, the opposite
    of embed's incremental-reuse case), and the whole corpus is re-chunked
    from the source PDFs every run (chunking is cheap; keeping it identical
    to embed's own chunking is what makes the chunk-set sync check in
    _validate_retrieve_inputs meaningful).

    Writes `results_path` (parquet, overwrite, with a provenance record —
    EMBED_MODEL, _RETRIEVE_DEPTH and each memo's index fingerprint — that
    eval_pipeline checks) and its .xlsx sibling; returns the combined
    DataFrame (columns: _RESULTS_COLUMNS).
    One row per (memo_id, section, phrase, method, chunk_id): every phrase is
    searched three ways — "dense" (cosine), "keyword" (BM25 with
    golden_set_pipeline's own _tokenize, chunks scoring 0 omitted, so it can
    return fewer than k rows or none) and "both" (_fuse_rrf of the two).
    NOTE the byte-identical-re-run property holds for a fixed index and fixed
    phrase embeddings — the mocked suite guarantees both; a real re-run
    re-embeds the phrases and an embedding API is not bit-identical
    call-to-call.

    Raises ValueError naming the memo and both dimensions if the phrase
    embeddings come back a different dimension than the index (the index and
    EMBED_MODEL disagree) — checked before the cosine matmul, which would
    otherwise fail with a bare NumPy shape error naming nothing.
    """
    paths = _discover_phrase_configs(retrieval_dir, memo_ids=None)
    prepared = _validate_retrieve_inputs(paths, index_dir, claims_dir, client.model)

    frames: list[pd.DataFrame] = []
    fingerprints: dict[str, str] = {}
    for cfg, index_df in prepared:
        memo_id = cfg["memo_id"]
        fingerprints[memo_id] = _index_fingerprint(index_df)
        chunk_ids, doc_ids, chunk_texts, M = _load_search_matrix(index_df)
        n_chunks = len(chunk_ids)
        k = min(_RETRIEVE_DEPTH, n_chunks)

        flat_phrases: list[tuple[str, int, str]] = [
            (section_name, pidx, phrase)
            for section_name, phrases in cfg["sections"].items()
            for pidx, phrase in enumerate(phrases)
        ]
        phrase_vectors = _embed_texts(
            [p for _, _, p in flat_phrases],
            client,
            input_type="query" if client.asymmetric else None,
            label=memo_id,
        )
        _check_query_dim(memo_id, phrase_vectors, M, "phrase")

        bm25 = BM25Okapi([_tokenize(t) for t in chunk_texts])  # once per memo
        rows: list[dict] = []
        for (section_name, pidx, phrase), pvec in zip(flat_phrases, phrase_vectors):
            scores = _cosine_scores(M, pvec)
            if float(np.max(scores)) <= 0.0:
                logger.warning(
                    "retrieve %s / %r / phrase %r: every cosine score <= 0",
                    memo_id, section_name, phrase,
                )
            dense = _rank_by_score(scores, chunk_ids, k, drop_nonpositive=False)
            keyword = _rank_by_score(
                bm25.get_scores(_tokenize(phrase)), chunk_ids, k, drop_nonpositive=True
            )
            if not keyword:
                logger.info(
                    "retrieve %s / %r / phrase %r: no chunk shares a word with the "
                    "phrase — keyword search returns nothing for it",
                    memo_id, section_name, phrase,
                )
            both = _fuse_rrf(dense, keyword, chunk_ids, k)
            for method, ranked in (("dense", dense), ("keyword", keyword), ("both", both)):
                for rank, (i, score) in enumerate(ranked, start=1):
                    rows.append({
                        "memo_id": memo_id,
                        "section": section_name,
                        "phrase": phrase,
                        "phrase_index": pidx,
                        "method": method,
                        "doc_id": doc_ids[i],
                        "chunk_id": chunk_ids[i],
                        "chunk_text": chunk_texts[i],
                        "rank": rank,
                        "score": score,
                    })
        frames.append(pd.DataFrame(rows, columns=_RESULTS_COLUMNS))
        logger.info("retrieve %s: %d phrase(s), %d chunks, k=%d", memo_id, len(flat_phrases), n_chunks, k)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_RESULTS_COLUMNS)
    df = df.sort_values(
        ["memo_id", "section", "method", "phrase_index", "rank", "chunk_id"], kind="stable"
    ).reset_index(drop=True)

    _write_with_provenance(df, results_path, {"command": "retrieve", "model": client.model,
                                              "depth": _RETRIEVE_DEPTH, "indexes": fingerprints})
    xlsx_path = (results_path[:-len(".parquet")] if results_path.endswith(".parquet") else results_path) + ".xlsx"
    # openpyxl raises on the control characters pypdf leaves in some chunk
    # text, so the xlsx copy is sanitized while the parquet keeps it raw.
    xlsx_df = df.copy()
    xlsx_df["chunk_text"] = xlsx_df["chunk_text"].map(_excel_safe)
    xlsx_df.to_excel(xlsx_path, index=False, sheet_name="retrieval")
    _widen_columns(xlsx_path, df.columns, wide={"phrase", "chunk_text"})
    logger.info("retrieve: wrote %d rows to %s (+ %s)", len(df), results_path, os.path.basename(xlsx_path))
    return df


def dedupe_by_section(df: pd.DataFrame, *, top_k: int = _TOP_K) -> pd.DataFrame:
    """
    Collapses the row-per-(phrase, chunk) results to the per-section union a
    recall metric would consume: filter to rank <= top_k, then keep one row
    per (memo_id, section, method, chunk_id) — the one with the best (lowest)
    rank, ties broken by score. `chunk_id` is also in the sort key, but only
    for deterministic output ROW ORDERING — within one (memo_id, section,
    method, chunk_id) group `chunk_id` is constant, so it can never itself
    break a tie between rows in that group. This is where _TOP_K is applied;
    the on-disk artifact keeps all _RETRIEVE_DEPTH rows. Methods are never
    merged: dense, keyword and both each keep their own union.
    """
    filtered = df[df["rank"] <= top_k]
    return (
        filtered.sort_values(
            ["memo_id", "section", "method", "rank", "score", "chunk_id"],
            ascending=[True, True, True, True, False, True],
            kind="stable",
        )
        .drop_duplicates(["memo_id", "section", "method", "chunk_id"], keep="first")
        .reset_index(drop=True)
    )


def run_recheck(
    retrieval_dir: str = "retrieval",
    index_dir: str = "retrieval_index",
    claims_dir: str = "claims",
    out_path: str = "claim_queries.parquet",
    *,
    client: EmbeddingClient,
) -> pd.DataFrame:
    """
    For every memo with a retrieval/<memo_id>.yaml, embed the text of every
    claim in claims/<memo_id>.md as a dense query and keep its top
    min(_TOP_K, n_chunks) chunks. Writes out_path (parquet, overwrite, with
    the same kind of provenance record as run_retrieve) and returns it
    (columns: _CLAIM_QUERY_COLUMNS). claim_id is recomputed with
    _derive_claim_id, so it joins to the golden set exactly.

    This is the eval's meaning-based cross-check for claims the golden set
    could not verify (spec: "The dense cross-check"). It reads claims files
    only — never tags, never reviewed/ — so retrieval stays blind to the
    ground truth; eval_pipeline decides which claims and chunks matter.

    Pre-pass first, as in run_retrieve: every phrase config, index and chunk
    set is validated (_validate_retrieve_inputs) and every claims file parsed
    before any claim is embedded. A memo with a phrase config but no claims
    file raises ValueError.
    """
    paths = _discover_phrase_configs(retrieval_dir, memo_ids=None)
    prepared = _validate_retrieve_inputs(paths, index_dir, claims_dir, client.model)

    work: list[tuple[str, pd.DataFrame, list[tuple[str, str, str]]]] = []
    for cfg, index_df in prepared:
        memo_id = cfg["memo_id"]
        path = os.path.join(claims_dir, f"{memo_id}.md")
        if not os.path.exists(path):
            raise ValueError(
                f"{path}: not found — recheck embeds each claim's text, so every memo "
                f"with a retrieval/{memo_id}.yaml needs its claims file"
            )
        _, _, _, sections = parse_claims_file(path)
        claims = [
            (section, _derive_claim_id(memo_id, section, text, n), text)
            for section, texts in sections
            for text, n in _claims_with_occurrence(texts)
        ]
        work.append((memo_id, index_df, claims))

    frames: list[pd.DataFrame] = []
    for memo_id, index_df, claims in work:
        chunk_ids, doc_ids, chunk_texts, M = _load_search_matrix(index_df)
        k = min(_TOP_K, len(chunk_ids))
        vectors = _embed_texts(
            [text for _, _, text in claims],
            client,
            input_type="query" if client.asymmetric else None,
            label=memo_id,
        )
        _check_query_dim(memo_id, vectors, M, "claim")
        rows: list[dict] = []
        for (section, claim_id, text), vector in zip(claims, vectors):
            ranked = _rank_by_score(_cosine_scores(M, vector), chunk_ids, k, drop_nonpositive=False)
            for rank, (i, score) in enumerate(ranked, start=1):
                rows.append({
                    "memo_id": memo_id,
                    "section": section,
                    "claim_id": claim_id,
                    "claim_text": text,
                    "doc_id": doc_ids[i],
                    "chunk_id": chunk_ids[i],
                    "chunk_text": chunk_texts[i],
                    "rank": rank,
                    "score": score,
                })
        frames.append(pd.DataFrame(rows, columns=_CLAIM_QUERY_COLUMNS))
        logger.info("recheck %s: %d claim(s), top %d chunk(s) each", memo_id, len(claims), k)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_CLAIM_QUERY_COLUMNS)
    df = df.sort_values(["memo_id", "section", "claim_id", "rank"], kind="stable").reset_index(drop=True)
    _write_with_provenance(df, out_path, {"command": "recheck", "model": client.model, "depth": _TOP_K,
                                          "indexes": {memo_id: _index_fingerprint(index_df)
                                                      for memo_id, index_df, _ in work}})
    logger.info("recheck: wrote %d rows to %s", len(df), out_path)
    return df


# %% [markdown]
# ## 7. xlsx column widening (local — the golden-set helper's wide set differs)

# %%
def _widen_columns(path: str, columns, wide: set[str]) -> None:
    """
    Widen `wide` columns to 60 and the rest to 18, wrap-text all data rows.
    Mirrors golden_set_pipeline._widen_review_columns, including the
    `ws is None` narrowing that commit 5e7a3d9 added there (pyright sees
    Workbook.active as `... | None`; openpyxl only returns None for a
    sheet-less workbook, which df.to_excel can't produce — the guard beats a
    deep AttributeError).
    """
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment
    from openpyxl.utils import get_column_letter

    wb = load_workbook(path)
    ws = wb.active
    if ws is None:
        raise ValueError(f"{path} has no active worksheet to widen")

    wrap = Alignment(wrap_text=True, vertical="top")
    last_row = ws.max_row
    for idx, col_name in enumerate(columns, start=1):
        letter = get_column_letter(idx)
        ws.column_dimensions[letter].width = 60 if col_name in wide else 18
        for row in range(2, last_row + 1):
            ws[f"{letter}{row}"].alignment = wrap
    wb.save(path)


# %% [markdown]
# ## 8. CLI dispatch + demo
#
#   python retrieval_pipeline.py embed [memo_id ...]   # build/refresh the index
#   python retrieval_pipeline.py retrieve              # run every phrase, write results
#   python retrieval_pipeline.py recheck               # each claim's own text as a query
#
# `retrieve` takes NO memo filter — it is whole-corpus by design (a partial
# retrieval_results.parquet looks complete and poisons the eval).

# %%
def _main(argv: list[str]) -> None:
    """
    `python retrieval_pipeline.py [embed [memo_id ...] | retrieve | recheck]` dispatch,
    factored out of `if __name__ == "__main__":` for unit testing.

    No argument, an unrecognised command, or memo args passed to `retrieve`
    or `recheck` all raise SystemExit with a usage message BEFORE
    EmbeddingClient.from_env runs — a typo shouldn't need credentials to
    diagnose (same as golden_set_pipeline._main).
    """
    command = argv[1] if len(argv) > 1 else None
    extra = argv[2:]
    if command not in ("embed", "retrieve", "recheck"):
        raise SystemExit(
            f"usage: python retrieval_pipeline.py [embed [memo_id ...] | retrieve | recheck] "
            f"(got {command!r})"
        )
    if command in ("retrieve", "recheck") and extra:
        raise SystemExit(
            f"'{command}' takes no arguments — it always processes every retrieval/*.yaml "
            f"(got {extra!r}). Use 'embed {' '.join(extra)}' to restrict the embed step."
        )

    client = EmbeddingClient.from_env()
    if command == "embed":
        run_embed("retrieval", "retrieval_index", extra or None, client=client)
    elif command == "recheck":
        df = run_recheck("retrieval", "retrieval_index", "claims", "claim_queries.parquet", client=client)
        print(f"recheck: {df['claim_id'].nunique()} claim(s) searched, {len(df)} row(s) in claim_queries.parquet")
    else:
        df = run_retrieve("retrieval", "retrieval_index", "retrieval_results.parquet", client=client)
        deduped = dedupe_by_section(df)
        for (memo_id, section, method), grp in deduped.groupby(["memo_id", "section", "method"]):
            print(f"{memo_id} / {section} / {method}: {len(grp)} unique chunk(s) in the top-{_TOP_K} union")


if __name__ == "__main__":
    _main(sys.argv)
