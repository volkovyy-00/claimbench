# %% [markdown]
# # Retrieval eval harness (Phase 3)
#
# Scores `retrieval_pipeline.py`'s results against the hand-reviewed golden
# set and renders a self-contained HTML report. Design:
# `docs/superpowers/specs/2026-09-10-retrieval-eval-harness-design.md`.
#
# **This file measures; it never searches.** It reads four kinds of artifact
# — `claims/<memo_id>.md`, `reviewed/<memo_id>.xlsx`,
# `retrieval_results.parquet` (+ the index, for a parity check) and
# `claim_queries.parquet` — and writes `eval_runs/<run_id>/`. It imports
# `_same_evidence` and `_group_equivalent_chunks` from golden_set_pipeline:
# deciding what counts as *the same evidence* is part of defining a correct
# answer, not of producing one, so the import is measurement.
#
# **Loud refusal.** Anything that would make a number silently wrong — an
# untagged row, a claim the review never saw, a section with no search
# results, a golden set chunked differently from the index — stops the run
# with every problem listed, rather than scoring around it.
#
#     python eval_pipeline.py score [label]
#     python eval_pipeline.py report <run_id|latest> [baseline_run_id]

# %%
import hashlib
import html
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from golden_set_pipeline import (
    _HUMAN_ADDED_PREFIX,
    _QUOTE_SEPARATOR,
    _UnionFind,
    _chunk_index_of,
    _claims_with_occurrence,
    _derive_claim_id,
    _group_equivalent_chunks,
    _is_deterministic_claim_id,
    _normalize_span,
    _same_evidence,
    _valid_memo_id,
    parse_claims_file,
)
from retrieval_pipeline import (
    _CLAIM_QUERY_COLUMNS,
    _RESULTS_COLUMNS,
    _RETRIEVAL_METHODS,
    _RETRIEVE_DEPTH,
    _TOP_K,
    _excel_safe,
    _index_fingerprint,
    _load_index_model,
    _read_phrase_config,
    dedupe_by_section,
    read_provenance,
)

logger = logging.getLogger("eval")

# %% [markdown]
# ## 1. Ground truth — the claim census and the evidence
#
# The census comes from the claims file (ids recomputed, design decision 17);
# evidence and tags come from `reviewed/<memo_id>.xlsx` (written by
# `tag_pipeline.py finalize`). Two independent sources, so a claim cannot
# silently vanish from the census because its rows were deleted.

# %%
_CLOSED_TAGS = ("extractive", "synthesized", "unverifiable")
_VERIFIED_TAGS = ("extractive", "synthesized")
_SHEET_REQUIRED = ("claim_id", "section", "claim_text", "doc_id", "chunk_id",
                   "chunk_text", "evidence_span", "found", "tag", "tag_rationale")
_CLAIM_COLUMNS = ["memo_id", "section", "claim_id", "claim_text", "bucket"]
_EVIDENCE_COLUMNS = ["memo_id", "section", "claim_id", "doc_id", "chunk_id", "evidence_span", "tag",
                     "human_added"]
_JUDGED_COLUMNS = ["memo_id", "claim_id", "chunk_id", "chunk_text"]


class EvalInputError(ValueError):
    """Every problem found in the inputs, one per line, raised together
    instead of producing a number computed around them."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("eval refused:\n  " + "\n  ".join(self.problems))


@dataclass
class GroundTruth:
    """One memo's ground truth. Every text cell is str or None, never NaN.

    claims:   the census, claims-file order (_CLAIM_COLUMNS), with bucket
              EXTRACTIVE / SYNTHESIZED / UNVERIFIABLE.
    evidence: the relevant set — rows with found true AND tag extractive or
              synthesized (_EVIDENCE_COLUMNS); human_added marks a chunk a
              person added in review, whose span may join several quotes.
    judged:   every (claim, chunk) a human judged, with the sheet's chunk
              text (_JUDGED_COLUMNS)."""
    memo_id: str
    claims: pd.DataFrame
    evidence: pd.DataFrame
    judged: pd.DataFrame


def _text(value) -> str | None:
    """A cell as str, or None for a blank. An xlsx blank reads back as NaN;
    a whitespace-only cell counts as blank too."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    text = str(value)
    return text if text.strip() else None


def claim_buckets(verified: pd.DataFrame) -> pd.Series:
    """
    claim_id -> EXTRACTIVE if any of the claim's verified rows is extractive,
    else SYNTHESIZED (verified rows carry only those two tags). Ordered by
    what retrieval has to accomplish: one directly-stating chunk makes the
    claim reachable in one hop. Claims with no verified row are absent; the
    caller buckets them UNVERIFIABLE.
    """
    return verified.groupby("claim_id")["tag"].agg(
        lambda tags: "EXTRACTIVE" if (tags == "extractive").any() else "SYNTHESIZED"
    )


def _warn_quotes_not_in_their_chunk(sheet_path: str, verified: pd.DataFrame) -> None:
    """
    Logs a WARNING when verified quotes are not in their own chunk's text
    after _normalize_span — a quote that runs past the chunk, or PDF text
    with a split word or a curly apostrophe (finalize matches ignoring
    whitespace, the eval does not). Such a quote is still hit by its own
    chunk, but never by a neighbour holding the same passage, and never
    grouped with another row's copy of it. Not a refusal: the score is
    right for the chunk that was cited; the warning keeps the loss visible.
    `verified` keeps the sheet's row positions (Excel row = index + 2).
    """
    missing = [(n + 2, row.claim_id) for n, row in verified.iterrows()
               for quote in _quotes(row)
               if (_normalize_span(quote) or "") not in (_normalize_span(row.chunk_text) or "")]
    if missing:
        logger.warning("eval: %s: %d verified quote(s) are not in their own chunk's text as written (first: "
                       "row %d, claim %s) — only that exact chunk can hit them, not a neighbour holding the "
                       "same passage", sheet_path, len(missing), *missing[0])


def load_ground_truth(memo_id: str, claims_dir: str = "claims", reviewed_dir: str = "reviewed") -> GroundTruth:
    """
    Reads claims/<memo_id>.md (the census) and reviewed/<memo_id>.xlsx (the
    evidence), checks them against each other, and returns a GroundTruth.

    Raises EvalInputError listing every problem (sheet rows numbered as Excel
    shows them) — never scores around one:
      - a missing file or column;
      - claim_ids that are not deterministic uuid5 (a golden set built before
        design decision 17 — reported once, since every row would also fail
        to match the census), or a blank claim_id;
      - a blank tag, an unknown tag, or `rejected` (written by
        import_reviewed, which this workflow does not use);
      - extractive/synthesized on a found=False row, or on a row with no
        chunk_id, doc_id or evidence_span to score against (without a doc_id
        _same_evidence never matches, so neighbour hits would silently drop);
      - the same (claim, chunk) twice;
      - a census claim with no row in the sheet, or a sheet claim not in the
        census (its wording changed after tagging, rotating its id);
      - a claims file parse_claims_file cannot read (its message names the
        file and line).
    """
    claims_path = os.path.join(claims_dir, f"{memo_id}.md")
    sheet_path = os.path.join(reviewed_dir, f"{memo_id}.xlsx")
    missing = [f"{path}: not found" for path in (claims_path, sheet_path) if not os.path.exists(path)]
    if missing:
        raise EvalInputError(missing)

    try:
        _, _, _, sections = parse_claims_file(claims_path)
    except ValueError as e:
        raise EvalInputError([str(e)])
    census = pd.DataFrame(
        [{"memo_id": memo_id, "section": section,
          "claim_id": _derive_claim_id(memo_id, section, text, n), "claim_text": text}
         for section, texts in sections
         for text, n in _claims_with_occurrence(texts)],
        columns=["memo_id", "section", "claim_id", "claim_text"],
    )

    sheet = pd.read_excel(sheet_path).reset_index(drop=True)
    absent = [c for c in _SHEET_REQUIRED if c not in sheet.columns]
    if absent:
        raise EvalInputError([f"{sheet_path}: missing column(s) {absent} — not a tag_pipeline.py finalize file"])

    # Cleaned into plain lists, not Series: pandas 3 stores a None returned by
    # .map() in a string column as NaN, and every check below relies on None.
    excel_rows = [i + 2 for i in range(len(sheet))]  # header is row 1
    cell = {c: [_text(v) for v in sheet[c]]
            for c in ("claim_id", "doc_id", "chunk_id", "chunk_text", "evidence_span", "tag", "tag_rationale")}
    claim_ids, chunk_ids, spans, raw_tags = cell["claim_id"], cell["chunk_id"], cell["evidence_span"], cell["tag"]
    tags = [t.strip().lower() if t else None for t in raw_tags]
    found = sheet["found"].eq(True).tolist()

    legacy = [n for n, c in zip(excel_rows, claim_ids) if c is not None and not _is_deterministic_claim_id(c)]
    if legacy:
        raise EvalInputError([
            f"{sheet_path}: {len(legacy)} row(s) carry a claim_id that is not a deterministic "
            f"uuid5 id (first: row {legacy[0]}) — this golden set predates design decision 17; rebuild it"
        ])

    problems: list[str] = []
    for n, claim_id, raw, tag, is_found, chunk_id, span, doc_id in zip(
        excel_rows, claim_ids, raw_tags, tags, found, chunk_ids, spans, cell["doc_id"]
    ):
        if claim_id is None:
            problems.append(f"{sheet_path} row {n}: claim_id is blank — the row cannot be matched to any claim")
            continue
        where = f"{sheet_path} row {n} (claim {claim_id})"
        if tag is None:
            problems.append(f"{where}: tag is blank — tag it; the eval never scores around an untagged row")
        elif tag == "rejected":
            problems.append(f"{where}: tag 'rejected' is written by import_reviewed, which this workflow does not use")
        elif tag not in _CLOSED_TAGS:
            problems.append(f"{where}: tag {raw!r} is not one of {', '.join(_CLOSED_TAGS)}")
        elif tag in _VERIFIED_TAGS and not is_found:
            problems.append(f"{where}: tag {tag!r} on a row with found=False — finalize never writes this")
        elif tag in _VERIFIED_TAGS and (chunk_id is None or span is None or doc_id is None):
            problems.append(f"{where}: tag {tag!r} but no chunk_id, doc_id or evidence_span to score against")

    seen: set[tuple[str, str]] = set()
    for n, claim_id, chunk_id in zip(excel_rows, claim_ids, chunk_ids):
        if chunk_id is not None:
            if (claim_id, chunk_id) in seen:
                problems.append(f"{sheet_path} row {n}: claim {claim_id} lists chunk {chunk_id} twice")
            seen.add((claim_id, chunk_id))

    sheet_ids = {c for c in claim_ids if c is not None}
    for claim in census[~census["claim_id"].isin(sheet_ids)].itertuples(index=False):
        problems.append(
            f"{claims_path}: claim {claim.claim_text!r} ({claim.section}) has no row in "
            f"{sheet_path} — the review is out of date for it"
        )
    first_row: dict[str, int] = {}
    for n, claim_id in zip(excel_rows, claim_ids):
        first_row.setdefault(claim_id, n)
    for claim_id in sorted(sheet_ids - set(census["claim_id"])):
        problems.append(
            f"{sheet_path} row {first_row[claim_id]}: claim_id {claim_id} is not a claim in "
            f"{claims_path} — its wording changed after tagging; rebuild, draft and review it again"
        )
    if problems:
        raise EvalInputError(problems)

    # Every sheet claim_id is in the census by now, and the id encodes its
    # section, so each row's section comes from the census, never from the
    # sheet's section cell: an edited or blank cell can't move evidence.
    section_of = dict(zip(census["claim_id"], census["section"]))
    rows = pd.DataFrame({"memo_id": memo_id, **cell, "tag": tags,  # cleaned tags replace raw ones
                         "section": [section_of[c] for c in claim_ids],
                         "human_added": [r is not None and r.startswith(_HUMAN_ADDED_PREFIX)
                                         for r in cell["tag_rationale"]]}, dtype=object)
    verified = rows[[f and t in _VERIFIED_TAGS for f, t in zip(found, tags)]]
    _warn_quotes_not_in_their_chunk(sheet_path, verified)
    census["bucket"] = census["claim_id"].map(claim_buckets(verified)).fillna("UNVERIFIABLE")
    return GroundTruth(
        memo_id=memo_id,
        claims=census[_CLAIM_COLUMNS].reset_index(drop=True),
        evidence=verified[_EVIDENCE_COLUMNS].reset_index(drop=True),
        judged=rows[[c is not None for c in chunk_ids]][_JUDGED_COLUMNS].reset_index(drop=True),
    )


# %% [markdown]
# ## 2. The retrieval side, and scoring
#
# A retrieved chunk hits a golden evidence row when it IS that chunk, or when
# design decision 1's rule (_same_evidence) says it carries the same
# evidence — the golden quote compared with the retrieved chunk's text.
# Golden groups come from golden rows alone; hits are computed once at
# _RETRIEVE_DEPTH and every k is read off by best rank.

# %%
_K_VALUES = tuple(range(1, _RETRIEVE_DEPTH + 1))
_CLAIM_HIT_COLUMNS = ["memo_id", "section", "claim_id", "bucket", "method",
                      "group", "group_size", "best_rank", "best_chunk_id", "best_golden_chunk_id"]
_CHUNK_HIT_COLUMNS = ["memo_id", "section", "method", "chunk_id", "best_rank", "golden"]
_METRIC_COLUMNS = ["scope", "memo_id", "section", "method", "k",
                   "claims", "covered", "coverage",
                   "claims_extractive", "covered_extractive",
                   "claims_synthesized", "covered_synthesized", "complete_synthesized",
                   "recall_macro", "recall_micro", "mrr", "passages",
                   "retrieved", "retrieved_golden", "precision", "f1"]
_NO_PRECISION = {"retrieved": np.nan, "retrieved_golden": np.nan, "precision": np.nan, "f1": np.nan}


def _unreadable(path: str, error: Exception, command: str) -> str:
    """The problem line for a file that exists but cannot be read (a truncated
    copy, a provenance record that is not JSON) — listed with the others,
    not a traceback that hides them."""
    return f"{path}: unreadable ({error}) — re-run python retrieval_pipeline.py {command}"


def load_results(path: str = "retrieval_results.parquet") -> pd.DataFrame:
    """retrieval_results.parquet, refusing a file that cannot be read,
    without the method column (written before keyword and combined search
    existed), with a missing column, or with a method name this file does
    not know."""
    if not os.path.exists(path):
        raise EvalInputError([f"{path}: not found — run python retrieval_pipeline.py retrieve"])
    try:
        df = pd.read_parquet(path)
    except (OSError, ValueError) as e:   # pyarrow's ArrowInvalid is a ValueError
        raise EvalInputError([_unreadable(path, e, "retrieve")])
    if "method" not in df.columns:
        raise EvalInputError([
            f"{path}: no 'method' column — it was written before keyword and combined search "
            f"existed; re-run python retrieval_pipeline.py retrieve"
        ])
    absent = [c for c in _RESULTS_COLUMNS if c not in df.columns]
    unknown = sorted(set(df["method"]) - set(_RETRIEVAL_METHODS))
    problems = ([f"{path}: missing column(s) {absent}"] if absent else []) + \
               ([f"{path}: unknown method(s) {unknown}"] if unknown else [])
    if problems:
        raise EvalInputError(problems)
    return df


def check_index_parity(gt: GroundTruth, index_dir: str = "retrieval_index") -> None:
    """
    Every chunk a human judged must exist in retrieval_index/<memo_id>.parquet
    with the same text (after _excel_safe) — otherwise the golden set and the
    index were chunked differently and every join would miss silently.
    Replaces a "chunk settings" fingerprint with a direct check. Each
    evidence row's doc_id must also be its chunk's doc_id in the index: the
    hit rule (_same_evidence) only matches within one document, so a wrong
    doc_id would silently stop neighbour hits for that row. (The index's
    model is checked by check_provenance, against the one retrieve used.)
    """
    path = os.path.join(index_dir, f"{gt.memo_id}.parquet")
    if not os.path.exists(path):
        raise EvalInputError([f"{path}: not found — run python retrieval_pipeline.py embed {gt.memo_id}"])
    index = pd.read_parquet(path, columns=["chunk_id", "doc_id", "chunk_text"])
    text_by_id = dict(zip(index["chunk_id"], index["chunk_text"]))
    doc_by_id = dict(zip(index["chunk_id"], index["doc_id"]))
    bad: list[str] = []
    for row in gt.judged.itertuples(index=False):
        if row.chunk_id not in text_by_id:
            bad.append(f"{row.chunk_id} is not in the index")
        elif _excel_safe(text_by_id[row.chunk_id]) != (row.chunk_text or ""):
            bad.append(f"{row.chunk_id}'s text differs from the index")
    problems = []
    if bad:
        problems.append(
            f"{path}: {len(bad)} golden chunk(s) do not match the retrieval index (first: {bad[0]}) — "
            f"the golden set and the index were chunked differently (chunk size, overlap or PDF "
            f"reader); rebuild one to match the other"
        )
    problems += [
        f"{path}: evidence row for claim {row.claim_id} gives chunk {row.chunk_id} doc_id "
        f"{row.doc_id!r}, but the index says {doc_by_id[row.chunk_id]!r} — the reviewed sheet was "
        f"edited after finalize; correct the doc_id"
        for row in gt.evidence.itertuples(index=False)
        if row.chunk_id in doc_by_id and row.doc_id != doc_by_id[row.chunk_id]
    ]
    if problems:
        raise EvalInputError(problems)


def check_sections_retrieved(gt: GroundTruth, results: pd.DataFrame) -> None:
    """
    Every section holding a verifiable claim must have dense rows in the
    results — else it would score zero, indistinguishable from a retriever
    that failed there, when really the section is missing from (or spelled
    differently in) retrieval/<memo_id>.yaml. Dense always returns rows for a
    phrase, so dense is the test.

    The other two methods must be present too, or they would score 0% just
    as silently: "both" fuses the dense list, so every phrase with dense rows
    must have "both" rows; keyword legitimately returns none for a phrase
    sharing no word with the corpus, so it is only required somewhere in
    the memo.
    """
    memo = results[results["memo_id"] == gt.memo_id]
    need = sorted(set(gt.claims.loc[gt.claims["bucket"] != "UNVERIFIABLE", "section"]))
    have = set(memo.loc[memo["method"] == "dense", "section"])
    problems = [
        f"retrieval results have no rows for {gt.memo_id} / {s!r}, which has claims to score — "
        f"is the section missing from retrieval/{gt.memo_id}.yaml, or spelled differently there?"
        for s in need if s not in have
    ]
    dense_phrases = set(zip(memo.loc[memo["method"] == "dense", "section"],
                            memo.loc[memo["method"] == "dense", "phrase_index"]))
    both_phrases = set(zip(memo.loc[memo["method"] == "both", "section"],
                           memo.loc[memo["method"] == "both", "phrase_index"]))
    problems += [
        f"retrieval results for {gt.memo_id} / {s!r} phrase {i}: dense rows but no 'both' rows — "
        f"the file is partial; re-run python retrieval_pipeline.py retrieve"
        for s, i in sorted(dense_phrases - both_phrases)
    ]
    if have and not (memo["method"] == "keyword").any():
        problems.append(f"retrieval results for {gt.memo_id} have no keyword rows at all — the file is "
                        f"partial; re-run python retrieval_pipeline.py retrieve")
    if problems:
        raise EvalInputError(problems)


def check_provenance(memo_id: str, provenance: dict, path: str, command: str,
                     index_dir: str = "retrieval_index") -> None:
    """
    `path` (retrieval_results.parquet from retrieve, or claim_queries.parquet
    from recheck) must have been produced from this memo's index as it is
    now. Its provenance record (retrieval_pipeline.read_provenance) must hold
    the index's fingerprint — else the index was rebuilt since, and a
    chunk_id in the file may name different text — and the model the index
    was embedded with — else the queries came from another vector space. The
    recorded depth must be the one the eval reads the file to:
    _RETRIEVE_DEPTH for retrieve, since every k up to it is read off the
    results, and _TOP_K for recheck, the re-review list's depth. A file with
    no record at all is refused by the caller, once; a missing index by
    check_index_parity.
    """
    index_path = os.path.join(index_dir, f"{memo_id}.parquet")
    if not os.path.exists(index_path):
        return
    rerun = f"re-run python retrieval_pipeline.py {command}"
    problems = []
    recorded = provenance.get("indexes", {}).get(memo_id)
    if recorded is None:
        raise EvalInputError([f"{path}: records no search for {memo_id} — it was produced before "
                              f"{memo_id} was set up; {rerun}"])
    fingerprint = _index_fingerprint(pd.read_parquet(index_path, columns=["chunk_id", "chunk_text"]))
    if recorded != fingerprint:
        problems.append(f"{path}: produced from a different {index_path} (its chunks changed since) — a "
                        f"chunk_id may now name different text; {rerun}")
    index_model = _load_index_model(index_path)
    if index_model is not None and provenance.get("model") != index_model:
        problems.append(f"{path}: produced with model {provenance.get('model')!r}, but {index_path} was "
                        f"embedded with {index_model!r}; {rerun}")
    if command == "retrieve" and provenance.get("depth") != _RETRIEVE_DEPTH:
        problems.append(f"{path}: retrieved at depth {provenance.get('depth')}, but _RETRIEVE_DEPTH is "
                        f"{_RETRIEVE_DEPTH} — deeper k would repeat the last value; {rerun}")
    if command == "recheck" and provenance.get("depth") != _TOP_K:
        problems.append(f"{path}: rechecked at depth {provenance.get('depth')}, but _TOP_K is {_TOP_K} — "
                        f"the re-review list would use the old depth; {rerun}")
    if problems:
        raise EvalInputError(problems)


def _no_provenance(path: str, command: str) -> str:
    return (f"{path}: records no provenance (written before {command} recorded its model, depth and "
            f"index) — re-run python retrieval_pipeline.py {command}")


def ranked_chunks(results: pd.DataFrame, memo_id: str) -> dict[tuple[str, str], pd.DataFrame]:
    """(section, method) -> one row per retrieved chunk with its best rank to
    _RETRIEVE_DEPTH (dedupe_by_section), chunk text made _excel_safe so it
    compares with the sheet's quotes."""
    best = dedupe_by_section(results[results["memo_id"] == memo_id], top_k=_RETRIEVE_DEPTH)
    best = best.assign(chunk_text=best["chunk_text"].map(_excel_safe))
    return {(section, method): group.reset_index(drop=True)
            for (section, method), group in best.groupby(["section", "method"])}


def _shared_edge(left: str, right: str) -> str:
    """The text two adjacent chunks share: the longest end of `left` that is
    also the start of `right` ("" if none). Chunks come from one sliding
    window, so this is exactly their overlap."""
    for k in range(min(len(left), len(right)), 0, -1):
        if left.endswith(right[:k]):
            return right[:k]
    return ""


def _quotes(golden) -> list[str]:
    """The quotes one golden evidence row holds. A chunk a person added in
    review joins the quotes they pasted with _QUOTE_SEPARATOR, and the joined
    text appears in no chunk, so each is evidence on its own. Any other row's
    span is one quote, kept whole: it is text from the chunk, which may itself
    contain the separator (a table row), and split it would match on a
    fragment."""
    span = golden.evidence_span or ""
    return span.split(_QUOTE_SEPARATOR) if golden.human_added else [span]


def _evidence_groups(golden: list) -> list[list[int]]:
    """
    Decision 1's groups over one claim's golden rows (_group_equivalent_chunks),
    formed from the same quotes _is_hit tests (_quotes): a human-added row
    joins a group when any one of its quotes is the same evidence as a
    member's. Grouped on the joined text instead, it would stand alone and
    count as a separate piece of evidence. Groups in order of their first
    member, members ascending — _group_equivalent_chunks' order, which this
    reproduces exactly when no row holds several quotes.
    """
    owner, doc_ids, chunk_ids, texts = [], [], [], []
    for i, g in enumerate(golden):
        for quote in _quotes(g):
            owner.append(i)
            doc_ids.append(g.doc_id)
            chunk_ids.append(g.chunk_id)
            texts.append(quote)
    uf = _UnionFind(len(golden))
    for pieces in _group_equivalent_chunks(doc_ids, chunk_ids, texts):
        for p in pieces[1:]:
            uf.union(owner[pieces[0]], owner[p])
    groups: dict[int, list[int]] = {}
    for i in range(len(golden)):
        groups.setdefault(uf.find(i), []).append(i)
    return list(groups.values())


def _is_hit(golden, retrieved, golden_text: dict[str, str]) -> bool:
    """
    A retrieved chunk hits a golden evidence row when it IS that chunk, or
    when decision 1's rule (_same_evidence) says it carries the same
    evidence. The chunk_id test comes first because _same_evidence never
    matches a chunk with itself.

    For a neighbouring chunk that rule alone is too loose, because one side
    is a whole ~1000-character chunk, not a quote: a short quote ("5%")
    repeated anywhere in the neighbour would count. So a neighbour must also
    hold the WHOLE quote inside the text the two chunks share (_shared_edge
    with the golden chunk's text, golden_text[chunk_id]) — the only place the
    same passage can sit in both. One direction only: a long quote that
    merely contains the shared text is not wholly in the neighbour, so it is
    not a hit. In _rows_for_claim both sides are short quotes, so this extra
    condition is the eval's alone.

    A human-added row can hold several quotes (_quotes); each is tested on
    its own and any one of them makes a hit.
    """
    if golden.chunk_id == retrieved.chunk_id:
        return True
    g, r = _chunk_index_of(golden.chunk_id), _chunk_index_of(retrieved.chunk_id)
    for quote in _quotes(golden):
        if not _same_evidence(golden.doc_id, golden.chunk_id, quote,
                              retrieved.doc_id, retrieved.chunk_id, retrieved.chunk_text):
            continue
        if g is None or r is None or abs(g - r) != 1 or golden.chunk_id not in golden_text:
            return True   # an identical-text match, not a neighbour
        left, right = ((golden_text[golden.chunk_id], retrieved.chunk_text) if r == g + 1
                       else (retrieved.chunk_text, golden_text[golden.chunk_id]))
        normalized = _normalize_span(quote)
        if normalized and normalized in (_normalize_span(_shared_edge(left, right)) or ""):
            return True
    return False


def score_claims(gt: GroundTruth, ranked: dict) -> pd.DataFrame:
    """One row per (verifiable claim, golden evidence group, method): the best
    rank at which any retrieved chunk hits any row of the group, that
    retrieved chunk, and the golden row it matched (NaN / None when nothing
    hits within _RETRIEVE_DEPTH). The golden row is recorded because a claim
    can have many evidence rows, and the report must quote the one that was
    actually found, not the claim's first."""
    evidence_by_claim = {c: list(g.itertuples(index=False)) for c, g in gt.evidence.groupby("claim_id")}
    golden_text = dict(zip(gt.judged["chunk_id"], gt.judged["chunk_text"]))
    rows: list[dict] = []
    for claim in gt.claims[gt.claims["bucket"] != "UNVERIFIABLE"].itertuples(index=False):
        golden = evidence_by_claim[claim.claim_id]
        groups = _evidence_groups(golden)
        for method in _RETRIEVAL_METHODS:
            retrieved = ranked.get((claim.section, method))
            candidates = [] if retrieved is None else list(retrieved.itertuples(index=False))
            for index, members in enumerate(groups):
                hits = []
                for r in candidates:
                    matched = next((golden[m].chunk_id for m in members if _is_hit(golden[m], r, golden_text)), None)
                    if matched is not None:
                        hits.append((int(r.rank), r.chunk_id, matched))
                best_rank, best_chunk, best_golden = min(hits) if hits else (None, None, None)
                rows.append({"memo_id": gt.memo_id, "section": claim.section, "claim_id": claim.claim_id,
                             "bucket": claim.bucket, "method": method, "group": index,
                             "group_size": len(members), "best_rank": best_rank, "best_chunk_id": best_chunk,
                             "best_golden_chunk_id": best_golden})
    return pd.DataFrame(rows, columns=_CLAIM_HIT_COLUMNS).astype({"best_rank": "float64"})


def score_chunks(gt: GroundTruth, ranked: dict) -> pd.DataFrame:
    """One row per (section, method, retrieved chunk): its best rank, and
    whether it hits any evidence row of any verifiable claim in its section —
    the numerator of citation precision."""
    evidence_by_section = {s: list(g.itertuples(index=False)) for s, g in gt.evidence.groupby("section")}
    golden_text = dict(zip(gt.judged["chunk_id"], gt.judged["chunk_text"]))
    rows: list[dict] = []
    for (section, method), retrieved in ranked.items():
        golden = evidence_by_section.get(section, [])
        for r in retrieved.itertuples(index=False):
            rows.append({"memo_id": gt.memo_id, "section": section, "method": method, "chunk_id": r.chunk_id,
                         "best_rank": int(r.rank), "golden": any(_is_hit(g, r, golden_text) for g in golden)})
    return pd.DataFrame(rows, columns=_CHUNK_HIT_COLUMNS)


_NO_CLAIMS = {"claims": 0, "covered": 0, "coverage": np.nan,
              "claims_extractive": 0, "covered_extractive": 0,
              "claims_synthesized": 0, "covered_synthesized": 0, "complete_synthesized": 0,
              "recall_macro": np.nan, "recall_micro": np.nan, "mrr": np.nan}


def _claim_metrics_by_k(hits: pd.DataFrame) -> list[dict]:
    """
    Coverage, cohort split, recall and MRR@k over one scope and method, one
    dict per k in _K_VALUES. Grouped once; each k is then numpy arithmetic on
    the per-group best ranks (a missing rank counts as infinitely deep) —
    regrouping per k made scoring ~20x slower for identical numbers.
    """
    if hits.empty:
        return [dict(_NO_CLAIMS) for _ in _K_VALUES]
    per_claim = hits.groupby("claim_id", sort=False).agg(bucket=("bucket", "first"), groups=("best_rank", "size"))
    n = len(per_claim)
    codes = pd.Categorical(hits["claim_id"], categories=per_claim.index).codes
    ranks = hits["best_rank"].fillna(np.inf).to_numpy(dtype=float)
    first = np.full(n, np.inf)
    np.minimum.at(first, codes, ranks)          # each claim's best rank over its groups
    groups = per_claim["groups"].to_numpy(dtype=float)
    extractive = (per_claim["bucket"] == "EXTRACTIVE").to_numpy()
    synthesized = (per_claim["bucket"] == "SYNTHESIZED").to_numpy()
    out = []
    for k in _K_VALUES:
        hit_groups = np.bincount(codes, weights=(ranks <= k).astype(float), minlength=n)
        covered = hit_groups > 0
        out.append({
            "claims": n, "covered": int(covered.sum()), "coverage": float(covered.mean()),
            "claims_extractive": int(extractive.sum()), "covered_extractive": int((covered & extractive).sum()),
            "claims_synthesized": int(synthesized.sum()), "covered_synthesized": int((covered & synthesized).sum()),
            "complete_synthesized": int(((hit_groups == groups) & synthesized).sum()),
            "recall_macro": float((hit_groups / groups).mean()),
            "recall_micro": float(hit_groups.sum() / groups.sum()),
            "mrr": float(np.where(first <= k, 1.0 / first, 0.0).mean()),
        })
    return out


def _precision_by_k(chunks: pd.DataFrame, recall_by_k: list[float]) -> list[dict]:
    """Citation precision@k pooled over one memo's sections, and F1 against
    macro recall, one dict per k. A floor, not a grade: the golden set covers
    only passages someone cited."""
    ranks = chunks["best_rank"].to_numpy(dtype=float)
    golden = chunks["golden"].to_numpy(dtype=bool)
    out = []
    for k, recall in zip(_K_VALUES, recall_by_k):
        within = ranks <= k
        retrieved, hit = int(within.sum()), int((within & golden).sum())
        precision = hit / retrieved if retrieved else np.nan
        if np.isnan(precision) or np.isnan(recall):
            f1 = np.nan
        elif precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        out.append({"retrieved": retrieved, "retrieved_golden": hit, "precision": precision, "f1": f1})
    return out


def compute_metrics(claim_hits: pd.DataFrame, chunk_hits: pd.DataFrame) -> pd.DataFrame:
    """Every metric for scope all / memo / section x method x k = 1.._RETRIEVE_DEPTH
    (_METRIC_COLUMNS). `passages` (distinct passages handed over at k, summed
    over the scope's sections) is filled everywhere. Precision and F1 are
    filled on memo rows only — the spec pools them per memo, never per
    section or across the corpus.

    A memo or section gets rows when it has a verifiable claim OR was
    searched: a section holding only unverifiable claims still had passages
    handed to the model, so its rows carry those passages with claims 0 and
    coverage NaN. Every passage count therefore adds up across scopes."""
    memos = sorted(set(claim_hits["memo_id"]) | set(chunk_hits["memo_id"]))
    sections = sorted(set(zip(claim_hits["memo_id"], claim_hits["section"]))
                      | set(zip(chunk_hits["memo_id"], chunk_hits["section"])))
    scopes = [("all", "ALL", "ALL", claim_hits, chunk_hits)]
    scopes += [("memo", m, "ALL", claim_hits[claim_hits["memo_id"] == m], chunk_hits[chunk_hits["memo_id"] == m])
               for m in memos]
    scopes += [("section", m, s,
                claim_hits[(claim_hits["memo_id"] == m) & (claim_hits["section"] == s)],
                chunk_hits[(chunk_hits["memo_id"] == m) & (chunk_hits["section"] == s)])
               for m, s in sections]
    rows: list[dict] = []
    for scope, memo_id, section, hits, chunks in scopes:
        for method in _RETRIEVAL_METHODS:
            method_chunks = chunks[chunks["method"] == method]
            chunk_ranks = method_chunks["best_rank"].to_numpy(dtype=float)
            claim_rows = _claim_metrics_by_k(hits[hits["method"] == method])
            precision_rows = (_precision_by_k(method_chunks, [r["recall_macro"] for r in claim_rows])
                              if scope == "memo" else [_NO_PRECISION] * len(_K_VALUES))
            for k, claim_row, precision_row in zip(_K_VALUES, claim_rows, precision_rows):
                rows.append({"scope": scope, "memo_id": memo_id, "section": section, "method": method,
                             "k": k, **claim_row, "passages": int((chunk_ranks <= k).sum()), **precision_row})
    return pd.DataFrame(rows, columns=_METRIC_COLUMNS)


# %% [markdown]
# ## 3. Re-review candidates and run records
#
# A run is one scoring of one retrieval_results.parquet, written to
# eval_runs/<run_id>/ and never overwritten. The k and method levers are
# views inside one run; the phrase lever makes a new run.

# %%
_CANDIDATE_COLUMNS = list(_CLAIM_QUERY_COLUMNS)
_RUN_TABLES = ("claims", "evidence", "claim_hits", "chunk_hits", "metrics", "candidates")
_RUN_LABEL_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _round_half_away(x: float) -> int:
    """x rounded to a whole number, halves away from zero (12.5 -> 13,
    -2.5 -> -3). Python's round() sends halves to the even neighbour, so 1/8
    would show as 12% but 3/8 as 38%."""
    return int(np.sign(x) * np.floor(abs(x) + 0.5))


def _pct(num, den) -> str:
    """A percentage with its counts, as every number on the report is shown."""
    return f"{_round_half_away(100 * num / den)}% ({int(num)}/{int(den)})" if den else "n/a (0/0)"


def _metric_row(metrics: pd.DataFrame, scope: str, memo_id: str, section: str, method: str, k: int):
    """The one metrics row for a scope, method and k, or None if absent."""
    m = metrics
    rows = m[(m["scope"] == scope) & (m["memo_id"] == memo_id) & (m["section"] == section)
             & (m["method"] == method) & (m["k"] == k)]
    return rows.iloc[0] if len(rows) else None


def results_phrases(results: pd.DataFrame, memo_id: str) -> dict[str, list[str]]:
    """The phrases the results were actually retrieved with:
    {section: [phrase, ...]} in phrase_index order, read from the dense rows
    (dense always returns rows for every phrase)."""
    rows = (results[(results["memo_id"] == memo_id) & (results["method"] == "dense")]
            [["section", "phrase_index", "phrase"]].drop_duplicates()
            .sort_values(["section", "phrase_index"]))
    return {section: g["phrase"].tolist() for section, g in rows.groupby("section", sort=False)}


def check_results_match_phrases(memo_id: str, results: pd.DataFrame, phrase_path: str) -> dict[str, list[str]]:
    """
    Refuses when retrieval/<memo_id>.yaml no longer holds the phrases the
    results were retrieved with — an edited phrase file whose retrieve was
    skipped or failed. Scoring then would stamp new phrases on old results.
    Returns the phrases, taken from the results.

    The file is read with retrieve's own reader (_read_phrase_config), so a
    phrase is compared exactly as retrieve stored it — stripped of
    surrounding whitespace, which a quoted phrase or a folded '>' line can
    carry. Content that reader rejects (its ValueError: a wrong memo_id, a
    blank phrase, …) is refused with its message.
    """
    try:
        configured = _read_phrase_config(phrase_path)["sections"]
    except ValueError as e:
        raise EvalInputError([str(e)])
    used = results_phrases(results, memo_id)
    if configured != used:
        differ = sorted(s for s in set(configured) | set(used) if configured.get(s) != used.get(s))
        raise EvalInputError([
            f"{phrase_path}: the phrases for {', '.join(repr(s) for s in differ)} differ from those "
            f"the retrieval results were produced with — re-run python retrieval_pipeline.py retrieve, "
            f"or the run would score old results under new phrases"
        ])
    return used


def load_claim_queries(path: str, gts: list[GroundTruth], index_dir: str = "retrieval_index") -> pd.DataFrame:
    """claim_queries.parquet (retrieval_pipeline.py recheck), refused when
    missing — the unverifiable count is never shown without it — when its
    claims no longer match a memo's census (claims edited since recheck), or
    when it was not produced from a memo's current index and model
    (check_provenance)."""
    if not os.path.exists(path):
        raise EvalInputError([
            f"{path}: not found — run python retrieval_pipeline.py recheck "
            f"(the unverifiable count is never reported without it)"
        ])
    try:
        df = pd.read_parquet(path)
        provenance = read_provenance(path)
    except (OSError, ValueError) as e:
        raise EvalInputError([_unreadable(path, e, "recheck")])
    absent = [c for c in _CLAIM_QUERY_COLUMNS if c not in df.columns]
    if absent:
        raise EvalInputError([f"{path}: missing column(s) {absent}"])
    problems = []
    for gt in gts:
        have = set(df.loc[df["memo_id"] == gt.memo_id, "claim_id"])
        want = set(gt.claims["claim_id"])
        if have != want:
            problems.append(
                f"{path}: {gt.memo_id} has {len(want - have)} claim(s) with no recheck row and "
                f"{len(have - want)} recheck claim(s) outside the census — stale; "
                f"re-run python retrieval_pipeline.py recheck"
            )
    if provenance is None:
        problems.append(_no_provenance(path, "recheck"))
    else:
        for gt in gts:
            try:
                check_provenance(gt.memo_id, provenance, path, "recheck", index_dir)
            except EvalInputError as e:
                problems += e.problems
    if problems:
        raise EvalInputError(problems)
    return df


def rereview_candidates(gt: GroundTruth, claim_queries: pd.DataFrame) -> pd.DataFrame:
    """
    For each UNVERIFIABLE claim: the chunks meaning-based search offers for
    its text that no human judged for that claim, strongest (highest cosine)
    first. A worklist for re-review, never a count of errors — the eval
    cannot know whether a chunk supports the claim; only a person can.
    """
    unverifiable = set(gt.claims.loc[gt.claims["bucket"] == "UNVERIFIABLE", "claim_id"])
    judged = set(zip(gt.judged["claim_id"], gt.judged["chunk_id"]))
    q = claim_queries[claim_queries["memo_id"] == gt.memo_id]
    new = pd.Series([(c, k) not in judged for c, k in zip(q["claim_id"], q["chunk_id"])], index=q.index, dtype=bool)
    out = q[q["claim_id"].isin(unverifiable) & new]
    return out.sort_values(["score", "claim_id", "rank"], ascending=[False, True, True], kind="stable")[
        _CANDIDATE_COLUMNS].reset_index(drop=True)


def golden_fingerprint(gts: list[GroundTruth]) -> str:
    """A hash of the ground truth's content — buckets, evidence (with
    human_added, which decides how a span is split into quotes), judged
    pairs — not of the xlsx bytes, so re-saving a sheet unchanged keeps it
    and changing one verdict changes it. The ruler two runs must share."""
    parts = []
    for gt in sorted(gts, key=lambda g: g.memo_id):
        parts.append(gt.claims[["claim_id", "bucket"]].sort_values("claim_id").to_csv(index=False))
        parts.append(gt.evidence[["claim_id", "chunk_id", "tag", "evidence_span", "human_added"]]
                     .sort_values(["claim_id", "chunk_id"]).to_csv(index=False))
        parts.append(gt.judged[["claim_id", "chunk_id"]].sort_values(["claim_id", "chunk_id"]).to_csv(index=False))
    return _sha256("\n".join(parts).encode("utf-8"))


@dataclass
class Run:
    """One scored run, as read back from eval_runs/<run_id>/."""
    run_id: str
    meta: dict
    claims: pd.DataFrame
    evidence: pd.DataFrame
    claim_hits: pd.DataFrame
    chunk_hits: pd.DataFrame
    metrics: pd.DataFrame
    candidates: pd.DataFrame


def _write_run(runs_dir: str, run_id: str, tables: dict, meta: dict) -> str:
    """
    Writes the run into a temporary folder of its own (a unique
    <run_id>.….tmp, so two runs never touch each other's work), then renames
    it to <run_id>. The rename is what claims the id: if another run took it
    in the meantime, the next free suffix (_2, _3, …) is used and meta.json
    rewritten to match. An interrupted write never leaves a half run under a
    real name, and no run is ever overwritten. A write that fails removes
    its temporary folder (it holds client claim and chunk text, and
    latest_run_id would never show it). Returns the run directory.
    """
    os.makedirs(runs_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=f"{run_id}.", suffix=".tmp", dir=runs_dir)
    try:
        for name in _RUN_TABLES:
            tables[name].to_parquet(os.path.join(tmp, f"{name}.parquet"), index=False)
        base, n = run_id, 2
        while True:
            with open(os.path.join(tmp, "meta.json"), "w", encoding="utf-8") as f:
                json.dump({**meta, "run_id": run_id}, f, indent=2, ensure_ascii=False)
            run_dir = os.path.join(runs_dir, run_id)
            try:
                os.rename(tmp, run_dir)   # refused when run_dir already holds a run
                return run_dir
            except OSError:
                if not os.path.isdir(run_dir):
                    raise
                run_id, n = f"{base}_{n}", n + 1
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def score_run(
    label: str | None = None,
    *,
    claims_dir: str = "claims",
    reviewed_dir: str = "reviewed",
    results_path: str = "retrieval_results.parquet",
    index_dir: str = "retrieval_index",
    claim_queries_path: str = "claim_queries.parquet",
    retrieval_dir: str = "retrieval",
    runs_dir: str = "eval_runs",
    now: datetime | None = None,
) -> str:
    """
    Scores every memo with a reviewed/<memo_id>.xlsx and writes one run to
    eval_runs/<YYYY-MM-DD_HHMMSS>[_<label>]/ (tables _RUN_TABLES + meta.json).
    Returns the run directory.

    Every input is checked first and every problem, from every memo, raised
    together as EvalInputError: ground truth (load_ground_truth), index
    parity (text and doc_id), sections without results or with a method
    missing, results retrieved with the phrases on disk
    (check_results_match_phrases), results and claim queries produced from
    each memo's current index and its model, at the depth each is read to
    (_RETRIEVE_DEPTH for the results, _TOP_K for the claim queries —
    check_provenance), which also rules out mixed models across memos,
    claim_queries.parquet present and current. Each check runs even when
    another failed; only index parity and the sections check skip a memo
    whose ground truth could not be read, since they need it. A memo in the
    results without a reviewed sheet is logged and skipped. In reviewed/, an
    Excel lock file ("~$…") or a hidden file is ignored silently; finalize's
    temporary "<memo_id>.<random>.tmp.xlsx", left by an interrupted run, and
    any other .xlsx whose name is not a valid memo id (e.g. "MEMO-005
    copy.xlsx") are skipped with a WARNING naming them.
    """
    if label is not None and not _RUN_LABEL_RE.match(label):
        raise EvalInputError([f"run label {label!r}: use only letters, digits, '.', '_' or '-'"])
    names = sorted(os.listdir(reviewed_dir)) if os.path.isdir(reviewed_dir) else []
    sheets = [f for f in names if f.endswith(".xlsx") and not f.startswith(("~$", "."))]
    for f in [f for f in sheets if f.endswith(".tmp.xlsx")]:
        logger.warning("eval: %s/%s is a temporary file an interrupted finalize left behind — not scored; "
                       "delete it", reviewed_dir, f)
    sheets = [f for f in sheets if not f.endswith(".tmp.xlsx")]
    memo_ids = [f[: -len(".xlsx")] for f in sheets if _valid_memo_id(f[: -len(".xlsx")])]
    for f in sheets:
        if not _valid_memo_id(f[: -len(".xlsx")]):
            logger.warning("eval: %s/%s is not named <memo_id>.xlsx — not scored", reviewed_dir, f)
    if not memo_ids:
        raise EvalInputError([f"{reviewed_dir}/: no reviewed sheets — run python tag_pipeline.py finalize <memo_id>"])

    problems: list[str] = []
    results = results_provenance = None
    try:
        results = load_results(results_path)
        try:
            results_provenance = read_provenance(results_path)
        except (OSError, ValueError) as e:
            raise EvalInputError([_unreadable(results_path, e, "retrieve")])
        if results_provenance is None:
            problems.append(_no_provenance(results_path, "retrieve"))
    except EvalInputError as e:
        problems += e.problems
    gts: list[GroundTruth] = []
    phrases: dict[str, dict[str, list[str]]] = {}
    for memo_id in memo_ids:
        # Each check has its own try, so one failing never hides another's
        # problems; only the checks that need the ground truth wait for it.
        phrase_path = os.path.join(retrieval_dir, f"{memo_id}.yaml")
        if not os.path.exists(phrase_path):
            problems.append(f"{phrase_path}: not found — every scored memo needs its phrase config")
        gt = None
        try:
            gt = load_ground_truth(memo_id, claims_dir, reviewed_dir)
            gts.append(gt)
        except EvalInputError as e:
            problems += e.problems
        if gt is not None:
            try:
                check_index_parity(gt, index_dir)
            except EvalInputError as e:
                problems += e.problems
            if results is not None:
                try:
                    check_sections_retrieved(gt, results)
                except EvalInputError as e:
                    problems += e.problems
        if results_provenance is not None:
            try:
                check_provenance(memo_id, results_provenance, results_path, "retrieve", index_dir)
            except EvalInputError as e:
                problems += e.problems
        if results is not None and os.path.exists(phrase_path):
            try:
                phrases[memo_id] = check_results_match_phrases(memo_id, results, phrase_path)
            except EvalInputError as e:
                problems += e.problems
    queries = None
    if gts:
        try:
            queries = load_claim_queries(claim_queries_path, gts, index_dir)
        except EvalInputError as e:
            problems += e.problems
    if problems:
        raise EvalInputError(problems)
    skipped = sorted(set(results["memo_id"]) - set(memo_ids))
    if skipped:
        logger.warning("eval: retrieval results also cover %s with no reviewed sheet — not scored", skipped)

    claim_hits, chunk_hits, candidates = [], [], []
    for gt in gts:
        ranked = ranked_chunks(results, gt.memo_id)
        claim_hits.append(score_claims(gt, ranked))
        chunk_hits.append(score_chunks(gt, ranked))
        candidates.append(rereview_candidates(gt, queries))
    claim_hits_df = pd.concat(claim_hits, ignore_index=True)
    chunk_hits_df = pd.concat(chunk_hits, ignore_index=True)
    tables = {
        "claims": pd.concat([g.claims for g in gts], ignore_index=True),
        "evidence": pd.concat([g.evidence for g in gts], ignore_index=True),
        "claim_hits": claim_hits_df,
        "chunk_hits": chunk_hits_df,
        "metrics": compute_metrics(claim_hits_df, chunk_hits_df),
        "candidates": pd.concat(candidates, ignore_index=True)
                        .sort_values(["score", "memo_id", "claim_id", "rank"],
                                     ascending=[False, True, True, True], kind="stable")
                        .reset_index(drop=True),
    }

    now = now or datetime.now()
    run_id = now.strftime("%Y-%m-%d_%H%M%S") + (f"_{label}" if label else "")   # _write_run adds _2, … if taken
    with open(results_path, "rb") as f:
        results_hash = _sha256(f.read())
    meta = {
        "run_id": run_id,
        "created": now.isoformat(timespec="seconds"),
        "memos": memo_ids,
        "embedding_model": results_provenance["model"],   # what retrieve embedded the phrases with
        "golden_hash": golden_fingerprint(gts),
        "phrase_hash": _sha256(json.dumps(phrases, sort_keys=True, ensure_ascii=False).encode("utf-8")),
        "results_hash": results_hash,
        "results_provenance": results_provenance,
        "phrases": phrases,
        "phrase_counts": {m: {s: len(p) for s, p in secs.items()} for m, secs in phrases.items()},
        "top_k": _TOP_K,
        "depth": _RETRIEVE_DEPTH,
    }
    run_dir = _write_run(runs_dir, run_id, tables, meta)
    logger.info("eval: wrote %s (%d memo(s))", run_dir, len(memo_ids))
    return run_dir


def latest_run_id(runs_dir: str = "eval_runs") -> str:
    """
    The newest run's id. Only folders holding a meta.json are runs (a stray
    folder is ignored), ordered by the meta's `created` time and then by when
    meta.json was written — not by folder name, where "_10" sorts before "_9"
    and a folder like "zz-archive" after every timestamp.
    """
    runs = []
    for d in (os.listdir(runs_dir) if os.path.isdir(runs_dir) else []):
        meta_path = os.path.join(runs_dir, d, "meta.json")
        if d.endswith(".tmp") or not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path, encoding="utf-8") as f:
                created = json.load(f)["created"]
        except (OSError, ValueError, KeyError) as e:
            raise EvalInputError([f"{meta_path}: unreadable ({e}) — fix or remove that run folder"])
        runs.append((created, os.stat(meta_path).st_mtime_ns, d))
    if not runs:
        raise EvalInputError([f"{runs_dir}/: no runs yet — run python eval_pipeline.py score"])
    return max(runs)[2]


def load_run(run_id: str, runs_dir: str = "eval_runs") -> Run:
    """Reads eval_runs/<run_id>/ back; run_id "latest" means latest_run_id().
    A missing or unreadable file is refused as EvalInputError naming it."""
    run_id = latest_run_id(runs_dir) if run_id == "latest" else run_id
    run_dir = os.path.join(runs_dir, run_id)
    if not os.path.isdir(run_dir):
        raise EvalInputError([f"{run_dir}: no such run"])
    try:
        with open(os.path.join(run_dir, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        tables = {name: pd.read_parquet(os.path.join(run_dir, f"{name}.parquet")) for name in _RUN_TABLES}
    except (OSError, ValueError) as e:
        raise EvalInputError([f"{run_dir}: incomplete or damaged run ({e}) — score it again"])
    return Run(run_id=run_id, meta=meta, **tables)


def check_comparable(run: Run, baseline: Run) -> None:
    """Refuses to compare runs measured with different rulers: the golden set
    and the embedding model must match. The phrase config is expected to
    differ — it is the phrase lever. Method and k are compared within one
    run, never across runs."""
    problems = []
    if run.meta["golden_hash"] != baseline.meta["golden_hash"]:
        problems.append(f"{run.run_id} and {baseline.run_id} were scored against different golden sets — not comparable")
    if run.meta["embedding_model"] != baseline.meta["embedding_model"]:
        problems.append(
            f"{run.run_id} used {run.meta['embedding_model']!r}, {baseline.run_id} used "
            f"{baseline.meta['embedding_model']!r} — not comparable"
        )
    if problems:
        raise EvalInputError(problems)


# %% [markdown]
# ## 4. The report — one self-contained HTML page per (run, method, k)
#
# Written from run records only: no external asset, no script, opens with no
# network. Every percentage carries its counts. The unverifiable count never
# appears without the re-review sentence beside it (spec: "Unverifiable, and
# its honesty check").

# %%
_METHOD_LABEL = {"dense": "meaning search (dense)", "keyword": "keyword search (BM25)",
                 "both": "both combined (RRF)"}
_METHOD_COLOR = {"dense": "#2563eb", "keyword": "#d97706", "both": "#059669"}
_MISS_CATEGORIES = ("found by another method at this depth",
                    f"found only deeper (by k={_RETRIEVE_DEPTH})",
                    f"not found by any method within {_RETRIEVE_DEPTH}")
_CANDIDATES_SHOWN = 25
_CSS = """
body{font:15px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#1f2937;background:#fafaf9;
     max-width:1040px;margin:0 auto;padding:24px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 8px;border-bottom:1px solid #e5e7eb;padding-bottom:4px}
.sub{color:#6b7280;margin:0 0 12px}.frame{background:#eef2ff;border-left:4px solid #6366f1;padding:8px 12px}
.tiles{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}.tile{background:#fff;border:1px solid #e5e7eb;
     border-radius:8px;padding:12px 16px;min-width:180px}.tile .v{font-size:26px;font-weight:600}
.tile .l{color:#6b7280;font-size:13px}.delta{font-size:13px;color:#059669}.delta.neg{color:#dc2626}
.note{background:#fffbeb;border-left:4px solid #f59e0b;padding:8px 12px}
table{border-collapse:collapse;width:100%;font-size:13px;background:#fff}
th,td{border:1px solid #e5e7eb;padding:4px 8px;text-align:left;vertical-align:top}
th{background:#f3f4f6}.bar{background:#e5e7eb;height:10px;width:160px;display:inline-block;vertical-align:middle}
.bar span{display:block;height:10px;background:#2563eb}.quote{color:#374151;font-style:italic}
.scroll{overflow-x:auto}
"""


def miss_taxonomy(run: Run, method: str, k: int) -> pd.DataFrame:
    """
    One row per verifiable claim NOT covered at (method, k), with why:
      - "found by another method at this depth": another method covers it at k;
      - "found only deeper (by k=_RETRIEVE_DEPTH)": no method at k, some
        method by k=_RETRIEVE_DEPTH;
      - "not found by any method within _RETRIEVE_DEPTH".
    """
    if run.claim_hits.empty:
        return pd.DataFrame(columns=["claim_id", "bucket", "category"])
    first = (run.claim_hits.groupby(["claim_id", "method"])["best_rank"].min()
             .unstack("method").reindex(columns=list(_RETRIEVAL_METHODS)))
    buckets = run.claim_hits.groupby("claim_id")["bucket"].first()
    rows = []
    for claim_id, ranks in first.iterrows():
        if ranks[method] <= k:
            continue
        if any(ranks[m] <= k for m in _RETRIEVAL_METHODS if m != method):
            category = _MISS_CATEGORIES[0]
        elif (ranks <= _RETRIEVE_DEPTH).any():
            category = _MISS_CATEGORIES[1]
        else:
            category = _MISS_CATEGORIES[2]
        rows.append({"claim_id": claim_id, "bucket": buckets[claim_id], "category": category})
    return pd.DataFrame(rows, columns=["claim_id", "bucket", "category"])


def _e(value) -> str:
    """HTML-escaped text; control characters pypdf may leave are removed first."""
    return html.escape(_excel_safe(value) if isinstance(value, str) else str(value))


def _delta(current, base) -> str:
    """'+33 pts vs <baseline>' for two coverage fractions, or '' without a baseline."""
    if base is None or current is None or pd.isna(base.coverage) or pd.isna(current.coverage):
        return ""
    points = _round_half_away(100 * (current.coverage - base.coverage))
    cls = "delta neg" if points < 0 else "delta"
    return f'<div class="{cls}">{points:+d} pts vs baseline</div>'


def _coverage_svg(run: Run, method: str, k: int, baseline: Run | None) -> str:
    """Coverage@k, k = 1.._RETRIEVE_DEPTH, one line per method (the chosen
    one heavier), a marker at k, and the baseline's line for the chosen
    method dashed."""
    width, height, left, right, top, bottom = 640, 260, 48, 16, 16, 36

    def x(kk):
        return left + (kk - 1) * (width - left - right) / (_RETRIEVE_DEPTH - 1)

    def y(v):
        return top + (1 - v) * (height - top - bottom)

    def points(metrics, m):
        out = []
        for kk in _K_VALUES:
            row = _metric_row(metrics, "all", "ALL", "ALL", m, kk)
            if row is not None and not pd.isna(row.coverage):
                out.append(f"{x(kk):.1f},{y(row.coverage):.1f}")
        return " ".join(out)

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
             f'aria-label="Claim coverage by passages per search phrase, for each search method">']
    for v in (0, 0.25, 0.5, 0.75, 1):
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#e5e7eb"/>'
                     f'<text x="{left - 6}" y="{y(v) + 4:.1f}" font-size="11" text-anchor="end" fill="#6b7280">'
                     f'{int(v * 100)}%</text>')
    for kk in sorted({1, *range(5, _RETRIEVE_DEPTH + 1, 5), _RETRIEVE_DEPTH}):
        parts.append(f'<text x="{x(kk):.1f}" y="{height - bottom + 16}" font-size="11" text-anchor="middle" '
                     f'fill="#6b7280">{kk}</text>')
    parts.append(f'<text x="{(left + width - right) / 2:.1f}" y="{height - 4}" font-size="11" text-anchor="middle" '
                 f'fill="#6b7280">k — passages per search phrase</text>')
    parts.append(f'<line x1="{x(k):.1f}" x2="{x(k):.1f}" y1="{top}" y2="{height - bottom}" '
                 f'stroke="#9ca3af" stroke-dasharray="2 3"/>')
    if baseline is not None:
        parts.append(f'<polyline fill="none" stroke="#9ca3af" stroke-width="2" stroke-dasharray="6 4" '
                     f'points="{points(baseline.metrics, method)}"/>')
    for m in _RETRIEVAL_METHODS:
        weight = 3 if m == method else 1.5
        parts.append(f'<polyline fill="none" stroke="{_METHOD_COLOR[m]}" stroke-width="{weight}" '
                     f'points="{points(run.metrics, m)}"/>')
    for i, m in enumerate(_RETRIEVAL_METHODS):
        parts.append(f'<text x="{left + 8}" y="{top + 14 + 14 * i}" font-size="12" fill="{_METHOD_COLOR[m]}">'
                     f'{_e(_METHOD_LABEL[m])}</text>')
    if baseline is not None:
        parts.append(f'<text x="{left + 8}" y="{top + 14 + 14 * 3}" font-size="12" fill="#6b7280">'
                     f'baseline {_e(baseline.run_id)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _phrase_count_note(run: Run, baseline: Run | None) -> str:
    """A warning when the baseline searched some section with a different
    number of phrases: more phrases hand the model more passages, so part of
    any change there is simply more text, not better questions."""
    if baseline is None:
        return ""
    counts, base_counts = run.meta.get("phrase_counts", {}), baseline.meta.get("phrase_counts", {})
    # Both runs' sections: one only the baseline searched lost its passages too.
    keys = sorted({(m, s) for c in (counts, base_counts) for m, sections in c.items() for s in sections})
    changed = [f"{m} / {s}" for m, s in keys if counts.get(m, {}).get(s) != base_counts.get(m, {}).get(s)]
    if not changed:
        return ""
    more = "…" if len(changed) > 5 else ""
    return (f'<p class="note">The baseline used a different number of search phrases in {len(changed)} '
            f'section(s) ({_e(", ".join(changed[:5]))}{more}). More phrases hand the model more passages, '
            f'so part of any change there is simply more text, not better questions.</p>')


def _bar(row) -> str:
    if row is None or not row.claims:
        return "n/a"
    pct = 100 * row.covered / row.claims
    return f'<span class="bar"><span style="width:{pct:.0f}%"></span></span> {_pct(row.covered, row.claims)}'


def render_report(run: Run, *, method: str = "dense", k: int = _TOP_K, baseline: Run | None = None) -> str:
    """The whole report for one run at (method, k), optionally with deltas
    against a baseline run (check_comparable must pass). Returns the HTML."""
    if method not in _RETRIEVAL_METHODS:
        raise ValueError(f"method {method!r}: use one of {', '.join(_RETRIEVAL_METHODS)}")
    if not 1 <= k <= _RETRIEVE_DEPTH:
        raise ValueError(f"k={k}: use 1..{_RETRIEVE_DEPTH}")
    if baseline is not None:
        check_comparable(run, baseline)

    def row(r, scope="all", memo_id="ALL", section="ALL"):
        return _metric_row(r.metrics, scope, memo_id, section, method, k)

    head = row(run)
    base = row(baseline) if baseline is not None else None
    covered, claims = (int(head.covered), int(head.claims)) if head is not None else (0, 0)
    ext = (int(head.covered_extractive), int(head.claims_extractive)) if head is not None else (0, 0)
    syn = (int(head.covered_synthesized), int(head.claims_synthesized)) if head is not None else (0, 0)
    syn_complete = int(head.complete_synthesized) if head is not None else 0
    passages = int(head.passages) if head is not None else 0
    # section rows at this (method, k) — every section searched, verifiable claims or not
    section_rows = run.metrics[(run.metrics["scope"] == "section") & (run.metrics["method"] == method)
                               & (run.metrics["k"] == k)]
    n_sections = int((section_rows["passages"] > 0).sum())
    census = len(run.claims)
    unverifiable = int((run.claims["bucket"] == "UNVERIFIABLE").sum())
    text_by_claim = dict(zip(run.claims["claim_id"], run.claims["claim_text"]))
    out: list[str] = []

    out.append(f"<h1>Retrieval eval — {_e(run.run_id)}</h1>")
    out.append(f'<p class="sub">{_e(", ".join(run.meta["memos"]))} · embedding model '
               f'{_e(run.meta["embedding_model"])} · {_e(_METHOD_LABEL[method])} · k = {k} passages per search phrase'
               + (f" · compared with {_e(baseline.run_id)}" if baseline is not None else "") + "</p>")
    out.append('<p class="frame">This measures a retrieval <b>prototype</b> against a hand-reviewed ruler. '
               'It shows what the instrument produces — it is not a verdict on any production system.</p>')

    out.append('<div class="tiles">')
    out.append(f'<div class="tile"><div class="l">claim coverage</div><div class="v">{_pct(covered, claims)}</div>'
               f'{_delta(head, base)}<div class="l">verifiable claims with supporting evidence retrieved</div></div>')
    out.append(f'<div class="tile"><div class="l">stated in one passage (extractive)</div>'
               f'<div class="v">{_pct(*ext)}</div></div>')
    out.append(f'<div class="tile"><div class="l">needs several passages (synthesized): at least one retrieved</div>'
               f'<div class="v">{_pct(*syn)}</div>'
               f'<div class="l">all pieces retrieved: {_pct(syn_complete, syn[1])}</div></div>')
    out.append(f'<div class="tile"><div class="l">passages handed to the model</div><div class="v">{passages}</div>'
               f'<div class="l">across {n_sections} sections, all search phrases together</div></div>')
    out.append("</div>")
    if claims:
        out.append(f'<p class="sub">With {claims} verifiable claims, one claim moves coverage by about '
                   f'{_round_half_away(100 / claims)} points — treat small changes as noise.</p>')
    out.append(_phrase_count_note(run, baseline))
    with_candidates = run.candidates["claim_id"].nunique()
    out.append(f'<p class="note">{unverifiable} of {census} claims were not found in the sources by this process. '
               f'For {with_candidates} of them, meaning-based search offers passages the review never showed a '
               f'human — {len(run.candidates)} listed under Re-review candidates. They are not scored above.</p>')

    out.append("<h2>Coverage as each search phrase returns more passages</h2>")
    out.append(_coverage_svg(run, method, k, baseline))

    out.append("<h2>By memo and section</h2>")
    table = ["<table><tr><th>memo</th><th>section</th><th>claim coverage</th><th>passages</th>"
             "<th>change</th></tr>"]
    for memo_id in run.meta["memos"]:
        m = row(run, "memo", memo_id)
        mb = row(baseline, "memo", memo_id) if baseline is not None else None
        # every claims-file section, plus any section searched that the claims file lacks
        sections = sorted(set(run.claims.loc[run.claims["memo_id"] == memo_id, "section"])
                          | set(section_rows.loc[section_rows["memo_id"] == memo_id, "section"]))
        table.append(f"<tr><th>{_e(memo_id)}</th><th>all</th><td>{_bar(m)}</td>"
                     f"<td>{int(m.passages) if m is not None else 0}</td><td>{_delta(m, mb)}</td></tr>")
        for section in sections:
            s = row(run, "section", memo_id, section)
            sb = row(baseline, "section", memo_id, section) if baseline is not None else None
            table.append(f"<tr><td></td><td>{_e(section)}</td><td>{_bar(s)}</td>"
                         f"<td>{int(s.passages) if s is not None else 0}</td><td>{_delta(s, sb)}</td></tr>")
    table.append("</table>")
    out.append("".join(table))

    out.append(f"<h2>Why the missed claims were missed (k = {k})</h2>")
    misses = miss_taxonomy(run, method, k)
    table = ["<table><tr><th>reason</th><th>extractive</th><th>synthesized</th></tr>"]
    for category in _MISS_CATEGORIES:
        rows = misses[misses["category"] == category]
        table.append(f"<tr><td>{_e(category)}</td><td>{int((rows['bucket'] == 'EXTRACTIVE').sum())}</td>"
                     f"<td>{int((rows['bucket'] == 'SYNTHESIZED').sum())}</td></tr>")
    table.append("</table>")
    out.append("".join(table))

    out.append("<h2>Traced examples</h2>")
    hits = run.claim_hits[run.claim_hits["method"] == method]
    best = hits.sort_values(["best_rank"], kind="stable").groupby("claim_id").first()
    first_quote = run.evidence.groupby("claim_id").first()
    quote_of = run.evidence.set_index(["claim_id", "chunk_id"])["evidence_span"]
    table = ["<table><tr><th>memo</th><th>claim</th><th>evidence a human confirmed</th><th>retrieved?</th></tr>"]
    for memo_id in run.meta["memos"]:
        verifiable = run.claims[(run.claims["memo_id"] == memo_id) & (run.claims["bucket"] != "UNVERIFIABLE")]
        found = [c for c in verifiable["claim_id"] if best.loc[c, "best_rank"] <= k]
        missed = [c for c in verifiable["claim_id"] if not best.loc[c, "best_rank"] <= k]
        for claim_id in found[:1] + missed[:1]:
            b = best.loc[claim_id]
            if b.best_rank <= k:
                # quote the evidence row that was actually found — a claim can have many
                golden = b.best_golden_chunk_id
                quote = quote_of.loc[(claim_id, golden)]
                g, r = _chunk_index_of(golden), _chunk_index_of(b.best_chunk_id)
                if b.best_chunk_id == golden:
                    where = "the same passage"
                elif g is not None and r is not None and abs(g - r) == 1:
                    where = f"the neighbouring passage {_e(b.best_chunk_id)}, which contains this quote"
                else:   # _is_hit's identical-text case: another place holding the same text
                    where = f"passage {_e(b.best_chunk_id)}, whose text is identical to this quote"
                outcome = f"yes — rank {int(b.best_rank)}: {where}"
            else:
                golden = first_quote.loc[claim_id, "chunk_id"]
                quote = first_quote.loc[claim_id, "evidence_span"]
                outcome = f"no — not in the top {k} for any search phrase"
            table.append(f"<tr><td>{_e(memo_id)}</td><td>{_e(text_by_claim[claim_id])}</td>"
                         f'<td class="quote">{_e(quote)} ({_e(golden)})</td><td>{outcome}</td></tr>')
    table.append("</table>")
    out.append(f'<div class="scroll">{"".join(table)}</div>')

    out.append("<h2>Re-review candidates</h2>")
    out.append(f"<p>{len(run.candidates)} passage(s) for {run.candidates['claim_id'].nunique()} unverifiable "
               f"claim(s), strongest first; the first {min(_CANDIDATES_SHOWN, len(run.candidates))} shown. "
               f"A worklist for a person, not a count of errors.</p>")
    table = ["<table><tr><th>score</th><th>memo</th><th>claim</th><th>passage</th></tr>"]
    for c in run.candidates.head(_CANDIDATES_SHOWN).itertuples(index=False):
        table.append(f"<tr><td>{c.score:.3f}</td><td>{_e(c.memo_id)}</td><td>{_e(c.claim_text)}</td>"
                     f'<td class="quote">{_e(c.chunk_text[:300])}{"…" if len(c.chunk_text) > 300 else ""} '
                     f'({_e(c.chunk_id)})</td></tr>')
    table.append("</table>")
    out.append(f'<div class="scroll">{"".join(table)}</div>')

    out.append("<h2>Details</h2>")
    detail = run.metrics[(run.metrics["method"] == method) & (run.metrics["k"] == k)].drop(columns=["mrr"])
    out.append('<div class="scroll">' + detail.to_html(index=False, escape=True, na_rep="—",
                                                       float_format=lambda v: f"{v:.3f}") + "</div>")
    out.append('<p class="sub">Precision is <b>citation</b> precision, a floor: the golden set covers only passages '
               'someone cited, so a relevant but uncited passage counts against it. It is pooled per memo. '
               'Passages are counted per section and summed. Macro recall averages, per claim, the share of its '
               'separate pieces of evidence retrieved; where duplicate source documents repeat a passage, each copy '
               'is its own piece, so a claim found in only one copy reaches at most 50% (coverage is unaffected when '
               'both copies were cited; a copy nobody cited is never credited). '
               'MRR is stored in the run but not shown: it takes '
               "each claim's best rank over several separate phrase rankings, so it rises with the number of "
               'phrases and is not comparable between sections.</p>')

    return ("<!doctype html><html><head><meta charset=\"utf-8\">"
            f"<title>Retrieval eval {_e(run.run_id)}</title><style>{_CSS}</style></head><body>"
            + "".join(out) + "</body></html>")


def write_report(run_id: str, *, method: str = "dense", k: int = _TOP_K,
                 baseline_id: str | None = None, runs_dir: str = "eval_runs") -> str:
    """Renders and writes eval_runs/<run_id>/report_<method>_k<k>[_vs_<baseline>].html; returns the path."""
    run = load_run(run_id, runs_dir)
    baseline = load_run(baseline_id, runs_dir) if baseline_id else None
    name = f"report_{method}_k{k}" + (f"_vs_{baseline.run_id}" if baseline is not None else "") + ".html"
    page = render_report(run, method=method, k=k, baseline=baseline)
    path = os.path.join(runs_dir, run.run_id, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path


def show_report(run_id: str | None = None, method: str = "dense", k: int = _TOP_K,
                baseline: str | None = None, runs_dir: str = "eval_runs") -> str:
    """For the notebook demo: write the report (run_id None means the latest
    run), display it inline when running under IPython, and return its path."""
    path = write_report(run_id or "latest", method=method, k=k, baseline_id=baseline, runs_dir=runs_dir)
    try:
        # Optional (notebook only), so not in requirements.txt; the ignore keeps
        # pyright quiet where IPython is not installed, as in CI.
        from IPython.display import HTML, display  # pyright: ignore[reportMissingImports]
    except ImportError:
        pass
    else:
        with open(path, encoding="utf-8") as f:
            display(HTML(f.read()))
    print(f"report: {path}")
    return path


# %% [markdown]
# ## Demo controls
#
# In a notebook, flip RUN_DEMO_REPORT to True and re-run this cell after
# changing METHOD or K — nothing is recomputed, the stored run is redrawn.
# Kept False so a plain script run and the test suite never render.

# %%
RUN_DEMO_REPORT = False
if RUN_DEMO_REPORT:
    RUN = None          # None = the latest run
    METHOD = "dense"    # dense | keyword | both
    K = 5               # passages per search phrase, 1.._RETRIEVE_DEPTH
    BASELINE = None     # a run id to show changes against, or None
    show_report(RUN, method=METHOD, k=K, baseline=BASELINE)


# %% [markdown]
# ## Command line

# %%
_USAGE = ("usage: python eval_pipeline.py score [label] | "
          f"report <run_id|latest> [baseline_run_id] [--method=dense|keyword|both] [--k=1..{_RETRIEVE_DEPTH}]")


def _main(argv: list[str]) -> None:
    """`python eval_pipeline.py score [label] | report <run_id|latest> [baseline_run_id]
    [--method=…] [--k=…]`. An input problem (EvalInputError, or a bad method or
    k) exits with its full text."""
    command, extra = (argv[1] if len(argv) > 1 else None), argv[2:]
    try:
        if command == "score" and len(extra) <= 1:
            run = load_run(os.path.basename(score_run(extra[0] if extra else None)))
            print(f"eval: wrote eval_runs/{run.run_id}")
            for method in _RETRIEVAL_METHODS:
                m = _metric_row(run.metrics, "all", "ALL", "ALL", method, _TOP_K)
                print(f"  {method:8} claim coverage at k={_TOP_K}: "
                      f"{_pct(m.covered, m.claims) if m is not None else 'n/a'}")
            return
        if command == "report":
            positional = [a for a in extra if not a.startswith("--")]
            flags = dict(a[2:].split("=", 1) for a in extra if a.startswith("--") and "=" in a)
            if 1 <= len(positional) <= 2 and set(flags) <= {"method", "k"} and len(positional) + len(flags) == len(extra):
                path = write_report(positional[0], method=flags.get("method", "dense"),
                                    k=int(flags.get("k", _TOP_K)),
                                    baseline_id=positional[1] if len(positional) == 2 else None)
                print(f"eval: wrote {path}")
                return
    except ValueError as e:  # EvalInputError is a ValueError, as are a bad --method or --k
        raise SystemExit(str(e))
    raise SystemExit(_USAGE)


if __name__ == "__main__":
    _main(sys.argv)
