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
    `verified` carries each row's own `excel_row` column (its position in
    the sheet, header is row 1) rather than relying on the DataFrame index.
    """
    missing = [(row.excel_row, row.claim_id) for _, row in verified.iterrows()
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
        if claim_id is not None and chunk_id is not None:  # a blank claim_id is reported above
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
        if claim_id is not None:
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
                                         for r in cell["tag_rationale"]],
                         "excel_row": excel_rows}, dtype=object)
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
                      "group", "group_chunk_ids", "best_rank", "best_chunk_id",
                      "best_golden_chunk_id"]
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
    span = golden.evidence_span
    if span is None:
        return [""]
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
    """One row per (verifiable claim, golden evidence group, method): the
    group's golden chunk ids (group_chunk_ids), the best rank at which any
    retrieved chunk hits any row of the group, that retrieved chunk, and the
    golden row it matched (NaN / None when nothing hits within
    _RETRIEVE_DEPTH). The golden row is recorded because a claim can have
    many evidence rows, and the report must quote the one that was actually
    found, not the claim's first. The chunk ids are recorded so the claims
    page shows each quote under its own group's rank without grouping again
    (a claim never lists a chunk twice, so the ids name its rows)."""
    evidence_by_claim = {c: list(g.itertuples(index=False)) for c, g in gt.evidence.groupby("claim_id")}
    golden_text = dict(zip(gt.judged["chunk_id"], gt.judged["chunk_text"]))
    rows: list[dict] = []
    for claim in gt.claims[gt.claims["bucket"] != "UNVERIFIABLE"].itertuples(index=False):
        golden = evidence_by_claim[claim.claim_id]
        groups = _evidence_groups(golden)
        group_ids = [[golden[m].chunk_id for m in members] for members in groups]
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
                             "group_chunk_ids": group_ids[index],
                             "best_rank": best_rank, "best_chunk_id": best_chunk,
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
_META_FILENAME = "meta.json"
_XLSX_EXT = ".xlsx"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _round_half_away(x: float) -> int:
    """x rounded to a whole number, halves away from zero (12.5 -> 13,
    -2.5 -> -3). Python's round() sends halves to the even neighbour, so 1/8
    would show as 12% but 3/8 as 38%. x is first snapped to 12 decimals,
    so a value within 5e-13 of a half counts as that half: a stored fraction
    scaled by 100 carries float error near 1e-14 (100 * 0.145 is
    14.499999999999998) and would otherwise round the wrong way. A count
    ratio (_pct) never sits that close to a half without being one. A mean —
    MRR, a mean of 1/rank, or recall averaged per claim, a mean of per-claim
    fractions — can, once a run's depth or pieces per claim grow large (its
    smallest gap to a half is about 1 / (2 · claims · lcm of the ranks or
    piece counts)): such a figure is shown as the half, rounded away from
    zero, a difference no reader could see at two places."""
    x = round(x, 12)
    return int(np.sign(x) * np.floor(abs(x) + 0.5))


def _pct(num, den) -> str:
    """A percentage with its counts, as every number on the report is shown."""
    return f"{_round_half_away(100 * num / den)}% ({int(num)}/{int(den)})" if den else "n/a (0/0)"


def _mean_pct(x) -> str:
    """A mean of per-claim fractions (recall averaged per claim) as a whole
    percent, rounded as _pct rounds, or 'n/a' for NaN. No (x/y): a mean of
    fractions has none, so callers name what it averages."""
    return "n/a" if pd.isna(x) else f"{_round_half_away(100 * x)}%"


def _score(x) -> str:
    """A score such as MRR to two places, halves away from zero like every
    percentage here (0.125 -> 0.13, where format() gives 0.12), or 'n/a'
    for NaN — a scope with no verifiable claims."""
    return "n/a" if pd.isna(x) else f"{_round_half_away(100 * x) / 100:.2f}"


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
            with open(os.path.join(tmp, _META_FILENAME), "w", encoding="utf-8") as f:
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
    sheets = [f for f in names if f.endswith(_XLSX_EXT) and not f.startswith(("~$", "."))]
    for f in [f for f in sheets if f.endswith(".tmp.xlsx")]:
        logger.warning("eval: %s/%s is a temporary file an interrupted finalize left behind — not scored; "
                       "delete it", reviewed_dir, f)
    sheets = [f for f in sheets if not f.endswith(".tmp.xlsx")]
    memo_ids = [f[: -len(_XLSX_EXT)] for f in sheets if _valid_memo_id(f[: -len(_XLSX_EXT)])]
    for f in sheets:
        if not _valid_memo_id(f[: -len(_XLSX_EXT)]):
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
    # Every path that leaves results, queries or results_provenance None has
    # added to problems, so the None tests change nothing at runtime; they let
    # the checker see all three are set below. Should a later edit break that,
    # the fallback message says so instead of an empty refusal.
    if problems or results is None or queries is None or results_provenance is None:
        raise EvalInputError(problems or ["internal: retrieval results, claim queries or their provenance "
                                          "are unset, yet no problem was recorded"])
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
        meta_path = os.path.join(runs_dir, d, _META_FILENAME)
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
        with open(os.path.join(run_dir, _META_FILENAME), encoding="utf-8") as f:
            meta = json.load(f)
        tables = {name: pd.read_parquet(os.path.join(run_dir, f"{name}.parquet")) for name in _RUN_TABLES}
    except (OSError, ValueError) as e:
        raise EvalInputError([f"{run_dir}: incomplete or damaged run ({e}) — score it again"])
    return Run(run_id=run_id, meta=meta, **tables)


def check_comparable(run: Run, baseline: Run) -> None:
    """Refuses to compare runs measured with different rulers: the golden
    set, the embedding model and the retrieval depth must match. Depth is
    part of the ruler because RRF fuses the two depth-cut lists — a chunk
    inside both lists at one depth may be inside neither at another, so
    "both"'s top k differs between depths even at the same k. The phrase
    config is expected to differ — it is the phrase lever. Method and k are
    compared within one run, never across runs."""
    problems = []
    if run.meta.get("depth") != baseline.meta.get("depth"):
        problems.append(f"{run.run_id} was scored at retrieval depth {run.meta.get('depth')}, "
                        f"{baseline.run_id} at {baseline.meta.get('depth')} — not comparable: the combined "
                        f"method's top k depends on the depth the fused lists were cut to")
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
# ## 4. The report — three linked pages per (run, method, k)
#
# A summary for a non-technical audience, a claims page tracing every
# verifiable claim, and a re-review page listing every candidate (the same
# for every method and k, so one file per run). Written from run records
# only: no external asset, no script, opens with no network — a folded quote
# is plain HTML <details>. Every percentage carries its counts — except
# recall averaged per claim, a mean of per-claim fractions with no single
# (x/y), which names what it averages instead (a heading's, _heading_counts;
# the summary's "avg over N claims", _metrics_block). The
# unverifiable count never appears without the re-review sentence beside it
# (spec: "Unverifiable, and its honesty check"). The claims page shows recall
# averaged per claim (EV-19); the summary shows it per search method with
# pooled pieces, MRR and each memo's citation precision (EV-20). All are
# read from the stored metrics rows, never recomputed — pooled pieces'
# counts are the one measure that is not (_pooled_pieces) — and each is
# explained on its page in plain words. Precision and MRR appear on the
# summary only.

# %%
_METHOD_LABEL = {"dense": "meaning search (dense)", "keyword": "keyword search (BM25)",
                 "both": "both combined (RRF)"}
_METHOD_COLOR = {"dense": "#2563eb", "keyword": "#d97706", "both": "#059669"}
# The three miss reasons' stable keys — each is also its badge class on
# the claims page and the summary (.badge.other/.deep/.none; "retrieved at
# k" is .hit — both rendered by _status_badge; the piece-level grey .miss is
# separate on purpose, _piece_badge says why). miss_taxonomy assigns the key; every caption is built from
# it, so rewording a caption can never break a lookup.
_MISS_KEYS = ("other", "deep", "none")


def _miss_caption(key: str, depth: int) -> str:
    """A miss key's display text, written from the run's own recorded
    depth (meta["depth"]), never today's _RETRIEVE_DEPTH, so a run scored
    under another depth still captions itself correctly."""
    return {"other": "found by another method at this depth",
            "deep": f"found only deeper (by k={depth})",
            "none": f"not found by any method within {depth}"}[key]
_TABLE_CLOSE = "</table>"
_CSS = """
/* Shared by the three pages; colour tokens on :root (light only). The
   summary shares the tokens, sticky nav and folded-summary styling by
   design (EV-19: shared styling, content and order unchanged); only
   span.quote pins its traced examples to the pre-EV-19 inline look. The
   status badges also mark the summary's traced examples and miss reasons
   (EV-20); the claim, piece, rank, toc and passage classes serve only the
   claims and re-review pages. No class or comment here may name the
   summary's finer measures: this sheet is part of every page, and the
   claims and re-review pages must never mention them. */
:root{--bg:#fafaf9;--card:#fff;--ink:#1f2937;--muted:#6b7280;--line:#e5e7eb;--line2:#d1d5db;--head:#f3f4f6;
     --accent:#6366f1;--dense:#2563eb;--keyword:#d97706;--both:#059669;
     --hit:#166534;--hit-bg:#dcfce7;--other:#1e40af;--other-bg:#dbeafe;--deep:#92400e;--deep-bg:#fef3c7;
     --none:#991b1b;--none-bg:#fee2e2}
a{color:var(--dense)}
body{font:15px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg);
     max-width:1040px;margin:0 auto;padding:0 24px 48px}
h1{font-size:24px;margin:24px 0 4px}
h2{font-size:18px;margin:36px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px;scroll-margin-top:56px}
h3{font-size:15px;margin:20px 0 6px;scroll-margin-top:56px}
.cnt{color:var(--muted);font-weight:400;font-size:13px;margin-left:8px}
.sub{color:var(--muted);margin:0 0 12px}.frame{background:var(--other-bg);border-left:4px solid var(--accent);padding:8px 12px}
.note{background:var(--deep-bg);border-left:4px solid var(--keyword);padding:8px 12px}
/* one line always (scrolls sideways if long): headings reserve 56px of scroll-margin, a wrapped nav would be taller */
.nav{position:sticky;top:0;z-index:1;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 0;margin:0 0 12px;
     white-space:nowrap;overflow-x:auto}
.nav a,.nav b{margin-right:16px}.nav .cnt{margin-left:0}
.toc{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 16px;margin:12px 0 24px;
     display:flex;flex-wrap:wrap;gap:4px 32px}
.toc ul{list-style:none;margin:4px 0 0;padding:0}.toc li{padding:1px 0}
.tiles{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}.tile{background:var(--card);border:1px solid var(--line);
     border-radius:8px;padding:12px 16px;min-width:180px}.tile .v{font-size:26px;font-weight:600}
.tile .l{color:var(--muted);font-size:13px}.delta{font-size:13px;color:var(--both)}.delta.neg{color:var(--none)}
table{border-collapse:collapse;width:100%;font-size:13px;background:var(--card)}
th,td{border:1px solid var(--line2);padding:4px 8px;text-align:left;vertical-align:top}
table.narrow{width:auto;min-width:min(420px,100%);margin-top:12px}
/* muted explanations under a table — not .note, the amber warning box */
.explain{margin:8px 0 0;font-size:13px;color:var(--muted)}.explain p{margin:2px 0}.explain b{color:var(--ink)}
th{background:var(--head)}.bar{background:var(--line);height:10px;width:160px;display:inline-block;vertical-align:middle}
.bar span{display:block;height:10px;background:var(--dense)}
.scroll{overflow-x:auto}
/* one claim: number column + body; :target highlights a claim reached by link */
.claim{display:grid;grid-template-columns:2.6em minmax(0,1fr);background:var(--card);border:1px solid var(--line);
     border-radius:8px;padding:12px 14px 10px 10px;margin:0 0 10px;scroll-margin-top:56px}
.claim:target{border-color:var(--accent);box-shadow:0 0 0 2px var(--accent)}
.n{color:var(--muted);font-size:13px;padding-top:3px}
.n a{color:inherit;text-decoration:none}.n a:hover{text-decoration:underline}
.text{margin:0 0 6px;font-size:15px}
.meta{display:flex;flex-wrap:wrap;align-items:center;gap:6px 8px;margin:0 0 6px;font-size:13px;color:var(--muted)}
.tag{border:1px solid var(--line2);border-radius:4px;padding:0 6px;font-size:12px;color:var(--muted)}
.badge{border-radius:4px;padding:1px 8px;font-size:12px;font-weight:600}
.badge.hit{color:var(--hit);background:var(--hit-bg)}.badge.other{color:var(--other);background:var(--other-bg)}
.badge.deep{color:var(--deep);background:var(--deep-bg)}.badge.none{color:var(--none);background:var(--none-bg)}
.badge.miss{color:var(--muted);background:var(--head)}
/* rank chips: one per method, method colour, dashed when never within the retrieval depth, ring on the scored method */
.rank{font:12px/1.6 ui-monospace,Menlo,Consolas,monospace;border:1px solid currentColor;border-radius:999px;padding:0 8px}
.rank.dense{color:var(--dense)}.rank.keyword{color:var(--keyword)}.rank.both{color:var(--both)}
.rank.nf{color:var(--muted);border-style:dashed}.rank.cur{box-shadow:0 0 0 2px var(--line2)}
/* one line, tail clipped: keeps a long chunk id from pushing the preview onto a second line; display stays
   list-item so the disclosure triangle survives (a flex summary would drop it) */
details summary{cursor:pointer;color:var(--muted);font-size:13px;padding:2px 0;white-space:nowrap;overflow:hidden;
     text-overflow:ellipsis}
details summary:hover{color:var(--ink)}
.piece{border-top:1px dashed var(--line);padding:8px 0 4px 0}
.piece .meta{margin-bottom:2px}.piece .meta b{color:var(--ink)}
/* the summary's traced examples (span, pre-EV-19 look) vs the claims page's quotes (blockquote): same class, split
   by element so the claims-page block layout cannot restyle the summary */
span.quote{color:#374151;font-style:italic;margin:4px 0}
blockquote.quote{margin:4px 0 6px 12px;padding:0 0 0 10px;border-left:3px solid var(--line2);white-space:pre-wrap;
     font-variant-numeric:tabular-nums;font-size:13px;line-height:1.45}
.quote code,summary code{font:12px ui-monospace,Menlo,Consolas,monospace;color:var(--muted)}
/* a re-review passage: line breaks kept, capped height, scroll inside */
.passage{white-space:pre-wrap;font-family:inherit;font-size:13px;line-height:1.45;font-variant-numeric:tabular-nums;
     margin:6px 0 10px;padding:8px 10px;max-height:20em;overflow:auto;background:var(--bg);
     border:1px solid var(--line);border-radius:6px}
.score{font:600 12px/1.6 ui-monospace,Menlo,Consolas,monospace;color:var(--ink);background:var(--head);border-radius:4px;
     padding:0 6px;margin-right:6px}
.preview{margin-left:8px}
details[open]>summary .preview{display:none}
@media (max-width:640px){.preview{display:none}.claim{grid-template-columns:2em minmax(0,1fr);padding:10px}}
"""


def miss_taxonomy(run: Run, method: str, k: int) -> pd.DataFrame:
    """
    One row per verifiable claim NOT covered at (method, k), with why —
    the stable `key` (_MISS_KEYS) plus its `category` caption
    (_miss_caption at run.meta["depth"]), hit-tested by _within_k so this
    and the claims page's piece counts share one rule.
    """
    if run.claim_hits.empty:
        return pd.DataFrame(columns=["claim_id", "bucket", "category", "key"])
    depth = run.meta["depth"]
    first = (run.claim_hits.groupby(["claim_id", "method"])["best_rank"].min()
             .unstack("method").reindex(columns=list(_RETRIEVAL_METHODS)))
    buckets = run.claim_hits.groupby("claim_id")["bucket"].first()
    rows = []
    for claim_id, ranks in first.iterrows():
        if _within_k(ranks[method], k):
            continue
        if any(_within_k(ranks[m], k) for m in _RETRIEVAL_METHODS if m != method):
            key = "other"
        elif any(_within_k(r, depth) for r in ranks):
            key = "deep"
        else:
            key = "none"
        rows.append({"claim_id": claim_id, "bucket": buckets[claim_id],
                     "category": _miss_caption(key, depth), "key": key})
    return pd.DataFrame(rows, columns=["claim_id", "bucket", "category", "key"])


def _e(value) -> str:
    """HTML-escaped text; control characters pypdf may leave are removed first."""
    return html.escape(_excel_safe(value) if isinstance(value, str) else str(value))


def _count(n: int, noun: str) -> str:
    """'1 quote', '2 quotes'."""
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _check_view(run: Run, method: str, k: int) -> None:
    """Refuses a method the run does not hold, or a k beyond the depth the
    run was scored to (ValueError), and a run that records no depth at all
    (EvalInputError): the pages caption every rank from meta["depth"] —
    ">10", "not within 10", the miss categories — so a run scored under
    another _RETRIEVE_DEPTH still reports, wearing its own depth."""
    if method not in _RETRIEVAL_METHODS:
        raise ValueError(f"method {method!r}: use one of {', '.join(_RETRIEVAL_METHODS)}")
    depth = run.meta.get("depth")
    if depth is None:
        raise EvalInputError([f"{run.run_id}: meta.json records no retrieval depth, so no rank can be "
                              f"captioned — score the run again: python eval_pipeline.py score"])
    if not 1 <= k <= depth:
        raise ValueError(f"k={k}: use 1..{depth}, the depth this run was scored to")


def _page_names(method: str, k: int, baseline_id: str | None) -> dict[str, str]:
    """The three pages' file names, keyed summary / claims / rereview: the
    summary report_<method>_k<k>[_vs_<baseline>].html, the claims page the
    same name ending _claims. The re-review page depends on none of method,
    k or baseline, so every call names the same file."""
    stem = f"report_{method}_k{k}" + (f"_vs_{baseline_id}" if baseline_id else "")
    return {"summary": f"{stem}.html", "claims": f"{stem}_claims.html", "rereview": "report_rereview.html"}


def _memo_anchor(memo_id: str) -> str:
    """The one spelling of a memo heading's anchor id, shared by the jump
    line, both tables of contents and both pages' memo headings."""
    return f"memo-{_e(memo_id)}"


def _memo_heading_and_toc(memo_id: str, heading_cnt: str, toc_cnt: str, items: str = "") -> tuple[str, str]:
    """One memo's <h2> heading and its table-of-contents entry, built as a
    pair around _memo_anchor so the two pages' markup cannot drift apart;
    `items` is the claims page's nested <ul> of section links."""
    heading = f'<h2 id="{_memo_anchor(memo_id)}">{_e(memo_id)} <span class="cnt">{heading_cnt}</span></h2>'
    entry = (f'<div><b><a href="#{_memo_anchor(memo_id)}">{_e(memo_id)}</a></b> '
             f'<span class="cnt">{toc_cnt}</span>{items}</div>')
    return heading, entry


def _jump_line(memo_ids: list[str]) -> str:
    """'Jump to:' links to each memo's heading (_memo_anchor), for a page's
    sticky link line. Empty when there is no memo to link."""
    if not memo_ids:
        return ""
    links = "".join(f'<a href="#{_memo_anchor(m)}">{_e(m)}</a>' for m in memo_ids)
    return f'<span class="cnt">Jump to: {links}</span>'


def _nav(run: Run, pages: dict[str, str] | None, current: str, jump: str = "") -> str:
    """The link line between the three pages, the current one in bold, the
    others linked by relative name so a copied run folder keeps working;
    `jump` (a _jump_line) appends that page's own memo links. With `pages`
    None only the current page's bold label and the jump links are shown —
    the re-review page, which links to no other page — and without a jump
    line either, nothing at all: the notebook's inline copy, where a
    relative link resolves against the notebook, not the run folder, and so
    is dead."""
    verifiable = int((run.claims["bucket"] != "UNVERIFIABLE").sum())
    labels = {"summary": "Summary", "claims": f"Claims ({verifiable} verifiable)",
              "rereview": f"Re-review ({_count(len(run.candidates), 'passage')})"}
    if pages is None:
        return f'<p class="nav"><b>{labels[current]}</b>{jump}</p>' if jump else ""
    items = [f"<b>{labels[p]}</b>" if p == current else f'<a href="{_e(pages[p])}">{labels[p]}</a>' for p in labels]
    return f'<p class="nav">{"".join(items)}{jump}</p>'


def _view_line(run: Run, method: str, k: int, baseline: Run | None = None) -> str:
    """The line under a page's title: memos, model, method, k, baseline."""
    return (f'<p class="sub">{_e(", ".join(run.meta["memos"]))} · embedding model '
            f'{_e(run.meta["embedding_model"])} · {_e(_METHOD_LABEL[method])} · k = {k} passages per search phrase'
            + (f" · compared with {_e(baseline.run_id)}" if baseline is not None else "") + "</p>")


def _unverifiable_note(run: Run, listed: str) -> str:
    """The unverifiable count with the re-review sentence beside it — the only
    form in which a page may state that count. `listed` says where the
    candidates are, as seen from the page it is on ("on the Re-review page",
    "below")."""
    census = len(run.claims)
    unverifiable = int((run.claims["bucket"] == "UNVERIFIABLE").sum())
    with_candidates = run.candidates["claim_id"].nunique()
    return (f'<p class="note">{unverifiable} of {census} claims were not found in the sources by this process. '
            f'For {with_candidates} of them, meaning-based search offers passages the review never showed a '
            f'human — {len(run.candidates)} listed {listed}. They are not part of the coverage '
            f'figures.</p>')


def _page(title: str, body: list[str]) -> str:
    """A whole self-contained page: the styles inline, nothing fetched."""
    return ("<!doctype html><html><head><meta charset=\"utf-8\">"
            f"<title>{_e(title)}</title><style>{_CSS}</style></head><body>" + "".join(body) + "</body></html>")


def _required_row(run: Run, scope: str, memo_id: str, section: str, method: str, k: int,
                  *, baseline: bool = False):
    """The metrics row a page must read, or a refusal naming it. A run —
    or baseline — whose metrics.parquet lost rows (damaged, or filtered by
    hand) would otherwise render as zeros or crash mid-page. A section row
    may legitimately be absent (a section holding only unverifiable claims
    that was never searched), so the summary reads section rows with
    _metric_row; the claims page lists only sections holding verifiable
    claims, whose rows every run writes, so it requires them here.
    `baseline` changes only the advice: `score` rebuilds a run from today's
    phrases and results, so it can repair the run being reported, never an
    earlier run serving as its baseline."""
    row = _metric_row(run.metrics, scope, memo_id, section, method, k)
    if row is None:
        where = ("the all-memo total" if scope == "all"
                 else f"{scope} {memo_id}" + (f" {section!r}" if scope == "section" else ""))
        advice = ("this baseline run's metrics table is incomplete; choose another baseline run, or score it "
                  "again only after restoring the phrases and results it was scored from" if baseline else
                  "the run's metrics table is incomplete; score the run again: python eval_pipeline.py score")
        raise EvalInputError([f"{run.run_id}: no metrics row for {where} at method={method}, k={k} — {advice}"])
    return row


def _baseline_row(baseline: Run | None, scope: str, memo_id: str, section: str, method: str, k: int):
    """The baseline's row for the same view, None without a baseline; a
    missing row is refused with the advice a baseline needs (_required_row)."""
    return None if baseline is None else _required_row(baseline, scope, memo_id, section, method, k, baseline=True)


def _delta(current, base, field: str = "coverage", *, score: bool = False) -> str:
    """A metrics row's change in `field` against the baseline's: '+33 pts
    vs baseline' for a fraction (coverage, recall_macro), or '+0.33 vs
    baseline' when the caller passes score=True (MRR). The change is the gap between the two figures as the page shows
    them — each x 100, rounded halves away from zero, as _pct and _score
    round — so it always adds up with the numbers beside it (67% against
    33% reads +34, not the unrounded +33). Both are whole numbers, so equal
    figures read '+0.00', never '-0.00', and only a fall wears the falling
    colour. '' without a baseline row, or when either value is NaN (a scope
    with no verifiable claims)."""
    if base is None or current is None:
        return ""
    now, then = getattr(current, field), getattr(base, field)
    if pd.isna(now) or pd.isna(then):
        return ""
    n = _round_half_away(100 * now) - _round_half_away(100 * then)
    text = f"{n / 100:+.2f}" if score else f"{n:+d} pts"
    cls = "delta neg" if n < 0 else "delta"
    return f'<div class="{cls}">{text} vs baseline</div>'


def _coverage_svg(run: Run, method: str, k: int, baseline: Run | None) -> str:
    """Coverage@k, k = 1..the run's recorded depth, one line per method
    (the chosen one heavier), a marker at k, and the baseline's line for
    the chosen method dashed (check_comparable holds the depths equal)."""
    width, height, left, right, top, bottom = 640, 260, 48, 16, 16, 36
    depth = run.meta["depth"]

    def x(kk):
        return left + (kk - 1) * (width - left - right) / max(depth - 1, 1)   # depth 1: a single column, no /0

    def y(v):
        return top + (1 - v) * (height - top - bottom)

    def points(r: Run, m: str, *, is_baseline: bool = False):
        # every k is read, not just the page's: a lost row would otherwise join its neighbours silently
        out = []
        for kk in range(1, depth + 1):
            row = _required_row(r, "all", "ALL", "ALL", m, kk, baseline=is_baseline)
            if not pd.isna(row.coverage):   # NaN: no verifiable claims, nothing to draw
                out.append(f"{x(kk):.1f},{y(row.coverage):.1f}")
        return " ".join(out)

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
             f'aria-label="Claim coverage by passages per search phrase, for each search method">']
    for v in (0, 0.25, 0.5, 0.75, 1):
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#e5e7eb"/>'
                     f'<text x="{left - 6}" y="{y(v) + 4:.1f}" font-size="11" text-anchor="end" fill="#6b7280">'
                     f'{int(v * 100)}%</text>')
    for kk in sorted({1, *range(5, depth + 1, 5), depth}):
        parts.append(f'<text x="{x(kk):.1f}" y="{height - bottom + 16}" font-size="11" text-anchor="middle" '
                     f'fill="#6b7280">{kk}</text>')
    parts.append(f'<text x="{(left + width - right) / 2:.1f}" y="{height - 4}" font-size="11" text-anchor="middle" '
                 f'fill="#6b7280">k — passages per search phrase</text>')
    parts.append(f'<line x1="{x(k):.1f}" x2="{x(k):.1f}" y1="{top}" y2="{height - bottom}" '
                 f'stroke="#9ca3af" stroke-dasharray="2 3"/>')
    if baseline is not None:
        parts.append(f'<polyline fill="none" stroke="#9ca3af" stroke-width="2" stroke-dasharray="6 4" '
                     f'points="{points(baseline, method, is_baseline=True)}"/>')
    for m in _RETRIEVAL_METHODS:
        weight = 3 if m == method else 1.5
        parts.append(f'<polyline fill="none" stroke="{_METHOD_COLOR[m]}" stroke-width="{weight}" '
                     f'points="{points(run, m)}"/>')
    for i, m in enumerate(_RETRIEVAL_METHODS):
        parts.append(f'<text x="{left + 8}" y="{top + 14 + 14 * i}" font-size="12" fill="{_METHOD_COLOR[m]}">'
                     f'{_e(_METHOD_LABEL[m])}</text>')
    if baseline is not None:
        parts.append(f'<text x="{left + 8}" y="{top + 14 + 14 * 3}" font-size="12" fill="#6b7280">'
                     f'baseline {_e(baseline.run_id)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _phrase_counts(run: Run) -> dict:
    """memo -> section -> number of search phrases, as the run recorded
    them; {} for a run that recorded none (a baseline needs only meta.json
    and metrics, so its counts may be missing)."""
    return run.meta.get("phrase_counts", {})


def _phrase_count_changes(run: Run, baseline: Run) -> list[tuple[str, str]]:
    """(memo, section) pairs, sorted, that the two runs searched with a
    different number of phrases — over both runs' sections, since one only
    the baseline searched lost its passages too. A count one run did not
    record differs. The one comparison behind _phrase_count_note and the
    MRR note's caveat."""
    counts, base_counts = _phrase_counts(run), _phrase_counts(baseline)
    keys = sorted({(m, s) for c in (counts, base_counts) for m, sections in c.items() for s in sections})
    return [(m, s) for m, s in keys if counts.get(m, {}).get(s) != base_counts.get(m, {}).get(s)]


def _phrase_count_note(run: Run, baseline: Run | None) -> str:
    """A warning when the baseline searched some section with a different
    number of phrases: more phrases hand the model more passages, so part of
    any change there is simply more text, not better questions."""
    if baseline is None:
        return ""
    changed = [f"{m} / {s}" for m, s in _phrase_count_changes(run, baseline)]
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


def _traced_examples(run: Run, method: str, k: int, misses: pd.DataFrame) -> str:
    """One found and one missed verifiable claim per memo at (method, k),
    each with the evidence a human confirmed, folded. A found claim quotes
    the evidence row that was actually hit (best_golden_chunk_id) — a claim
    can have many — and says where the hit was. Each outcome wears the
    claims page's status badge (_status_badge): green when retrieved, else
    its miss reason's colour and caption (`misses`: the page's
    miss_taxonomy, shared with its miss-reason table), so a claim another method found
    is never shown as found by none."""
    text_by_claim = dict(zip(run.claims["claim_id"], run.claims["claim_text"]))
    hits = run.claim_hits[run.claim_hits["method"] == method]
    best = hits.sort_values(["best_rank"], kind="stable").groupby("claim_id").first()
    first_quote = run.evidence.groupby("claim_id").first()
    quote_of = run.evidence.set_index(["claim_id", "chunk_id"])["evidence_span"]
    reason_of = dict(zip(misses["claim_id"], misses["key"]))
    depth = run.meta["depth"]
    table = ["<table><tr><th>memo</th><th>claim</th><th>evidence a human confirmed</th><th>retrieved?</th></tr>"]
    for memo_id in run.meta["memos"]:
        verifiable = run.claims[(run.claims["memo_id"] == memo_id) & (run.claims["bucket"] != "UNVERIFIABLE")]
        found = [c for c in verifiable["claim_id"] if _within_k(best.loc[c, "best_rank"], k)]
        missed = [c for c in verifiable["claim_id"] if not _within_k(best.loc[c, "best_rank"], k)]
        for claim_id in found[:1] + missed[:1]:
            b = best.loc[claim_id]
            if _within_k(b.best_rank, k):
                golden = b.best_golden_chunk_id
                quote = quote_of.loc[(claim_id, golden)]
                g, r = _chunk_index_of(golden), _chunk_index_of(b.best_chunk_id)
                if b.best_chunk_id == golden:
                    where = "the same passage"
                elif g is not None and r is not None and abs(g - r) == 1:
                    where = f"the neighbouring passage {_e(b.best_chunk_id)}, which contains this quote"
                else:   # _is_hit's identical-text case: another place holding the same text
                    where = f"passage {_e(b.best_chunk_id)}, whose text is identical to this quote"
                outcome = f"{_status_badge(None, k, depth)} — rank {int(b.best_rank)}: {where}"
            else:
                golden = first_quote.loc[claim_id, "chunk_id"]
                quote = first_quote.loc[claim_id, "evidence_span"]
                outcome = _status_badge(reason_of[claim_id], k, depth)
            table.append(f"<tr><td>{_e(memo_id)}</td><td>{_e(text_by_claim[claim_id])}</td>"
                         f'<td><details><summary>show quote</summary><span class="quote">{_e(quote)} '
                         f"({_e(golden)})</span></details></td><td>{outcome}</td></tr>")
    table.append(_TABLE_CLOSE)
    return f'<div class="scroll">{"".join(table)}</div>'


def _pooled_pieces(run: Run, method: str, k: int) -> tuple[int, int]:
    """(pieces of evidence retrieved within k, pieces there are), pooled
    over every verifiable claim. The one summary measure not read from a
    stored metrics row: metrics.parquet keeps only the fraction
    (recall_micro), and every percentage shows its counts.
    claim_hits holds one row per claim x method x piece, hit or not, and a
    piece counts when its best rank is <= k (a NaN rank never is) — the
    comparison _claim_metrics_by_k makes for recall_micro, so the fraction
    equals the stored one."""
    ranks = run.claim_hits.loc[run.claim_hits["method"] == method, "best_rank"]
    return int((ranks <= k).sum()), len(ranks)


def _metrics_block(run: Run, method: str, k: int, baseline: Run | None) -> str:
    """The summary's finer measures at k (EV-20). Per search method: its
    claim coverage (repeated from the chart, so recall reads beside it),
    recall averaged per claim, pieces of evidence retrieved pooled with
    counts (_pooled_pieces) and MRR — recall and MRR with their change
    against the baseline. Then citation precision per memo for the chosen
    method.
    Read from the all-memo and memo rows, which _required_row refuses by
    name when missing — in the baseline too. Each measure comes with a
    plain-words note that keeps it from being misread; a run with no
    verifiable claims shows 'n/a' and leaves out the recall and MRR notes,
    which would have nothing to describe."""
    out = [f"<h2>Finer measures by search method (k = {k})</h2>",
           '<div class="scroll"><table><tr><th>search method</th><th>claim coverage</th>'
           "<th>evidence recall, averaged per claim</th><th>pieces of evidence retrieved, pooled</th>"
           "<th>MRR</th></tr>"]
    for m in _RETRIEVAL_METHODS:
        row = _required_row(run, "all", "ALL", "ALL", m, k)
        base = _baseline_row(baseline, "all", "ALL", "ALL", m, k)
        name = f"<th>{_e(_METHOD_LABEL[m])}</th>" if m == method else f"<td>{_e(_METHOD_LABEL[m])}</td>"
        recall = _mean_pct(row.recall_macro) + ("" if pd.isna(row.recall_macro)
                                                else f" avg over {_count(int(row.claims), 'claim')}")
        out.append(f"<tr>{name}<td>{_pct(row.covered, row.claims)}</td>"
                   f"<td>{recall}{_delta(row, base, 'recall_macro')}</td>"
                   f"<td>{_pct(*_pooled_pieces(run, m, k))}</td>"
                   f"<td>{_score(row.mrr)}{_delta(row, base, 'mrr', score=True)}</td></tr>")
    out.append("</table></div>")
    verifiable = run.claims[run.claims["bucket"] != "UNVERIFIABLE"]
    if not verifiable.empty:   # the MRR note's ranges are taken over these claims' sections
        sections = set(zip(verifiable["memo_id"], verifiable["section"]))

        def phrases(r: Run) -> str | None:
            recorded = _phrase_counts(r)
            counts = sorted({n for memo_id, section in sections
                             if (n := recorded.get(memo_id, {}).get(section)) is not None})
            if not counts:
                return None
            return _count(counts[0], "phrase") if len(counts) == 1 else f"{counts[0]}–{counts[-1]} phrases"

        # MRR rises with the phrase count alone, so a change against a baseline searched otherwise says so
        caveat = ""
        if baseline is not None and sections & set(_phrase_count_changes(run, baseline)):
            searched = phrases(baseline)
            caveat = ((f"; the baseline's sections were searched with {searched} each" if searched else
                       "; the baseline searched these sections with a different number of phrases")
                      + ", so part of the MRR change comes from the number of phrases, not better phrases")
        out.append('<div class="explain">'
                   "<p><b>Evidence recall</b> counts every piece of evidence a claim lists; alternative sources "
                   "for the same fact pull it down, so read it beside the same method's claim coverage, in the "
                   "column before it.</p>"
                   "<p><b>MRR</b> (mean reciprocal rank) takes each claim's best rank over all of its section's "
                   f"search phrases (sections here are searched with {phrases(run) or 'an unrecorded number of phrases'} "
                   "each), so it is optimistic "
                   f"and not comparable between sections{caveat}.</p></div>")
    out.append(f'<table class="narrow"><tr><th>memo</th><th>citation precision, {_e(_METHOD_LABEL[method])}</th></tr>')
    for memo_id in run.meta["memos"]:
        row = _required_row(run, "memo", memo_id, "ALL", method, k)
        out.append(f"<tr><td>{_e(memo_id)}</td><td>{_pct(row.retrieved_golden, row.retrieved)}</td></tr>")
    out.append(_TABLE_CLOSE)
    out.append('<div class="explain"><p><b>Citation precision</b> counts only the retrieved passages someone cited '
               "as evidence, so it is a floor: an uncited passage may still have been useful.</p></div>")
    return "".join(out)


def render_summary(run: Run, *, method: str = "dense", k: int = _TOP_K, baseline: Run | None = None,
                   pages: dict[str, str] | None = None) -> str:
    """The summary page for one run at (method, k), for a non-technical
    audience: traced examples first, then the headline tiles, the coverage
    chart, the finer measures (_metrics_block: recall, pooled pieces, MRR,
    citation precision), the memo/section table and the miss reasons —
    optionally with changes against a baseline run (check_comparable must
    pass). The all-memo and memo rows it reads are refused by name when
    missing, in the run or the baseline (_required_row); only a section row
    may be absent. `pages` names the three files to link between
    (_page_names); None leaves the link line out. Returns the HTML."""
    _check_view(run, method, k)
    if baseline is not None:
        check_comparable(run, baseline)

    def section_row(r: Run, memo_id: str, section: str):
        # None is legitimate here: a section holding only unverifiable claims, never searched, has no row
        return _metric_row(r.metrics, "section", memo_id, section, method, k)

    head = _required_row(run, "all", "ALL", "ALL", method, k)
    base = _baseline_row(baseline, "all", "ALL", "ALL", method, k)
    covered, claims = int(head.covered), int(head.claims)
    ext = (int(head.covered_extractive), int(head.claims_extractive))
    syn = (int(head.covered_synthesized), int(head.claims_synthesized))
    syn_complete = int(head.complete_synthesized)
    passages = int(head.passages)
    # section rows at this (method, k) — every section searched, verifiable claims or not
    section_rows = run.metrics[(run.metrics["scope"] == "section") & (run.metrics["method"] == method)
                               & (run.metrics["k"] == k)]
    n_sections = int((section_rows["passages"] > 0).sum())
    out: list[str] = []

    out.append(f"<h1>Retrieval eval — {_e(run.run_id)}</h1>")
    out.append(_view_line(run, method, k, baseline))
    out.append(_nav(run, pages, "summary"))
    out.append('<p class="frame">This measures a retrieval <b>prototype</b> against a hand-reviewed ruler. '
               'It shows what the instrument produces — it is not a verdict on any production system.</p>')

    out.append("<h2>Traced examples</h2>")
    misses = miss_taxonomy(run, method, k)
    out.append(_traced_examples(run, method, k, misses))

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
    out.append(_unverifiable_note(run, "on the Re-review page"))

    out.append("<h2>Coverage as each search phrase returns more passages</h2>")
    out.append(_coverage_svg(run, method, k, baseline))

    out.append(_metrics_block(run, method, k, baseline))

    out.append("<h2>By memo and section</h2>")
    table = ["<table><tr><th>memo</th><th>section</th><th>claim coverage</th><th>passages</th>"
             "<th>change</th></tr>"]
    for memo_id in run.meta["memos"]:
        m = _required_row(run, "memo", memo_id, "ALL", method, k)
        mb = _baseline_row(baseline, "memo", memo_id, "ALL", method, k)
        # every claims-file section, plus any section searched that the claims file lacks
        sections = sorted(set(run.claims.loc[run.claims["memo_id"] == memo_id, "section"])
                          | set(section_rows.loc[section_rows["memo_id"] == memo_id, "section"]))
        table.append(f"<tr><th>{_e(memo_id)}</th><th>all</th><td>{_bar(m)}</td>"
                     f"<td>{int(m.passages)}</td><td>{_delta(m, mb)}</td></tr>")
        for section in sections:
            s = section_row(run, memo_id, section)
            sb = section_row(baseline, memo_id, section) if baseline is not None else None
            table.append(f"<tr><td></td><td>{_e(section)}</td><td>{_bar(s)}</td>"
                         f"<td>{int(s.passages) if s is not None else 0}</td><td>{_delta(s, sb)}</td></tr>")
    table.append(_TABLE_CLOSE)
    out.append("".join(table))

    out.append(f"<h2>Why the missed claims were missed (k = {k})</h2>")
    table = ["<table><tr><th>reason</th><th>extractive</th><th>synthesized</th></tr>"]
    for key in _MISS_KEYS:
        rows = misses[misses["key"] == key]
        table.append(f'<tr><td>{_badge(key, _miss_caption(key, run.meta["depth"]))}</td>'
                     f"<td>{int((rows['bucket'] == 'EXTRACTIVE').sum())}</td>"
                     f"<td>{int((rows['bucket'] == 'SYNTHESIZED').sum())}</td></tr>")
    table.append(_TABLE_CLOSE)
    out.append("".join(table))

    return _page(f"Retrieval eval {run.run_id}", out)


def _evidence_pieces(run: Run) -> dict[str, list[list]]:
    """
    Each verifiable claim's confirmed evidence rows, split into the pieces of
    evidence (decision 1's groups) scoring ranked, in scoring's group order.
    The pieces are read from the chunk ids scoring stored for each group
    (claim_hits' group_chunk_ids), never grouped again here, so a later
    change to the grouping code cannot put a quote under another piece's
    rank. Halts (EvalInputError) for a run scored before those ids were
    stored, and for a claim whose pieces do not hold exactly its evidence
    rows, so no quote is silently left off the page. Only the claims page
    needs the ids: such a run still serves as a baseline, which reads only
    its meta and metrics.
    """
    if "group_chunk_ids" not in run.claim_hits.columns:
        raise EvalInputError([f"{run.run_id}: scored before runs recorded each piece of evidence's chunks, "
                              f"which the claims page needs — it can still be a baseline; to report on the "
                              f"current retrieval results, run python eval_pipeline.py score"])
    evidence_by_claim = {c: {r.chunk_id: r for r in g.itertuples(index=False)}
                         for c, g in run.evidence.groupby("claim_id")}
    groups = run.claim_hits.drop_duplicates(["claim_id", "group"]).sort_values(["claim_id", "group"])
    members_by_claim = {c: list(g["group_chunk_ids"]) for c, g in groups.groupby("claim_id")}
    pieces: dict[str, list[list]] = {}
    problems: list[str] = []
    # every claim either table knows: one missing from the hits has evidence but no pieces
    for claim_id in [*evidence_by_claim, *(c for c in members_by_claim if c not in evidence_by_claim)]:
        rows = evidence_by_claim.get(claim_id, {})
        members = members_by_claim.get(claim_id, [])
        listed = [c for ids in members for c in ids]
        if len(listed) != len(rows) or set(listed) != set(rows):
            problems.append(f"{run.run_id}: claim {claim_id}: its pieces of evidence do not hold exactly its "
                            f"evidence rows — the run's files were changed after scoring")
            continue
        pieces[str(claim_id)] = [[rows[c] for c in ids] for ids in members]
    if problems:
        raise EvalInputError(problems)
    return pieces


def _rank_chips(ranks: dict[str, float], method: str, depth: int) -> str:
    """One chip per method — 'dense 2', or a dashed 'dense >20' (class nf)
    when the method never ranks it within the run's recorded depth — with
    a ring (class cur) on the method this page scores."""
    chips = []
    for m in _RETRIEVAL_METHODS:
        rank = ranks[m]
        cls = f"rank {m}" + (" cur" if m == method else "") + (" nf" if pd.isna(rank) else "")
        label = f"{m} >{depth}" if pd.isna(rank) else f"{m} {int(rank)}"
        chips.append(f'<span class="{cls}">{_e(label)}</span>')
    return "".join(chips)


def _within_k(rank, k: int) -> bool:
    """True when a best rank places within the page's k (NaN — never within
    the run's depth — does not). The one hit rule behind a piece's badge,
    its claim's 'X of Y pieces retrieved' count, miss_taxonomy's buckets and
    the traced examples, so they cannot drift."""
    return bool(pd.notna(rank) and rank <= k)


def _badge(cls: str, text: str) -> str:
    """One coloured label (.badge.<cls>) — the markup every status and
    piece badge on the pages shares."""
    return f'<span class="badge {cls}">{_e(text)}</span>'


def _status_badge(reason: str | None, k: int, depth: int) -> str:
    """A claim's status at k as one of the four coloured badges: reason
    None is 'retrieved', otherwise the miss_taxonomy key — which is the
    badge class — captioned by _miss_caption. Shared by the claims page and
    the summary's traced examples, so a claim wears the same colour on both."""
    status = f"retrieved at k = {k}" if reason is None else f"missed at k = {k} — {_miss_caption(reason, depth)}"
    return _badge(reason or "hit", status)


def _piece_badge(rank: float, k: int, depth: int) -> str:
    """The scored method's verdict on one piece of evidence: green
    'retrieved' within k, otherwise grey — 'rank N' when within the run's
    recorded depth, 'not within <depth>' when not. Grey on purpose, never
    the claim-level amber or red: those mean NO method found the claim, and
    a piece another method finds at this depth would wear the wrong meaning."""
    if _within_k(rank, k):
        return _badge("hit", "retrieved")
    return _badge("miss", f"rank {int(rank)}" if pd.notna(rank) else f"not within {depth}")


def _heading_counts(row, k: int) -> str:
    """A memo or section heading's counts — '3 claims · 67% (2/3) retrieved
    at k = 2 · recall 50%, the mean of each claim's piece recall' — read
    from its metrics row (_metric_row), never recomputed, so every heading
    matches the run's metrics.parquet. The recall carries no (x/y): a mean
    of per-claim fractions has none, so it names what it averages, over the
    claim count that opens the line."""
    return (f"{_count(int(row.claims), 'claim')} · {_pct(row.covered, row.claims)} retrieved at k = {k} · "
            f"recall {_mean_pct(row.recall_macro)}, the mean of each claim's piece recall")


def _claim_block(number: int, claim, pieces: list[list], ranks: dict, reason: str | None,
                 method: str, k: int, depth: int) -> str:
    """One claim on the claims page: its number, text and type, its status
    badge (_status_badge: reason None means retrieved; otherwise the
    miss_taxonomy key), one rank chip per method, and its recall at k with
    counts, then — folded — each piece of evidence with its own badge and
    chips and its quotes (_quotes: a human-added row can hold several)."""
    piece_ranks = [{m: ranks[(claim.claim_id, g, m)] for m in _RETRIEVAL_METHODS} for g in range(len(pieces))]
    best = {m: min((r[m] for r in piece_ranks if pd.notna(r[m])), default=np.nan) for m in _RETRIEVAL_METHODS}
    hit = sum(1 for r in piece_ranks if _within_k(r[method], k))
    folded, n_quotes = [], 0
    for i, (piece, piece_rank) in enumerate(zip(pieces, piece_ranks), start=1):
        quotes = [f'<blockquote class="quote">{_e(q)} <code>{_e(row.chunk_id)}</code></blockquote>'
                  for row in piece for q in _quotes(row)]
        n_quotes += len(quotes)
        folded.append(f'<div class="piece"><p class="meta"><b>Piece {i}</b>{_piece_badge(piece_rank[method], k, depth)}'
                      f'{_rank_chips(piece_rank, method, depth)}</p>{"".join(quotes)}</div>')
    return (f'<article class="claim" id="c{number}"><div class="n"><a href="#c{number}">{number}</a></div><div>'
            f'<p class="text">{_e(claim.claim_text)} <span class="tag">{_e(claim.bucket.lower())}</span></p>'
            f'<p class="meta">{_status_badge(reason, k, depth)}{_rank_chips(best, method, depth)}'
            f'<span>{hit} of {_count(len(pieces), "piece")} retrieved — {_pct(hit, len(pieces))}</span></p>'
            f'<details><summary>{_count(len(pieces), "piece")} of evidence, {_count(n_quotes, "quote")}</summary>'
            f'{"".join(folded)}</details></div></article>')


def render_claims(run: Run, *, method: str = "dense", k: int = _TOP_K, pages: dict[str, str] | None = None) -> str:
    """The claims page for one run at (method, k): every verifiable claim,
    numbered, grouped by memo and section in claims-file order (the order
    the memo reads, the same for every method and k). Each claim wears its
    status at (method, k) as one of four badges (retrieved, or its
    miss_taxonomy reason), one rank chip per method, and its recall at k
    with counts; folded under it, each piece of evidence with its own badge,
    chips and confirmed quotes. Quotes the pipeline groups as one piece
    (decision 1) share its rank. Memo and section headings carry claim
    coverage and per-claim recall read from the run's metrics
    (_heading_counts), a table of contents links every memo and section
    heading, and the page explains once, in plain words, what recall counts
    (decision 19's duplicated-document residual). No baseline changes here;
    the summary carries them. `pages` names the files to link between, as in
    render_summary — but None keeps this page's bold label and Jump-to links
    (_nav), leaving out only the links to the other pages. Returns the HTML."""
    _check_view(run, method, k)
    depth = run.meta["depth"]
    pieces = _evidence_pieces(run)
    ranks = run.claim_hits.set_index(["claim_id", "group", "method"])["best_rank"].to_dict()
    misses = miss_taxonomy(run, method, k)
    reason_of = dict(zip(misses["claim_id"], misses["key"]))
    verifiable = run.claims[run.claims["bucket"] != "UNVERIFIABLE"]

    def counts(scope: str, memo_id: str, section: str = "ALL") -> str:
        return _heading_counts(_required_row(run, scope, memo_id, section, method, k), k)

    legend = ", ".join(f"{m} = {_METHOD_LABEL[m]}" for m in _RETRIEVAL_METHODS)
    intro = (
        f'<p class="sub">Every verifiable claim ({len(verifiable)}), with the evidence a human confirmed and '
        f'the best rank at which each search method retrieved it within the top {depth} passages '
        f'per search phrase ({_e(legend)}). A chip reads <span class="rank dense">dense 2</span> = found at '
        f'rank 2; <span class="rank keyword nf">keyword &gt;{depth}</span> = not within the top '
        f'{depth}; the ringed chip is the method this page scores. A piece of evidence counts as '
        f'retrieved when that method ranks it within k = {k}; quotes the pipeline groups as one piece share '
        f'its rank. Recall counts every piece a claim lists: when the review lists alternative or duplicate '
        f'sources for the same fact, each counts, so recall understates how often the fact itself was found; '
        f'claim coverage does not.</p>')
    toc, bodies, memo_ids = [], [], []
    number = 0
    for memo_id in run.meta["memos"]:
        memo_claims = verifiable[verifiable["memo_id"] == memo_id]
        if memo_claims.empty:
            continue
        memo_ids.append(memo_id)
        memo_counts = counts("memo", memo_id)
        items, sections = [], []
        for i, (section, claims) in enumerate(memo_claims.groupby("section", sort=False), start=1):
            section_id = f"s-{_e(memo_id)}-{i}"
            section_counts = counts("section", memo_id, str(section))
            items.append(f'<li><a href="#{section_id}">{_e(section)}</a> '
                         f'<span class="cnt">{section_counts}</span></li>')
            sections.append(f'<h3 id="{section_id}">{_e(section)} <span class="cnt">{section_counts}</span></h3>')
            for claim in claims.itertuples(index=False):
                number += 1
                sections.append(_claim_block(number, claim, pieces[claim.claim_id], ranks,
                                             reason_of.get(claim.claim_id), method, k, depth))
        heading, entry = _memo_heading_and_toc(memo_id, memo_counts, memo_counts, f'<ul>{"".join(items)}</ul>')
        bodies.extend([heading, *sections])
        toc.append(entry)
    out = [f"<h1>Claims — {_e(run.run_id)}</h1>", _view_line(run, method, k),
           _nav(run, pages, "claims", _jump_line(memo_ids)), intro,
           f'<nav class="toc">{"".join(toc)}</nav>', *bodies]
    return _page(f"Claims {run.run_id}", out)


_PREVIEW_CHARS = 200


def _preview(text) -> str:
    """The folded one-line preview of a passage: whitespace collapsed to
    single spaces (many passages open with a table header), cut to
    _PREVIEW_CHARS characters, and only then escaped, so the cut cannot fall
    inside an HTML entity. CSS clips it to its line with an ellipsis; no
    literal '…' enters the page (the full text is always right below)."""
    return _e(" ".join(str(text).split())[:_PREVIEW_CHARS])


def render_rereview(run: Run) -> str:
    """The re-review page: every candidate (rereview_candidates), none cut,
    grouped by memo and under its claim — claims numbered, the one with the
    strongest passage first, its passages strongest first. A folded passage
    shows its score, its passage id and a one-line preview of its text
    (_preview); opened, the full text keeps its line breaks and a long one
    scrolls inside its own box (.passage). A table of contents links every
    memo heading. The same for every method and k, so one file per run
    (_page_names). It links to no other page: it cannot know which summary
    it was opened from, and the browser's Back button returns there. Returns
    the HTML."""
    toc, bodies, memo_ids = [], [], []
    number = 0
    for memo_id in run.meta["memos"]:
        memo = run.candidates[run.candidates["memo_id"] == memo_id]
        if memo.empty:
            continue
        memo_ids.append(memo_id)
        memo_counts = f'{_count(memo["claim_id"].nunique(), "claim")} · {_count(len(memo), "passage")}'
        heading, entry = _memo_heading_and_toc(memo_id, f"{memo_counts} · strongest passage first",
                                               f'{memo_counts} · strongest {memo["score"].max():.3f}')
        toc.append(entry)
        bodies.append(heading)
        # stored strongest first, so each claim first appears at its strongest passage
        for _, group in memo.groupby("claim_id", sort=False):
            number += 1
            passages = list(group.itertuples(index=False))
            folded = "".join(
                f'<details><summary><span class="score">{p.score:.3f}</span><code>{_e(p.chunk_id)}</code>'
                f'<span class="preview">{_preview(p.chunk_text)}</span></summary>'
                f'<pre class="passage">{_e(p.chunk_text)}</pre></details>' for p in passages)
            bodies.append(f'<article class="claim" id="r{number}"><div class="n"><a href="#r{number}">{number}</a></div><div>'
                          f'<p class="text">{_e(passages[0].claim_text)}</p>'
                          f'<p class="meta"><span class="tag">{_e(passages[0].section)}</span>'
                          f'<span>{_count(len(passages), "passage")} · strongest {passages[0].score:.3f}</span></p>'
                          f'{folded}</div></article>')
    out = [f"<h1>Re-review candidates — {_e(run.run_id)}</h1>",
           f'<p class="sub">{_e(", ".join(run.meta["memos"]))} · the same for every search method and k</p>',
           _nav(run, None, "rereview", _jump_line(memo_ids)),
           _unverifiable_note(run, "below"),
           "<p>A worklist for a person, not a count of errors: for each claim, the passages meaning-based search "
           "offers for its own text that the review never showed a human, strongest first. Only a person can "
           "say whether one supports the claim. The number beside each passage is its similarity to the claim "
           "(0–1); open a passage to read it in full.</p>",
           f'<nav class="toc">{"".join(toc)}</nav>', *bodies]
    return _page(f"Re-review {run.run_id}", out)


def write_report(run_id: str, *, method: str = "dense", k: int = _TOP_K,
                 baseline_id: str | None = None, runs_dir: str = "eval_runs") -> list[str]:
    """Renders the three pages and writes them into eval_runs/<run_id>/ under
    _page_names; returns their paths, summary first."""
    run = load_run(run_id, runs_dir)
    baseline = load_run(baseline_id, runs_dir) if baseline_id else None
    return _write_pages(run, baseline, method, k, runs_dir)


def _write_pages(run: Run, baseline: Run | None, method: str, k: int, runs_dir: str) -> list[str]:
    """write_report's work on runs already loaded. All three pages are
    rendered before any is written, so a refused page leaves no half-updated
    set."""
    pages = _page_names(method, k, baseline.run_id if baseline is not None else None)
    rendered = {"summary": render_summary(run, method=method, k=k, baseline=baseline, pages=pages),
                "claims": render_claims(run, method=method, k=k, pages=pages),
                "rereview": render_rereview(run)}
    paths = []
    for page, text in rendered.items():
        path = os.path.join(runs_dir, run.run_id, pages[page])
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        paths.append(path)
    return paths


def show_report(run_id: str | None = None, method: str = "dense", k: int = _TOP_K,
                baseline: str | None = None, runs_dir: str = "eval_runs") -> list[str]:
    """For the notebook demo: write the three pages (run_id None means the
    latest run), print their paths, and, when running under IPython, display
    the summary inline. The inline copy is rendered again without its link
    line — in a notebook a relative link resolves against the notebook, not
    the run folder, so it would be dead — and so differs from the file on
    disk by that line, on purpose. Open the printed paths in a browser to
    move between pages. Returns the paths, summary first."""
    run = load_run(run_id or "latest", runs_dir)
    base = load_run(baseline, runs_dir) if baseline else None
    paths = _write_pages(run, base, method, k, runs_dir)
    try:
        # Optional (notebook only), so not in requirements.txt; the ignore keeps
        # pyright quiet where IPython is not installed, as in CI.
        from IPython.display import HTML, display  # pyright: ignore[reportMissingImports]
    except ImportError:
        pass
    else:
        display(HTML(render_summary(run, method=method, k=k, baseline=base)))
    for path in paths:
        print(f"report: {path}")
    return paths


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
    K = 5               # passages per search phrase, 1..the run's scored depth
    BASELINE = None     # a run id to show changes against, or None
    show_report(RUN, method=METHOD, k=K, baseline=BASELINE)


# %% [markdown]
# ## Command line

# %%
_USAGE = ("usage: python eval_pipeline.py score [label] | "
          "report <run_id|latest> [baseline_run_id] [--method=dense|keyword|both] [--k=N, 1..the run's scored depth]")


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
            flags = {}
            for a in extra:
                if a.startswith("--") and "=" in a:
                    name, value = a[2:].split("=", 1)
                    flags[name] = value
            if 1 <= len(positional) <= 2 and set(flags) <= {"method", "k"} and len(positional) + len(flags) == len(extra):
                paths = write_report(positional[0], method=flags.get("method", "dense"),
                                     k=int(flags.get("k", _TOP_K)),
                                     baseline_id=positional[1] if len(positional) == 2 else None)
                for path in paths:
                    print(f"eval: wrote {path}")
                return
    except ValueError as e:  # EvalInputError is a ValueError, as are a bad --method or --k
        raise SystemExit(str(e))
    raise SystemExit(_USAGE)


if __name__ == "__main__":
    _main(sys.argv)
