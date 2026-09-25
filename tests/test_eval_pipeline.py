"""
eval_pipeline.py — scoring retrieval against the golden set (spec:
docs/superpowers/specs/2026-09-10-retrieval-eval-harness-design.md).

One small fictional world, built fresh per test in tmp_path:

  Business Profile: C1 extractive  (evidence d.pdf_1 "sells widgets")
                    C2 synthesized (evidence d.pdf_3 "five plants";
                                    d.pdf_7 found but REJECTED -> unverifiable)
                    C3 unverifiable (found=False)
                    C5 unverifiable (d.pdf_6 found but rejected)
  Ownership:        C4 extractive  (evidence d.pdf_5 "40% stake")

Dense retrieval returns, for Business Profile, d.pdf_0, d.pdf_2 (the
neighbour whose overlap with d.pdf_1 CONTAINS C1's quote), d.pdf_4 (neighbour of d.pdf_3 WITHOUT
the quote), d.pdf_6, d.pdf_7 (C2's rejected chunk); for Ownership d.pdf_6,
d.pdf_5. So dense covers 0/3 at k=1 and 2/3 from k=2 on.
"""
import ast
import os
import re
import sys
import types
import uuid
from pathlib import Path

import pandas as pd
import pytest
import yaml

import eval_pipeline as ep
import golden_set_pipeline as gsp
import retrieval_pipeline as rp

MEMO = "MEMO-T"
BP, OWN = "Business Profile", "Ownership"
C1 = "Acme sells widgets."
C2 = "Acme runs five plants."
C3 = "Acme was founded in 1900."
C4 = "Globex owns 40% of Acme."
C5 = "Acme is the market leader."
CHUNKS = {
    "d.pdf_0": "Intro page of the annual report.",
    "d.pdf_1": "Acme sells widgets worldwide.",
    "d.pdf_2": "sells widgets worldwide. Retail sales grew.",   # starts with the end of d.pdf_1, as chunks overlap
    "d.pdf_3": "Acme runs five plants in total.",
    "d.pdf_4": "Acme runs plants in Ohio.",
    "d.pdf_5": "Globex holds a 40% stake in Acme.",
    "d.pdf_6": "Unrelated market commentary.",
    "d.pdf_7": "Plant count table.",
}
REVIEWED_COLUMNS = ["claim_id", "memo_id", "section", "claim_text", "doc_id", "chunk_id", "chunk_text",
                    "bm25_score", "evidence_span", "found", "confidence", "ambiguous_match",
                    "verbatim_match", "human_reviewed", "tag", "tag_draft", "tag_rationale"]


def cid(section, text):
    return gsp._derive_claim_id(MEMO, section, text, 0)


def _row(section, text, chunk_id, span, tag, found=True,
         rationale="bundle review 2026-09-22: checked"):
    return {"claim_id": cid(section, text), "memo_id": MEMO, "section": section, "claim_text": text,
            "doc_id": "d.pdf" if chunk_id else None, "chunk_id": chunk_id,
            "chunk_text": CHUNKS.get(chunk_id), "bm25_score": 1.0 if chunk_id else None,
            "evidence_span": span, "found": found, "confidence": "high" if found else None,
            "ambiguous_match": False, "verbatim_match": True if found else None,
            "human_reviewed": False, "tag": tag, "tag_draft": tag, "tag_rationale": rationale}


def _reviewed_rows():
    return [
        _row(BP, C1, "d.pdf_1", "sells widgets", "extractive"),
        _row(BP, C2, "d.pdf_3", "five plants", "synthesized"),
        _row(BP, C2, "d.pdf_7", "Plant count", "unverifiable"),
        _row(BP, C3, None, None, "unverifiable", found=False,
             rationale="auto: found=False, nothing was found by search"),
        _row(BP, C5, "d.pdf_6", "market commentary", "unverifiable"),
        _row(OWN, C4, "d.pdf_5", "40% stake", "extractive"),
    ]


def _result(section, method, phrase, rank, chunk_id, score=1.0):
    return {"memo_id": MEMO, "section": section, "phrase": phrase, "phrase_index": 0, "method": method,
            "doc_id": "d.pdf", "chunk_id": chunk_id, "chunk_text": CHUNKS[chunk_id],
            "rank": rank, "score": score}


def _results_rows():
    rows = [_result(BP, "dense", "business profile", r, c, 1 - r / 10)
            for r, c in enumerate(["d.pdf_0", "d.pdf_2", "d.pdf_4", "d.pdf_6", "d.pdf_7"], start=1)]
    rows += [_result(OWN, "dense", "shareholders", 1, "d.pdf_6", 0.9),
             _result(OWN, "dense", "shareholders", 2, "d.pdf_5", 0.8)]
    rows += [_result(BP, "keyword", "business profile", 1, "d.pdf_3", 5.0)]  # Ownership: no keyword rows
    rows += [_result(BP, "both", "business profile", r, c)
             for r, c in enumerate(["d.pdf_0", "d.pdf_3", "d.pdf_2"], start=1)]
    rows += [_result(OWN, "both", "shareholders", r, c) for r, c in enumerate(["d.pdf_6", "d.pdf_5"], start=1)]
    return rows


def _claim_query_rows():
    def q(section, text, rank, chunk_id, score):
        return {"memo_id": MEMO, "section": section, "claim_id": cid(section, text), "claim_text": text,
                "doc_id": "d.pdf", "chunk_id": chunk_id, "chunk_text": CHUNKS[chunk_id],
                "rank": rank, "score": score}
    return [
        q(BP, C1, 1, "d.pdf_1", 0.95),
        q(BP, C2, 1, "d.pdf_3", 0.93),
        q(BP, C3, 1, "d.pdf_0", 0.80), q(BP, C3, 2, "d.pdf_6", 0.60),
        q(BP, C5, 1, "d.pdf_6", 0.90), q(BP, C5, 2, "d.pdf_0", 0.70),
        q(OWN, C4, 1, "d.pdf_5", 0.97),
    ]


def _write_reviewed(rows):
    pd.DataFrame(rows, columns=REVIEWED_COLUMNS).to_excel(os.path.join("reviewed", f"{MEMO}.xlsx"), index=False)


def _write_index(texts=None, model="test-model"):
    texts = CHUNKS if texts is None else texts
    pd.DataFrame([{"chunk_id": c, "doc_id": "d.pdf", "chunk_text": t, "start_offset": 0,
                   "embedding": [0.0, 1.0], "model": model} for c, t in texts.items()],
                 columns=rp._INDEX_COLUMNS).to_parquet(os.path.join("retrieval_index", f"{MEMO}.parquet"), index=False)


def _provenance(command, rows, default_depth, **override):
    """What retrieve / recheck record: model, depth and the fingerprint of
    each memo's index as it is on disk now."""
    memos = {r["memo_id"] for r in rows} | {MEMO}
    indexes = {m: rp._index_fingerprint(pd.read_parquet(os.path.join("retrieval_index", f"{m}.parquet")))
               for m in sorted(memos) if os.path.exists(os.path.join("retrieval_index", f"{m}.parquet"))}
    return {"command": command, "model": "test-model", "depth": default_depth, "indexes": indexes, **override}


def _write_results(rows, **override):
    rp._write_with_provenance(pd.DataFrame(rows, columns=rp._RESULTS_COLUMNS), "retrieval_results.parquet",
                              _provenance("retrieve", rows, rp._RETRIEVE_DEPTH, **override))


def _write_claim_queries(rows, **override):
    rp._write_with_provenance(pd.DataFrame(rows, columns=rp._CLAIM_QUERY_COLUMNS), "claim_queries.parquet",
                              _provenance("recheck", rows, rp._TOP_K, **override))


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for d in ("claims", "reviewed", "retrieval_index", "retrieval"):
        os.makedirs(d)
    gsp.write_claims_file(os.path.join("claims", f"{MEMO}.md"), MEMO, "src", {},
                          [(BP, [C1, C2, C3, C5]), (OWN, [C4])])
    _write_reviewed(_reviewed_rows())
    _write_index()
    _write_results(_results_rows())
    _write_claim_queries(_claim_query_rows())
    with open(os.path.join("retrieval", f"{MEMO}.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"memo_id": MEMO, "source_folder": "src",
                        "sections": {BP: ["business profile"], OWN: ["shareholders"]}}, f, sort_keys=False)
    return tmp_path


# --- structure --------------------------------------------------------------------------------

def test_imports_from_sibling_modules_resolve():
    # A rename in golden_set_pipeline or retrieval_pipeline must fail loudly here, not at first run.
    from golden_set_pipeline import (  # noqa: F401
        _HUMAN_ADDED_PREFIX, _QUOTE_SEPARATOR, _UnionFind, _chunk_index_of, _claims_with_occurrence, _derive_claim_id, _group_equivalent_chunks,
        _is_deterministic_claim_id, _normalize_span, _same_evidence, _valid_memo_id, parse_claims_file,
    )
    from retrieval_pipeline import (  # noqa: F401
        _CLAIM_QUERY_COLUMNS, _RESULTS_COLUMNS, _RETRIEVAL_METHODS, _RETRIEVE_DEPTH, _TOP_K,
        _excel_safe, _index_fingerprint, _load_index_model, _read_phrase_config, dedupe_by_section,
        read_provenance,
    )


def test_the_scorer_never_retrieves():
    # Structural, and an ALLOWLIST rather than a blocklist: eval_pipeline.py
    # imports exactly these names from its siblings and only these modules, so
    # nothing that searches (_embed_texts, BM25Okapi, _fuse_rrf, _rank_by_score,
    # _cosine_scores, bm25_shortlist) or reaches the network (requests, urllib,
    # socket, subprocess) can slip in. A new import must be added here on purpose.
    # The runtime counterpart, with sockets disabled, is in the report section.
    tree = ast.parse(open(ep.__file__, encoding="utf-8").read())
    from_names, modules = {}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            from_names.setdefault(node.module, set()).update(a.name for a in node.names)
            modules.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            modules.update(a.name.split(".")[0] for a in node.names)
    assert from_names["golden_set_pipeline"] == {
        "_HUMAN_ADDED_PREFIX", "_QUOTE_SEPARATOR", "_UnionFind", "_chunk_index_of", "_claims_with_occurrence", "_derive_claim_id", "_group_equivalent_chunks",
        "_is_deterministic_claim_id", "_normalize_span", "_same_evidence", "_valid_memo_id", "parse_claims_file"}
    assert from_names["retrieval_pipeline"] == {
        "_CLAIM_QUERY_COLUMNS", "_RESULTS_COLUMNS", "_RETRIEVAL_METHODS", "_RETRIEVE_DEPTH",
        "_TOP_K", "_excel_safe", "_index_fingerprint", "_load_index_model", "_read_phrase_config",
        "dedupe_by_section", "read_provenance"}
    assert modules <= {"hashlib", "html", "json", "logging", "os", "re", "shutil", "sys", "tempfile", "dataclasses",
                       "datetime", "numpy", "pandas", "yaml", "openpyxl", "golden_set_pipeline",
                       "retrieval_pipeline", "IPython"}


# --- ground truth -----------------------------------------------------------------------------

def test_ground_truth_census_buckets_and_evidence(world):
    gt = ep.load_ground_truth(MEMO)
    assert list(gt.claims.columns) == ep._CLAIM_COLUMNS
    assert gt.claims["claim_text"].tolist() == [C1, C2, C3, C5, C4]  # census (claims-file) order
    buckets = dict(zip(gt.claims["claim_text"], gt.claims["bucket"]))
    assert buckets == {C1: "EXTRACTIVE", C2: "SYNTHESIZED", C3: "UNVERIFIABLE",
                       C5: "UNVERIFIABLE", C4: "EXTRACTIVE"}
    # the relevant set: found AND extractive/synthesized — d.pdf_7 and d.pdf_6 were rejected
    assert sorted(gt.evidence["chunk_id"]) == ["d.pdf_1", "d.pdf_3", "d.pdf_5"]
    assert sorted(gt.judged["chunk_id"]) == ["d.pdf_1", "d.pdf_3", "d.pdf_5", "d.pdf_6", "d.pdf_7"]
    assert all(v is None or isinstance(v, str) for v in gt.evidence["doc_id"])


def test_tag_case_and_whitespace_are_tolerated(world):
    rows = _reviewed_rows()
    rows[0]["tag"] = "  Extractive "
    _write_reviewed(rows)
    gt = ep.load_ground_truth(MEMO)
    assert gt.claims.set_index("claim_text").loc[C1, "bucket"] == "EXTRACTIVE"


def test_blank_tags_halt_and_list_every_row(world):
    rows = _reviewed_rows()
    rows[0]["tag"] = None
    rows[5]["tag"] = "   "
    _write_reviewed(rows)
    with pytest.raises(ep.EvalInputError) as e:
        ep.load_ground_truth(MEMO)
    blank = [p for p in e.value.problems if "blank" in p]
    assert len(blank) == 2
    assert "row 2" in blank[0]
    assert "row 7" in blank[1]


def test_unknown_tag_halts(world):
    rows = _reviewed_rows()
    rows[0]["tag"] = "extractve"
    _write_reviewed(rows)
    with pytest.raises(ep.EvalInputError, match="'extractve' is not one of"):
        ep.load_ground_truth(MEMO)


def test_rejected_tag_halts_with_its_own_reason(world):
    rows = _reviewed_rows()
    rows[2]["tag"] = "rejected"
    _write_reviewed(rows)
    with pytest.raises(ep.EvalInputError, match="import_reviewed"):
        ep.load_ground_truth(MEMO)


def test_verified_tag_on_found_false_row_halts(world):
    rows = _reviewed_rows()
    rows[3]["tag"] = "extractive"
    _write_reviewed(rows)
    with pytest.raises(ep.EvalInputError, match="found=False"):
        ep.load_ground_truth(MEMO)


def test_census_claim_without_a_row_halts(world):
    _write_reviewed(_reviewed_rows()[:-1])  # drop C4's only row
    with pytest.raises(ep.EvalInputError, match="has no row"):
        ep.load_ground_truth(MEMO)


def test_sheet_row_outside_the_census_halts(world):
    rows = _reviewed_rows()
    rows.append(_row(BP, "Acme sells gadgets.", "d.pdf_1", "sells", "extractive"))
    _write_reviewed(rows)
    with pytest.raises(ep.EvalInputError, match="wording changed"):
        ep.load_ground_truth(MEMO)


def test_pre_decision_17_ids_halt_once(world):
    rows = _reviewed_rows()
    for r in rows:
        r["claim_id"] = str(uuid.uuid4())
    _write_reviewed(rows)
    with pytest.raises(ep.EvalInputError) as e:
        ep.load_ground_truth(MEMO)
    assert len(e.value.problems) == 1
    assert "predates design decision 17" in e.value.problems[0]


def test_blank_claim_id_is_named_as_such(world):
    rows = _reviewed_rows()
    rows[0]["claim_id"] = None
    _write_reviewed(rows)
    with pytest.raises(ep.EvalInputError) as e:
        ep.load_ground_truth(MEMO)
    assert any("row 2: claim_id is blank" in p for p in e.value.problems)
    assert not any("predates" in p for p in e.value.problems)


def test_duplicate_claim_chunk_pair_halts(world):
    rows = _reviewed_rows()
    rows.append(dict(rows[0]))
    _write_reviewed(rows)
    with pytest.raises(ep.EvalInputError, match="twice"):
        ep.load_ground_truth(MEMO)


def test_all_problems_are_reported_together(world):
    rows = _reviewed_rows()
    rows[0]["tag"] = None
    rows[2]["tag"] = "rejected"
    _write_reviewed(rows)
    with pytest.raises(ep.EvalInputError) as e:
        ep.load_ground_truth(MEMO)
    assert len(e.value.problems) == 2


def test_missing_files_halt(world):
    os.remove(os.path.join("reviewed", f"{MEMO}.xlsx"))
    with pytest.raises(ep.EvalInputError, match="not found"):
        ep.load_ground_truth(MEMO)


# --- retrieval checks -------------------------------------------------------------------------

def test_index_parity_passes(world):
    ep.check_index_parity(ep.load_ground_truth(MEMO))   # no EvalInputError


def test_index_parity_missing_chunk_halts(world):
    _write_index({c: t for c, t in CHUNKS.items() if c != "d.pdf_3"})
    gt = ep.load_ground_truth(MEMO)
    with pytest.raises(ep.EvalInputError, match="chunked differently"):
        ep.check_index_parity(gt)


def test_index_parity_differing_text_halts(world):
    _write_index({**CHUNKS, "d.pdf_1": "Acme sells gadgets worldwide."})
    gt = ep.load_ground_truth(MEMO)
    with pytest.raises(ep.EvalInputError, match="text differs"):
        ep.check_index_parity(gt)


def test_index_parity_ignores_control_characters_the_sheet_cannot_hold(world):
    _write_index({**CHUNKS, "d.pdf_1": "Acme sells\x00 widgets worldwide."})
    ep.check_index_parity(ep.load_ground_truth(MEMO))   # no EvalInputError


def test_section_with_claims_but_no_dense_rows_halts(world):
    _write_results([r for r in _results_rows() if not (r["section"] == OWN and r["method"] == "dense")])
    gt, results = ep.load_ground_truth(MEMO), ep.load_results()
    with pytest.raises(ep.EvalInputError, match="Ownership"):
        ep.check_sections_retrieved(gt, results)


def test_section_where_keyword_found_nothing_is_not_a_halt(world):
    # The fixture has no keyword rows for Ownership: a legitimate zero, not drift.
    ep.check_sections_retrieved(ep.load_ground_truth(MEMO), ep.load_results())


def test_results_written_before_the_method_column_halt(world):
    pd.DataFrame(_results_rows()).drop(columns=["method"]).to_parquet("retrieval_results.parquet", index=False)
    with pytest.raises(ep.EvalInputError, match="re-run python retrieval_pipeline.py retrieve"):
        ep.load_results()


# --- scoring ----------------------------------------------------------------------------------

def _scored():
    gt = ep.load_ground_truth(MEMO)
    ranked = ep.ranked_chunks(ep.load_results(), MEMO)
    claim_hits = ep.score_claims(gt, ranked)
    chunk_hits = ep.score_chunks(gt, ranked)
    return gt, ranked, claim_hits, chunk_hits, ep.compute_metrics(claim_hits, chunk_hits)


def _hit(claim_hits, text, section, method):
    return claim_hits[(claim_hits.claim_id == cid(section, text)) & (claim_hits.method == method)].iloc[0]


def _metric(metrics, method, k, scope="all", memo_id="ALL", section="ALL"):
    m = metrics
    return m[(m.scope == scope) & (m.memo_id == memo_id) & (m.section == section)
             & (m.method == method) & (m.k == k)].iloc[0]


def test_a_neighbour_containing_the_quote_is_a_hit(world):
    _, _, claim_hits, _, _ = _scored()
    h = _hit(claim_hits, C1, BP, "dense")
    assert (h.best_rank, h.best_chunk_id, h.best_golden_chunk_id) == (2, "d.pdf_2", "d.pdf_1")


def test_the_same_chunk_is_a_hit(world):
    _, _, claim_hits, _, _ = _scored()
    h = _hit(claim_hits, C4, OWN, "dense")
    assert (h.best_rank, h.best_chunk_id, h.best_golden_chunk_id) == (2, "d.pdf_5", "d.pdf_5")


def test_adjacency_without_containment_and_rejected_chunks_are_not_hits(world):
    # dense retrieves d.pdf_4 (next to C2's evidence, without its quote) and
    # d.pdf_7 (a chunk the reviewer rejected for C2): neither counts.
    _, _, claim_hits, _, _ = _scored()
    assert pd.isna(_hit(claim_hits, C2, BP, "dense").best_rank)


def test_unverifiable_claims_are_not_scored(world):
    _, _, claim_hits, _, _ = _scored()
    assert not claim_hits.claim_id.isin([cid(BP, C3), cid(BP, C5)]).any()


def test_dense_coverage_by_depth(world):
    *_, metrics = _scored()
    assert [(_metric(metrics, "dense", k).covered, _metric(metrics, "dense", k).claims) for k in (1, 2, 3, 5, 20)] == \
        [(0, 3), (2, 3), (2, 3), (2, 3), (2, 3)]


def test_keyword_and_both_coverage(world):
    *_, metrics = _scored()
    assert _metric(metrics, "keyword", 1).covered == 1   # d.pdf_3 is C2's own evidence
    assert _metric(metrics, "both", 2).covered == 2      # C2 (d.pdf_3) and C4 (d.pdf_5)
    assert _metric(metrics, "both", 3).covered == 3      # + C1 via d.pdf_2


def test_cohort_split(world):
    *_, metrics = _scored()
    m = _metric(metrics, "dense", 2)
    assert (m.claims_extractive, m.covered_extractive, m.claims_synthesized, m.covered_synthesized) == (2, 2, 1, 0)


def test_recall_and_mrr(world):
    *_, metrics = _scored()
    m = _metric(metrics, "dense", 2)
    assert m.recall_macro == pytest.approx(2 / 3)
    assert m.recall_micro == pytest.approx(2 / 3)
    assert m.mrr == pytest.approx((1 / 2 + 1 / 2 + 0) / 3)
    assert _metric(metrics, "dense", 1).mrr == 0


def test_precision_and_f1_are_pooled_per_memo_only(world):
    *_, metrics = _scored()
    m = _metric(metrics, "dense", 2, scope="memo", memo_id=MEMO)
    assert (m.retrieved, m.retrieved_golden) == (4, 2)   # d.pdf_0, d.pdf_2, d.pdf_6, d.pdf_5
    assert m.precision == pytest.approx(0.5)
    assert m.f1 == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))
    assert pd.isna(_metric(metrics, "dense", 2).precision)


def test_complete_synthesized_needs_every_group(world):
    *_, metrics = _scored()
    assert _metric(metrics, "keyword", 1).complete_synthesized == 1   # C2's only group
    assert _metric(metrics, "dense", 2).complete_synthesized == 0


def test_passages_count_what_the_model_is_handed(world):
    *_, metrics = _scored()
    assert _metric(metrics, "dense", 2).passages == 4                      # 2 per section
    assert _metric(metrics, "dense", 5, scope="section", memo_id=MEMO, section=BP).passages == 5
    assert _metric(metrics, "keyword", 5, scope="section", memo_id=MEMO, section=OWN).passages == 0


def test_section_scope(world):
    *_, metrics = _scored()
    m = _metric(metrics, "dense", 2, scope="section", memo_id=MEMO, section=BP)
    assert (m.covered, m.claims) == (1, 2)


def test_reading_best_rank_equals_deduping_at_each_k(world):
    # A second phrase ranks d.pdf_6 first (the first phrase has it at 4), so a
    # chunk's best rank is a real minimum over phrases and the two readings
    # could differ if the code were wrong.
    second = [dict(_result(BP, "dense", "second phrase", r, c), phrase_index=1)
              for r, c in ((1, "d.pdf_6"), (2, "d.pdf_1"))]
    _write_results(_results_rows() + second)
    ranked = ep.ranked_chunks(ep.load_results(), MEMO)
    assert ranked[(BP, "dense")].set_index("chunk_id").loc["d.pdf_6", "rank"] == 1
    results = ep.load_results()
    sub = results[(results.memo_id == MEMO) & (results.section == BP) & (results.method == "dense")]
    for k in range(1, 6):
        by_rank = set(ranked[(BP, "dense")].query("rank <= @k")["chunk_id"])
        assert by_rank == set(rp.dedupe_by_section(sub, top_k=k)["chunk_id"]), k


def test_multi_group_evidence_exercises_complete_synthesized_and_recall_differences(world):
    # C2 has two evidence groups: d.pdf_3 "five plants" and d.pdf_4 "plants in Ohio".
    # d.pdf_3 and d.pdf_4 are adjacent chunks; neither span contains the other, so they form
    # two separate groups in _group_equivalent_chunks.
    rows = _reviewed_rows()
    rows.insert(2, _row(BP, C2, "d.pdf_4", "plants in Ohio", "synthesized"))
    _write_reviewed(rows)

    # Add keyword result for d.pdf_4 at rank 2 so keyword retrieves both groups.
    # keyword already has d.pdf_3 at rank 1, so with d.pdf_4 at rank 2, keyword hits both groups at k=2.
    results = _results_rows()
    results.append(_result(BP, "keyword", "business profile", 2, "d.pdf_4", 4.5))
    _write_results(results)

    gt, ranked, claim_hits, chunk_hits, metrics = _scored()

    # C2 should have 2 claim_hits rows per method (one per group).
    c2_hits = claim_hits[claim_hits.claim_id == cid(BP, C2)]
    for method in rp._RETRIEVAL_METHODS:
        c2_method = c2_hits[c2_hits.method == method]
        assert len(c2_method) == 2, f"{method}: expected 2 groups for C2, got {len(c2_method)}"

    # Dense at k=3: dense returns d.pdf_4 at rank 3 (hits group 2 only).
    # C1 dense k=3: d.pdf_2 rank 2 hits group 1 → 1/1
    # C2 dense k=3: d.pdf_4 rank 3 hits group 2 → 1/2
    # C4 dense k=3: d.pdf_5 rank 2 hits group 1 → 1/1
    # recall_macro = mean((1, 0.5, 1)) = 2.5/3 ≈ 0.833
    # complete_synthesized = 0 (C2 is synthesized but only 1/2 groups hit)
    m_dense_k3 = _metric(metrics, "dense", 3)
    assert m_dense_k3.recall_macro == pytest.approx(2.5 / 3)
    assert m_dense_k3.complete_synthesized == 0
    # micro pools the groups instead: 3 of the 4 groups hit, which macro (2.5/3) does not equal
    assert m_dense_k3.recall_micro == pytest.approx(3 / 4)
    assert m_dense_k3.recall_micro != pytest.approx(m_dense_k3.recall_macro)
    assert m_dense_k3.covered_synthesized == 1   # C2 counts as covered with one group of two

    # Dense at k=2: dense returns d.pdf_2 rank 2 and d.pdf_5 rank 2 (neither reaches C2's groups).
    # C1 dense k=2: 1/1
    # C2 dense k=2: 0/2
    # C4 dense k=2: 1/1
    # recall_macro = 2/3 (unchanged from fixture)
    m_dense_k2 = _metric(metrics, "dense", 2)
    assert m_dense_k2.recall_macro == pytest.approx(2 / 3)
    assert m_dense_k2.complete_synthesized == 0

    # Keyword at k=2: keyword returns d.pdf_3 rank 1 and d.pdf_4 rank 2 (hits both C2 groups).
    # C1 keyword k=2: nothing (d.pdf_1 not in results) → 0/1
    # C2 keyword k=2: d.pdf_3 rank 1 and d.pdf_4 rank 2 both hit → 2/2 (BOTH groups!)
    # C4 keyword k=2: nothing (OWN has no keyword results) → 0/1
    # recall_macro = mean((0, 1, 0)) = 1/3
    # complete_synthesized = 1 (C2 is synthesized and ALL groups are hit at k=2)
    m_keyword_k2 = _metric(metrics, "keyword", 2)
    assert m_keyword_k2.recall_macro == pytest.approx(1 / 3)
    assert m_keyword_k2.complete_synthesized == 1

    # Keyword at k=1: keyword returns only d.pdf_3 rank 1 (hits group 1 only).
    # C1 keyword k=1: 0/1
    # C2 keyword k=1: 1/2
    # C4 keyword k=1: 0/1
    # recall_macro = mean((0, 0.5, 0)) = 1/6
    # complete_synthesized = 0 (only 1/2 groups hit)
    m_keyword_k1 = _metric(metrics, "keyword", 1)
    assert m_keyword_k1.recall_macro == pytest.approx(1 / 6)
    assert m_keyword_k1.complete_synthesized == 0


# --- re-review candidates and runs ------------------------------------------------------------
from datetime import datetime, timedelta  # noqa: E402

NOW = datetime(2026, 9, 23, 10, 0, 0)


def test_rereview_candidates_skip_judged_chunks_and_verifiable_claims(world):
    gt = ep.load_ground_truth(MEMO)
    cands = ep.rereview_candidates(gt, ep.load_claim_queries("claim_queries.parquet", [gt]))
    # C5's d.pdf_6 was already judged (and rejected) by a human; C1/C2/C4 are verifiable
    assert list(zip(cands["claim_text"], cands["chunk_id"])) == [(C3, "d.pdf_0"), (C5, "d.pdf_0"), (C3, "d.pdf_6")]


def test_stale_claim_queries_halt(world):
    q = pd.read_parquet("claim_queries.parquet")
    _write_claim_queries(q[q.claim_id != cid(BP, C3)].to_dict("records"))
    gt = ep.load_ground_truth(MEMO)
    with pytest.raises(ep.EvalInputError, match="re-run python retrieval_pipeline.py recheck"):
        ep.load_claim_queries("claim_queries.parquet", [gt])


def test_missing_claim_queries_halt(world):
    os.remove("claim_queries.parquet")
    gt = ep.load_ground_truth(MEMO)
    with pytest.raises(ep.EvalInputError, match="recheck"):
        ep.load_claim_queries("claim_queries.parquet", [gt])


def _rephrase(bp_phrase):
    """The phrase lever done properly: new phrase file AND results retrieved with it."""
    with open(os.path.join("retrieval", f"{MEMO}.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"memo_id": MEMO, "source_folder": "src",
                        "sections": {BP: [bp_phrase], OWN: ["shareholders"]}}, f)
    r = pd.read_parquet("retrieval_results.parquet")
    r.loc[r.section == BP, "phrase"] = bp_phrase
    _write_results(r.to_dict("records"))


def test_score_run_writes_a_complete_run(world):
    run_dir = ep.score_run("template", now=NOW)
    assert os.path.basename(run_dir) == "2026-09-23_100000_template"
    assert sorted(os.listdir(run_dir)) == sorted([f"{t}.parquet" for t in ep._RUN_TABLES] + ["meta.json"])
    assert not any(n.endswith(".tmp") for n in os.listdir("eval_runs"))
    run = ep.load_run("latest")
    assert run.run_id == "2026-09-23_100000_template"
    assert run.meta["memos"] == [MEMO]
    assert run.meta["embedding_model"] == "test-model"
    assert run.meta["phrases"][MEMO] == {BP: ["business profile"], OWN: ["shareholders"]}
    assert run.meta["phrase_counts"][MEMO] == {BP: 1, OWN: 1}
    m = ep._metric_row(run.metrics, "all", "ALL", "ALL", "dense", 2)
    assert (m.covered, m.claims) == (2, 3)
    assert len(run.candidates) == 3


def test_a_failed_write_leaves_no_temporary_run_folder(world, monkeypatch):
    # The temporary folder holds client claim and chunk text; latest_run_id never shows it.
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
    with pytest.raises(OSError, match="disk full"):
        ep.score_run(now=NOW)
    assert os.listdir("eval_runs") == []


def test_runs_are_never_overwritten(world):
    first = ep.score_run(now=NOW)
    second = ep.score_run(now=NOW)                       # same second
    assert os.path.basename(second) == os.path.basename(first) + "_2"
    assert ep.load_run(os.path.basename(first)).run_id == os.path.basename(first)


def test_edited_phrases_without_a_new_retrieve_halt(world):
    # The phrase lever done wrong: the phrase file changed, retrieve did not run.
    with open(os.path.join("retrieval", f"{MEMO}.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"memo_id": MEMO, "source_folder": "src",
                        "sections": {BP: ["plants"], OWN: ["shareholders"]}}, f)
    with pytest.raises(ep.EvalInputError, match="re-run python retrieval_pipeline.py retrieve"):
        ep.score_run(now=NOW)


def test_excel_lock_files_in_reviewed_are_ignored(world):
    with open(os.path.join("reviewed", f"{MEMO}.xlsx"), "rb") as src, \
         open(os.path.join("reviewed", f"~${MEMO}.xlsx"), "wb") as dst:
        dst.write(src.read())
    run = ep.load_run(os.path.basename(ep.score_run(now=NOW)))
    assert run.meta["memos"] == [MEMO]


def test_score_run_reports_problems_from_every_input_together(world):
    rows = _reviewed_rows()
    rows[0]["tag"] = None
    _write_reviewed(rows)
    os.remove(os.path.join("retrieval", f"{MEMO}.yaml"))
    with pytest.raises(ep.EvalInputError) as e:
        ep.score_run(now=NOW)
    text = str(e.value)
    assert "blank" in text
    assert "phrase config" in text


def test_score_run_lists_missing_sections_and_phrase_drift_together(world):
    # A section with no dense rows must not hide that the phrase file changed too.
    _write_results([r for r in _results_rows() if not (r["section"] == OWN and r["method"] == "dense")])
    with open(os.path.join("retrieval", f"{MEMO}.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"memo_id": MEMO, "source_folder": "src",
                        "sections": {BP: ["plants"], OWN: ["shareholders"]}}, f)
    with pytest.raises(ep.EvalInputError) as e:
        ep.score_run(now=NOW)
    assert any("has claims to score" in p for p in e.value.problems)
    assert any("re-run python retrieval_pipeline.py retrieve" in p for p in e.value.problems)


def test_malformed_claims_file_is_listed_with_the_other_problems(world):
    with open(os.path.join("claims", f"{MEMO}.md"), "w", encoding="utf-8") as f:
        f.write("no frontmatter here\n")
    with open(os.path.join("retrieval", f"{MEMO}.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"memo_id": MEMO, "source_folder": "src",
                        "sections": {BP: ["plants"], OWN: ["shareholders"]}}, f)
    with pytest.raises(ep.EvalInputError) as e:
        ep.score_run(now=NOW)
    assert any("frontmatter" in p for p in e.value.problems)
    assert any("re-run python retrieval_pipeline.py retrieve" in p for p in e.value.problems)


def test_phrases_are_compared_as_retrieve_reads_them(world):
    # retrieve strips each phrase; a quoted trailing space or a folded '>' scalar
    # (which ends in a newline) must not read as an edited phrase file.
    with open(os.path.join("retrieval", f"{MEMO}.yaml"), "w", encoding="utf-8") as f:
        f.write(f"memo_id: {MEMO}\nsource_folder: src\nsections:\n"
                f"  {BP}:\n    - \"business profile  \"\n"
                f"  {OWN}:\n    - >\n      shareholders\n")
    phrase_path = os.path.join("retrieval", f"{MEMO}.yaml")
    assert ep.check_results_match_phrases(MEMO, ep.load_results(), phrase_path) == \
        {BP: ["business profile"], OWN: ["shareholders"]}


def test_an_invalid_phrase_config_is_an_input_problem(world):
    with open(os.path.join("retrieval", f"{MEMO}.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"memo_id": "OTHER", "source_folder": "src", "sections": {BP: ["business profile"]}}, f)
    with pytest.raises(ep.EvalInputError, match="filename stem"):
        ep.score_run(now=NOW)


def test_a_phrase_file_that_is_not_yaml_is_listed_with_the_other_problems(world):
    # yaml.YAMLError is not a ValueError; it must not escape as a traceback
    # and hide the other problems already collected.
    rows = _reviewed_rows()
    rows[0]["tag"] = None
    _write_reviewed(rows)
    with open(os.path.join("retrieval", f"{MEMO}.yaml"), "w", encoding="utf-8") as f:
        f.write(f"memo_id: {MEMO}\nsource_folder: 'src\n")          # unclosed quote
    with pytest.raises(ep.EvalInputError) as e:
        ep.score_run(now=NOW)
    assert any("not valid YAML" in p for p in e.value.problems)
    assert any("tag is blank" in p for p in e.value.problems)


def test_score_run_refuses_indexes_embedded_with_different_models(world):
    # A second memo, MEMO-U, whose index reports another embedding model.
    other = "MEMO-U"
    other_cid = gsp._derive_claim_id(other, OWN, C4, 0)
    gsp.write_claims_file(os.path.join("claims", f"{other}.md"), other, "src", {}, [(OWN, [C4])])
    row = dict(_row(OWN, C4, "d.pdf_5", "40% stake", "extractive"), claim_id=other_cid, memo_id=other)
    pd.DataFrame([row], columns=REVIEWED_COLUMNS).to_excel(os.path.join("reviewed", f"{other}.xlsx"), index=False)
    pd.DataFrame([{"chunk_id": c, "doc_id": "d.pdf", "chunk_text": t, "start_offset": 0,
                   "embedding": [0.0, 1.0], "model": "other-model"} for c, t in CHUNKS.items()],
                 columns=rp._INDEX_COLUMNS).to_parquet(os.path.join("retrieval_index", f"{other}.parquet"), index=False)
    _write_results(_results_rows() + [dict(_result(OWN, m, "shareholders", 1, "d.pdf_5"), memo_id=other)
                                      for m in rp._RETRIEVAL_METHODS])   # a complete results set
    queries = pd.read_parquet("claim_queries.parquet")
    extra = {"memo_id": other, "section": OWN, "claim_id": other_cid, "claim_text": C4, "doc_id": "d.pdf",
             "chunk_id": "d.pdf_5", "chunk_text": CHUNKS["d.pdf_5"], "rank": 1, "score": 0.9}
    _write_claim_queries(queries.to_dict("records") + [extra])
    with open(os.path.join("retrieval", f"{other}.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"memo_id": other, "source_folder": "src", "sections": {OWN: ["shareholders"]}}, f)
    with pytest.raises(ep.EvalInputError) as e:
        ep.score_run(now=NOW)
    # check_provenance reports it, once per file that used the other model
    assert [p for p in e.value.problems if "other-model" in p] == [
        f"retrieval_results.parquet: produced with model 'test-model', but "
        f"{os.path.join('retrieval_index', other + '.parquet')} was embedded with 'other-model'; "
        f"re-run python retrieval_pipeline.py retrieve",
        f"claim_queries.parquet: produced with model 'test-model', but "
        f"{os.path.join('retrieval_index', other + '.parquet')} was embedded with 'other-model'; "
        f"re-run python retrieval_pipeline.py recheck"]


def test_bad_run_label_halts(world):
    with pytest.raises(ep.EvalInputError, match="run label"):
        ep.score_run("bad label", now=NOW)


def test_golden_fingerprint_follows_the_tags_not_the_retrieval(world):
    before = ep.golden_fingerprint([ep.load_ground_truth(MEMO)])
    _write_results(_results_rows()[:3] + _results_rows()[5:])  # retrieval changed
    assert ep.golden_fingerprint([ep.load_ground_truth(MEMO)]) == before
    rows = _reviewed_rows()
    rows[1]["tag"] = "extractive"                               # a verdict changed
    _write_reviewed(rows)
    assert ep.golden_fingerprint([ep.load_ground_truth(MEMO)]) != before


def test_runs_compare_across_a_phrase_change_but_not_a_golden_change(world):
    first = ep.load_run(os.path.basename(ep.score_run("a", now=NOW)))
    _rephrase("plants")
    second = ep.load_run(os.path.basename(ep.score_run("b", now=NOW + timedelta(seconds=1))))
    assert second.meta["phrase_hash"] != first.meta["phrase_hash"]
    ep.check_comparable(second, first)                          # the phrase lever: allowed
    rows = _reviewed_rows()
    rows[1]["tag"] = "extractive"
    _write_reviewed(rows)
    third = ep.load_run(os.path.basename(ep.score_run("c", now=NOW + timedelta(seconds=2))))
    with pytest.raises(ep.EvalInputError, match="golden set"):
        ep.check_comparable(third, first)


def test_runs_embedded_with_different_models_are_not_comparable(world):
    first = ep.load_run(os.path.basename(ep.score_run("a", now=NOW)))
    second = ep.load_run(os.path.basename(ep.score_run("b", now=NOW + timedelta(seconds=1))))
    second.meta = {**second.meta, "embedding_model": "other-model"}
    with pytest.raises(ep.EvalInputError) as e:
        ep.check_comparable(second, first)
    assert e.value.problems == [
        f"{second.run_id} used 'other-model', {first.run_id} used 'test-model' — not comparable"]


def test_main_score_prints_the_headline(world, capsys):
    ep._main(["eval_pipeline.py", "score", "cli"])
    out = capsys.readouterr().out
    assert "dense" in out
    assert "67% (2/3)" in out


def test_main_usage_and_refusal(world):
    with pytest.raises(SystemExit, match="usage"):
        ep._main(["eval_pipeline.py"])
    os.remove("claim_queries.parquet")
    with pytest.raises(SystemExit, match="eval refused"):
        ep._main(["eval_pipeline.py", "score"])


# --- the report -------------------------------------------------------------------------------

def _run(label="a", now=NOW):
    return ep.load_run(os.path.basename(ep.score_run(label, now=now)))


def _pages(run, k=5):
    """The three pages of a run at dense, k — summary, claims, re-review."""
    return [ep.render_summary(run, k=k), ep.render_claims(run, k=k), ep.render_rereview(run)]


def _with_second_evidence_row_for_c1():
    """C1 gets a second evidence row listed FIRST (d.pdf_7, which dense
    retrieves at rank 5): two pieces of evidence, group 0 = d.pdf_7,
    group 1 = d.pdf_1 (the one dense finds at rank 2 via its neighbour)."""
    rows = _reviewed_rows()
    rows.insert(0, _row(BP, C1, "d.pdf_7", "Plant count", "extractive"))
    _write_reviewed(rows)


def test_report_headline_counts_and_cohort_split(world):
    page = ep.render_summary(_run(), method="dense", k=2)
    assert "67% (2/3)" in page      # claim coverage
    assert "100% (2/2)" in page     # extractive
    assert "0% (0/1)" in page       # synthesized
    assert "at least one retrieved" in page
    assert "all pieces retrieved: 0% (0/1)" in page
    assert "k = 2 passages per search phrase" in page
    assert "per query" not in page
    assert "one claim moves coverage by about 33 points" in page


def test_report_counts_passages_of_sections_without_verifiable_claims(world):
    # History holds only an unverifiable claim — nothing to score — but dense
    # still handed the model 3 passages for it.
    hist, c6 = "History", "Acme was founded in 1850."
    gsp.write_claims_file(os.path.join("claims", f"{MEMO}.md"), MEMO, "src", {},
                          [(BP, [C1, C2, C3, C5]), (OWN, [C4]), (hist, [c6])])
    _write_reviewed(_reviewed_rows() + [_row(hist, c6, None, None, "unverifiable", found=False,
                                             rationale="auto: found=False, nothing was found by search")])
    _write_results(_results_rows() + [_result(hist, m, "history", r, c)
                                      for m in ("dense", "both")
                                      for r, c in enumerate(["d.pdf_0", "d.pdf_6", "d.pdf_7"], start=1)])
    queries = pd.read_parquet("claim_queries.parquet")
    extra = {"memo_id": MEMO, "section": hist, "claim_id": cid(hist, c6), "claim_text": c6, "doc_id": "d.pdf",
             "chunk_id": "d.pdf_0", "chunk_text": CHUNKS["d.pdf_0"], "rank": 1, "score": 0.5}
    _write_claim_queries(queries.to_dict("records") + [extra])
    with open(os.path.join("retrieval", f"{MEMO}.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"memo_id": MEMO, "source_folder": "src",
                        "sections": {BP: ["business profile"], OWN: ["shareholders"], hist: ["history"]}}, f)
    run = _run()
    # the metrics table itself carries the section, so the stored run and the page agree
    m = ep._metric_row(run.metrics, "section", MEMO, hist, "dense", 5)
    assert (m.claims, m.passages) == (0, 3)
    assert pd.isna(m.coverage)
    page = ep.render_summary(run, method="dense", k=5)
    assert "<div class=\"v\">10</div><div class=\"l\">across 3 sections" in page   # 5 + 2 + 3
    assert f"<td>{hist}</td><td>n/a</td><td>3</td>" in page
    assert re.search(rf"<th>{MEMO}</th><th>all</th><td>.*?</td><td>10</td>", page)   # memo row = its sections' sum
    # keyword returned passages for one section only; the tile counts that one
    assert "<div class=\"v\">1</div><div class=\"l\">across 1 sections" in ep.render_summary(_run("b"), method="keyword")


def test_summary_opens_with_the_traced_examples(world):
    page = ep.render_summary(_run())
    assert page.index("<h2>Traced examples</h2>") < page.index('<div class="tiles">')


def test_summary_folds_the_traced_quotes(world):
    page = ep.render_summary(_run(), k=2)
    traced = page.split("<h2>Traced examples</h2>")[1].split("<h2>")[0]
    assert "<details><summary>" in traced


def test_summary_leaves_the_candidates_to_the_rereview_page(world):
    page = ep.render_summary(_run())
    assert "Intro page of the annual report." not in page   # a candidate passage


def test_claims_page_lists_every_verifiable_claim_once_in_claims_file_order(world):
    page = ep.render_claims(_run(), k=2)
    assert [page.count(c) for c in (C1, C2, C4)] == [1, 1, 1]
    assert page.index(C1) < page.index(C2) < page.index(C4)


def test_claims_page_leaves_out_unverifiable_claims(world):
    page = ep.render_claims(_run(), k=2)
    assert C3 not in page
    assert C5 not in page


def test_claims_page_headline_gives_each_methods_best_rank(world):
    page = ep.render_claims(_run(), k=2)
    assert "dense: rank 2 · keyword: not found · both: rank 3" in page         # C1
    assert "dense: not found · keyword: rank 1 · both: rank 2" in page         # C2


def test_claims_page_gives_the_miss_reason_at_the_chosen_k(world):
    page = ep.render_claims(_run(), k=2)
    assert "missed at k = 2: found by another method at this depth" in page   # C2
    assert page.count("retrieved at k = 2") == 2                               # C1, C4


def test_claims_page_ranks_each_piece_of_evidence_on_its_own(world):
    _with_second_evidence_row_for_c1()
    page = ep.render_claims(_run(), k=2)
    assert "2 pieces of evidence, 2 quotes" in page
    first = page.index("dense: rank 5 · keyword: not found · both: not found")    # piece 1: d.pdf_7
    assert first < page.index("Plant count (d.pdf_7)") < page.index("sells widgets (d.pdf_1)")


def test_rereview_page_lists_every_candidate_under_its_claim_strongest_first(world):
    page = ep.render_rereview(_run())
    order = [page.index(s) for s in (f"<b>1.</b> {C3}", "0.800 · d.pdf_0", "0.600 · d.pdf_6",
                                     f"<b>2.</b> {C5}", "0.700 · d.pdf_0")]
    assert order == sorted(order)


def test_rereview_page_folds_each_candidate(world):
    page = ep.render_rereview(_run())
    assert page.count("<details>") == 3


def test_rereview_page_shows_every_passage_in_full(world):
    run = _run()
    run.candidates.loc[0, "chunk_text"] = "x" * 400
    page = ep.render_rereview(run)
    assert "x" * 400 in page
    assert "…" not in page


def test_rereview_page_says_it_is_the_same_for_every_method_and_k(world):
    assert "the same for every search method and k" in ep.render_rereview(_run())


def test_report_pages_are_self_contained(world):
    for page in _pages(_run()):
        for banned in ("http://", "https://", "<script", "<link", "@import"):
            assert banned not in page


def test_only_the_summary_draws_the_chart(world):
    assert [page.count("<svg") for page in _pages(_run())] == [1, 0, 0]


@pytest.mark.parametrize("render, listed", [("render_summary", "on the Re-review page"),
                                            ("render_rereview", "below")])
def test_unverifiable_count_comes_with_the_rereview_sentence(world, render, listed):
    page = getattr(ep, render)(_run())
    assert "2 of 5 claims were not found in the sources by this process" in page
    assert "For 2 of them" in page
    assert f"3 listed {listed}." in page
    assert "absent from the sources" not in page


def test_claims_page_shows_no_unverifiable_count(world):
    assert "not found in the sources" not in ep.render_claims(_run())


def test_report_escapes_every_text(world):
    run = _run()
    run.claims.loc[run.claims.claim_text == C1, "claim_text"] = "<script>alert(1)</script>"
    run.candidates.loc[0, "chunk_text"] = "<script>alert(2)</script>"
    for page in _pages(run, k=2):
        assert "<script" not in page
    assert "&lt;script&gt;alert(1)" in ep.render_claims(run, k=2)
    assert "&lt;script&gt;alert(2)" in ep.render_rereview(run)


def test_report_shows_a_delta_against_a_baseline(world):
    base = _run("base", NOW)
    rows = _results_rows()
    for r in rows:  # the phrase lever: C2's own evidence now ranks first for dense
        if r["section"] == BP and r["method"] == "dense" and r["rank"] == 1:
            r.update(chunk_id="d.pdf_3", chunk_text=CHUNKS["d.pdf_3"])
    _write_results(rows)
    _rephrase("plants")
    current = _run("edited", NOW + timedelta(seconds=1))
    page = ep.render_summary(current, k=2, baseline=base)
    assert "100% (3/3)" in page
    assert "+33 pts" in page
    assert base.run_id in page
    assert "different number of search phrases" not in page   # same count: reworded, not added


def test_report_refuses_an_incomparable_baseline(world):
    base = _run("base", NOW)
    rows = _reviewed_rows()
    rows[1]["tag"] = "extractive"
    _write_reviewed(rows)
    current = _run("retagged", NOW + timedelta(seconds=1))
    with pytest.raises(ep.EvalInputError, match="golden set"):
        ep.render_summary(current, baseline=base)


def test_traced_example_quotes_the_evidence_that_was_found(world):
    # the hit is on d.pdf_1's quote via the neighbour d.pdf_2, though C1's
    # first evidence row is d.pdf_7. The summary must quote d.pdf_1.
    _with_second_evidence_row_for_c1()
    page = ep.render_summary(_run(), k=2)
    assert "sells widgets (d.pdf_1)" in page
    assert "Plant count (d.pdf_7)" not in page
    assert "the neighbouring passage d.pdf_2, which contains this quote" in page


def test_report_shows_no_raw_metrics(world):
    # MRR rises with the number of phrases and is not comparable between
    # sections; precision and macro recall need notes the pages no longer carry.
    # All stay in metrics.parquet, none on a page.
    for page in _pages(_run()):
        for banned in ("<h2>Details</h2>", "mrr", "MRR", "precision", "recall_macro", "macro recall"):
            assert banned not in page


def test_report_warns_when_the_phrase_count_changed(world):
    base = _run("base", NOW)
    extra = [dict(_result(BP, m, "second", 1, "d.pdf_3"), phrase_index=1) for m in ("dense", "both")]
    _write_results(_results_rows() + extra)
    with open(os.path.join("retrieval", f"{MEMO}.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"memo_id": MEMO, "source_folder": "src",
                        "sections": {BP: ["business profile", "second"], OWN: ["shareholders"]}}, f)
    current = _run("more", NOW + timedelta(seconds=1))
    page = ep.render_summary(current, baseline=base)
    assert "different number of search phrases in 1 section(s)" in page


def test_scoring_and_reporting_never_touch_the_network(world, monkeypatch):
    # Runtime counterpart of test_the_scorer_never_retrieves: any socket at all fails the test.
    import socket

    def refuse(*a, **k):
        raise AssertionError("eval_pipeline tried to open a network connection")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    _run()
    ep.write_report("latest")


def test_miss_taxonomy(world):
    t = ep.miss_taxonomy(_run(), "dense", 1)
    assert dict(zip(t["claim_id"], t["category"])) == {
        cid(BP, C1): "found only deeper (by k=20)",
        cid(BP, C2): "found by another method at this depth",
        cid(OWN, C4): "found only deeper (by k=20)",
    }


@pytest.mark.parametrize("render", ["render_summary", "render_claims"])
def test_render_rejects_an_unknown_method(world, render):
    with pytest.raises(ValueError, match="method"):
        getattr(ep, render)(_run(), method="semantic")


@pytest.mark.parametrize("render", ["render_summary", "render_claims"])
def test_render_rejects_an_unknown_k(world, render):
    with pytest.raises(ValueError, match="k"):
        getattr(ep, render)(_run(), k=21)


def test_scoring_records_each_pieces_chunk_ids(world):
    _with_second_evidence_row_for_c1()
    hits = _run().claim_hits    # read back from disk
    c1 = hits[(hits.claim_id == cid(BP, C1)) & (hits.method == "dense")].sort_values("group")
    assert [list(ids) for ids in c1["group_chunk_ids"]] == [["d.pdf_7"], ["d.pdf_1"]]


def test_claims_page_reads_the_pieces_from_the_run_not_a_regrouping(world, monkeypatch):
    # The grouping code changes after the run was scored: the page must still
    # show each quote under the rank scoring gave its own piece.
    _with_second_evidence_row_for_c1()
    run = _run()
    regroup = ep._evidence_groups
    monkeypatch.setattr(ep, "_evidence_groups", lambda golden: regroup(golden)[::-1])
    page = ep.render_claims(run, k=2)
    order = [page.index(s) for s in ("piece 1 — dense: rank 5", "Plant count (d.pdf_7)",
                                     "piece 2 — dense: rank 2", "sells widgets (d.pdf_1)")]
    assert order == sorted(order)


def _drop_piece_members_on_disk(run_id):
    """Make the stored run look like one scored before claim_hits recorded
    each piece's chunk ids."""
    path = os.path.join("eval_runs", run_id, "claim_hits.parquet")
    pd.read_parquet(path).drop(columns=["group_chunk_ids"]).to_parquet(path, index=False)


def test_claims_page_refuses_a_run_scored_without_piece_members(world):
    run = _run()
    _drop_piece_members_on_disk(run.run_id)
    with pytest.raises(ep.EvalInputError, match="claims page needs"):
        ep.render_claims(ep.load_run(run.run_id))


def test_a_run_scored_without_piece_members_still_serves_as_a_baseline(world):
    # A baseline reads only meta and metrics; re-scoring could not recreate it
    # once the retrieval results have moved on.
    base = _run("base", NOW)
    _drop_piece_members_on_disk(base.run_id)
    current = _run("again", NOW + timedelta(seconds=1))
    paths = ep.write_report(current.run_id, baseline_id=base.run_id)
    assert os.path.basename(paths[0]) == f"report_dense_k5_vs_{base.run_id}.html"


def test_claims_page_refuses_pieces_that_do_not_match_the_evidence(world):
    _with_second_evidence_row_for_c1()
    run = _run()
    run.evidence = run.evidence[run.evidence["chunk_id"] != "d.pdf_7"]
    with pytest.raises(ep.EvalInputError, match="pieces of evidence"):
        ep.render_claims(run)


def test_a_refused_page_leaves_no_report_files(world):
    run = _run()
    _drop_piece_members_on_disk(run.run_id)
    with pytest.raises(ep.EvalInputError):
        ep.write_report(run.run_id)
    assert not [f for f in os.listdir(os.path.join("eval_runs", run.run_id)) if f.endswith(".html")]


def test_report_pages_link_to_each_other_by_relative_name(world):
    _run()
    summary, claims, rereview = (Path(p).read_text(encoding="utf-8") for p in ep.write_report("latest"))
    assert 'href="report_dense_k5_claims.html"' in summary
    assert 'href="report_rereview.html"' in summary
    assert 'href="report_dense_k5.html"' in claims
    assert 'href="report_rereview.html"' in claims
    assert "href=" not in rereview


def test_claims_page_links_back_to_the_summary_with_its_baseline(world):
    base = _run("base", NOW)
    current = _run("again", NOW + timedelta(seconds=1))
    paths = ep.write_report(current.run_id, baseline_id=base.run_id)
    stem = f"report_dense_k5_vs_{base.run_id}"
    assert [os.path.basename(p) for p in paths] == [f"{stem}.html", f"{stem}_claims.html", "report_rereview.html"]
    assert f'href="{stem}.html"' in Path(paths[1]).read_text(encoding="utf-8")


def test_summary_without_page_names_has_no_links(world):
    assert "href=" not in ep.render_summary(_run())


def test_show_report_displays_the_summary_without_links(world, monkeypatch):
    shown = []
    display = types.ModuleType("IPython.display")
    display.HTML = str
    display.display = shown.append
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    _run()
    ep.show_report()
    assert len(shown) == 1
    assert "<h2>Traced examples</h2>" in shown[0]
    assert "href=" not in shown[0]


def test_report_command_writes_the_three_pages(world, capsys):
    run = _run()
    ep._main(["eval_pipeline.py", "report", "latest"])
    out = capsys.readouterr().out
    for name in ("report_dense_k5.html", "report_dense_k5_claims.html", "report_rereview.html"):
        assert os.path.exists(os.path.join("eval_runs", run.run_id, name))
        assert name in out


def test_show_report_writes_and_prints_the_three_pages(world, capsys):
    _run()
    paths = ep.show_report(method="both", k=3)
    assert [os.path.basename(p) for p in paths] == ["report_both_k3.html", "report_both_k3_claims.html",
                                                     "report_rereview.html"]
    assert all(os.path.exists(p) for p in paths)
    assert "report_both_k3_claims.html" in capsys.readouterr().out


def test_report_command_takes_method_and_k(world):
    run = _run()
    ep._main(["eval_pipeline.py", "report", "latest", "--method=keyword", "--k=7"])
    assert os.path.exists(os.path.join("eval_runs", run.run_id, "report_keyword_k7_claims.html"))


def test_report_command_refuses_bad_flags(world):
    _run()
    with pytest.raises(SystemExit, match="usage"):
        ep._main(["eval_pipeline.py", "report", "latest", "--colour=red"])
    with pytest.raises(SystemExit, match="method"):
        ep._main(["eval_pipeline.py", "report", "latest", "--method=semantic"])


def test_demo_cell_is_off_by_default():
    assert ep.RUN_DEMO_REPORT is False


# --- final-verification audit: doc_id and depth ---------------------------------------------

def test_verified_row_with_blank_doc_id_halts(world):
    # _same_evidence never matches without a doc_id, so neighbour hits would
    # silently stop counting for this row.
    rows = _reviewed_rows()
    rows[0]["doc_id"] = None
    _write_reviewed(rows)
    with pytest.raises(ep.EvalInputError, match="doc_id"):
        ep.load_ground_truth(MEMO)


def test_evidence_doc_id_must_match_the_index(world):
    rows = _reviewed_rows()
    rows[0]["doc_id"] = "other.pdf"
    _write_reviewed(rows)
    gt = ep.load_ground_truth(MEMO)
    with pytest.raises(ep.EvalInputError, match="doc_id"):
        ep.check_index_parity(gt)


def test_results_without_provenance_halt(world):
    # written before retrieve recorded what produced it
    pd.DataFrame(_results_rows(), columns=rp._RESULTS_COLUMNS).to_parquet("retrieval_results.parquet", index=False)
    with pytest.raises(ep.EvalInputError, match="retrieval_results.parquet: records no provenance"):
        ep.score_run(now=NOW)


def test_results_retrieved_at_another_depth_halt(world):
    _write_results(_results_rows(), depth=rp._RETRIEVE_DEPTH - 5)
    with pytest.raises(ep.EvalInputError, match="_RETRIEVE_DEPTH"):
        ep.score_run(now=NOW)


def test_claim_queries_rechecked_at_another_depth_halt(world):
    _write_claim_queries(_claim_query_rows(), depth=rp._TOP_K + 5)
    with pytest.raises(ep.EvalInputError, match="claim_queries.parquet: rechecked at depth 10, but _TOP_K is 5"):
        ep.score_run(now=NOW)


def test_index_re_embedded_with_another_model_after_retrieve_halts(world):
    # Same chunks, new model: index parity passes, yet the results (and the
    # claim queries) came from the old model's vectors.
    _write_index(model="new-model")
    with pytest.raises(ep.EvalInputError) as e:
        ep.score_run(now=NOW)
    text = str(e.value)
    assert "retrieval_results.parquet: produced with model 'test-model'" in text
    assert "new-model" in text
    assert "claim_queries.parquet: produced with model 'test-model'" in text


def test_index_rebuilt_with_other_chunks_after_retrieve_halts(world):
    # d.pdf_0 is no golden chunk, so parity passes — but a result chunk_id
    # could now name different text.
    _write_index({**CHUNKS, "d.pdf_0": "A different intro page."})
    with pytest.raises(ep.EvalInputError, match="its chunks changed since"):
        ep.score_run(now=NOW)


def test_miss_labels_follow_the_depth():
    assert all(str(rp._RETRIEVE_DEPTH) in c for c in ep._MISS_CATEGORIES[1:])


# --- PR #2 review ------------------------------------------------------------------------------

@pytest.mark.parametrize("sheet_section", ["Somewhere Else", None])
def test_evidence_section_comes_from_the_census_not_the_sheet(world, sheet_section):
    # claim_id already encodes the section; an edited or blank sheet cell must
    # not move the evidence (score_chunks groups evidence by section).
    rows = _reviewed_rows()
    rows[0]["section"] = sheet_section
    _write_reviewed(rows)
    gt = ep.load_ground_truth(MEMO)
    assert gt.evidence.set_index("chunk_id").loc["d.pdf_1", "section"] == BP


def test_a_phrase_without_combined_rows_halts(world):
    # "both" fuses the dense list, so it has rows whenever dense does; a
    # missing method would otherwise score 0% as if the retriever failed.
    _write_results([r for r in _results_rows() if not (r["section"] == OWN and r["method"] == "both")])
    gt, results = ep.load_ground_truth(MEMO), ep.load_results()
    with pytest.raises(ep.EvalInputError, match="'both'"):
        ep.check_sections_retrieved(gt, results)


def test_a_memo_without_any_keyword_rows_halts(world):
    _write_results([r for r in _results_rows() if r["method"] != "keyword"])
    gt, results = ep.load_ground_truth(MEMO), ep.load_results()
    with pytest.raises(ep.EvalInputError, match="keyword"):
        ep.check_sections_retrieved(gt, results)


def test_latest_ignores_folders_that_are_not_runs(world):
    run_id = os.path.basename(ep.score_run(now=NOW))
    os.makedirs(os.path.join("eval_runs", "zz-archive"))   # sorts after every timestamp
    assert ep.latest_run_id() == run_id


def test_latest_orders_same_second_runs_past_nine(world):
    # _write_run's suffixes sort as text "_10" < "_9"; latest must not.
    ids = [os.path.basename(ep.score_run(now=NOW)) for _ in range(11)]
    assert ids[-1].endswith("_11")
    assert ep.latest_run_id() == ids[-1]


def test_a_run_missing_a_table_is_refused_cleanly(world):
    run_dir = ep.score_run(now=NOW)
    os.remove(os.path.join(run_dir, "metrics.parquet"))
    run_id = os.path.basename(run_dir)
    with pytest.raises(ep.EvalInputError, match="metrics.parquet"):
        ep.load_run(run_id)


def test_a_neighbour_with_the_quote_outside_the_shared_text_is_not_a_hit(world):
    # The same passage can only sit where two adjacent chunks overlap. Here
    # d.pdf_2 repeats "sells widgets" in a sentence of its own, past the text
    # it shares with d.pdf_1 ("worldwide."), so it is not C1's evidence.
    elsewhere = "worldwide. Later, Acme sells widgets to retailers."
    rows = [dict(r, chunk_text=elsewhere) if r["chunk_id"] == "d.pdf_2" else r for r in _results_rows()]
    _write_results(rows)
    _, _, claim_hits, _, _ = _scored()
    assert pd.isna(_hit(claim_hits, C1, BP, "dense").best_rank)


def test_a_quote_longer_than_the_shared_text_is_not_hit_by_the_neighbour(world):
    # C1's quote is all of d.pdf_1; the neighbour d.pdf_2 shares only its end
    # ("sells widgets worldwide."). The shared text lies inside the quote, but
    # the quote does not lie inside the neighbour — it is not C1's evidence.
    rows = _reviewed_rows()
    rows[0]["evidence_span"] = CHUNKS["d.pdf_1"]
    _write_reviewed(rows)
    _, _, claim_hits, _, _ = _scored()
    assert pd.isna(_hit(claim_hits, C1, BP, "dense").best_rank)


# --- PR #2 review, round 2 -------------------------------------------------------------------

def test_percentages_round_halves_away_from_zero():
    assert ep._pct(1, 8) == "13% (1/8)"      # 12.5 -> 13, not banker's 12
    assert ep._pct(3, 8) == "38% (3/8)"      # 37.5 -> 38, the same way
    assert ep._round_half_away(-2.5) == -3
    assert ep._round_half_away(2.5) == 3


def test_a_human_added_row_with_several_quotes_is_hit_by_its_neighbour(world):
    # tag_pipeline joins a human's pasted quotes with " | "; the joined text
    # appears in no chunk, so each quote must be tested on its own.
    rows = _reviewed_rows()
    rows[0]["evidence_span"] = "Plant count | sells widgets"
    rows[0]["tag_rationale"] = "human-added 2026-09-22: quote 'Plant count'; quote 'sells widgets'"
    _write_reviewed(rows)
    _, _, claim_hits, _, _ = _scored()
    h = _hit(claim_hits, C1, BP, "dense")
    assert (h.best_rank, h.best_chunk_id) == (2, "d.pdf_2")


def test_a_found_quote_holding_the_separator_is_not_split(world):
    # Only a human-added row joins quotes; a found quote is chunk text, and a
    # table row can carry " | " itself. Split, its fragment "sells widgets"
    # alone would make the neighbour a hit.
    rows = _reviewed_rows()
    rows[0]["evidence_span"] = "Plant count | sells widgets"
    _write_reviewed(rows)
    _, _, claim_hits, _, _ = _scored()
    assert _hit(claim_hits, C1, BP, "dense").best_chunk_id != "d.pdf_2"


def test_a_human_added_row_is_grouped_by_its_quotes(world):
    # The person's first quote is the passage found in the neighbouring chunk:
    # one piece of evidence, one group. Grouped on the joined text, which
    # neither contains nor is contained by the found quote, it would be two.
    rows = _reviewed_rows()
    rows.insert(1, dict(_row(BP, C1, "d.pdf_2", "sells widgets worldwide | Retail sales grew", "extractive",
                             rationale="human-added 2026-09-22: two quotes")))
    rows[0]["evidence_span"] = "Acme sells widgets worldwide."
    _write_reviewed(rows)
    _, _, claim_hits, _, _ = _scored()
    c1 = claim_hits[(claim_hits["claim_id"] == cid(BP, C1)) & (claim_hits["method"] == "dense")]
    assert len(c1) == 1
    assert len(c1.iloc[0]["group_chunk_ids"]) == 2


def test_a_reviewed_sheet_with_an_invalid_name_is_warned_about(world, caplog):
    import shutil
    shutil.copy(os.path.join("reviewed", f"{MEMO}.xlsx"), os.path.join("reviewed", f"{MEMO} copy.xlsx"))
    with caplog.at_level("WARNING", logger="eval"):
        run = ep.load_run(os.path.basename(ep.score_run(now=NOW)))
    assert run.meta["memos"] == [MEMO]
    assert f"{MEMO} copy.xlsx" in caplog.text


def test_a_temporary_finalize_sheet_in_reviewed_is_skipped_with_a_warning(world, caplog):
    # tag_pipeline writes reviewed/<memo_id>.<random>.tmp.xlsx and renames it;
    # a killed finalize leaves it, and its stem is a valid memo id.
    import shutil
    shutil.copy(os.path.join("reviewed", f"{MEMO}.xlsx"), os.path.join("reviewed", f"{MEMO}.k3j_x9a1.tmp.xlsx"))
    with caplog.at_level("WARNING", logger="eval"):
        run = ep.load_run(os.path.basename(ep.score_run(now=NOW)))
    assert run.meta["memos"] == [MEMO]
    assert "k3j_x9a1.tmp.xlsx is a temporary file" in caplog.text


def test_a_quote_not_in_its_own_chunk_is_warned_about(world, caplog):
    rows = _reviewed_rows()
    rows[0]["evidence_span"] = "Acme sells widgets worldwide and more"   # runs past d.pdf_1's text
    _write_reviewed(rows)
    with caplog.at_level("WARNING", logger="eval"):
        ep.load_ground_truth(MEMO)
    assert f"1 verified quote(s) are not in their own chunk's text as written (first: row 2, claim {cid(BP, C1)})" \
        in caplog.text


def test_a_damaged_results_file_is_listed_with_the_other_problems(world):
    rows = _reviewed_rows()
    rows[0]["tag"] = None
    _write_reviewed(rows)
    with open("retrieval_results.parquet", "wb") as f:
        f.write(b"PAR1 truncated by a copy")
    with pytest.raises(ep.EvalInputError) as e:
        ep.score_run(now=NOW)
    assert any(p.startswith("retrieval_results.parquet: unreadable (") for p in e.value.problems)
    assert any("tag is blank" in p for p in e.value.problems)


def test_a_damaged_claim_queries_file_is_refused_cleanly(world):
    with open("claim_queries.parquet", "wb") as f:
        f.write(b"not parquet")
    with pytest.raises(ep.EvalInputError, match="claim_queries.parquet: unreadable .* re-run python "
                                                "retrieval_pipeline.py recheck"):
        ep.score_run(now=NOW)


def test_a_memo_the_results_never_searched_gets_its_own_reason(world):
    _write_results(_results_rows(), indexes={})
    with pytest.raises(ep.EvalInputError) as e:
        ep.score_run(now=NOW)
    assert any(f"records no search for {MEMO}" in p for p in e.value.problems)
    assert not any("its chunks changed since" in p for p in e.value.problems)


def test_the_golden_hash_changes_when_a_row_becomes_human_added(world):
    before = ep.golden_fingerprint([ep.load_ground_truth(MEMO)])
    rows = _reviewed_rows()
    rows[0]["tag_rationale"] = "human-added 2026-09-22: quote 'sells widgets'"
    _write_reviewed(rows)
    assert ep.golden_fingerprint([ep.load_ground_truth(MEMO)]) != before


def test_the_phrase_count_note_names_a_section_only_the_baseline_searched(world):
    run = ep.load_run(os.path.basename(ep.score_run(now=NOW)))
    baseline = ep.load_run(os.path.basename(ep.score_run(now=NOW)))
    baseline.meta["phrase_counts"][MEMO]["Dropped section"] = 2
    assert "Dropped section" in ep._phrase_count_note(run, baseline)


def test_an_index_without_a_recorded_model_is_not_a_model_name(world):
    # retrieve only warns about such an index; the run's model comes from
    # what retrieve recorded, not a placeholder.
    _write_index(model=None)
    run = ep.load_run(os.path.basename(ep.score_run(now=NOW)))
    assert run.meta["embedding_model"] == "test-model"


def test_scoring_never_touches_another_runs_work_in_progress(world):
    foreign = os.path.join("eval_runs", "2026-09-23_100000.tmp")   # what a second score at NOW used to delete
    os.makedirs(foreign)
    open(os.path.join(foreign, "half-written"), "w").close()
    run_dir = ep.score_run(now=NOW)
    assert os.path.exists(os.path.join(foreign, "half-written"))
    assert os.path.basename(run_dir) == "2026-09-23_100000"


def test_a_run_id_already_taken_moves_to_the_next(world):
    first = os.path.basename(ep.score_run(now=NOW))
    second = os.path.basename(ep.score_run(now=NOW))
    assert second == first + "_2"
    assert ep.load_run(second).meta["run_id"] == second
    assert ep.load_run(first).run_id == first
