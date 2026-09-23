"""
_rows_for_claim collapses two matches that share a chunk_id into one row
(spec 2026-09-10-golden-set-row-tagging-design.md §7 edit 3). Fixes the 2
duplicate (claim_id, chunk_id) rows in the golden set at the source, and a
latent spurious ambiguous_match. Mocked -- no LLM.
"""
import json
import logging

import pandas as pd

import golden_set_pipeline as gsp

GOLDEN_SET_LOGGER = gsp.logger.name


def _match(chunk_id, span, confidence="high"):
    return {
        "chunk_id": chunk_id,
        "evidence_span": span,
        "confidence": confidence,
        "chunk_text": f"text of {chunk_id}",
        "bm25_score": 10.0,
        "verbatim_match": True,
    }


def test_same_chunk_id_collapses_keeping_higher_confidence(caplog):
    caplog.set_level(logging.WARNING, logger=GOLDEN_SET_LOGGER)
    matches = [_match("d.pdf_3", "same span", "low"),
               _match("d.pdf_3", "same span", "high")]
    chunk_id_to_doc_id = {"d.pdf_3": "d.pdf"}
    rows = gsp._rows_for_claim("M1", "Ownership", "cid", "claim", matches, chunk_id_to_doc_id)
    assert len(rows) == 1
    assert rows[0]["chunk_id"] == "d.pdf_3"
    assert rows[0]["confidence"] == "high"
    assert rows[0]["ambiguous_match"] is False
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("d.pdf_3" in m and "collapsed" in m.lower() for m in msgs)


def test_same_chunk_id_different_span_still_collapses_no_spurious_ambiguity(caplog):
    caplog.set_level(logging.WARNING, logger=GOLDEN_SET_LOGGER)
    # the hallucinated-recovery shape: one chunk_id, two different quoted spans
    matches = [_match("d.pdf_3", "quote A"), _match("d.pdf_3", "quote B")]
    rows = gsp._rows_for_claim("M1", "Ownership", "cid", "claim", matches, {"d.pdf_3": "d.pdf"})
    assert len(rows) == 1
    assert rows[0]["ambiguous_match"] is False  # would be True without the collapse


def test_distinct_chunks_in_two_docs_still_two_ambiguous_rows():
    matches = [_match("a.pdf_1", "span a"), _match("b.pdf_1", "span b")]
    chunk_id_to_doc_id = {"a.pdf_1": "a.pdf", "b.pdf_1": "b.pdf"}
    rows = gsp._rows_for_claim("M1", "Ownership", "cid", "claim", matches, chunk_id_to_doc_id)
    assert len(rows) == 2
    assert all(r["ambiguous_match"] is True for r in rows)


def test_missing_confidence_ranks_lowest():
    matches = [_match("d.pdf_3", "s", "medium"), _match("d.pdf_3", "s", None)]
    rows = gsp._rows_for_claim("M1", "Ownership", "cid", "claim", matches, {"d.pdf_3": "d.pdf"})
    assert len(rows) == 1
    assert rows[0]["confidence"] == "medium"


def test_non_string_confidence_ranks_lowest_instead_of_raising():
    """A malformed answer can carry a list or dict as confidence; looking it up
    in _CONFIDENCE_RANK raised TypeError (unhashable), turning the whole claim
    into an error row. It ranks like a missing confidence instead."""
    for bad in (["high"], {"level": "high"}):
        matches = [_match("d.pdf_3", "s", bad), _match("d.pdf_3", "s", "low"), _match("d.pdf_4", "t", "high")]
        rows = gsp._rows_for_claim("M1", "Ownership", "cid", "claim", matches, {"d.pdf_3": "d.pdf", "d.pdf_4": "d.pdf"})
        assert [(r["chunk_id"], r["confidence"]) for r in rows] == [("d.pdf_3", "low"), ("d.pdf_4", "high")]


def test_non_string_confidence_from_the_llm_is_stored_as_none(monkeypatch, tmp_path):
    """Upstream of the ranking guard above: propose_evidence_from_chunks keeps
    the match but stores a list/dict confidence as None. As the only match it
    used to reach the golden-set rows, and writing the checkpoint then raised
    ArrowTypeError, crashing a build run on one malformed answer."""
    candidates = [{"chunk_id": "d.pdf_3", "doc_id": "d.pdf", "chunk_text": "Revenue grew 12% to $9.2 billion.",
                   "start_offset": 0, "bm25_score": 5.0}]
    for bad in (["high"], {"level": "high"}):
        response = json.dumps([{"chunk_id": "d.pdf_3", "evidence_span": "Revenue grew 12% to $9.2 billion.",
                                "confidence": bad}])
        monkeypatch.setattr(gsp, "call_llm", lambda *a, response=response, **k: response)
        matches = gsp.propose_evidence_from_chunks("claim", candidates, object())
        assert [m["confidence"] for m in matches] == [None]
        rows = gsp._rows_for_claim("M1", "Ownership", "cid", "claim", matches, {"d.pdf_3": "d.pdf"})
        pd.DataFrame(rows).to_parquet(tmp_path / "checkpoint.parquet")
