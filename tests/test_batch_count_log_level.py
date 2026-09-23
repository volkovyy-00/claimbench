"""
Tests for the log-level escalation on propose_evidence_from_chunks_batched's
pre-loop candidate/batch-count line (see
docs/tickets/004-log-oversized-candidate-shortlists.md and CLAUDE.md design
decision 13).

Mocks golden_set_pipeline.call_llm so no real LLM provider is contacted,
consistent with this repo's testing convention (CLAUDE.md, "Testing
convention").
"""
import json
import logging

import golden_set_pipeline as gsp

GOLDEN_SET_LOGGER = gsp.logger.name


def _make_candidates(n):
    return [
        {
            "chunk_id": f"doc_{i}",
            "doc_id": "doc",
            "chunk_text": f"Candidate text number {i}.",
            "start_offset": i * 100,
            "bm25_score": 1.0,
        }
        for i in range(n)
    ]


def test_ten_batches_logs_at_info(monkeypatch, caplog):
    candidates = _make_candidates(10)
    monkeypatch.setattr(gsp, "call_llm", lambda *a, **k: json.dumps([]))
    caplog.set_level(logging.INFO, logger=GOLDEN_SET_LOGGER)

    gsp.propose_evidence_from_chunks_batched("claim text", candidates, object(), batch_size=1)

    split_records = [r for r in caplog.records if "split into" in r.getMessage()]
    assert len(split_records) == 1
    assert split_records[0].levelno == logging.INFO
    assert "split into 10 batch(es)" in split_records[0].getMessage()


def test_eleven_batches_logs_at_warning(monkeypatch, caplog):
    candidates = _make_candidates(11)
    monkeypatch.setattr(gsp, "call_llm", lambda *a, **k: json.dumps([]))
    caplog.set_level(logging.INFO, logger=GOLDEN_SET_LOGGER)

    gsp.propose_evidence_from_chunks_batched("claim text", candidates, object(), batch_size=1)

    split_records = [r for r in caplog.records if "split into" in r.getMessage()]
    assert len(split_records) == 1
    assert split_records[0].levelno == logging.WARNING
    assert "split into 11 batch(es)" in split_records[0].getMessage()

    # The unrelated post-loop dropped/recovered line must be unaffected --
    # still logged, still at INFO, regardless of the pre-loop line's level.
    summary_records = [
        r for r in caplog.records if "dropped" in r.getMessage() and "recovered" in r.getMessage()
    ]
    assert len(summary_records) == 1
    assert summary_records[0].levelno == logging.INFO
