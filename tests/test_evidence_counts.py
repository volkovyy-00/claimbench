"""
Tests for the dropped/recovered evidence-entry counting and logging added to
propose_evidence_from_chunks, propose_evidence_from_chunks_batched, and
build_golden_set_draft (see CLAUDE.md design decision 11).

These mock golden_set_pipeline.call_llm (and, where a claim needs to be
supplied, extract_atomic_claims) so no real LLM provider is contacted --
consistent with this repo's testing convention (CLAUDE.md, "Testing
convention"): mock everything except prompt-text behavior itself, which
these changes don't touch.
"""
import json
import logging

import golden_set_pipeline as gsp

GOLDEN_SET_LOGGER = gsp.logger.name


def test_propose_evidence_from_chunks_counts_dropped_and_recovered(monkeypatch):
    candidates = [
        {
            "chunk_id": "docA_0",
            "doc_id": "docA",
            "chunk_text": "Net income was $774.1 million in fiscal 2024.",
            "start_offset": 0,
            "bm25_score": 5.0,
        },
        {
            "chunk_id": "docA_1",
            "doc_id": "docA",
            "chunk_text": "Revenue grew 12% year over year to $9.2 billion.",
            "start_offset": 500,
            "bm25_score": 4.0,
        },
        {
            "chunk_id": "docB_0",
            "doc_id": "docB",
            "chunk_text": "The Austria segment reported a balance of 5% of total.",
            "start_offset": 0,
            "bm25_score": 3.0,
        },
        {
            "chunk_id": "docB_1",
            "doc_id": "docB",
            "chunk_text": "The Austria segment reported a balance of 5% of total.",
            "start_offset": 400,
            "bm25_score": 2.0,
        },
    ]
    llm_response = json.dumps(
        [
            {"evidence_span": "some span but no chunk_id at all"},  # malformed: missing chunk_id -> dropped
            {
                "chunk_id": "hallucinated_1",
                "evidence_span": "Net income was $774.1 million in fiscal 2024.",
                "confidence": "high",
            },  # recoverable: unique substring match to docA_0
            {
                "chunk_id": "hallucinated_2",
                "evidence_span": "The Austria segment reported a balance of 5% of total.",
                "confidence": "medium",
            },  # unrecoverable: matches BOTH docB_0 and docB_1 -> dropped
            {
                "chunk_id": "docA_1",
                "evidence_span": "Revenue grew 12% year over year to $9.2 billion.",
                "confidence": "high",
            },  # clean match
        ]
    )
    monkeypatch.setattr(gsp, "call_llm", lambda *a, **k: llm_response)
    llm_client = object()

    # Omitting counts must behave exactly as before a standalone/manual call.
    matches_no_counts = gsp.propose_evidence_from_chunks("claim text", candidates, llm_client)
    assert len(matches_no_counts) == 2

    counts = {"dropped": 0, "recovered": 0}
    matches = gsp.propose_evidence_from_chunks("claim text", candidates, llm_client, counts=counts)
    assert len(matches) == 2
    assert counts == {"dropped": 2, "recovered": 1}

    recovered_match = next(m for m in matches if m["evidence_span"].startswith("Net income"))
    assert recovered_match["chunk_id"] == "docA_0"


def test_propose_evidence_from_chunks_batched_aggregates_per_claim(monkeypatch, caplog):
    candidates = [
        {"chunk_id": "docA_0", "doc_id": "docA", "chunk_text": "Batch one candidate zero text.", "start_offset": 0, "bm25_score": 5.0},
        {"chunk_id": "docA_1", "doc_id": "docA", "chunk_text": "Batch one candidate one text.", "start_offset": 500, "bm25_score": 4.0},
        {"chunk_id": "docB_0", "doc_id": "docB", "chunk_text": "Batch two candidate zero text.", "start_offset": 0, "bm25_score": 3.0},
        {"chunk_id": "docB_1", "doc_id": "docB", "chunk_text": "Batch two candidate one text.", "start_offset": 400, "bm25_score": 2.0},
    ]
    # batch_size=2 -> batch 1 = [docA_0, docA_1], batch 2 = [docB_0, docB_1]
    batch1_response = json.dumps(
        [
            {"chunk_id": "nonexistent", "evidence_span": "not a substring of anything here"},  # dropped
            {"chunk_id": "docA_1", "evidence_span": "Batch one candidate one text.", "confidence": "high"},  # clean
        ]
    )
    batch2_response = json.dumps(
        [
            {
                "chunk_id": "hallucinated_b",
                "evidence_span": "Batch two candidate zero text.",
                "confidence": "medium",
            },  # recovered -> docB_0
            {"chunk_id": "docB_1", "evidence_span": "Batch two candidate one text.", "confidence": "high"},  # clean
        ]
    )
    responses = iter([batch1_response, batch2_response])
    monkeypatch.setattr(gsp, "call_llm", lambda *a, **k: next(responses))

    caplog.set_level(logging.INFO, logger=GOLDEN_SET_LOGGER)
    llm_client = object()

    matches, counts = gsp.propose_evidence_from_chunks_batched("claim text", candidates, llm_client, batch_size=2)

    assert len(matches) == 3  # 2 clean + 1 recovered
    assert counts == {"dropped": 1, "recovered": 1}

    info_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any("1 dropped, 1 recovered (2/2 batch(es) succeeded)" in m for m in info_messages)
    assert any("split into 2 batch(es)" in m for m in info_messages)

    warning_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_messages) == 2  # 1 drop + 1 recovery, existing per-entry lines unchanged


def test_build_golden_set_draft_section_wide_totals(monkeypatch, caplog):
    source_documents = [
        ("doc1.pdf", "Widgets are manufactured in Ohio. " * 5 + " Gadgets are assembled in Texas. " * 5),
    ]
    chunk_index = gsp.build_chunk_index(source_documents)
    real_chunk_id = chunk_index[0]["chunk_id"]
    real_chunk_text = chunk_index[0]["chunk_text"]

    def fake_call_llm(prompt, llm_client, temperature=0.0, max_tokens=4096):
        if "Claim one" in prompt:
            return json.dumps(
                [
                    {"evidence_span": "no chunk_id here"},  # dropped: malformed
                    {"chunk_id": real_chunk_id, "evidence_span": real_chunk_text[:20], "confidence": "high"},  # clean
                ]
            )
        return json.dumps(
            [
                {
                    "chunk_id": "hallucinated_for_claim_two",
                    "evidence_span": real_chunk_text[:20],
                    "confidence": "medium",
                },  # recovered
            ]
        )

    monkeypatch.setattr(gsp, "call_llm", fake_call_llm)
    caplog.set_level(logging.INFO, logger=GOLDEN_SET_LOGGER)
    llm_client = object()

    df = gsp.build_golden_set_draft(
        "memo1", "section1", ["Claim one about widgets.", "Claim two about gadgets."], source_documents, llm_client
    )

    expected_schema = [
        "claim_id", "memo_id", "section", "claim_text", "doc_id", "chunk_id",
        "chunk_text", "bm25_score", "evidence_span", "found", "confidence",
        "ambiguous_match", "verbatim_match", "human_reviewed", "tag",
    ]
    assert list(df.columns) == expected_schema  # no leaked dropped/recovered columns

    info_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    section_lines = [m for m in info_messages if "2 claim(s)" in m and "dropped" in m]
    assert len(section_lines) == 1
    assert "1 dropped, 1 recovered" in section_lines[0]
    assert "memo1" in section_lines[0]
    assert "section1" in section_lines[0]


def test_build_golden_set_draft_failed_claim_contributes_no_counts(monkeypatch, caplog):
    source_documents = [
        ("doc1.pdf", "Widgets are manufactured in Ohio. " * 5 + " Gadgets are assembled in Texas. " * 5),
    ]

    def fake_call_llm(prompt, llm_client, temperature=0.0, max_tokens=4096):
        if "Claim one" in prompt:
            raise RuntimeError("simulated upstream provider failure")
        return json.dumps([{"evidence_span": "no chunk_id here"}])  # claim two: 1 dropped, 0 recovered

    monkeypatch.setattr(gsp, "call_llm", fake_call_llm)
    caplog.set_level(logging.INFO, logger=GOLDEN_SET_LOGGER)
    llm_client = object()

    df = gsp.build_golden_set_draft(
        "memo1", "section1", ["Claim one about widgets.", "Claim two about gadgets."], source_documents, llm_client
    )

    error_rows = df[df["confidence"] == "error"]
    assert len(error_rows) == 1  # claim one's total-batch-failure becomes one error row

    info_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    section_lines = [m for m in info_messages if "2 claim(s)" in m and "dropped" in m]
    assert len(section_lines) == 1
    assert "1 dropped, 0 recovered" in section_lines[0]  # only claim two's counts

    # A fully-failed claim must never log a misleading "0 dropped, 0 recovered" per-claim line.
    claim_one_count_lines = [m for m in info_messages if "Claim one" in m and "dropped" in m]
    assert claim_one_count_lines == []
