"""
Tests for deterministic claim_id derivation (_derive_claim_id) and the
in-section duplicate tie-breaker (_claims_with_occurrence), added for the
two-stage split (design decision 17). Mocked -- no LLM.
"""
import uuid

import pytest

import golden_set_pipeline as gsp


def test_claims_with_occurrence_numbers_only_repeats():
    assert gsp._claims_with_occurrence(["a", "b", "a"]) == [("a", 0), ("b", 0), ("a", 1)]
    assert gsp._claims_with_occurrence(["a", "a", "a"]) == [("a", 0), ("a", 1), ("a", 2)]
    assert gsp._claims_with_occurrence([]) == []


def test_derive_claim_id_is_deterministic_and_a_uuid():
    a = gsp._derive_claim_id("MEMO-1", "Ownership", "Acme is listed.", 0)
    b = gsp._derive_claim_id("MEMO-1", "Ownership", "Acme is listed.", 0)
    assert a == b
    uuid.UUID(a)  # parses as a UUID string; raises if not


def test_derive_claim_id_matches_pinned_golden_value():
    # Final-review finding 3: this is the branch's headline invariant --
    # _CLAIM_ID_NAMESPACE + memo_id + section_name + claim_text + occurrence,
    # joined by "\x1f" -- and until now nothing pinned the field ORDER or the
    # delimiter itself. Reordering the join or swapping the delimiter changes
    # every claim_id while every other test in this suite stays green, since
    # they only check internal consistency (equal/unequal), never an actual
    # value. This literal was computed once from the current implementation:
    #   uuid.uuid5(_CLAIM_ID_NAMESPACE,
    #              "\x1f".join(("MEMO-1", "Ownership", "Acme is listed.", "0")))
    # If this assertion ever fails, every previously-built golden set's
    # claim_ids have rotated -- confirm that's a deliberate, intended change
    # (e.g. a namespace/delimiter/field-order edit) before accepting it, not
    # an incidental refactor.
    assert (
        gsp._derive_claim_id("MEMO-1", "Ownership", "Acme is listed.", 0)
        == "59d5251c-46fa-5260-a569-a0342216ae6b"
    )


def test_derive_claim_id_distinguishes_every_field():
    base = gsp._derive_claim_id("MEMO-1", "Ownership", "Acme is listed.", 0)
    assert base != gsp._derive_claim_id("MEMO-1", "Ownership", "Acme is listed.", 1)
    assert base != gsp._derive_claim_id("MEMO-2", "Ownership", "Acme is listed.", 0)
    assert base != gsp._derive_claim_id("MEMO-1", "Business", "Acme is listed.", 0)
    assert base != gsp._derive_claim_id("MEMO-1", "Ownership", "Acme is listed", 0)


def test_derive_claim_id_delimiter_prevents_collision():
    # ("ab","c",...) must not collide with ("a","bc",...)
    assert gsp._derive_claim_id("ab", "c", "x", 0) != gsp._derive_claim_id("a", "bc", "x", 0)


def _stub_evidence(monkeypatch):
    monkeypatch.setattr(gsp, "build_chunk_index", lambda docs, **kw: [{"chunk_id": "d_0", "doc_id": "d", "chunk_text": "t", "start_offset": 0}])
    monkeypatch.setattr(gsp, "bm25_threshold_shortlist", lambda *a, **k: [])
    monkeypatch.setattr(gsp, "propose_evidence_from_chunks_batched", lambda *a, **k: ([], {"dropped": 0, "recovered": 0}))


def test_build_draft_takes_claims_list_and_derives_ids(monkeypatch):
    _stub_evidence(monkeypatch)
    df = gsp.build_golden_set_draft("MEMO-1", "Ownership", ["c one", "c two", "c one"], [("d", "text")], llm_client=None)
    ids = list(df["claim_id"])
    assert len(ids) == 3
    assert ids[0] != ids[2]  # same text, occurrence 0 vs 1 -> distinct
    # deterministic: a second identical run yields the same ids
    df2 = gsp.build_golden_set_draft("MEMO-1", "Ownership", ["c one", "c two", "c one"], [("d", "text")], llm_client=None)
    assert list(df2["claim_id"]) == ids


def test_build_draft_empty_claims_yields_schema_frame(monkeypatch):
    _stub_evidence(monkeypatch)
    df = gsp.build_golden_set_draft("MEMO-1", "Ownership", [], [("d", "text")], llm_client=None)
    assert df.empty
    assert list(df.columns) == gsp._SCHEMA_COLUMNS


def test_build_draft_per_claim_error_row_carries_derived_id(monkeypatch):
    monkeypatch.setattr(gsp, "build_chunk_index", lambda docs, **kw: [{"chunk_id": "d_0", "doc_id": "d", "chunk_text": "t", "start_offset": 0}])
    monkeypatch.setattr(gsp, "bm25_threshold_shortlist", lambda *a, **k: [])

    def _raise(*a, **k):
        raise RuntimeError("simulated evidence lookup failure")

    monkeypatch.setattr(gsp, "propose_evidence_from_chunks_batched", _raise)

    df = gsp.build_golden_set_draft("MEMO-1", "Ownership", ["c one"], [("d", "text")], llm_client=None)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["confidence"] == "error"
    assert row["claim_id"] == gsp._derive_claim_id("MEMO-1", "Ownership", "c one", 0)


def test_build_draft_does_not_restrip_claim_text(monkeypatch):
    _stub_evidence(monkeypatch)
    raw = "  spaced   claim  "
    df = gsp.build_golden_set_draft("MEMO-1", "Ownership", [raw], [("d", "text")], llm_client=None)

    row = df.iloc[0]
    # 1. claim_text is passed through untouched -- no re-strip, no internal-
    #    whitespace collapse.
    assert row["claim_text"] == raw

    # 2. claim_id is derived from the raw string, not the stripped one -- the
    #    assertion that actually catches a "tidy up with .strip()" mutation.
    assert row["claim_id"] == gsp._derive_claim_id("MEMO-1", "Ownership", raw, 0)
    assert row["claim_id"] != gsp._derive_claim_id("MEMO-1", "Ownership", raw.strip(), 0)


def test_build_draft_rejects_non_list_claims_argument():
    with pytest.raises(TypeError, match="list of claim strings"):
        gsp.build_golden_set_draft("MEMO-1", "Ownership", "not a list", [("d", "text")], llm_client=None)


def test_is_deterministic_claim_id_truth_table():
    real = gsp._derive_claim_id("MEMO-1", "Ownership", "Acme is listed.", 0)
    assert gsp._is_deterministic_claim_id(real) is True
    # a uuid4 string (what pre-decision-17 golden sets used)
    assert gsp._is_deterministic_claim_id("f47ac10b-58cc-4372-a567-0e02b2c3d479") is False
    # garbage / empty / wrong type all refuse, never raise -- including a
    # version-5 uuid.UUID object: the contract is a UUID *string*
    for bad in ("", "xyz", "not-a-uuid", None, 12345, uuid.UUID(real)):
        assert gsp._is_deterministic_claim_id(bad) is False
