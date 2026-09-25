"""
Design decision 1's "same evidence" rule, extracted from _rows_for_claim so
eval_pipeline.py can share it: _same_evidence (pairwise) and
_group_equivalent_chunks (the union-find partition over it).
"""
import pytest

import golden_set_pipeline as gsp


def test_identical_normalized_text_same_doc_is_same_evidence():
    assert gsp._same_evidence("d", "d_1", "Revenue  was   5", "d", "d_9", "Revenue was 5")


def test_leader_runs_are_normalized_before_comparing():
    assert gsp._same_evidence("d", "d_1", "Revenue........5", "d", "d_7", "Revenue...5")


def test_different_doc_is_never_same_evidence():
    assert not gsp._same_evidence("d", "d_1", "same text", "e", "e_2", "same text")


def test_missing_doc_is_never_same_evidence():
    assert not gsp._same_evidence(None, "d_1", "x", None, "d_2", "x")


@pytest.mark.parametrize("a, b", [
    ("sells widgets", "Acme sells widgets worldwide."),
    ("Acme sells widgets worldwide.", "sells widgets"),
])
def test_adjacent_chunks_with_containment_are_same_evidence(a, b):
    assert gsp._same_evidence("d", "d_1", a, "d", "d_2", b)


def test_adjacent_chunks_without_containment_are_not_same_evidence():
    assert not gsp._same_evidence("d", "d_3", "five plants", "d", "d_4", "Acme runs plants in Ohio.")


def test_containment_two_chunks_apart_is_not_same_evidence():
    assert not gsp._same_evidence("d", "d_1", "sells widgets", "d", "d_3", "Acme sells widgets.")


def test_same_chunk_is_not_same_evidence_by_this_rule():
    # Delta 0 never matches: callers test chunk_id equality themselves.
    assert not gsp._same_evidence("d", "d_1", "sells widgets", "d", "d_1", "Acme sells widgets worldwide.")


def test_unparseable_chunk_id_is_never_adjacent():
    assert not gsp._same_evidence("d", "nonsense", "ab", "d", "d_2", "xaby")


@pytest.mark.parametrize("chunk_id", [None, float("nan"), 12])
def test_chunk_index_of_a_non_string_is_none(chunk_id):
    assert gsp._chunk_index_of(chunk_id) is None


def test_group_chains_transitively():
    groups = gsp._group_equivalent_chunks(["d", "d", "d"], ["d_0", "d_1", "d_2"], ["abc", "abcd", "abcde"])
    assert groups == [[0, 1, 2]]


def test_group_order_is_first_member_order():
    groups = gsp._group_equivalent_chunks(["e", "d", "e"], ["e_5", "d_1", "e_6"], ["q", "x", "q r"])
    assert groups == [[0, 2], [1]]


def _matches():
    return [
        {"chunk_id": "d_0", "evidence_span": "Revenue grew 5%", "confidence": "high",
         "chunk_text": "t0", "bm25_score": 1.0},
        {"chunk_id": "d_1", "evidence_span": "Revenue grew 5% in 2024", "confidence": "medium",
         "chunk_text": "t1", "bm25_score": 0.9},
        {"chunk_id": "e_4", "evidence_span": "Revenue grew 5%", "confidence": "low",
         "chunk_text": "t4", "bm25_score": 0.5},
    ]


def test_rows_for_claim_two_groups_is_ambiguous():
    # Passes before and after the extraction — it pins _rows_for_claim's output.
    rows = gsp._rows_for_claim("M", "S", "cid", "claim", _matches(), {"d_0": "d", "d_1": "d", "e_4": "e"})
    assert [r["chunk_id"] for r in rows] == ["d_0", "d_1", "e_4"]
    assert all(r["ambiguous_match"] for r in rows)


def test_rows_for_claim_one_group_is_not_ambiguous():
    rows = gsp._rows_for_claim("M", "S", "cid", "claim", _matches()[:2], {"d_0": "d", "d_1": "d"})
    assert [r["chunk_id"] for r in rows] == ["d_0", "d_1"]
    assert not any(r["ambiguous_match"] for r in rows)
