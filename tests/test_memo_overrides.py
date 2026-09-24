"""
Tests for the per-memo relative_threshold/min_candidates/batch_size overrides
read from memos.yaml by load_memo_sections_from_config and applied by
build_golden_set_batch.

These mock golden_set_pipeline.load_pdf_text (so no real PDF is parsed) and
golden_set_pipeline.build_golden_set_draft (so no LLM provider is contacted),
consistent with this repo's testing convention (CLAUDE.md, "Testing
convention"). No prompt text changes here, so the live-call exception doesn't
apply.
"""
import textwrap

import pandas as pd
import pytest

import golden_set_pipeline as gsp


def write_config(tmp_path, body: str) -> str:
    """
    Writes a memos.yaml holding `body` plus a source folder containing one
    stub PDF, and returns the config path. Write SOURCE where the config needs
    the source folder and it's substituted here. The stub PDF only has to exist
    and end in .pdf -- load_pdf_text is mocked out by the fixture below.
    """
    source_folder = tmp_path / "sources"
    source_folder.mkdir(exist_ok=True)
    (source_folder / "doc.pdf").write_bytes(b"%PDF-1.4 stub")
    config_path = tmp_path / "memos.yaml"
    config_path.write_text(textwrap.dedent(body).replace("SOURCE", str(source_folder)), encoding="utf-8")
    return str(config_path)


@pytest.fixture(autouse=True)
def stub_pdf_reader(monkeypatch):
    monkeypatch.setattr(gsp, "load_pdf_text", lambda path: "Some source document text, long enough to pass the length check.")


@pytest.fixture
def recorded_draft_calls(monkeypatch):
    """
    Replaces build_golden_set_draft with a recorder, so a batch run's
    per-section arguments can be asserted on without any LLM contact.
    """
    calls: list[dict] = []

    def fake_draft(memo_id, section_name, claims, source_documents, llm_client, **kwargs):
        calls.append({"memo_id": memo_id, "section_name": section_name, **kwargs})
        return pd.DataFrame([{"claim_id": "c1", "memo_id": memo_id, "section": section_name}])

    monkeypatch.setattr(gsp, "build_golden_set_draft", fake_draft)
    return calls


# --- load_memo_sections_from_config: parsing ---------------------------------


def test_memo_without_overrides_gets_empty_override_dict(tmp_path):
    config_path = write_config(
        tmp_path,
        """
        memos:
          - id: MEMO-001
            source_folder: SOURCE
            sections:
              Business Profile: Some section text.
        """,
    )

    memo_sections = gsp.load_memo_sections_from_config(config_path)

    assert len(memo_sections) == 1
    assert len(memo_sections[0]) == 5
    assert memo_sections[0][4] == {}


def test_overrides_are_parsed_and_scoped_to_their_own_memo(tmp_path):
    config_path = write_config(
        tmp_path,
        """
        memos:
          - id: MEMO-001
            source_folder: SOURCE
            relative_threshold: 0.45
            min_candidates: 8
            batch_size: 25
            sections:
              Business Profile: Some section text.
          - id: MEMO-002
            source_folder: SOURCE
            sections:
              Ownership: Other section text.
        """,
    )

    memo_sections = gsp.load_memo_sections_from_config(config_path)

    overrides_by_memo = {entry[0]: entry[4] for entry in memo_sections}
    assert overrides_by_memo["MEMO-001"] == {
        "relative_threshold": 0.45,
        "min_candidates": 8,
        "batch_size": 25,
    }
    assert overrides_by_memo["MEMO-002"] == {}


def test_partial_overrides_omit_the_keys_that_were_not_set(tmp_path):
    """Only the keys actually written are returned, so the rest fall back to the batch run's own defaults."""
    config_path = write_config(
        tmp_path,
        """
        memos:
          - id: MEMO-001
            source_folder: SOURCE
            batch_size: 12
            sections:
              Business Profile: Some section text.
        """,
    )

    memo_sections = gsp.load_memo_sections_from_config(config_path)

    assert memo_sections[0][4] == {"batch_size": 12}


def test_an_integer_relative_threshold_is_accepted_as_a_float(tmp_path):
    config_path = write_config(
        tmp_path,
        """
        memos:
          - id: MEMO-001
            source_folder: SOURCE
            relative_threshold: 1
            sections:
              Business Profile: Some section text.
        """,
    )

    memo_sections = gsp.load_memo_sections_from_config(config_path)

    assert memo_sections[0][4] == {"relative_threshold": 1.0}
    assert isinstance(memo_sections[0][4]["relative_threshold"], float)


# --- load_memo_sections_from_config: validation ------------------------------


@pytest.mark.parametrize(
    "override_line, field",
    [
        ("relative_threshold: 1.5", "relative_threshold"),
        ("relative_threshold: -0.1", "relative_threshold"),
        ("relative_threshold: high", "relative_threshold"),
        ("relative_threshold: yes", "relative_threshold"),  # YAML bool; bool is an int subclass
        ("min_candidates: -1", "min_candidates"),
        ("min_candidates: 2.5", "min_candidates"),
        ("min_candidates: yes", "min_candidates"),
        ("batch_size: 0", "batch_size"),
        ("batch_size: big", "batch_size"),
        ("batch_size:", "batch_size"),  # written with no value -> None, not "absent"
    ],
)
def test_malformed_override_raises_naming_memo_and_field(tmp_path, override_line, field):
    config_path = write_config(
        tmp_path,
        f"""
        memos:
          - id: MEMO-001
            source_folder: SOURCE
            {override_line}
            sections:
              Business Profile: Some section text.
        """,
    )

    with pytest.raises(ValueError) as excinfo:
        gsp.load_memo_sections_from_config(config_path)

    assert "MEMO-001" in str(excinfo.value)
    assert field in str(excinfo.value)


def test_unrecognized_memo_key_raises_naming_memo_and_key(tmp_path):
    """A typo'd knob would otherwise be ignored and the memo would silently run on the defaults."""
    config_path = write_config(
        tmp_path,
        """
        memos:
          - id: MEMO-001
            source_folder: SOURCE
            batchsize: 25
            sections:
              Business Profile: Some section text.
        """,
    )

    with pytest.raises(ValueError) as excinfo:
        gsp.load_memo_sections_from_config(config_path)

    assert "MEMO-001" in str(excinfo.value)
    assert "batchsize" in str(excinfo.value)


# --- build_golden_set_batch: application -------------------------------------


def test_overrides_apply_to_their_own_memo_only(tmp_path, recorded_draft_calls):
    """Memo A's sections run on A's values; memo B's, in the same batch run, run on the globals."""
    # One dict shared by both of memo A's sections, as the config loader builds it.
    overrides_a = {"relative_threshold": 0.45, "min_candidates": 8, "batch_size": 25}
    memo_sections = [
        ("MEMO-A", "Business Profile", ["claim"], [], overrides_a),
        ("MEMO-A", "Ownership", ["claim"], [], overrides_a),
        ("MEMO-B", "Business Profile", ["claim"], [], {}),
    ]

    gsp.build_golden_set_batch(
        memo_sections,
        llm_client=None,
        checkpoint_path=str(tmp_path / "checkpoint.parquet"),
        relative_threshold=0.3,
        min_candidates=5,
        batch_size=40,
    )

    assert [(c["relative_threshold"], c["min_candidates"], c["batch_size"]) for c in recorded_draft_calls] == [
        (0.45, 8, 25),
        (0.45, 8, 25),
        (0.3, 5, 40),
    ]


def test_partial_override_falls_back_to_globals_for_unset_knobs(tmp_path, recorded_draft_calls):
    memo_sections = [("MEMO-A", "Business Profile", ["claim"], [], {"batch_size": 12})]

    gsp.build_golden_set_batch(
        memo_sections,
        llm_client=None,
        checkpoint_path=str(tmp_path / "checkpoint.parquet"),
        relative_threshold=0.3,
        min_candidates=5,
        batch_size=40,
    )

    assert recorded_draft_calls[0]["batch_size"] == 12
    assert recorded_draft_calls[0]["relative_threshold"] == 0.3
    assert recorded_draft_calls[0]["min_candidates"] == 5


def test_hand_written_four_tuples_still_run_unchanged(tmp_path, recorded_draft_calls):
    """A caller who builds memo_sections by hand never has to know overrides exist."""
    memo_sections = [
        ("MEMO-A", "Business Profile", ["claim"], []),
        ("MEMO-B", "Ownership", ["claim"], []),
    ]

    result = gsp.build_golden_set_batch(
        memo_sections,
        llm_client=None,
        checkpoint_path=str(tmp_path / "checkpoint.parquet"),
        relative_threshold=0.3,
        min_candidates=5,
    )

    assert len(result) == 2
    for call in recorded_draft_calls:
        assert call["relative_threshold"] == 0.3
        assert call["min_candidates"] == 5
        assert call["batch_size"] == 40


def test_batch_size_argument_reaches_build_golden_set_draft(tmp_path, recorded_draft_calls):
    """build_golden_set_batch had no batch_size parameter at all before this change."""
    memo_sections = [("MEMO-A", "Business Profile", ["claim"], [])]

    gsp.build_golden_set_batch(
        memo_sections,
        llm_client=None,
        checkpoint_path=str(tmp_path / "checkpoint.parquet"),
        batch_size=17,
    )

    assert recorded_draft_calls[0]["batch_size"] == 17


def test_oversized_tuple_raises_instead_of_silently_dropping_overrides(tmp_path, recorded_draft_calls):
    """
    A 6-tuple must not fall through to "no overrides" -- that would run the memo
    on the batch defaults with nothing raised and nothing logged.
    """
    memo_sections = [("MEMO-A", "Business Profile", ["claim"], [], {"batch_size": 12}, "extra")]

    with pytest.raises(ValueError) as excinfo:
        gsp.build_golden_set_batch(
            memo_sections,
            llm_client=None,
            checkpoint_path=str(tmp_path / "checkpoint.parquet"),
        )

    assert "memo_sections[0]" in str(excinfo.value)
    assert recorded_draft_calls == []


def test_non_dict_fifth_element_raises_naming_the_memo(tmp_path, recorded_draft_calls):
    memo_sections = [("MEMO-A", "Business Profile", ["claim"], [], ["batch_size", 12])]

    with pytest.raises(ValueError) as excinfo:
        gsp.build_golden_set_batch(
            memo_sections,
            llm_client=None,
            checkpoint_path=str(tmp_path / "checkpoint.parquet"),
        )

    assert "MEMO-A" in str(excinfo.value)
    assert recorded_draft_calls == []


def test_unrecognized_override_key_in_a_hand_built_tuple_raises(tmp_path, recorded_draft_calls):
    """
    A typo'd key in a hand-built override dict must not fall through to the
    defaults -- the run would otherwise log "overrides in effect" while
    reporting the default values.
    """
    memo_sections = [("MEMO-A", "Business Profile", ["claim"], [], {"batch_sze": 12})]

    with pytest.raises(ValueError) as excinfo:
        gsp.build_golden_set_batch(
            memo_sections,
            llm_client=None,
            checkpoint_path=str(tmp_path / "checkpoint.parquet"),
        )

    assert "batch_sze" in str(excinfo.value)
    assert recorded_draft_calls == []


def test_entry_shapes_are_checked_before_any_section_runs(tmp_path, recorded_draft_calls):
    """
    A malformed entry late in the list must fail before the earlier sections are
    processed -- in a live run those are multi-minute LLM calls whose results the
    periodic checkpoint may not have saved yet.
    """
    memo_sections = [("MEMO-A", f"Section {n}", ["claim"], []) for n in range(1, 5)]
    memo_sections.append(("MEMO-A", "Section 5", ["claim"], [], {}, "unexpected sixth element"))

    with pytest.raises(ValueError) as excinfo:
        gsp.build_golden_set_batch(
            memo_sections,
            llm_client=None,
            checkpoint_path=str(tmp_path / "checkpoint.parquet"),
            checkpoint_every=3,
        )

    assert "memo_sections[4]" in str(excinfo.value)
    assert recorded_draft_calls == []


def test_mixed_type_unrecognized_keys_still_raise_valueerror_naming_the_memo(tmp_path):
    """
    YAML 1.1 resolves a bare `on:` key to the boolean True. Sorting that
    alongside a string key for the error message must not itself blow up with a
    TypeError naming no memo.
    """
    config_path = write_config(
        tmp_path,
        """
        memos:
          - id: MEMO-001
            source_folder: SOURCE
            on: 5
            batchsize: 3
            sections:
              Business Profile: Some section text.
        """,
    )

    with pytest.raises(ValueError) as excinfo:
        gsp.load_memo_sections_from_config(config_path)

    assert "MEMO-001" in str(excinfo.value)
    assert "batchsize" in str(excinfo.value)


def test_override_roundtrips_memos_yaml_to_claims_file_to_build(tmp_path, monkeypatch, recorded_draft_calls):
    monkeypatch.setattr(gsp, "extract_atomic_claims", lambda text, c: ["one claim"])
    monkeypatch.setattr(gsp, "load_pdf_text", lambda p: "long enough source text here")
    src = tmp_path / "src"
    src.mkdir()
    (src / "d.pdf").write_bytes(b"%PDF stub")
    cfg = tmp_path / "memos.yaml"
    cfg.write_text(textwrap.dedent(f"""
        memos:
          - id: MEMO-1
            source_folder: {src}
            relative_threshold: 0.45
            sections: {{Only: "text"}}
    """), encoding="utf-8")

    gsp.run_extract(str(cfg), str(tmp_path / "claims"), llm_client=None)
    written = (tmp_path / "claims" / "MEMO-1.md").read_text(encoding="utf-8")
    assert "relative_threshold: 0.45" in written
    assert "min_candidates" not in written
    assert "batch_size" not in written

    tuples = gsp.load_memo_sections_from_claims(str(tmp_path / "claims"))
    assert tuples[0][4] == {"relative_threshold": 0.45}

    gsp.build_golden_set_batch(tuples, None, checkpoint_path=str(tmp_path / "cp.parquet"))
    assert recorded_draft_calls[0]["relative_threshold"] == 0.45


def test_pre_pass_rejects_raw_section_text_in_slot_3(tmp_path, recorded_draft_calls):
    """
    Since the two-stage split, slot 3 holds already-split claims, not raw
    section text. build_golden_set_draft enforces that too -- but only once
    its entry is reached, so a list whose LATER entry still carries section
    text would raise after paying for every earlier section. Design decision
    12's pre-pass exists precisely to catch shape errors before any spend,
    so it must cover this shape as well.
    """
    entries = [
        ("M1", ["doc text"], ["a claim."], "memo one"),
        ("M2", ["doc text"], "raw section text, not a list", "memo two"),
    ]
    with pytest.raises(ValueError, match=r"memo_sections\[1\] \(M2\).*3rd element must be a list"):
        gsp.build_golden_set_batch(
            entries, llm_client=object(), checkpoint_path=str(tmp_path / "cp.parquet"),
        )
    assert recorded_draft_calls == [], "no section should have run before the shape check"
