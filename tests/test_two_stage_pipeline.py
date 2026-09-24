"""
Tests for the two-stage split: _load_source_documents / _read_memo_config /
load_memo_sections_from_claims, run_extract, run_build, and __main__
dispatch. Mocked -- gsp.load_pdf_text and gsp.extract_atomic_claims are
stubbed; no LLM, no real PDF parsing (CLAUDE.md, "Testing convention").
"""
import os
import textwrap

import pandas as pd
import pytest

import golden_set_pipeline as gsp


@pytest.fixture(autouse=True)
def stub_pdf(monkeypatch):
    # >= warn_if_text_suspiciously_short's 200-char floor, so tests that DON'T
    # override this never trip a spurious "suspiciously short" WARNING.
    monkeypatch.setattr(gsp, "load_pdf_text", lambda p: "source document text. " * 20)


def _folder_with_pdfs(tmp_path, *names):
    d = tmp_path / "src"
    d.mkdir(exist_ok=True)
    for n in names:
        (d / n).write_bytes(b"%PDF-1.4 stub")
    return str(d)


def test_load_source_documents_matches_pdf_extension_case_insensitively(tmp_path):
    # The .pdf extension match is case-insensitive (a.PDF is included); the
    # sort itself is plain str sort (case-sensitive), lifted verbatim from
    # the current loader. Uppercase names sort before lowercase.
    folder = _folder_with_pdfs(tmp_path, "b.pdf", "A.PDF", "notes.txt")
    docs = gsp._load_source_documents(folder, label="MEMO-1.md")
    assert [d[0] for d in docs] == ["A.PDF", "b.pdf"]


def test_load_source_documents_errors(tmp_path):
    with pytest.raises(ValueError, match="MEMO-1.md.*not a directory"):
        gsp._load_source_documents(str(tmp_path / "missing"), label="MEMO-1.md")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="MEMO-1.md.*no PDF"):
        gsp._load_source_documents(str(empty), label="MEMO-1.md")


def test_load_source_documents_warns_on_short_doc(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(gsp, "load_pdf_text", lambda p: "tiny")  # under warn_if_text_suspiciously_short's floor
    folder = _folder_with_pdfs(tmp_path, "a.pdf")
    with caplog.at_level("WARNING", logger=gsp.logger.name):
        gsp._load_source_documents(folder, label="MEMO-1.md")
    assert "a.pdf" in caplog.text


def _config(tmp_path, body):
    p = tmp_path / "memos.yaml"
    p.write_text(textwrap.dedent(body).replace("SRC", _folder_with_pdfs(tmp_path, "d.pdf")), encoding="utf-8")
    return str(p)


def test_read_memo_config_shape(tmp_path):
    memos = gsp._read_memo_config(_config(tmp_path, """
        memos:
          - id: MEMO-1
            source_folder: SRC
            relative_threshold: 0.4
            sections:
              Ownership: |
                Acme is listed.
    """))
    assert len(memos) == 1
    assert memos[0]["id"] == "MEMO-1"
    assert os.path.isdir(memos[0]["source_folder"])
    assert memos[0]["overrides"] == {"relative_threshold": 0.4}
    assert list(memos[0]["sections"]) == ["Ownership"]
    assert memos[0]["sections"]["Ownership"].strip() == "Acme is listed."


def test_read_memo_config_rejects_unsafe_memo_id(tmp_path):
    config_path = _config(tmp_path, """
        memos:
          - id: MEMO/1
            source_folder: SRC
            sections: {Ownership: "x"}
    """)
    with pytest.raises(ValueError, match="MEMO/1|must match"):
        gsp._read_memo_config(config_path)


def test_read_memo_config_rejects_padded_section_name(tmp_path):
    config_path = _config(tmp_path, """
        memos:
          - id: MEMO-1
            source_folder: SRC
            sections:
              "  Padded  ": "x"
    """)
    with pytest.raises(ValueError, match="section name"):
        gsp._read_memo_config(config_path)


def _claims_dir(tmp_path, **files):
    d = tmp_path / "claims"
    d.mkdir(exist_ok=True)
    for stem, text in files.items():
        (d / f"{stem}.md").write_text(textwrap.dedent(text), encoding="utf-8")
    return str(d)


def test_load_from_claims_builds_tuples(tmp_path):
    src = _folder_with_pdfs(tmp_path, "d.pdf")
    cd = _claims_dir(tmp_path, **{"MEMO-1": f"""
        ---
        memo_id: MEMO-1
        source_folder: {src}
        batch_size: 20
        ---

        ## Ownership

        1. Acme is listed.
        2. Acme trades in London.
    """})
    tuples = gsp.load_memo_sections_from_claims(cd)
    assert len(tuples) == 1
    memo_id, section, claims, docs, overrides = tuples[0]
    assert (memo_id, section) == ("MEMO-1", "Ownership")
    assert claims == ["Acme is listed.", "Acme trades in London."]
    assert overrides == {"batch_size": 20}
    assert docs[0][0] == "d.pdf"


def test_load_from_claims_empty_dir_errors(tmp_path):
    d = tmp_path / "claims"
    d.mkdir()
    with pytest.raises(ValueError, match="no .md claims files"):
        gsp.load_memo_sections_from_claims(str(d))


def test_load_from_claims_validates_all_before_reading_pdfs(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(gsp, "load_pdf_text", lambda p: calls.append(p) or "text long enough here")
    src = _folder_with_pdfs(tmp_path, "d.pdf")
    _claims_dir(tmp_path, **{
        "MEMO-1": f"---\nmemo_id: MEMO-1\nsource_folder: {src}\n---\n\n## A\n\n1. ok\n",
        "MEMO-2": "this file has no frontmatter fence at all\n",
    })
    with pytest.raises(ValueError, match="MEMO-2.md"):
        gsp.load_memo_sections_from_claims(str(tmp_path / "claims"))
    assert calls == []  # parse pass failed before any PDF read


@pytest.fixture
def stub_extract(monkeypatch):
    """extract_atomic_claims returns one claim per section, keyed off the text."""
    monkeypatch.setattr(gsp, "extract_atomic_claims", lambda text, client: [f"claim from: {text.strip()[:20]}"])


def test_run_extract_writes_one_file_per_memo(tmp_path, stub_extract):
    cfg = _config(tmp_path, """
        memos:
          - id: MEMO-1
            source_folder: SRC
            sections: {Ownership: "Acme is listed in London."}
    """)
    counts = gsp.run_extract(cfg, str(tmp_path / "claims"), llm_client=None)
    assert counts == {"written": 1, "skipped": 0, "drift": 0, "failed": 0}
    written = (tmp_path / "claims" / "MEMO-1.md").read_text(encoding="utf-8")
    assert "## Ownership" in written
    assert "1. claim from:" in written


def test_run_extract_never_overwrites_and_reports_drift(tmp_path, stub_extract):
    cfg = _config(tmp_path, """
        memos:
          - id: MEMO-1
            source_folder: SRC
            sections: {Ownership: "a", Governance: "b"}
    """)
    cd = tmp_path / "claims"
    cd.mkdir()
    (cd / "MEMO-1.md").write_text("---\nmemo_id: MEMO-1\nsource_folder: s\n---\n\n## Ownership\n\n1. hand-written\n", encoding="utf-8")
    counts = gsp.run_extract(cfg, str(cd), llm_client=None)
    assert counts == {"written": 0, "skipped": 1, "drift": 1, "failed": 0}
    assert "hand-written" in (cd / "MEMO-1.md").read_text(encoding="utf-8")  # untouched


def test_run_extract_skips_malformed_existing_file_without_crashing(tmp_path, stub_extract):
    cfg = _config(tmp_path, """
        memos:
          - id: MEMO-1
            source_folder: SRC
            sections: {Ownership: "a"}
    """)
    cd = tmp_path / "claims"
    cd.mkdir()
    (cd / "MEMO-1.md").write_text("### wrong heading level\nnot valid at all\n", encoding="utf-8")
    counts = gsp.run_extract(cfg, str(cd), llm_client=None)
    # _scan_claims_file_sections returns [] on the broken body, so every
    # memos.yaml section reads as "added" -> skipped-with-drift, never a crash.
    assert counts == {"written": 0, "skipped": 1, "drift": 1, "failed": 0}


def test_run_extract_aborts_memo_on_section_failure(tmp_path, monkeypatch):
    def boom(text, client):
        if "explode" in text:
            raise RuntimeError("provider down")
        return ["ok claim"]
    monkeypatch.setattr(gsp, "extract_atomic_claims", boom)
    cfg = _config(tmp_path, """
        memos:
          - id: MEMO-1
            source_folder: SRC
            sections: {A: "fine", B: "please explode"}
          - id: MEMO-2
            source_folder: SRC
            sections: {A: "fine"}
    """)
    counts = gsp.run_extract(cfg, str(tmp_path / "claims"), llm_client=None)
    assert counts["failed"] == 1
    assert counts["written"] == 1
    assert not (tmp_path / "claims" / "MEMO-1.md").exists()
    assert (tmp_path / "claims" / "MEMO-2.md").exists()


def test_run_extract_treats_empty_result_as_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(gsp, "extract_atomic_claims", lambda text, client: [])
    cfg = _config(tmp_path, "memos:\n  - id: MEMO-1\n    source_folder: SRC\n    sections: {A: \"x\"}\n")
    counts = gsp.run_extract(cfg, str(tmp_path / "claims"), llm_client=None)
    assert counts["failed"] == 1
    assert not (tmp_path / "claims" / "MEMO-1.md").exists()


def test_run_extract_bad_source_folder_fails_before_any_llm_call(tmp_path, monkeypatch):
    spy = []
    monkeypatch.setattr(gsp, "extract_atomic_claims", lambda *a, **k: spy.append(1) or ["c"])
    p = tmp_path / "memos.yaml"
    p.write_text("memos:\n  - id: MEMO-1\n    source_folder: /no/such/folder\n    sections: {A: \"x\"}\n", encoding="utf-8")
    counts = gsp.run_extract(str(p), str(tmp_path / "claims"), llm_client=None)
    assert counts == {"written": 0, "skipped": 0, "drift": 0, "failed": 1}
    assert spy == []  # no extraction call was made
    assert not (tmp_path / "claims" / "MEMO-1.md").exists()


def test_run_extract_write_failure_is_per_memo_not_fatal(tmp_path, monkeypatch):
    # extract_atomic_claims can return a claim with an interior newline;
    # write_claims_file rejects it -- that must fail this memo, not the run.
    def maybe_bad(text, client):
        return ["line one\nline two"] if "bad" in text else ["clean claim"]
    monkeypatch.setattr(gsp, "extract_atomic_claims", maybe_bad)
    cfg = _config(tmp_path, """
        memos:
          - id: MEMO-1
            source_folder: SRC
            sections: {A: "please be bad"}
          - id: MEMO-2
            source_folder: SRC
            sections: {A: "fine"}
    """)
    counts = gsp.run_extract(cfg, str(tmp_path / "claims"), llm_client=None)
    assert counts == {"written": 1, "skipped": 0, "drift": 0, "failed": 1}
    assert not (tmp_path / "claims" / "MEMO-1.md").exists()
    assert (tmp_path / "claims" / "MEMO-2.md").exists()


def test_run_extract_makes_no_evidence_calls(tmp_path, stub_extract, monkeypatch):
    spy = []
    monkeypatch.setattr(gsp, "build_golden_set_draft", lambda *a, **k: spy.append(1))
    monkeypatch.setattr(gsp, "propose_evidence_from_chunks_batched", lambda *a, **k: spy.append(1))
    cfg = _config(tmp_path, "memos:\n  - id: MEMO-1\n    source_folder: SRC\n    sections: {A: \"x\"}\n")
    gsp.run_extract(cfg, str(tmp_path / "claims"), llm_client=None)
    assert spy == []


def test_run_build_wires_loader_to_batch_and_export(tmp_path, monkeypatch):
    src = _folder_with_pdfs(tmp_path, "d.pdf")
    cd = _claims_dir(tmp_path, **{"MEMO-1": f"---\nmemo_id: MEMO-1\nsource_folder: {src}\n---\n\n## A\n\n1. a claim\n"})
    seen = {}
    fake_df = pd.DataFrame([{"claim_id": "x", "claim_text": "a claim"}])

    def fake_batch(ms, client, **kw):
        seen["sections"] = ms
        return fake_df

    monkeypatch.setattr(gsp, "build_golden_set_batch", fake_batch)
    monkeypatch.setattr(gsp, "export_for_review", lambda df, path: seen.update(review=path, review_df=df))
    out = gsp.run_build(cd, llm_client=None,
                        checkpoint_path=str(tmp_path / "cp.parquet"), review_path=str(tmp_path / "r.xlsx"))
    assert seen["sections"][0][2] == ["a claim"]      # slot 3 is the claims list
    assert seen["review"] == str(tmp_path / "r.xlsx")
    assert seen["review_df"] is fake_df  # the real frame flows through
    assert out is fake_df


def test_run_build_errors_on_empty_claims_dir(tmp_path):
    d = tmp_path / "claims"
    d.mkdir()
    with pytest.raises(ValueError, match="no .md claims files"):
        gsp.run_build(str(d), llm_client=None)


def test_hand_authored_file_builds_end_to_end(tmp_path, monkeypatch):
    """AC-2: a claims file written by hand (not by write_claims_file) runs through build."""
    src = _folder_with_pdfs(tmp_path, "d.pdf")
    cd = tmp_path / "claims"
    cd.mkdir()
    (cd / "MEMO-9.md").write_text(
        f"---\nmemo_id: MEMO-9\nsource_folder: {src}\n---\n\n## Handmade\n\n1. A hand-written claim.\n2. Another.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gsp, "build_chunk_index", lambda docs, **k: [{"chunk_id": "d_0", "doc_id": "d", "chunk_text": "t", "start_offset": 0}])
    monkeypatch.setattr(gsp, "bm25_threshold_shortlist", lambda *a, **k: [])
    monkeypatch.setattr(gsp, "propose_evidence_from_chunks_batched", lambda *a, **k: ([], {"dropped": 0, "recovered": 0}))
    monkeypatch.setattr(gsp, "export_for_review", lambda df, p: None)
    df = gsp.run_build(str(cd), llm_client=None,
                       checkpoint_path=str(tmp_path / "cp.parquet"), review_path=str(tmp_path / "r.xlsx"))
    assert sorted(df["claim_text"]) == ["A hand-written claim.", "Another."]


def test_extract_then_build_preserves_claim_text(tmp_path, monkeypatch):
    """AC-3 proxy: claims that go into extract come out as build's claim_text."""
    monkeypatch.setattr(gsp, "extract_atomic_claims", lambda text, c: ["Alpha fact.", "Beta fact."])
    cfg = _config(tmp_path, "memos:\n  - id: MEMO-1\n    source_folder: SRC\n    sections: {Only: \"whatever\"}\n")
    gsp.run_extract(cfg, str(tmp_path / "claims"), llm_client=None)
    monkeypatch.setattr(gsp, "build_chunk_index", lambda docs, **k: [{"chunk_id": "d_0", "doc_id": "d", "chunk_text": "t", "start_offset": 0}])
    monkeypatch.setattr(gsp, "bm25_threshold_shortlist", lambda *a, **k: [])
    monkeypatch.setattr(gsp, "propose_evidence_from_chunks_batched", lambda *a, **k: ([], {"dropped": 0, "recovered": 0}))
    monkeypatch.setattr(gsp, "export_for_review", lambda df, p: None)
    df = gsp.run_build(str(tmp_path / "claims"), llm_client=None,
                       checkpoint_path=str(tmp_path / "cp.parquet"))
    assert sorted(df["claim_text"]) == ["Alpha fact.", "Beta fact."]


def test_main_rejects_unknown_command(monkeypatch):
    monkeypatch.setattr(gsp.LLMClient, "from_env", classmethod(lambda cls: None))
    with pytest.raises(SystemExit, match="buld"):
        gsp._main(["golden_set_pipeline.py", "buld"])


def test_main_no_arg_runs_build(monkeypatch):
    monkeypatch.setattr(gsp.LLMClient, "from_env", classmethod(lambda cls: None))
    called = {}
    monkeypatch.setattr(gsp, "run_build", lambda claims_dir, *, llm_client: called.setdefault("build", claims_dir))
    gsp._main(["golden_set_pipeline.py"])
    assert called["build"] == "claims"


def test_run_extract_undecodable_existing_file_is_skipped_not_fatal(tmp_path, stub_extract, caplog):
    # Reviewer finding 1: a claims file that can no longer be decoded as
    # UTF-8 (e.g. re-saved as latin-1 by Word/Excel) must not abort the
    # whole run -- it's counted skipped (never overwritten -- we can't know
    # its sections), not drift (drift means we successfully read it and
    # found a gap), and every other memo still gets processed.
    cfg = _config(tmp_path, """
        memos:
          - id: MEMO-1
            source_folder: SRC
            sections: {Ownership: "a"}
          - id: MEMO-2
            source_folder: SRC
            sections: {Ownership: "b"}
    """)
    cd = tmp_path / "claims"
    cd.mkdir()
    (cd / "MEMO-1.md").write_bytes("## Ownership\n\n1. caf\xe9 latin-1\n".encode("latin-1"))

    with caplog.at_level("WARNING", logger=gsp.logger.name):
        counts = gsp.run_extract(cfg, str(cd), llm_client=None)

    assert counts == {"written": 1, "skipped": 1, "drift": 0, "failed": 0}
    assert (cd / "MEMO-2.md").exists()
    assert "MEMO-1" in caplog.text


def test_main_extract_arg_runs_extract_and_does_not_exit_on_success(monkeypatch):
    monkeypatch.setattr(gsp.LLMClient, "from_env", classmethod(lambda cls: None))
    called = {}

    def fake_extract(config_path, claims_dir, *, llm_client):
        called["extract"] = (config_path, claims_dir)
        return {"written": 1, "skipped": 0, "drift": 0, "failed": 0}

    monkeypatch.setattr(gsp, "run_extract", fake_extract)
    gsp._main(["golden_set_pipeline.py", "extract"])  # must not raise
    assert called["extract"] == ("memos.yaml", "claims")


def test_main_extract_arg_exits_1_when_any_memo_failed(monkeypatch):
    monkeypatch.setattr(gsp.LLMClient, "from_env", classmethod(lambda cls: None))
    monkeypatch.setattr(gsp, "run_extract", lambda *a, **k: {"written": 0, "skipped": 0, "drift": 0, "failed": 1})
    with pytest.raises(SystemExit) as excinfo:
        gsp._main(["golden_set_pipeline.py", "extract"])
    assert excinfo.value.code == 1


def test_read_memo_config_rejects_ids_differing_only_in_case(tmp_path):
    """
    The memo id doubles as the claims-file name. On a case-insensitive
    filesystem (APFS/HFS+/NTFS -- this repo's platform) 'Memo' and 'memo'
    are ONE file, so run_extract's never-overwrite guard would count the
    second memo as *skipped*: exit status 0, nothing reported failed, and a
    build that silently omits that memo from the golden set entirely. An
    exact-match uniqueness check cannot see that collision, so the check is
    case-folded.
    """
    cfg = _config(tmp_path, """
        memos:
          - id: Memo
            source_folder: SRC
            sections:
              SecA: Alpha text.
          - id: memo
            source_folder: SRC
            sections:
              SecX: Xray text.
        """)
    with pytest.raises(ValueError, match="reused by an earlier entry"):
        gsp._read_memo_config(cfg)


def test_read_memo_config_still_allows_ids_that_differ_beyond_case(tmp_path):
    """The case-folded check must not over-reach onto genuinely distinct ids."""
    cfg = _config(tmp_path, """
        memos:
          - id: MEMO-1
            source_folder: SRC
            sections:
              SecA: Alpha text.
          - id: memo-2
            source_folder: SRC
            sections:
              SecX: Xray text.
        """)
    assert [m["id"] for m in gsp._read_memo_config(cfg)] == ["MEMO-1", "memo-2"]
