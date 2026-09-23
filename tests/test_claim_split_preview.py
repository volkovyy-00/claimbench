"""
Tests for preview_claim_splits and the RUN_CLAIM_SPLIT_PREVIEW opt-in cell
(see local ticket 005, archived in the maintainer's notes repo, and
CLAUDE.md design decision 14).

These mock golden_set_pipeline.extract_atomic_claims (so no real LLM
provider is contacted) and builtins.input, consistent with this repo's
testing convention (CLAUDE.md, "Testing convention"). No prompt text
changes here, so the live-call exception doesn't apply.
"""
import ast

import pytest

import golden_set_pipeline as gsp


@pytest.fixture(autouse=True)
def _forbid_unmocked_input(monkeypatch):
    """
    A test that forgets to stub input() would hang instead of failing.
    Make an unmocked call raise loudly; tests that need an answer override
    builtins.input themselves (their monkeypatch.setattr wins).
    """
    monkeypatch.setattr(
        "builtins.input",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("unmocked input()")),
    )


class _Answers:
    """Feeds canned answers to input() and records the prompts it received."""

    def __init__(self, *answers):
        self._answers = iter(answers)
        self.prompts = []

    def __call__(self, prompt=""):
        self.prompts.append(prompt)
        try:
            return next(self._answers)
        except StopIteration:
            raise AssertionError("input() called more times than the test allows")


# --- the happy path ---------------------------------------------------------


def test_all_sections_are_shown_when_every_prompt_is_answered_yes(monkeypatch, capsys):
    sections = [
        ("MEMO-1", "Alpha", "text-a", []),
        ("MEMO-1", "Beta", "text-b", []),
        ("MEMO-1", "Gamma", "text-c", []),
    ]
    claims_by_text = {
        "text-a": ["A one.", "A two."],
        "text-b": ["B one."],
        "text-c": ["C one.", "C two.", "C three."],
    }
    monkeypatch.setattr(
        gsp, "extract_atomic_claims", lambda section_text, llm_client: claims_by_text[section_text]
    )
    answers = _Answers("y", "y")
    monkeypatch.setattr("builtins.input", answers)

    gsp.preview_claim_splits(sections, llm_client=object())

    out = capsys.readouterr().out
    assert "=== MEMO-1/Alpha (1/3) ===" in out
    assert "=== MEMO-1/Beta (2/3) ===" in out
    assert "=== MEMO-1/Gamma (3/3) ===" in out
    for claim in ("A one.", "A two.", "B one.", "C one.", "C two.", "C three."):
        assert claim in out
    assert "Preview complete." in out
    assert len(answers.prompts) == 2
    assert any("next section" in p for p in answers.prompts)


def test_claims_are_printed_numbered_in_the_extractors_exact_wording(monkeypatch, capsys):
    monkeypatch.setattr(
        gsp, "extract_atomic_claims", lambda section_text, llm_client: ["Claim one.", "Claim two."]
    )
    # one section -> no prompt, so the autouse input guard is never tripped

    gsp.preview_claim_splits([("M", "S", "t", [])], llm_client=object())

    out = capsys.readouterr().out
    assert "  1. Claim one." in out
    assert "  2. Claim two." in out


# --- stopping -------------------------------------------------------------


def test_answering_no_stops_before_the_next_section_is_shown(monkeypatch, capsys):
    sections = [
        ("M", "First", "t1", []),
        ("M", "Second", "t2", []),
        ("M", "Third", "t3", []),
    ]
    calls = []
    monkeypatch.setattr(
        gsp,
        "extract_atomic_claims",
        lambda section_text, llm_client: calls.append(section_text) or ["a claim"],
    )
    monkeypatch.setattr("builtins.input", _Answers("n"))

    gsp.preview_claim_splits(sections, llm_client=object())

    out = capsys.readouterr().out
    assert "=== M/First (1/3) ===" in out
    assert "=== M/Second" not in out
    assert "=== M/Third" not in out
    assert "Stopping preview." in out
    assert "Preview complete." not in out
    assert calls == ["t1"]  # extraction never ran for the unshown sections


@pytest.mark.parametrize("answer", ["", " ", "maybe", "q", "no", "n", "yes please"])
def test_any_answer_other_than_yes_stops_the_preview(monkeypatch, capsys, answer):
    sections = [("M", "First", "t1", []), ("M", "Second", "t2", [])]
    monkeypatch.setattr(gsp, "extract_atomic_claims", lambda s, c: ["a claim"])
    monkeypatch.setattr("builtins.input", _Answers(answer))

    gsp.preview_claim_splits(sections, llm_client=object())

    out = capsys.readouterr().out
    assert "=== M/Second" not in out
    assert "Stopping preview." in out


@pytest.mark.parametrize("answer", ["y", "Y", "yes", "YES", " y ", "  Yes  "])
def test_yes_variants_are_accepted(monkeypatch, capsys, answer):
    sections = [("M", "First", "t1", []), ("M", "Second", "t2", [])]
    monkeypatch.setattr(gsp, "extract_atomic_claims", lambda s, c: ["a claim"])
    monkeypatch.setattr("builtins.input", _Answers(answer))

    gsp.preview_claim_splits(sections, llm_client=object())

    out = capsys.readouterr().out
    assert "=== M/Second (2/2) ===" in out
    assert "Preview complete." in out


def test_a_section_whose_extraction_fails_shows_the_failure_and_stops(monkeypatch, capsys):
    sections = [
        ("M", "First", "t1", []),
        ("M", "Second", "t2", []),
        ("M", "Third", "t3", []),
    ]

    def extract(section_text, llm_client):
        if section_text == "t2":
            raise RuntimeError("boom")
        return ["a claim"]

    monkeypatch.setattr(gsp, "extract_atomic_claims", extract)
    answers = _Answers("y")  # only the prompt after section 1 is allowed
    monkeypatch.setattr("builtins.input", answers)

    gsp.preview_claim_splits(sections, llm_client=object())

    out = capsys.readouterr().out
    assert "=== M/Second (2/3) ===" in out
    assert "boom" in out
    assert "Stopping preview." in out
    assert "=== M/Third" not in out
    assert len(answers.prompts) == 1  # not prompted after the failed section


# --- shape / structure ---------------------------------------------------


def test_four_tuple_and_five_tuple_entries_are_both_accepted(monkeypatch, capsys):
    sections = [
        ("M", "S1", "t1", []),
        ("M", "S2", "t2", [], {"batch_size": 5}),
    ]
    monkeypatch.setattr(gsp, "extract_atomic_claims", lambda s, c: ["a claim"])
    monkeypatch.setattr("builtins.input", _Answers("y"))

    gsp.preview_claim_splits(sections, llm_client=object())

    out = capsys.readouterr().out
    assert "=== M/S1 (1/2) ===" in out
    assert "=== M/S2 (2/2) ===" in out


def test_the_final_section_does_not_prompt_to_continue(monkeypatch, capsys):
    sections = [("M", "S1", "t1", []), ("M", "S2", "t2", [])]
    monkeypatch.setattr(gsp, "extract_atomic_claims", lambda s, c: ["a claim"])
    answers = _Answers("y")  # a second call would raise "more times than allowed"
    monkeypatch.setattr("builtins.input", answers)

    gsp.preview_claim_splits(sections, llm_client=object())

    out = capsys.readouterr().out
    assert "=== M/S1 (1/2) ===" in out
    assert "=== M/S2 (2/2) ===" in out
    assert "Preview complete." in out
    assert len(answers.prompts) == 1


# --- the real batch run / plain script must never pause ------------------


def test_run_claim_split_preview_flag_defaults_to_false():
    assert gsp.RUN_CLAIM_SPLIT_PREVIEW is False


def test_preview_is_only_invoked_under_the_run_flag():
    """
    Every call to preview_claim_splits in the module source must sit
    inside `if RUN_CLAIM_SPLIT_PREVIEW:`. An unguarded module-scope call
    (or one under `if __name__ == "__main__":`) would run on plain
    `import`, `python golden_set_pipeline.py`, and every pytest run —
    hanging on input() and firing a real LLM call. Checked structurally
    against the source so it can't pass for the wrong reason (a live
    import check goes green whenever the stray LLM call happens to fail).
    """
    tree = ast.parse(open(gsp.__file__).read())
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}

    def under_run_flag(node):
        while node in parents:
            node = parents[node]
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Name)
                and node.test.id == "RUN_CLAIM_SPLIT_PREVIEW"
            ):
                return True
        return False

    call_sites = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "preview_claim_splits"
    ]
    assert call_sites, "expected the module to invoke preview_claim_splits somewhere"
    assert all(under_run_flag(node) for node in call_sites)
