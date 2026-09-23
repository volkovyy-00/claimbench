"""
Tests for the claims-file parser/writer and the drift scanner
(parse_claims_file, write_claims_file, _scan_claims_file_sections) added for
the two-stage split. All mocked -- no LLM, no real PDFs -- per this repo's
testing convention (CLAUDE.md, "Testing convention").
"""
import os
import shutil
import textwrap
import uuid

import pytest

import golden_set_pipeline as gsp


def test_module_constants_present_and_shaped():
    assert isinstance(gsp._CLAIM_ID_NAMESPACE, uuid.UUID)
    assert str(gsp._CLAIM_ID_NAMESPACE) == "ee12e9dc-1de2-4125-aabc-244988261980"
    assert gsp._CLAIMS_FRONTMATTER_FIELDS == {
        "memo_id", "source_folder", "filing_entity", "relative_threshold", "min_candidates", "batch_size",
    }
    # Explicit, hardcoded list -- not re-derived from _error_row(...).keys(),
    # which would just prove _SCHEMA_COLUMNS agrees with itself and catch
    # nothing if a column were silently added, removed, or reordered.
    assert gsp._SCHEMA_COLUMNS == [
        "claim_id",
        "memo_id",
        "section",
        "claim_text",
        "doc_id",
        "chunk_id",
        "chunk_text",
        "bm25_score",
        "evidence_span",
        "found",
        "confidence",
        "ambiguous_match",
        "verbatim_match",
        "human_reviewed",
        "tag",
    ]


def test_committed_example_parses_when_copied_into_place(tmp_path):
    """
    claims.example.md's stem is "claims.example", not "MEMO-001" -- so
    parse_claims_file (which enforces memo_id == filename stem) cannot parse
    it in place. This mirrors the real workflow: copy the template into
    claims/ under its memo id, then parse it there.
    """
    example = os.path.join(os.path.dirname(gsp.__file__), "claims.example.md")
    dest = tmp_path / "MEMO-001.md"
    shutil.copy(example, dest)
    memo_id, _, overrides, sections = gsp.parse_claims_file(str(dest))
    assert memo_id == "MEMO-001"
    assert overrides == {"relative_threshold": 0.45}
    assert [s[0] for s in sections] == ["Business Profile", "Ownership"]


@pytest.mark.parametrize("name", ["Business Profile", "Ownership", "Q4 2024 Results"])
def test_valid_section_name_accepts_clean_names(name):
    assert gsp._valid_section_name(name) is True


@pytest.mark.parametrize("name", [
    "", "   ", "  Padded  ", "Has\ttab", "Carriage\rreturn", "Bad\x1fchar",
    "# hashy", "## double", "Ends in ATX #", "Double ATX ##",
])
def test_valid_section_name_rejects_bad_names(name):
    assert gsp._valid_section_name(name) is False


def test_valid_section_name_allows_hash_without_preceding_space():
    # 'C#' is not an ATX close (no space before '#'), so it round-trips and is allowed.
    assert gsp._valid_section_name("C#") is True
    assert gsp._valid_section_name("F# and C#") is True


def test_scan_returns_heading_names(tmp_path):
    f = tmp_path / "MEMO-1.md"
    f.write_text("---\nmemo_id: MEMO-1\n---\n\n## Alpha\n\n1. x\n\n## Beta ##\n\n1. y\n", encoding="utf-8")
    assert gsp._scan_claims_file_sections(str(f)) == ["Alpha", "Beta"]


def test_scan_never_raises_on_malformed_body(tmp_path):
    f = tmp_path / "MEMO-1.md"
    f.write_text("no frontmatter\n### Wrong\ngarbage\x0c\n", encoding="utf-8")
    assert gsp._scan_claims_file_sections(str(f)) == []  # best-effort, no exception


def test_scan_strips_bom(tmp_path):
    f = tmp_path / "MEMO-1.md"
    f.write_bytes("﻿## Alpha\n1. x\n".encode("utf-8"))
    assert gsp._scan_claims_file_sections(str(f)) == ["Alpha"]


def _write(tmp_path, name, text):
    f = tmp_path / name
    f.write_text(textwrap.dedent(text), encoding="utf-8")
    return str(f)


def test_parse_frontmatter_happy(tmp_path):
    p = _write(tmp_path, "MEMO-1.md", """\
        ---
        memo_id: MEMO-1
        source_folder: sources/acme
        relative_threshold: 0.45
        ---

        ## Ownership

        1. Acme is listed.
        """)
    memo_id, source_folder, overrides, sections = gsp.parse_claims_file(p)
    assert memo_id == "MEMO-1"
    assert source_folder == "sources/acme"
    assert overrides == {"relative_threshold": 0.45}  # only the key that was set


def test_filing_entity_is_optional_and_read_on_its_own(tmp_path):
    with_entity = _write(tmp_path, "MEMO-1.md", "---\nmemo_id: MEMO-1\nfiling_entity: Acme Ltd\n"
                                                "source_folder: s\n---\n\n## A\n\n1. Acme is old.\n")
    assert gsp.read_filing_entity(with_entity) == "Acme Ltd"
    # parse_claims_file's return shape is unchanged: the key is validated, not returned
    assert gsp.parse_claims_file(with_entity) == ("MEMO-1", "s", {}, [("A", ["Acme is old."])])
    without = _write(tmp_path, "MEMO-2.md", "---\nmemo_id: MEMO-2\nsource_folder: s\n---\n")
    assert gsp.read_filing_entity(without) is None


@pytest.mark.parametrize("value", ["''", "", "3", "[Acme]", "|\n  Acme\n  Ltd"])
def test_filing_entity_must_be_a_non_blank_one_line_string(tmp_path, value):
    p = _write(tmp_path, "MEMO-1.md", f"---\nmemo_id: MEMO-1\nsource_folder: s\nfiling_entity: {value}\n---\n")
    with pytest.raises(ValueError, match="'filing_entity' must be a non-blank one-line string"):
        gsp.parse_claims_file(p)
    with pytest.raises(ValueError, match="'filing_entity' must be a non-blank one-line string"):
        gsp.read_filing_entity(p)


@pytest.mark.parametrize("body,msg", [
    ("no fence here\n", "must start with a '---' frontmatter fence"),
    ("---\nmemo_id: MEMO-1\n", "is never closed"),
    ("---\n---\n## A\n1. x\n", "frontmatter is empty"),
    ("---\n: : :\n---\n", "not valid YAML"),
    ("---\n- a\n- b\n---\n", "must be a mapping"),
    ("---\nmemo_id: MEMO-1\nsource_folder: s\nnope: 1\n---\n", "unrecognized frontmatter key"),
    ("---\nsource_folder: s\n---\n", "memo_id"),
    ("---\nmemo_id: MEMO/1\nsource_folder: s\n---\n", "memo_id"),
    ("---\nmemo_id: OTHER\nsource_folder: s\n---\n", "filename"),
    ("---\nmemo_id: MEMO-1\n---\n", "source_folder"),
])
def test_parse_frontmatter_errors(tmp_path, body, msg):
    p = _write(tmp_path, "MEMO-1.md", body)
    with pytest.raises(ValueError, match=msg):
        gsp.parse_claims_file(p)


FM = "---\nmemo_id: MEMO-1\nsource_folder: s\n---\n\n"


def test_body_happy_multisection(tmp_path):
    p = _write(tmp_path, "MEMO-1.md", FM + textwrap.dedent("""\
        ## Business Profile

        1. Acme is the largest listed widget maker in Europe.
        2. Acme's plants were valued at EUR 20bn at 31 December 2024.

        <!-- claim 3 removed 2026-08-30 -->

        ## Ownership

        1. Acme is listed in London and in Frankfurt.
        """))
    _, _, _, sections = gsp.parse_claims_file(p)
    assert sections == [
        ("Business Profile", [
            "Acme is the largest listed widget maker in Europe.",
            "Acme's plants were valued at EUR 20bn at 31 December 2024.",
        ]),
        ("Ownership", ["Acme is listed in London and in Frankfurt."]),
    ]


def test_body_claim_text_may_look_dangerous(tmp_path):
    p = _write(tmp_path, "MEMO-1.md", FM + "## A\n\n1. ## not a heading and --- not a fence <!-- not a comment\n")
    _, _, _, sections = gsp.parse_claims_file(p)
    assert sections == [("A", ["## not a heading and --- not a fence <!-- not a comment"])]


def test_body_list_number_is_cosmetic(tmp_path):
    p = _write(tmp_path, "MEMO-1.md", FM + "## A\n\n7. first\n7. second\n")
    _, _, _, sections = gsp.parse_claims_file(p)
    assert sections == [("A", ["first", "second"])]


@pytest.mark.parametrize("body,msg", [
    (FM + "1. orphan claim\n", "claim before the first"),
    (FM + "## A\n\n1. x\n\n## A\n\n1. y\n", "duplicate section heading"),
    (FM + "## A\n\n1. x\n### B\n\n1. y\n", "exactly '## '"),
    (FM + "##Nospace\n\n1. x\n", "no space after"),
    (FM + "  ## Indented\n\n1. x\n", "must not be indented"),
    (FM + "## \n\n1. x\n", "has no name"),
    (FM + "## A\n\n1. x\n   trailing note\n", "one line each"),
    (FM + "## A\n\n1. \n", "has no text"),
    (FM + "## A\n\n1. bad\x1fchar\n", "control character"),
    (FM + "## A\n\n1. x\n<!-- unterminated\n", "multi-line HTML comments"),
])
def test_body_errors(tmp_path, body, msg):
    p = _write(tmp_path, "MEMO-1.md", body)
    with pytest.raises(ValueError, match=msg):
        gsp.parse_claims_file(p)


def test_body_complete_comment_may_abut_a_claim(tmp_path):
    p = _write(tmp_path, "MEMO-1.md", FM + "## A\n\n1. x\n<!-- fine, single line -->\n2. y\n")
    _, _, _, sections = gsp.parse_claims_file(p)
    assert sections == [("A", ["x", "y"])]


def test_no_sections_at_all(tmp_path):
    p = _write(tmp_path, "MEMO-1.md", "---\nmemo_id: MEMO-1\nsource_folder: s\n---\n\njust prose, no headings\n")
    with pytest.raises(ValueError, match="no '## ' section headings found"):
        gsp.parse_claims_file(p)


def test_empty_section_names_itself_and_hints(tmp_path):
    p = _write(tmp_path, "MEMO-1.md", FM + "## Empty\n\n- I used a dash by mistake\n\n## Real\n\n1. x\n")
    with pytest.raises(ValueError, match=r"section 'Empty' has no claims .*dash"):
        gsp.parse_claims_file(p)


def test_mixed_marker_warns_but_parses(tmp_path, caplog):
    p = _write(tmp_path, "MEMO-1.md", FM + "## A\n\n1. real claim\n\n- a dash note\n")
    with caplog.at_level("WARNING", logger=gsp.logger.name):
        _, _, _, sections = gsp.parse_claims_file(p)
    assert sections == [("A", ["real claim"])]
    assert "were not parsed as claims" in caplog.text
    assert "a dash note" in caplog.text


def test_forgotten_claim_number_warns_and_is_not_silently_dropped(tmp_path, caplog):
    # Final-review finding 1: a plain sentence under a section, missing its
    # leading 'N. ', used to be silently dropped -- zero log lines, claim
    # gone. It must now warn, naming the section and quoting the dropped
    # line, so the omission is visible instead of poisoning every
    # downstream recall metric silently.
    dropped = "Acme's LTV was 38.0% at 31 Dec 2024."
    p = _write(tmp_path, "MEMO-1.md", FM + (
        "## Business Profile\n\n"
        "1. Acme is the largest listed widget maker in Europe.\n\n"
        f"{dropped}\n\n"
        "2. Acme's plants were valued at EUR 20bn.\n"
    ))
    with caplog.at_level("WARNING", logger=gsp.logger.name):
        _, _, _, sections = gsp.parse_claims_file(p)
    assert sections == [(
        "Business Profile",
        [
            "Acme is the largest listed widget maker in Europe.",
            "Acme's plants were valued at EUR 20bn.",
        ],
    )]
    assert "were not parsed as claims" in caplog.text
    assert "Business Profile" in caplog.text
    assert dropped in caplog.text


def test_single_line_comment_note_still_silent_no_warning(tmp_path, caplog):
    # Regression guard for the finding-1 fix: the intended silent-note
    # mechanism -- a complete single-line '<!-- ... -->' comment -- is
    # handled earlier (_COMPLETE_COMMENT_RE) and must remain unaffected;
    # only bare, uncommented prose should now warn.
    p = _write(tmp_path, "MEMO-1.md", FM + (
        "## A\n\n1. real claim\n\n<!-- a deliberate, silent review note -->\n\n2. another claim\n"
    ))
    with caplog.at_level("WARNING", logger=gsp.logger.name):
        _, _, _, sections = gsp.parse_claims_file(p)
    assert sections == [("A", ["real claim", "another claim"])]
    assert caplog.text == ""


def test_bom_and_crlf(tmp_path):
    f = tmp_path / "MEMO-1.md"
    f.write_bytes("﻿---\r\nmemo_id: MEMO-1\r\nsource_folder: s\r\n---\r\n\r\n## A\r\n\r\n1. x\r\n".encode("utf-8"))
    memo_id, _, _, sections = gsp.parse_claims_file(str(f))
    assert memo_id == "MEMO-1"
    assert sections == [("A", ["x"])]


def test_form_feed_interior_errors_boundary_is_stripped(tmp_path):
    # \x0c is in _CONTROL_CHARS_RE -> interior form-feed is a rejected control char
    p = _write(tmp_path, "MEMO-1.md", FM + "## A\n\n1. bad\x0cmiddle\n")
    with pytest.raises(ValueError, match="control character"):
        gsp.parse_claims_file(p)
    # a trailing form-feed is whitespace to .strip() -> silently removed, claim text clean
    p2 = _write(tmp_path, "MEMO-2.md", "---\nmemo_id: MEMO-2\nsource_folder: s\n---\n\n## A\n\n1. clean\x0c\n")
    assert gsp.parse_claims_file(p2)[3] == [("A", ["clean"])]


SECTIONS = [
    ("Business Profile", ["Acme is the largest listed operator in Europe.", "Plants were valued at EUR 20bn."]),
    ("Ownership", ["Acme is listed in London."]),
]


def test_write_then_parse_roundtrips(tmp_path):
    p = str(tmp_path / "MEMO-1.md")
    gsp.write_claims_file(p, "MEMO-1", "sources/acme", {"relative_threshold": 0.45}, SECTIONS)
    assert gsp.parse_claims_file(p) == ("MEMO-1", "sources/acme", {"relative_threshold": 0.45}, SECTIONS)


def test_write_roundtrips_adversarial_claim_text(tmp_path):
    hostile = [("A", ["## fake", "--- not a fence", "<!-- x", "3. leading digit-dot", 'has "quotes" and ünïcode'])]
    p = str(tmp_path / "MEMO-1.md")
    gsp.write_claims_file(p, "MEMO-1", "s", {}, hostile)
    assert gsp.parse_claims_file(p)[3] == hostile


def test_write_roundtrips_section_name_with_hash(tmp_path):
    # 'C#' has no space before the '#', so the ATX-close strip is a no-op and it round-trips.
    secs = [("C# language", ["A claim."]), ("F#", ["Another."])]
    p = str(tmp_path / "MEMO-1.md")
    gsp.write_claims_file(p, "MEMO-1", "s", {}, secs)
    assert gsp.parse_claims_file(p)[3] == secs
    # 'Foo ##' cannot round-trip, so write_claims_file must reject it.
    with pytest.raises(ValueError, match="invalid section name"):
        gsp.write_claims_file(p, "MEMO-1", "s", {}, [("Foo ##", ["x"])])


def test_write_frontmatter_key_order(tmp_path):
    p = str(tmp_path / "MEMO-1.md")
    gsp.write_claims_file(p, "MEMO-1", "s", {"batch_size": 20, "relative_threshold": 0.4}, SECTIONS)
    head = "\n".join(open(p, encoding="utf-8").read().splitlines()[:6])
    assert head.index("memo_id") < head.index("source_folder") < head.index("relative_threshold")
    assert head.index("relative_threshold") < head.index("batch_size")


@pytest.mark.parametrize("kwargs,msg", [
    (dict(sections=[("A", ["ok", "bad\nnewline"])]), "newline"),
    (dict(sections=[("A", ["x"]), ("A", ["y"])]), "duplicate section name"),          # duplicate section
    (dict(sections=[("A", ["ctrl\x1fchar"])]), "control character"),
    (dict(memo_id="OTHER", sections=SECTIONS), "must equal the filename stem"),       # stem mismatch
    (dict(overrides={"nope": 1}, sections=SECTIONS), "unrecognized override key"),
])
def test_write_rejects_bad_input(tmp_path, kwargs, msg):
    args = dict(memo_id="MEMO-1", source_folder="s", overrides={}, sections=SECTIONS)
    args.update(kwargs)
    with pytest.raises(ValueError, match=msg):
        gsp.write_claims_file(str(tmp_path / "MEMO-1.md"), args["memo_id"], args["source_folder"],
                              args["overrides"], args["sections"])


# --- Regression tests: fix round 1 findings 1-3 -----------------------------

def test_write_rejects_cr_in_claim_text(tmp_path):
    # Finding 1 (CRITICAL): an unrejected \r in claim text let the parser's
    # \r -> \n normalisation silently split one claim into two on read-back.
    p = str(tmp_path / "MEMO-1.md")
    with pytest.raises(ValueError, match="carriage return"):
        gsp.write_claims_file(p, "MEMO-1", "s", {}, [("A", ["foo\r2. injected"])])


def test_write_rejects_cr_in_source_folder(tmp_path):
    p = str(tmp_path / "MEMO-1.md")
    with pytest.raises(ValueError, match="carriage return"):
        gsp.write_claims_file(p, "MEMO-1", "s\rource", {}, SECTIONS)


def test_write_rejects_empty_sections(tmp_path):
    # Finding 2 (IMPORTANT): sections=[] used to write a frontmatter-only
    # file that parse_claims_file could never read back.
    p = str(tmp_path / "MEMO-1.md")
    with pytest.raises(ValueError, match="no sections to write"):
        gsp.write_claims_file(p, "MEMO-1", "s", {}, [])


def test_write_rejects_claim_ending_in_html_comment_open(tmp_path):
    # Finding 3 (IMPORTANT): a claim ending in '<!--' rendered a line the
    # parser reads as an unterminated multi-line HTML comment.
    p = str(tmp_path / "MEMO-1.md")
    with pytest.raises(ValueError, match="unterminated HTML comment"):
        gsp.write_claims_file(p, "MEMO-1", "s", {}, [("A", ["Note the marker <!--"])])


def test_write_rejects_section_name_ending_in_html_comment_open(tmp_path):
    # Finding 3 (IMPORTANT), section-name variant.
    p = str(tmp_path / "MEMO-1.md")
    with pytest.raises(ValueError, match="unterminated HTML comment"):
        gsp.write_claims_file(p, "MEMO-1", "s", {}, [("Foo <!--", ["x"])])


# --- Round 2: _CLAIM_MARKER_RE anchored at column 0 -------------------------

def test_indented_claim_marker_abutting_claim_raises(tmp_path):
    # An indented 'N.' directly under a claim, no blank line, is ambiguous
    # between a sub-point and a continuation -- it must raise, not be
    # silently promoted to its own top-level claim.
    p = _write(tmp_path, "MEMO-1.md", FM + "## A\n\n1. Real claim.\n   1. sub point\n")
    with pytest.raises(ValueError, match=r"column 0.*blank line before an indented list item"):
        gsp.parse_claims_file(p)


def test_indented_claim_marker_after_blank_line_warns_and_parses(tmp_path, caplog):
    # The same indented 'N.', but separated by a blank line, is just a loose
    # list item -- warned about and ignored, not an error, and the section
    # (with its one real claim) still parses.
    p = _write(tmp_path, "MEMO-1.md", FM + "## A\n\n1. Real claim.\n\n   1. sub point\n")
    with caplog.at_level("WARNING", logger=gsp.logger.name):
        _, _, _, sections = gsp.parse_claims_file(p)
    assert sections == [("A", ["Real claim."])]
    assert "were not parsed as claims" in caplog.text


def test_section_with_only_indented_claim_marker_has_no_claims(tmp_path):
    # An indented 'N.' is never a claim marker, so a section containing only
    # one falls out of the existing empty-section check.
    p = _write(tmp_path, "MEMO-1.md", FM + "## A\n\n   1. only this\n")
    with pytest.raises(ValueError, match=r"section 'A' has no claims .*only this"):
        gsp.parse_claims_file(p)


def test_column_zero_claims_and_roundtrip_unaffected(tmp_path):
    # Regression guard: write_claims_file always emits 'N. ' at column 0, so
    # the anchoring must not affect anything it produces.
    p = str(tmp_path / "MEMO-1.md")
    gsp.write_claims_file(p, "MEMO-1", "s", {}, SECTIONS)
    assert gsp.parse_claims_file(p) == ("MEMO-1", "s", {}, SECTIONS)
    # And a plain, unindented, hand-written file still parses as before.
    p2 = _write(tmp_path, "MEMO-2.md", "---\nmemo_id: MEMO-2\nsource_folder: s\n---\n\n"
                                        "## A\n\n1. first\n2. second\n")
    assert gsp.parse_claims_file(p2)[3] == [("A", ["first", "second"])]


def test_write_is_atomic_leaves_no_truncated_or_stray_tmp_file_on_mid_write_failure(tmp_path, monkeypatch):
    # Reviewer finding 2: a mid-write OSError (simulated ENOSPC, a
    # KeyboardInterrupt mid-write is the same shape) must not leave a
    # truncated file on disk -- the target must be either absent or
    # complete, and no sibling .tmp file should survive either.
    p = str(tmp_path / "MEMO-1.md")

    real_open = open

    def flaky_open(path, mode="r", *args, **kwargs):
        f = real_open(path, mode, *args, **kwargs)
        # Matches either a direct write to the target (pre-fix) or a write
        # to a sibling temp file (post-fix) -- whichever this
        # implementation actually opens for writing.
        if isinstance(path, str) and path.startswith(p) and "w" in mode:
            real_write = f.write

            def boom(data):
                real_write(data[:20])
                raise OSError("simulated ENOSPC")

            f.write = boom
        return f

    monkeypatch.setattr("builtins.open", flaky_open)

    with pytest.raises(OSError):
        gsp.write_claims_file(p, "MEMO-1", "s", {}, SECTIONS)

    assert not os.path.exists(p)
    assert not os.path.exists(p + ".tmp")


def test_unnumbered_line_before_first_heading_warns_instead_of_vanishing(tmp_path, caplog):
    """
    A claim whose 'N. ' was forgotten (or deleted during review) while it sits
    ABOVE the first '## ' heading is indistinguishable from prose, exactly as
    it is under a section. It is still ignored, but it must not be ignored
    *silently* -- that is the silent-claim-loss shape this parser exists to
    make visible.
    """
    path = _write(tmp_path, "M.md", (
        "---\nmemo_id: M\nsource_folder: src\n---\n\n"
        "this was meant to be a claim.\n\n"
        "## A\n\n1. one.\n"
    ))
    with caplog.at_level("WARNING"):
        _, _, _, sections = gsp.parse_claims_file(path)
    assert sections == [("A", ["one."])]
    assert "before the first '## ' heading" in caplog.text
    assert "this was meant to be a claim." in caplog.text


def test_complete_comment_before_first_heading_stays_silent(tmp_path, caplog):
    """The '<!-- ... -->' escape hatch must keep working in the preamble --
    it is where write_claims_file puts _EXTRACT_MARKER on every file it
    writes, so warning here would warn on every extract-produced file."""
    path = _write(tmp_path, "M.md", (
        "---\nmemo_id: M\nsource_folder: src\n---\n\n"
        f"{gsp._EXTRACT_MARKER}\n\n"
        "## A\n\n1. one.\n"
    ))
    with caplog.at_level("WARNING"):
        _, _, _, sections = gsp.parse_claims_file(path)
    assert sections == [("A", ["one."])]
    assert caplog.text == ""


def test_extract_written_file_round_trips_without_warning(tmp_path, caplog):
    """Guards the regression the test above describes, end to end."""
    path = str(tmp_path / "M.md")
    gsp.write_claims_file(path, "M", "src", {}, [("A", ["one."])])
    with caplog.at_level("WARNING"):
        memo_id, folder, ov, sections = gsp.parse_claims_file(path)
    assert (memo_id, folder, ov, sections) == ("M", "src", {}, [("A", ["one."])])
    assert caplog.text == ""
