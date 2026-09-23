"""
Tests for .github/scripts/release_check.py — the rules behind the required
`Release` check and the notes step of the `Publish release` workflow (see
CONTRIBUTING.md, sections 3 and 4). Pure text in, problems out: no git, no
network.
"""

import json

import pytest

import release_check as rc

BASE = """# Changelog

Intro paragraph.

## [0.2.0] - 2026-09-22

### Added

- Something older. (EV-1)

## [0.1.0] - 2026-09-08

- The first release.
"""

NEW_BODY = "### Added\n\n- A new thing. (EV-11)\n"


def with_release(heading, body=NEW_BODY, base=BASE):
    """`base` with `heading` and `body` inserted above its top version."""
    before, marker, after = base.partition("## [0.2.0]")
    return f"{before}{heading}\n\n{body}\n{marker}{after}"


def check(head, title="EV-11: Add a thing", labels=(), base=BASE):
    return rc.check_pr(title, list(labels), base, head)


# --- a release PR --------------------------------------------------------


def test_release_pr_that_follows_every_rule_passes():
    assert check(with_release("## [0.3.0] - 2026-09-23")) == []


@pytest.mark.parametrize("version", ["0.2.1", "0.3.0", "1.0.0"])
def test_patch_minor_and_major_steps_all_pass(version):
    assert check(with_release(f"## [{version}] - 2026-09-23")) == []


@pytest.mark.parametrize("version", ["0.2.0", "0.2.2", "0.3.1", "0.4.0", "2.0.0", "0.1.9"])
def test_any_other_version_fails(version):
    problems = check(with_release(f"## [{version}] - 2026-09-23"))
    assert len(problems) == 1
    assert "not one step above 0.2.0" in problems[0]
    assert "0.2.1, 0.3.0, 1.0.0" in problems[0]


def test_release_pr_without_a_new_heading_fails():
    problems = check(BASE)
    assert len(problems) == 1
    assert "adds exactly one new version heading" in problems[0]
    assert "no-release" in problems[0]


def test_two_new_headings_fail():
    head = with_release("## [0.3.0] - 2026-09-23", body=NEW_BODY + "\n## [0.2.1] - 2026-09-23\n\n- x (EV-11)\n")
    assert any("exactly one new version heading" in p for p in check(head))


def test_new_heading_below_the_top_fails():
    head = BASE.replace("## [0.1.0]", "## [0.3.0] - 2026-09-23\n\n- x (EV-11)\n\n## [0.1.0]")
    assert any("exactly one new version heading" in p for p in check(head))


def test_changing_an_existing_heading_fails():
    head = with_release("## [0.3.0] - 2026-09-23").replace("## [0.1.0] - 2026-09-08", "## [0.1.0] - 2026-09-09")
    assert any("exactly one new version heading" in p for p in check(head))


def test_date_that_is_not_a_real_date_fails_with_only_that_message():
    # Not also "adds exactly one new version heading ... add the no-release label".
    problems = check(with_release("## [0.3.0] - 2026-02-30"))
    assert problems == ["CHANGELOG.md line 5: 2026-02-30 is not a real date."]


def test_date_before_the_previous_release_fails():
    problems = check(with_release("## [0.3.0] - 2026-09-21"))
    assert problems == ["Version 0.3.0 is dated 2026-09-21, before 0.2.0 (2026-09-22)."]


def test_future_date_is_allowed():
    # No upper bound on purpose: the check runs in UTC, the maintainer may be ahead of it.
    assert check(with_release("## [0.3.0] - 2099-01-01")) == []


def test_section_without_entries_fails():
    problems = check(with_release("## [0.3.0] - 2026-09-23", body="### Added\n"))
    assert "The 0.3.0 section has no `- ` entries." in problems


def test_section_must_cite_the_title_key():
    problems = check(with_release("## [0.3.0] - 2026-09-23", body="### Added\n\n- A thing. (EV-5)\n"))
    assert any("must cite the PR's ticket, EV-11" in p for p in problems)


def test_key_is_matched_as_a_whole_word():
    # EV-1 must not be satisfied by EV-11 appearing in the section.
    problems = check(with_release("## [0.3.0] - 2026-09-23"), title="EV-1: Something")
    assert any("must cite the PR's ticket, EV-1" in p for p in problems)


# --- the PR title --------------------------------------------------------


@pytest.mark.parametrize("title", ["Add a thing", "ev-11: Add a thing", "EV-11 Add a thing", "EV-11:", "[EV-11] Add"])
def test_title_without_a_leading_key_fails(title):
    problems = check(with_release("## [0.3.0] - 2026-09-23"), title=title)
    assert any("must start with its Jira key" in p for p in problems)


def test_no_jira_label_waives_the_title_and_section_key():
    head = with_release("## [0.3.0] - 2026-09-23", body="### Changed\n\n- Bumped a dependency.\n")
    assert check(head, title="Bump requests", labels=["no-jira"]) == []


# --- a no-release PR -----------------------------------------------------


def test_no_release_pr_with_unchanged_headings_passes():
    head = BASE.replace("Something older.", "Something older, reworded.")
    assert check(head, title="EV-3: Tidy CI", labels=["no-release"]) == []


def test_no_release_pr_that_adds_a_heading_fails():
    problems = check(with_release("## [0.3.0] - 2026-09-23"), labels=["no-release"])
    assert len(problems) == 1
    assert "no-release label, but it adds or changes a version heading" in problems[0]


def test_no_release_pr_still_needs_a_title_key():
    problems = check(BASE, title="Tidy CI", labels=["no-release"])
    assert len(problems) == 1
    assert "must start with its Jira key" in problems[0]


# --- the Unreleased section and the bootstrap ----------------------------


def test_unreleased_left_in_the_pr_fails():
    head = with_release("## [0.3.0] - 2026-09-23").replace("## [0.3.0]", "## [Unreleased]\n\n## [0.3.0]")
    problems = check(head)
    assert any("'## [Unreleased]' is not a" in p and "CONTRIBUTING.md" in p for p in problems)


def test_bootstrap_base_with_unreleased_is_read_leniently():
    base = BASE.replace("## [0.2.0]", "## [Unreleased]\n\n### Added\n\n- Pending.\n\n## [0.2.0]")
    assert check(with_release("## [0.3.0] - 2026-09-23"), base=base) == []


def test_every_problem_is_reported_at_once():
    problems = check(BASE, title="No key here")
    assert len(problems) == 2


# --- malformed headings --------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "## 0.3.0 - 2026-09-23",
        "##[0.3.0] - 2026-09-23",
        "## v0.3.0 - 2026-09-23",
        "## Unreleased",
        " ## [0.3.0] - 2026-09-23",
        "### 0.3.0",
    ],
)
def test_heading_that_is_not_a_proper_version_heading_is_reported(line):
    # Before, only `## [` lines were looked at, so a no-release PR adding one of these passed.
    head = BASE.replace("## [0.2.0]", f"{line}\n\n## [0.2.0]")
    problems = check(head, labels=["no-release"])
    assert problems == [
        f"CHANGELOG.md line 5: {line!r} is not a `## [x.y.z] - YYYY-MM-DD` heading. Its `##` headings "
        "are version headings only, and there is no `## [Unreleased]` section: each release PR adds its "
        "own version heading (CONTRIBUTING.md, section 3)."
    ]


def test_lines_inside_a_fenced_code_block_are_not_headings():
    body = NEW_BODY + "\n```bash\n## step one\n# 1. run extract\n```\n"
    assert check(with_release("## [0.3.0] - 2026-09-23", body=body)) == []


def test_a_fenced_heading_example_does_not_end_the_section():
    # The key after the fence must still count, and the notes must include it.
    body = "### Added\n\n- Documents the heading format:\n\n  ```markdown\n  ## [x.y.z] - DATE\n  ```\n\n  (EV-11)\n"
    head = with_release("## [0.3.0] - 2026-09-23", body=body)
    assert check(head) == []
    assert rc.release_notes(head, "0.3.0").endswith("(EV-11)")


@pytest.mark.parametrize("block", ["````\n```\n````", "```\n~~~\n```"])
def test_a_fence_closes_only_on_the_same_character_and_at_least_its_length(block):
    # A 4-backtick block showing a 3-backtick line; a ~~~ line inside a ``` block.
    # Three fence-like lines each: naive toggling ends inside a fence and drops
    # every heading below.
    body = NEW_BODY + "\n" + block + "\n"
    assert check(with_release("## [0.3.0] - 2026-09-23", body=body)) == []


def test_list_lines_inside_a_fence_are_not_entries():
    base = BASE.replace("- Something older. (EV-1)\n", "- Something older. (EV-1)\n\n  ```yaml\n  - a\n  ```\n")
    head = base.replace("  - a\n", "  - a\n  - b\n").replace("```yaml\n  - a", "```yaml\n- a")
    assert check(head, title="EV-3: Tidy", labels=["no-release"], base=base) == []


def test_a_wrapped_line_starting_with_hash_number_is_not_a_heading():
    body = "### Fixed\n\n- Fixes the crash reported in pull request\n  #3 when a file is empty. (EV-11)\n"
    assert check(with_release("## [0.3.0] - 2026-09-23", body=body)) == []


def test_version_with_a_leading_zero_is_reported():
    # 0.03.0 would otherwise count as 0.3.0 and be tagged v0.3.0.
    problems = check(with_release("## [0.03.0] - 2026-09-23"))
    assert any("'## [0.03.0] - 2026-09-23' is not a" in p for p in problems)


# --- released entries ----------------------------------------------------


def test_no_release_pr_that_removes_a_released_entry_fails():
    head = BASE.replace("- Something older. (EV-1)\n", "")
    problems = check(head, title="EV-3: Tidy", labels=["no-release"])
    assert problems == [
        "The released 0.2.0 section has 0 `- ` entries, but 1 on the base branch. "
        "Released entries are never added or removed; rewording one is fine (CONTRIBUTING.md, section 3)."
    ]


def test_no_release_pr_that_adds_an_entry_to_a_released_section_fails():
    head = BASE.replace("- Something older. (EV-1)\n", "- Something older. (EV-1)\n- Added late. (EV-3)\n")
    problems = check(head, title="EV-3: Tidy", labels=["no-release"])
    assert any("0.2.0 section has 2 `- ` entries, but 1 on the base branch" in p for p in problems)


def test_release_pr_that_removes_an_older_entry_fails():
    head = with_release("## [0.3.0] - 2026-09-23").replace("- The first release.\n", "")
    problems = check(head)
    assert any("0.1.0 section has 0 `- ` entries, but 1 on the base branch" in p for p in problems)


# --- notes for the Publish release workflow ------------------------------


# 0.4.0 above 0.3.0 above the untagged 0.2.0 and 0.1.0.
TWO_RELEASES = with_release("## [0.3.0] - 2026-09-23").replace(
    "## [0.3.0]", "## [0.4.0] - 2026-10-01\n\n- Later. (EV-12)\n\n## [0.3.0]"
)


def test_tagged_versions_lists_0_3_0_and_later_oldest_first():
    assert rc.tagged_versions(TWO_RELEASES) == ["0.3.0", "0.4.0"]


def test_tagged_versions_is_empty_before_0_3_0():
    assert rc.tagged_versions(BASE) == []


def test_release_notes_returns_that_version_s_section():
    text = with_release("## [0.3.0] - 2026-09-23")
    assert rc.release_notes(text, "0.3.0") == "### Added\n\n- A new thing. (EV-11)"
    assert rc.release_notes(text, "0.2.0") == "### Added\n\n- Something older. (EV-1)"


def test_release_notes_for_a_missing_version_raises():
    with pytest.raises(ValueError):
        rc.release_notes(BASE, "0.3.0")


# --- the command line ----------------------------------------------------


def test_main_check_exits_0_when_clean_and_1_with_problems(tmp_path, monkeypatch, capsys):
    base = tmp_path / "base.md"
    head = tmp_path / "head.md"
    base.write_text(BASE)
    head.write_text(with_release("## [0.3.0] - 2026-09-23"))
    monkeypatch.setenv("PR_LABELS", json.dumps([]))

    monkeypatch.setenv("PR_TITLE", "EV-11: Add a thing")
    assert rc.main(["check", str(base), str(head)]) == 0
    assert "Release check passed." in capsys.readouterr().out

    monkeypatch.setenv("PR_TITLE", "Add a thing")
    assert rc.main(["check", str(base), str(head)]) == 1
    assert "must start with its Jira key" in capsys.readouterr().out


def test_main_check_treats_missing_labels_as_none(tmp_path, monkeypatch):
    base = tmp_path / "base.md"
    head = tmp_path / "head.md"
    base.write_text(BASE)
    head.write_text(with_release("## [0.3.0] - 2026-09-23"))
    monkeypatch.setenv("PR_TITLE", "EV-11: Add a thing")
    monkeypatch.delenv("PR_LABELS", raising=False)
    assert rc.main(["check", str(base), str(head)]) == 0


def test_main_versions_prints_one_version_per_line(tmp_path, capsys):
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(TWO_RELEASES)
    assert rc.main(["versions", str(changelog)]) == 0
    assert capsys.readouterr().out == "0.3.0\n0.4.0\n"


def test_main_notes_writes_that_version_s_section(tmp_path):
    changelog = tmp_path / "CHANGELOG.md"
    out = tmp_path / "notes.md"
    changelog.write_text(with_release("## [0.3.0] - 2026-09-23"))
    assert rc.main(["notes", str(changelog), "0.3.0", str(out)]) == 0
    assert out.read_text() == "### Added\n\n- A new thing. (EV-11)\n"


def test_main_with_bad_arguments_exits_2(capsys):
    assert rc.main(["publish"]) == 2
    assert "usage:" in capsys.readouterr().err
