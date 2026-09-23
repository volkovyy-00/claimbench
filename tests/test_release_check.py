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


def test_top_release_returns_the_version_and_its_section():
    version, section = rc.top_release(with_release("## [0.3.0] - 2026-09-23"))
    assert version == "0.3.0"
    assert section == "### Added\n\n- A new thing. (EV-11)"


def test_top_release_without_any_heading_raises():
    with pytest.raises(ValueError):
        rc.top_release("# Changelog\n\n## [Unreleased]\n")


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


def test_main_notes_prints_the_version_and_writes_the_section(tmp_path, capsys):
    changelog = tmp_path / "CHANGELOG.md"
    out = tmp_path / "notes.md"
    changelog.write_text(with_release("## [0.3.0] - 2026-09-23"))
    assert rc.main(["notes", str(changelog), str(out)]) == 0
    assert capsys.readouterr().out.strip() == "0.3.0"
    assert out.read_text() == "### Added\n\n- A new thing. (EV-11)\n"


def test_main_with_bad_arguments_exits_2(capsys):
    assert rc.main(["publish"]) == 2
    assert "usage:" in capsys.readouterr().err
