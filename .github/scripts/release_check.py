"""Checks a pull request against CONTRIBUTING.md's release rules, and reads release notes.

Two workflows call this file; it uses only the standard library.

- `.github/workflows/release-check.yml` (the required `Release` check) runs
  `python3 release_check.py check <base CHANGELOG.md> <head CHANGELOG.md>`
  with the PR title in the `PR_TITLE` environment variable and the PR's label
  names, as a JSON list, in `PR_LABELS`. It prints every problem it finds and
  exits 1 if there is any, 0 if there is none.
- `.github/workflows/release.yml` (runs on every push to `main`) runs
  `python3 release_check.py versions <CHANGELOG.md>`, which prints every
  version from 0.3.0 on (the first tagged one), oldest first, one per line.
  For each, `python3 release_check.py notes <CHANGELOG.md> <version> <output
  file>` writes that version's changelog section to the output file, for the
  tag message and the GitHub Release notes.

The rules (CONTRIBUTING.md, sections 3 and 4):

- The PR title starts with a Jira key, `EV-<n>: `, unless the PR carries the
  `no-jira` label.
- Every `##` heading in the PR's CHANGELOG.md (indented up to 3 spaces, as
  Markdown allows), and every `#` heading that starts with a version number
  (e.g. `### 0.4.0`), is exactly `## [x.y.z] - YYYY-MM-DD`: no leading zeros
  in x, y or z, and a real calendar date. Lines inside fenced code blocks
  are not headings; a fence closes only on the same character (``` or ~~~),
  at least as long as the one that opened it. `## [Unreleased]` is gone for good.
  Only the PR's own file is held to this; the base branch's file is read
  leniently, because the first PR under these rules still had
  `## [Unreleased]` in its base.
- A PR without the `no-release` label is a release: it adds exactly one
  heading, at the top, and leaves every existing heading unchanged. The new
  version is exactly one step above the base's top version (patch, minor or
  major), its date is not before the base's top date, its section has at
  least one `- ` entry, and (unless `no-jira`) the section names the title's
  key. There is deliberately no upper bound on the date: the check runs in
  UTC, and a maintainer ahead of UTC writing after local midnight would
  otherwise be rejected.
- Every section the base branch has already released keeps its number of
  `- ` entries: released entries are never added or removed. Rewording one,
  e.g. to fix a reference, is fine. Only the count is compared, so swapping
  one entry for another is not caught; review is the net for that.
- A malformed heading is reported on its own: while any heading line is
  wrong, the other CHANGELOG.md rules are not checked, because the
  malformed line may be the intended new version and comparing without it
  would only add misleading messages.
- A PR with the `no-release` label leaves every version heading (version and
  date) exactly as the base has it. Text inside released sections may still
  change, e.g. a typo or reference fix.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
from dataclasses import dataclass

NO_RELEASE = "no-release"
NO_JIRA = "no-jira"

_HEADING_RE = re.compile(
    r"^## \[(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)\] - (\d{4}-\d{2}-\d{2})$"
)
# A line that must be a version heading: any `##` heading (not `###`), or any
# `#` heading whose text starts with a version number, indented up to 3
# spaces as Markdown allows (`## Unreleased`, ` ## [0.4.0]`, `##[0.4.0]`).
# Markdown needs a space after the `#`s, so a wrapped `  #3 ...` is not one.
_HEADING_LIKE_RE = re.compile(r"^ {0,3}(##(?!#)|#{1,6}[ \t]+\[?v?\d)")
# A fence line of a fenced code block: the fence, then the rest of the line.
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_TITLE_RE = re.compile(r"^(EV-\d+): \S")

# The first version with a tag: 0.1.0 and 0.2.0 were released before this
# repository was published and stay untagged.
FIRST_TAGGED = (0, 3, 0)


def _format_version(version: tuple[int, int, int]) -> str:
    """`(0, 3, 0)` as `0.3.0`, the spelling used in headings and tag names."""
    return "{}.{}.{}".format(*version)


def _fenced(lines: list[str]) -> list[bool]:
    """For each line, whether it is part of a fenced code block (its fence
    lines included).

    As in CommonMark, a fence closes only on the same character, at least as
    long as the opening fence, with nothing but spaces after it.
    """
    flags: list[bool] = []
    opener = ""
    for line in lines:
        match = _FENCE_RE.match(line)
        if not opener:
            if match:
                opener = match[1]
            flags.append(bool(match))
            continue
        flags.append(True)
        if (
            match
            and match[1][0] == opener[0]
            and len(match[1]) >= len(opener)
            and not match[2].strip()
        ):
            opener = ""
    return flags


@dataclass(frozen=True)
class Heading:
    """One `## [x.y.z] - YYYY-MM-DD` line of CHANGELOG.md.

    `line` is 1-based, for error messages and for finding the section.
    """

    version: tuple[int, int, int]
    date: str
    line: int

    @property
    def label(self) -> str:
        return _format_version(self.version)


def parse_headings(text: str) -> tuple[list[Heading], list[str]]:
    """Return every valid version heading in `text`, top first, plus a
    problem message for each heading-like line (`_HEADING_LIKE_RE`) that is
    not one. Lines inside fenced code blocks are skipped.

    A line that is not a version heading, or whose date is not a real
    calendar date, is reported and left out of the returned headings.
    """
    headings: list[Heading] = []
    problems: list[str] = []
    lines = text.splitlines()
    for number, (line, fenced) in enumerate(zip(lines, _fenced(lines)), start=1):
        if fenced or not _HEADING_LIKE_RE.match(line):
            continue
        match = _HEADING_RE.match(line)
        if match is None:
            problems.append(
                f"CHANGELOG.md line {number}: {line!r} is not a "
                "`## [x.y.z] - YYYY-MM-DD` heading. Its `##` headings are "
                "version headings only, and there is no `## [Unreleased]` "
                "section: each release PR adds its own version heading "
                "(CONTRIBUTING.md, section 3)."
            )
            continue
        try:
            dt.date.fromisoformat(match[4])
        except ValueError:
            problems.append(
                f"CHANGELOG.md line {number}: {match[4]} is not a real date."
            )
            continue
        version = (int(match[1]), int(match[2]), int(match[3]))
        headings.append(Heading(version, match[4], number))
    return headings, problems


def next_versions(version: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    """The three versions one step above `version`: patch, minor, major."""
    major, minor, patch = version
    return [(major, minor, patch + 1), (major, minor + 1, 0), (major + 1, 0, 0)]


def _section_lines(text: str, heading: Heading) -> list[tuple[str, bool]]:
    """Each line between `heading` and the next heading-like line outside a
    fenced code block (or the end of the file), with whether it is fenced."""
    lines = text.splitlines()
    fenced = _fenced(lines)
    body: list[tuple[str, bool]] = []
    for line, in_fence in zip(lines[heading.line :], fenced[heading.line :]):
        if not in_fence and _HEADING_LIKE_RE.match(line):
            break
        body.append((line, in_fence))
    return body


def section_text(text: str, heading: Heading) -> str:
    """The section under `heading`, with surrounding blank lines removed."""
    return "\n".join(line for line, _ in _section_lines(text, heading)).strip("\n")


def entry_count(text: str, heading: Heading) -> int:
    """How many `- ` entries the section under `heading` has, not counting
    lines inside fenced code blocks."""
    return sum(
        line.startswith("- ") and not fenced for line, fenced in _section_lines(text, heading)
    )


def released_entry_changes(
    base_text: str, base: list[Heading], head_text: str, head: list[Heading]
) -> list[str]:
    """A problem for each released section whose number of `- ` entries
    differs between the base and the PR.

    `base` and `head` are the same released headings, in the same order,
    as found in `base_text` and `head_text`.
    """
    problems: list[str] = []
    for old, new in zip(base, head):
        before = entry_count(base_text, old)
        after = entry_count(head_text, new)
        if before != after:
            problems.append(
                f"The released {old.label} section has {after} `- ` entries, "
                f"but {before} on the base branch. Released entries are never "
                "added or removed; rewording one is fine (CONTRIBUTING.md, "
                "section 3)."
            )
    return problems


def check_pr(title: str, labels: list[str], base_text: str, head_text: str) -> list[str]:
    """Every way this PR breaks the release rules; an empty list means none.

    `base_text` is the base branch's CHANGELOG.md, `head_text` the PR's.
    All problems are collected, so one run reports everything to fix.
    """
    problems: list[str] = []

    key = None
    if NO_JIRA not in labels:
        title_match = _TITLE_RE.match(title)
        if title_match is None:
            problems.append(
                f"PR title {title!r} must start with its Jira key, e.g. "
                "'EV-12: Trim CLAUDE.md' (or add the no-jira label)."
            )
        else:
            key = title_match[1]

    base, _ = parse_headings(base_text)
    head, head_problems = parse_headings(head_text)
    problems.extend(head_problems)
    if head_problems:
        # The malformed line may be the intended new heading; comparing the
        # headings without it would only add misleading messages.
        return problems

    base_ids = [(h.version, h.date) for h in base]
    head_ids = [(h.version, h.date) for h in head]

    if NO_RELEASE in labels:
        if head_ids != base_ids:
            problems.append(
                "This PR has the no-release label, but it adds or changes a "
                "version heading in CHANGELOG.md. Remove the label if this is "
                "a release, or restore the headings."
            )
        else:
            problems.extend(released_entry_changes(base_text, base, head_text, head))
        return problems

    if len(head_ids) != len(base_ids) + 1 or head_ids[1:] != base_ids:
        problems.append(
            "A release PR adds exactly one new version heading at the top of "
            "CHANGELOG.md and leaves the existing ones unchanged. If users "
            "see no change, add the no-release label instead "
            "(CONTRIBUTING.md, section 3)."
        )
        return problems

    problems.extend(released_entry_changes(base_text, base, head_text, head[1:]))
    new = head[0]
    if base:
        top = base[0]
        allowed = next_versions(top.version)
        if new.version not in allowed:
            choices = ", ".join(_format_version(v) for v in allowed)
            problems.append(
                f"Version {new.label} is not one step above {top.label}; "
                f"use one of {choices}."
            )
        if new.date < top.date:
            problems.append(
                f"Version {new.label} is dated {new.date}, before "
                f"{top.label} ({top.date})."
            )

    if entry_count(head_text, new) == 0:
        problems.append(f"The {new.label} section has no `- ` entries.")
    if key is not None and not re.search(rf"\b{re.escape(key)}\b", section_text(head_text, new)):
        problems.append(
            f"The {new.label} section must cite the PR's ticket, {key}, "
            f"e.g. at the end of an entry: ({key})."
        )
    return problems


def tagged_versions(text: str) -> list[str]:
    """Every version from `FIRST_TAGGED` on (e.g. '0.3.0'), oldest first:
    the versions the `Publish release` workflow makes sure are tagged."""
    headings, _ = parse_headings(text)
    return [h.label for h in reversed(headings) if h.version >= FIRST_TAGGED]


def release_notes(text: str, version: str) -> str:
    """The changelog section of `version` (e.g. '0.3.0').

    Raises ValueError if CHANGELOG.md has no heading for it.
    """
    headings, _ = parse_headings(text)
    for heading in headings:
        if heading.label == version:
            return section_text(text, heading)
    raise ValueError(f"CHANGELOG.md has no heading for version {version}.")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; see the module docstring for usage."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) == 3 and args[0] == "check":
        title = os.environ.get("PR_TITLE", "")
        labels = json.loads(os.environ.get("PR_LABELS") or "[]")
        problems = check_pr(title, labels, _read(args[1]), _read(args[2]))
        for problem in problems:
            print(f"- {problem}")
        if problems:
            print(f"Release check failed: {len(problems)} problem(s). See CONTRIBUTING.md.")
            return 1
        print("Release check passed.")
        return 0
    if len(args) == 2 and args[0] == "versions":
        for version in tagged_versions(_read(args[1])):
            print(version)
        return 0
    if len(args) == 4 and args[0] == "notes":
        section = release_notes(_read(args[1]), args[2])
        with open(args[3], "w", encoding="utf-8") as handle:
            handle.write(section + "\n")
        return 0
    print(
        "usage: release_check.py check BASE_CHANGELOG HEAD_CHANGELOG\n"
        "       release_check.py versions CHANGELOG\n"
        "       release_check.py notes CHANGELOG VERSION OUTPUT_FILE",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
