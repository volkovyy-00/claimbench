"""Checks a pull request against CONTRIBUTING.md's release rules, and reads release notes.

Two workflows call this file; it uses only the standard library.

- `.github/workflows/release-check.yml` (the required `Release` check) runs
  `python3 release_check.py check <base CHANGELOG.md> <head CHANGELOG.md>`
  with the PR title in the `PR_TITLE` environment variable and the PR's label
  names, as a JSON list, in `PR_LABELS`. It prints every problem it finds and
  exits 1 if there is any, 0 if there is none.
- `.github/workflows/release.yml` (runs on every push to `main`) runs
  `python3 release_check.py notes <CHANGELOG.md> <output file>`, which prints
  the top version (e.g. `0.3.0`) and writes that version's changelog section
  to the output file, for the tag message and the GitHub Release notes.

The rules (CONTRIBUTING.md, sections 3 and 4):

- The PR title starts with a Jira key, `EV-<n>: `, unless the PR carries the
  `no-jira` label.
- Every `## [` line in the PR's CHANGELOG.md is a version heading,
  `## [x.y.z] - YYYY-MM-DD` with a real calendar date. `## [Unreleased]` is
  gone for good. Only the PR's own file is held to this; the base branch's
  file is read leniently, because the first PR under these rules still had
  `## [Unreleased]` in its base.
- A PR without the `no-release` label is a release: it adds exactly one
  heading, at the top, and leaves every existing heading unchanged. The new
  version is exactly one step above the base's top version (patch, minor or
  major), its date is not before the base's top date, its section has at
  least one `- ` entry, and (unless `no-jira`) the section names the title's
  key. There is deliberately no upper bound on the date: the check runs in
  UTC, and a maintainer ahead of UTC writing after local midnight would
  otherwise be rejected.
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

_HEADING_RE = re.compile(r"^## \[(\d+)\.(\d+)\.(\d+)\] - (\d{4}-\d{2}-\d{2})$")
_TITLE_RE = re.compile(r"^(EV-\d+): \S")


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
        return "{}.{}.{}".format(*self.version)


def parse_headings(text: str) -> tuple[list[Heading], list[str]]:
    """Return every valid version heading in `text`, top first, plus a
    problem message for each `## [` line that is not one.

    A line that is not a version heading, or whose date is not a real
    calendar date, is reported and left out of the returned headings.
    """
    headings: list[Heading] = []
    problems: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.startswith("## ["):
            continue
        match = _HEADING_RE.match(line)
        if match is None:
            problems.append(
                f"CHANGELOG.md line {number}: {line!r} is not a "
                "`## [x.y.z] - YYYY-MM-DD` heading. There is no "
                "`## [Unreleased]` section: each release PR adds its own "
                "version heading (CONTRIBUTING.md, section 3)."
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


def section_text(text: str, heading: Heading) -> str:
    """The lines between `heading` and the next `## [` line (or the end of
    the file), with surrounding blank lines removed."""
    lines = text.splitlines()[heading.line :]
    body: list[str] = []
    for line in lines:
        if line.startswith("## ["):
            break
        body.append(line)
    return "\n".join(body).strip("\n")


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

    base_ids = [(h.version, h.date) for h in base]
    head_ids = [(h.version, h.date) for h in head]

    if NO_RELEASE in labels:
        if head_ids != base_ids:
            problems.append(
                "This PR has the no-release label, but it adds or changes a "
                "version heading in CHANGELOG.md. Remove the label if this is "
                "a release, or restore the headings."
            )
        return problems

    if len(head_ids) != len(base_ids) + 1 or head_ids[1:] != base_ids:
        problems.append(
            "A release PR adds exactly one new version heading at the top of "
            "CHANGELOG.md and leaves the existing ones unchanged. If users "
            "see no change, add the no-release label instead "
            "(CONTRIBUTING.md, section 3)."
        )
        return problems

    new = head[0]
    if base:
        top = base[0]
        allowed = next_versions(top.version)
        if new.version not in allowed:
            choices = ", ".join("{}.{}.{}".format(*v) for v in allowed)
            problems.append(
                f"Version {new.label} is not one step above {top.label}; "
                f"use one of {choices}."
            )
        if new.date < top.date:
            problems.append(
                f"Version {new.label} is dated {new.date}, before "
                f"{top.label} ({top.date})."
            )

    section = section_text(head_text, new)
    if not any(line.startswith("- ") for line in section.splitlines()):
        problems.append(f"The {new.label} section has no `- ` entries.")
    if key is not None and not re.search(rf"\b{re.escape(key)}\b", section):
        problems.append(
            f"The {new.label} section must cite the PR's ticket, {key}, "
            f"e.g. at the end of an entry: ({key})."
        )
    return problems


def top_release(text: str) -> tuple[str, str]:
    """The top version heading's version (e.g. '0.3.0') and its section.

    Raises ValueError if CHANGELOG.md has no version heading.
    """
    headings, _ = parse_headings(text)
    if not headings:
        raise ValueError("CHANGELOG.md has no `## [x.y.z] - YYYY-MM-DD` heading.")
    top = headings[0]
    return top.label, section_text(text, top)


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
    if len(args) == 3 and args[0] == "notes":
        version, section = top_release(_read(args[1]))
        with open(args[2], "w", encoding="utf-8") as handle:
            handle.write(section + "\n")
        print(version)
        return 0
    print(
        "usage: release_check.py check BASE_CHANGELOG HEAD_CHANGELOG\n"
        "       release_check.py notes CHANGELOG OUTPUT_FILE",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
