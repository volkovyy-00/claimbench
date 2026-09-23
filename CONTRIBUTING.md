# Contributing to ClaimBench

ClaimBench has one maintainer. This file is the process they follow, written
down so every working session, human or AI, follows the same one.
`CLAUDE.md` holds the rules of the codebase; this file holds how a change is
made and shipped, and which file owns which kind of project knowledge.

## 1. Start from a Jira ticket

Every change starts from a ticket in the Jira project `EV`; create one if
none fits. Name the branch after it: `ev-<n>-<short-slug>`, for example
`ev-12-trim-claude-md`.

## 2. Make the change

- Run what CI runs before you push: `pytest`, `ruff check .` and
  `basedpyright` (see `CLAUDE.md`, "Commands").
- This repository is public. Committed files use fictional names only
  (Acme, Borealis, Gantry, …): never a real company, person, memo id or
  figure from a client memo.
- Update the file that owns each kind of knowledge you touched (section 6).

## 3. Version and changelog

First decide: will a user of the pipeline notice this change? A command,
flag, output column, file format, config key, or pipeline behaviour.

**Yes: the PR is a release.** Add a new heading at the top of
`CHANGELOG.md`, directly under the introduction:

```markdown
## [0.4.0] - 2026-10-02

### Added

- `python eval_pipeline.py report` accepts `--format=csv`. (EV-42)
```

- **Version:** one step above the current top heading. Before 1.0:
  - **minor** (0.3.0 → 0.4.0): a new command, flag or output column, or
    anything that breaks an existing command, file format or config;
  - **patch** (0.3.0 → 0.3.1): a fix or a behaviour correction;
  - **major** (→ 1.0.0): only when the CLI is declared stable.
- **Date:** the day you write the entry, `YYYY-MM-DD`. Refresh it if the
  merge slips by days.
- **Sections:** Keep a Changelog's `### Added`, `### Changed`,
  `### Deprecated`, `### Removed`, `### Fixed`, `### Security`. End each
  entry with its ticket key, e.g. `(EV-12)`.
- Released entries are never added or removed; the Release check counts
  each released section's entries. Fixing a reference inside one is fine.

**No: the PR is not a release.** CI, tests, dev-only dependencies, and
internal docs such as this file or `CLAUDE.md` don't need a version. Add the
`no-release` label to the PR and leave the version headings alone.

There is no `## [Unreleased]` section: every release PR carries its own
version.

## 4. Open and merge the pull request

- **Title:** `EV-<n>: <what changes>`, for example
  `EV-12: Trim CLAUDE.md under 200 lines`. The key comes first, so the
  Jira–GitHub integration can link the PR and the key survives in `main`'s
  history. A PR with no ticket (rare, e.g. an automated dependency bump)
  gets the `no-jira` label instead.
- **Checks:** five must pass: Ruff, Pyright, Pytest, SonarCloud Scan and
  **Release**. Release enforces the title and section 3. It re-runs when you
  edit the title or the labels. It runs the PR's own copy of
  `.github/scripts/release_check.py`, so a PR that edits that script is
  judged by the edited version. Review such a change with that in mind.
- **Merge:** `main` accepts squash merges only, titled from the PR title.
  Merge with the noreply author address, so no personal email enters the
  public history:

  ```bash
  gh pr merge <number> --squash --author-email 220916739+volkovyy-00@users.noreply.github.com
  ```

- **After the merge,** the `Publish release` workflow tags `v<version>` and
  publishes a GitHub Release from that changelog section. Nothing to do by
  hand. If it fails, re-run it: it only creates what is missing.
- **Two release PRs open at once** both claim the next version. After the
  first merges, the second conflicts on `CHANGELOG.md`. Update its branch
  from `main` and move its entry to the next version up.
- **Repository settings this relies on:** labels `no-release` and
  `no-jira`; squash merging only, with the PR title as the commit title;
  branch protection on `main` requiring the five checks (bound to the
  GitHub Actions app) and an up-to-date branch.

## 5. Before you merge: close out

A spec or a plan is working material, not the record. Before merging, move
anything a future session needs into the file that owns it (section 6):

- a new or changed rule → `CLAUDE.md`;
- why it was decided that way, and any residual it still has, stated as a
  fact → `docs/design-decisions/NN-*.md`;
- the work to fix a residual, or any other follow-up → a new `EV` ticket,
  cited by its key (never "ticket N (open)" prose, which goes stale);
- live-run measurements → `docs/prompt-verification-log.md`.

Then move the ticket to Done.

## 6. Where knowledge lives

Each kind of knowledge has one owner. When two places disagree, the owner
wins, and the other place is fixed.

| Question | Owner |
|---|---|
| What work is open or next, including the fix for a known residual? | Jira project `EV` |
| What is the rule; where does the code live? | `CLAUDE.md`: rules and map, no dated status ("currently …", "(open)") |
| Why was it decided that way, and what does it still get wrong? | `docs/design-decisions/NN-*.md`, citing the `EV` key of any fix |
| What does this function do? | its docstring |
| What did live runs measure? | `docs/prompt-verification-log.md` |
| What changed for users, in which version? | `CHANGELOG.md`, each entry citing its `EV` key |
| How is a change made and shipped? | this file |
| Plain-English walkthrough of the pipeline | `docs/pipeline-overview.md` (defers to `CLAUDE.md`) |
| How do I install and use it? | `README.md` |
| What does a valid input file look like? | the committed templates: `memos.yaml.example`, `claims.example.md`, `retrieval.example.yaml`, `.env.example` |
| Design specs, plans, acceptance records, real names | the maintainer's private notes repository (section 7) |
| An AI assistant's per-machine memory | personal working preferences only, never project status |

## 7. The maintainer's private notes

Design specs, implementation plans, acceptance-test records, the archive of
the pre-Jira ticket backlog, and `CLAUDE.local.md` (the real names behind
the fictional ones) live in a private repository. It is cloned into this
repository's gitignored `docs/superpowers/` folder:

```bash
git clone https://github.com/volkovyy-00/claimbench-notes docs/superpowers
ln -s docs/superpowers/CLAUDE.local.md CLAUDE.local.md
```

Claude Code reads `CLAUDE.local.md` through the symlink, and its edits land
in the notes repository. Commit and push them there.

## 8. References from before 0.3.0

Before 0.3.0 the backlog was a set of local files, cited as "ticket NNN"
or "local ticket NNN". Finished ones are archived in the notes
repository's `tickets/`
folder. The six still open moved to Jira on 2026-09-23:

| Local | Jira |
|---|---|
| 006 | EV-5 |
| 010 | EV-6 |
| 011 | EV-7 |
| 013 | EV-8 |
| 014 | EV-9 |
| 015 | EV-10 |
