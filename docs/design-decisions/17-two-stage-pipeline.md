# Design decision 17 — full narrative

Split out of `CLAUDE.md` to keep that file's per-session load down; the
condensed rule and current residual live there under design decision 17.
This file holds the full spec text and every place the shipped
implementation went further than that spec.

---

**Claim extraction and evidence matching are two stages joined only by a
reviewable claims file.** `build_golden_set_draft` used to call
`extract_atomic_claims` and loop the result straight into evidence matching. Now
`python golden_set_pipeline.py extract` reads `memos.yaml` and writes one Markdown
file per memo under `claims/` (gitignored); a human reviews, edits, or hand-authors
those; `python golden_set_pipeline.py build` (or no argument) reads only
`claims/*.md` and produces the golden set. An unrecognised argument errors. The
stages share nothing but the file and the source PDFs — a claims file written
entirely by hand, never having run `extract`, is a first-class input.

**Why.** Claim decomposition is open-ended work where model strength (or a human
editor) matters; evidence matching is a constrained "does this chunk support this
claim" judgement the weak self-hosted model handles well. Decoupling lets claim
decomposition be done by a stronger external model or by hand, and the claims
reviewed as a durable artifact, before any API spend on evidence matching — rather
than only by `extract_atomic_claims` against the self-hosted model, inline, with no
review point.

**The claims file is the sole seam.** `claims/<memo_id>.md`: `---`-fenced YAML
frontmatter (`memo_id`, `source_folder`, and only the tuning-override keys actually
set — design decision 12); `## ` sections; `N.` one-line claims (the number is
cosmetic). One strict hand-rolled parser (`parse_claims_file`) reads it, and every
ambiguous shape is an error naming the file and line — a silently mis-parsed claims
file poisons every downstream retrieval and hallucination metric, the same reasoning
as this file's opening promise. Stray prose and complete single-line `<!-- -->`
comments are ignored notes; a note line touching a claim, a multi-line comment
marker, and a heading that isn't exactly `## ` all error; a `-` bullet beside `N.`
claims warns. The frontmatter is delimited by the parser but read by `yaml.safe_load`
+ the existing `_parse_memo_overrides`.

**`extract` is complete-or-absent.** A memo's file is written only if its
`source_folder` exists (checked before the section loop, so a typo costs no API
spend) and every section extracted to at least one claim; either failure aborts
that file (not the run) and is reported. `extract` never overwrites an existing
file — it reports it as skipped, and warns if `memos.yaml` has since gained sections
the file lacks (best-effort `^## ` scan, `_scan_claims_file_sections`). No
`--force`; delete the file to re-extract. `extract_atomic_claims`'s own stranded-pronoun and
identical-text WARNINGs (design decision 5 and its ticket-007 addendum) now fire here, before the file is
written and reviewed, instead of mid-`build_golden_set_draft`.

**`claim_id` is now deterministic:** `uuid5` of a fixed namespace constant +
`memo_id` + section + canonical claim text + an occurrence index (count of earlier
byte-identical claims in the section). It replaces `str(uuid.uuid4())`. Consequences:
`import_reviewed` still only re-imports a single build's own output (an edited claim
gets a new id, so editing the claims file means re-reviewing from a fresh `build`);
deleting the first of two identical claims or renaming a heading rotates ids
(diff-only, still matched); a crashed `build` re-run is idempotent and its checkpoint
mergeable.

**`extract` and `preview_claim_splits` both exist on purpose.** `extract` produces
the durable, hand-editable artifact reviewed between the stages. `preview_claim_splits`
(design decision 14) stays the fast, opt-in, interactive throwaway that re-runs
extraction and prints the split — it writes nothing and is never wired into the real
path. Neither replaces the other.

**`load_memo_sections_from_config` keeps its name and behaviour** but is internally
split into `_read_memo_config` (pure YAML validation, used by `extract` so Stage 1
never reads a PDF) and `_load_source_documents` (the PDF-folder scan, shared with the
new `load_memo_sections_from_claims`).

**Implemented behaviour — where the running code goes further than the
paragraphs above** (the paragraphs above are spec text, pasted verbatim;
this paragraph is not — it documents where the actual implementation was
made stricter or more forgiving during build-out):

- `_CLAIM_MARKER_RE` is anchored at column 0 (`^\d+\.\s+`). An *indented*
  `N.` is never treated as a claim: if it directly abuts the claim above
  it (no blank line between them), that's a hard parse error telling the
  author to add a blank line; if a blank line separates it from the
  claim above, it's silently ignored except for a WARNING. This was a
  deliberate product decision by the repo owner, not an oversight — an
  indented number is genuinely ambiguous between a mis-indented claim, a
  nested sub-point, and a continuation of the claim above, and silently
  promoting it to a top-level claim would manufacture a claim with its
  own `claim_id`, BM25 shortlist and evidence LLM call.
- `write_claims_file` rejects strictly more than the paragraphs above
  describe: `\r` as well as `\n` in claim text and `source_folder`, an
  empty `sections` list, and any claim or section name that would itself
  render as a line the parser reads as an unterminated `<!-- -->`
  comment. The round-trip theorem (`write_claims_file` then
  `parse_claims_file` returns the input) holds for canonical
  (already-`.strip()`ed) input.
- `write_claims_file` writes atomically — to `path + ".tmp"`, then
  `os.replace(...)`. A failed or interrupted write leaves the target
  file absent, never truncated or half-written. **Known residual:** the
  `.tmp` name is fixed (not per-process), so two `run_extract`
  processes running concurrently against the same claims directory
  could still race on it. Single-process is this pipeline's documented
  usage model; this is a noted residual, not a bug being tracked for a
  fix.
- `run_extract` counts a memo as **skipped** (with a WARNING naming the
  file and reason) when an existing claims file cannot be read at all —
  undecodable bytes, a permissions error — rather than aborting the
  whole run. Same "one bad memo doesn't cost every other memo's work"
  principle as decision 8's pre-pass validation and decision 11's
  per-claim isolation, applied to Stage 1.
- `claim_id` is the deterministic `uuid5` described above at its one
  call site in `build_golden_set_draft`; `_error_row` still mints its
  own `uuid4` internally (unchanged legacy code), but that value is
  always overwritten with the derived `uuid5` before the row is
  returned, so it never actually reaches a caller.
- **Correction to the paragraph above:** "`load_memo_sections_from_config`
  keeps its name and behaviour" is true of its name and signature, but
  **not** its behaviour without qualification — `_read_memo_config`
  (the half of its old body that does the YAML validation) also
  enforces a memo `id` against `[A-Za-z0-9._-]+` and a section name
  against `_valid_section_name`, neither of which existed on `main`
  before this branch, and compares ids **case-insensitively** for
  the duplicate check. A `memos.yaml` that loaded cleanly before this
  change can now raise on any of the three. This is the same breaking
  change the CHANGELOG's `### Changed` entry documents for
  `memos.yaml` validation — the two must agree; if you edit one,
  check the other.

  The case-folded comparison is not fussiness: the id is also the
  claims-file name, and on a case-insensitive filesystem (APFS/HFS+
  — this repo's own platform — and NTFS) `Memo` and `memo` are one
  file. `_read_memo_config` therefore raises on such a pair before
  `run_extract` starts, so the collision cannot happen. Without it,
  `run_extract`'s never-overwrite guard (`os.path.exists`) would count
  the second memo **skipped**, not failed, exiting 0, and `build`
  would emit a golden set with that memo wholly absent — no error, no
  `found=False` rows, nothing. Found by the final-verification audit;
  the refusal is tested in `tests/test_two_stage_pipeline.py`.
- `_classify_claims_body`'s ignored-line handling: every non-blank
  line that is not blank, a comment, a heading, or a claim is now
  recorded and triggers a WARNING — not only lines that already look
  list-item-shaped, and **not only lines under a section**. A
  forgotten leading `N. ` on an ordinary sentence is otherwise
  indistinguishable from a deliberate note, and used to be dropped
  with zero log output. The trade this makes deliberately: a bare
  prose note now WARNs where it used to be silent. The silent-note
  mechanism is unaffected and unchanged — a complete single-line
  `<!-- ... -->` comment (handled earlier, by `_COMPLETE_COMMENT_RE`)
  still produces no warning at all.

  Lines **before the first `## ` heading** are tracked separately
  (`loose_preamble`, one WARNING naming no section) from lines under
  a section (`loose_under`, one WARNING per section). The preamble
  case was found by the final-verification audit *after* the
  under-a-section fix landed: the original `else` branch there
  carried a comment reasoning that the case was "unreachable for a
  would-be claim, which errors instead," which holds only for a line
  that still *has* its `N. ` marker — a claim whose number was
  forgotten reads as prose and fell straight through. This is why
  `_EXTRACT_MARKER` must stay a complete single-line comment: it sits
  in exactly that preamble region on every file `extract` writes, so
  anything less would WARN on every generated file.
- **Correction:** "every ambiguous shape is an error naming the file
  and line" (above) is true of body errors (`_classify_claims_body`),
  but not of every frontmatter error — `_parse_frontmatter` names only
  the file for most of its checks (missing `memo_id`, invalid YAML,
  an unrecognized key, and so on); only its unclosed-fence message
  quotes a line number, since that's the one check where a line
  number is knowable before the whole fenced block is even read.
