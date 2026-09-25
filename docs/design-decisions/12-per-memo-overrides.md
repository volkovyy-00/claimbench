# Design decision 12 — full narrative

Split out of `CLAUDE.md` to keep that file's per-session load down; the
condensed rule and current residual now live in `.claude/rules/golden-set-pipeline.md` under design decision 12.
This file holds the full text of that entry as it stood before the split.

---

**Per-memo overrides ride along as an optional 5th tuple element, and
the loader stores only the keys actually set.** `relative_threshold`/
`min_candidates`/`batch_size` are batch-wide defaults on
`build_golden_set_batch`, each overridable per memo, because observed
corpora range from ~1,000 to ~6-7,000 chunks per memo — too wide a
spread for one setting (design decision 7).

`memo_sections` entries are 4-tuples from a hand-written list, or
5-tuples from `load_memo_sections_from_config` with that memo's
override dict appended (shared across the memo's sections, like
`source_documents`). Chosen over a separate `memo_overrides={memo_id:
...}` parameter, which would have forced the loader to return a
2-tuple and broken every existing caller of a function whose whole
point is being easy to call.

`_validate_memo_sections_shape` checks every entry **before the first
section runs**, covering three silent-degrade shapes (all the shape
decision 8 fixed once): (a) a 6-tuple slipping down the "no overrides"
branch via a bare `entry[4] if len(entry) == 5 else {}`; (b) a
non-dict 5th element dying on a bare `AttributeError` naming no memo;
(c) worst — a misspelled key in a hand-built override dict
(`batch_sze`) dropped by `.get()` while `if overrides:` stays truthy,
so the run logs "per-memo overrides in effect" while using the
defaults. The YAML path is covered by `_MEMO_FIELDS`; this covers
hand-built 5-tuples, which the docstring supports equally.

A fourth was added by the final-verification audit: (d) slot 3 holding
raw section text instead of a claims list. `build_golden_set_draft`
raises `TypeError` on that too (design decision 17), but only once
that entry is reached — so a hand-built or concatenated list whose
*later* entry is stale would pay for every section ahead of it first,
which is the exact cost this pre-pass exists to avoid.

It's a **pre-pass, not a per-entry check in the run loop**, same
reason the loader validates before reading PDFs: pure structural
checks with no I/O, and a bad entry at position 40 would otherwise
throw away 39 sections of API spend the checkpoint may not have saved
(measured: `checkpoint_every=3` + a malformed 5th entry lost section
4's completed work).

Other deliberate choices:
- Error messages sort unrecognized keys with `key=repr` — YAML 1.1
  resolves a bare `on:` to `True`, and sorting that beside a string
  raises `TypeError` naming no memo.
- The override dict holds **only keys present in the YAML** — no
  filled-in defaults, so each default lives in exactly one place (the
  signatures) and "omitted → global default" falls out of
  `.get(key, default)`.
- Keys are **flat on the memo mapping, not under a `tuning:` block** —
  decision 7 treats these as three independent knobs, so a namespace
  would be config surface nothing has needed.
- **Unrecognized keys raise** (`_MEMO_FIELDS`) — a **product decision
  the user made explicitly**: a typo'd `batchsize: 25` is a malformed
  override silently yielding the default. No `notes:` escape hatch;
  YAML `#` comments cover per-memo annotation. (A *repeated* key is a
  separate gap — `yaml.safe_load` keeps the last value
  with no error, in both `memos.yaml` and a claims file's
  frontmatter; EV-8 tracks the fix.)
- **Value ranges validated at config-load**, not left to the runtime
  checks in `bm25_threshold_shortlist`/`propose_evidence_from_chunks_batched`
  (which fire 30+ min and real spend into a batch; `relative_threshold`
  has no runtime check at all). Those runtime checks are unchanged for
  direct/manual calls. Two YAML traps handled: `bool` is an `int`
  subclass, so `min_candidates: yes` slips past a plain `isinstance`;
  a valueless key (`batch_size:`) parses as `None` — treated as
  malformed, not absent.
