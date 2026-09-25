# Design decision 11 — full narrative

Split out of `CLAUDE.md` to keep that file's per-session load down; the
condensed rule now lives in `.claude/rules/golden-set-pipeline.md` under design decision 11. This file holds the
full text of that entry as it stood before the split.

---

**Dropped/recovered evidence counts are logged only, not added to the
returned DataFrame** — a per-claim and per-section running total makes
the impact of drops/recoveries visible without grepping logs, but the
schema (see `CLAUDE.md` → "DataFrame schema") stays claim/match-shaped,
not summary-shaped; a caller wanting the totals reads the log, not a new
column. "Dropped" is deliberately undifferentiated: a malformed entry, an
out-of-batch chunk_id that recovery couldn't resolve, and a missing
evidence_span all count as one dropped entry — the existing per-entry
WARNING line for each case (which does distinguish them) is unchanged, so
the detail isn't lost, just not re-summarized three ways.

Threaded via an optional `counts` accumulator dict on
`propose_evidence_from_chunks` (incremented in place only if passed in)
rather than changing its return type, since it's kept available for a
standalone manual pass and a manual caller shouldn't have to unpack a
tuple for a match list. Its sole caller,
`propose_evidence_from_chunks_batched`, has no such concern, so it
aggregates the per-batch counts and returns them as `(matches, counts)`;
`build_golden_set_draft` sums each claim's counts into a section-wide
total. A claim whose lookup raises entirely (all batches failed)
contributes no counts — this falls out of the per-claim try/except, and
`propose_evidence_from_chunks_batched` raises *before* logging its
per-claim summary, so a fully-failed claim never logs a misleading "0
dropped, 0 recovered."

The per-claim log line also reports how many of the claim's batches
succeeded (e.g. "1 dropped, 0 recovered (2/3 batch(es) succeeded)") — a
**product decision the user made explicitly**: when some but not all
batches fail, "0 dropped, 0 recovered" alone reads as "nothing lost" when
a whole batch's candidates were never evaluated, against the project's
opening promise (no silently dropped evidence). The section-wide line
stays exactly dropped/recovered totals with no ratio — a
partially/fully-failed claim there is already visible via the per-claim
ERROR log and the error row.
