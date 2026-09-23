# Design decision 7 — full narrative

Split out of `CLAUDE.md` to keep that file's per-session load down; the
condensed rule and current residual live there under design decision 7.
This file holds the full text of that entry as it stood before the split.

---

**`bm25_threshold_shortlist` has no upper bound on candidate count by
design** — on a real multi-hundred-chunk corpus with generic claim
text (dates, dollar figures), it can legitimately select hundreds of
chunks for one claim (observed: up to 420 on the Paxton shakedown corpus),
producing 10+ LLM batches per claim. This is intentional (a fixed cap
silently drops real evidence — the whole reason this replaced
`bm25_shortlist`), but it means `relative_threshold`/`min_candidates`
are real cost/latency tuning knobs on a large real corpus, not just
theoretical parameters. Both are passed through
`build_golden_set_draft`, not hardcoded — tune per corpus, and (with
`batch_size`) per memo within a batch run, see design decision 12.

**BM25 is lexical — some evidence is unreachable at any threshold.**
`bm25_threshold_shortlist` scores exact token overlap only, so a claim
that states a fact in different words than the source PDF ("physical
occupancy" vs. "occupancy", "headcount" vs. "employees") can leave the
correct chunk scoring too low to ever clear the cutoff — lowering
`relative_threshold` widens the net but cannot recover a zero-overlap
miss. EV-5 tracks a keyword-expansion mitigation.
