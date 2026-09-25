# Design decision 9 — full narrative

Split out of `CLAUDE.md` to keep that file's per-session load down; the
condensed rule and current residual now live in `.claude/rules/golden-set-pipeline.md` under design decision 9.
This file holds the full text of that entry as it stood before the split.

---

**`propose_evidence_from_chunks`'s prompt requires same-measure
support, not keyword overlap — and treats "says less than the claim" as
a separate, allowed case.** A chunk counts as evidence only if it is
about the same subject and the same quantity measured — sharing an
entity name and a similar-looking number is not enough (observed live:
"The balance in Austria (5%)." matched to an unrelated net-initial-yield
table row). The prompt names **exactly three** disqualifiers: different
measure/period/population; you can't tell what the figure refers to; it
contradicts the claim.

Keeping that list closed is load-bearing. The first version instead
demanded "the same measurement basis (units, period, **scope**)" *and*
separately allowed less-detailed passages — two rules that contradict
each other precisely when the claim carries a narrowing qualifier
("**full-time** employees", "**private** hospitals"), because omitting
it is both "less detail" and "different scope". The model resolved that
at random: the same partial match against the same single candidate
came back `low`, `low`, then `[]` on three identical runs. **Not**
batch contamination — a contradicting candidate in the same batch was
tested directly and didn't drive it (accepted 3/3 with the
contradiction present, rejected 3/3 in another batch without it).
Don't reintroduce a second, overlapping rule here.

**Product decision the user made explicitly** (asked as a
precision/recall question, answered "mark as found, low confidence"):
a passage that states the claim's fact but leaves one qualifier unsaid
is returned at `confidence="low"`, not dropped — a near-miss for
review beats real evidence going silently unlinked. Consequence: such
a row is `found=True`, so `low` is the reviewer's signal to check it.
`low` is reserved for this case; the prompt says explicitly it must
not pass along a different-measure lookalike (without that sentence,
the Austria lookalike leaked back in at `low` on 1 run in 2).

`confidence` is a review-priority hint, not a stable score — the same
candidate can come back `high` one run, `medium`/`low` the next.
**Residual:** a legitimate partial match is still dropped on roughly 1
run in 5 (measured across five out-of-sample batches). Re-check BOTH
directions if you touch this prompt, with a partial match, a
contradiction and a lookalike distractor in one batch — single-candidate
tests hide these interactions.
