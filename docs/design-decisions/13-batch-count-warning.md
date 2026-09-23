# Design decision 13 — full narrative

Split out of `CLAUDE.md` to keep that file's per-session load down; the
condensed rule and current residual live there under design decision 13.
This file holds the full text of that entry as it stood before the split.

---

**`propose_evidence_from_chunks_batched`'s pre-loop
candidate/batch-count line escalates from INFO to WARNING once a claim
splits into more than `_BATCH_COUNT_WARNING_THRESHOLD` (10) batches** —
the same single line, not a second one. The threshold is a module-level
constant next to the function, matching this file's convention for a
single-site documented literal (`_TOKEN_RE`, `_LEADER_RUN_RE`,
`_NO_CHUNK_SENTINEL`). Batch count, not candidate count, is the
trigger: decision 7 establishes that hundreds of candidates are routine
and a poor alarm, whereas batch count tracks a claim's actual LLM cost
and its exposure to the out-of-batch `chunk_id` failure (one chance per
batch, decision 8) — worth surfacing while a long run is in progress,
not only after the fact from a batch-failure warning.

**This deliberately overlaps with decision 7's own worst case — the
point, not an oversight.** At `batch_size=40`, decision 7's observed
maximum (420 candidates on the Paxton corpus) is 11 batches, already past
">10". That claim *is* the "extreme case" this decision exists to flag;
the WARNING firing on it is the threshold working. "420" is an observed
max for one claim on one corpus, not a typical batch count — but if a
future run shows the WARNING firing routinely, revisit the number 10,
not the batch-count-over-candidate-count choice.
