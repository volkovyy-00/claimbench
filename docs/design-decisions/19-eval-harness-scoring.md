# Design decision 19 — full narrative

Split out of `CLAUDE.md` to keep that file's per-session load down; the
condensed rule and current residuals live in `.claude/rules/eval-pipeline.md`
under design decision 19. Design:
`docs/superpowers/specs/2026-09-10-retrieval-eval-harness-design.md`
(gitignored, refreshed 2026-09-22).

## What counts as ground truth, and why

The eval scores claims against human-tagged evidence, and refuses rather
than scoring around a problem. The census comes from the claims file, the
evidence from `reviewed/`; the relevant set is `found` **and** a tag of
`extractive`/`synthesized` — `found` is the pipeline's guess, the tag the
human verdict. Defining the relevant set by `found` alone inflated
coverage and deflated recall on the first reviewed file. Unverifiable
claims leave the coverage denominator and are reported beside a re-review
worklist (`claim_queries.parquet`, chunks meaning search offers that no
human judged), never as "absent from the sources": the golden set was
built with lexical BM25 (decision 7).

## What counts as a hit

A hit is the same `chunk_id` **or** decision 1's rule (`_same_evidence`,
golden quote vs retrieved text) — the rule never matches a chunk with
itself, so the identity test is required. An adjacent chunk must also
hold the whole golden quote inside the text the two chunks share
(`_shared_edge`) — the only place the same passage can sit in both — so a
short quote repeated elsewhere in the neighbour is not a hit. One
direction only: feeding the shared text to the two-way `_same_evidence`
would count a long quote that merely contains it (**measured:** +12 false
hits on the real set). A human-added row's quotes, joined with
`_QUOTE_SEPARATOR` by `tag_pipeline`, are tested one by one, and grouped
by those same quotes; only such a row (its `tag_rationale` starts
`human-added`) is split — any other span is chunk text and may contain
the separator itself. Evidence takes its section from the census (the
claim_id encodes it), never from the sheet cell.

## k, MRR and precision

Hits are computed once at depth 20 and every k read off by best rank.
Each hit records the golden row it matched, so the report quotes the
evidence actually found, and each group records its golden chunk ids
(EV-1), so the claims page shows every quote under its own group's rank
without grouping again — regrouping at report time could, after a change
to the grouping code, move a quote while every group size and hit row
stayed the same. k is per search phrase (18–43 distinct passages
per section at k=5 on the real set, 2026-09-23); runs record phrase
counts and the report warns when a baseline's differ. A run is refused
unless the results were retrieved with the phrases on disk. MRR (query =
claim) is stored, not shown — a best rank over several phrase rankings
rises with phrase count. Precision is citation precision, a floor, pooled
per memo. Fusion ties go to the dense rank; with the app's single-word
phrases, fusion scores below dense at k=5 (a chunk in both lists outranks
dense's own #1).

**Measured 2026-09-22:** at 1000/200 chunking the overlap rule added no
claim over exact matching (the "up to 2×" first measured at 500/100 does
not reproduce) — kept for correctness. Text is compared after removing
the control characters the reviewed sheet cannot hold (`_excel_safe`).

## Residuals

**Duplicated documents.** MEMO-001's folder holds two documents that
repeat the same passages; decision 1 keeps different documents as
separate evidence groups (correct). When both copies were cited, coverage
is unaffected but a claim found in only one copy can reach at most 50%
macro recall; when the reviewer cited only one copy, a retrieved chunk
from the other is never credited (a miss, and against precision) — 0 such
claims on the 2026-09-23 sheets. Deliberately not fixed. Since EV-19 the
claims page shows recall averaged per claim (read from the run's
`metrics.parquet`) and carries a plain-words note that alternative or
duplicate sources for the same fact each count, so recall understates how
often the fact itself was found — that note is the page's explanation of
this cap.

**Quotes not verbatim in their chunk.** 20 of 169 golden quotes
(2026-09-23) are not in their own chunk's text after `_normalize_span`
(split words, curly apostrophes, quotes running past the chunk; finalize
matches ignoring whitespace). Only their own chunk can hit them, never a
neighbour; `score` logs a WARNING with the count. Matching the finalize
way changed no coverage and one group.

**Overlapping quotes in adjacent chunks.** Two golden rows whose quotes
both span the shared overlap, neither containing the other, stay two
groups under decision 1 (3 pairs on the real set; coverage unchanged,
macro recall ±0.02). Not widened in the eval alone: the golden set and
the eval must group alike.
