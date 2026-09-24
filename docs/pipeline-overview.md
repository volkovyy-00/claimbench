# Turning a written memo into a checked, sourced answer key

*A plain-English walkthrough of `golden_set_pipeline.py` — one file, read top to bottom.*

> **Note:** this is a snapshot overview, originally written as a Claude
> Artifact and saved here for offline/repo reference. It's a plain-English
> introduction, not the maintained technical reference — `CLAUDE.md` is
> kept current as the pipeline evolves and is the source of truth where
> the two disagree (e.g. it now also covers the `chunk_text`/`bm25_score`
> review columns and later prompt refinements not described below).

This whole project is **one Python file**. Its job: take a memo section a
person already wrote, break it into individual facts, and automatically go
find **exactly where in the source PDFs** each fact came from — then hand
you a spreadsheet to check its work.

> **In plain terms**
>
> Imagine handing a very fast, very literal intern a finished memo and a
> stack of source documents, and saying: *"For every single sentence in
> here, find me the exact page and quote that proves it — and tell me if
> you're not sure."* That's what this pipeline does, except the "intern"
> is an AI model, and everything it finds gets written into a spreadsheet
> for a human to double-check before anyone trusts it.

## The whole journey, at a glance

A memo section a person wrote goes in. Along the way it passes through one
file you get to read and edit by hand — the "claims file" — before the
source PDFs ever get searched. What comes out the other end is a
spreadsheet pairing every fact with its proof (or a flag that no proof was
found).

```
   Memo section
        │
        ▼
  STAGE 1: extract
  Split into single-fact claims
        │
        ▼
  claims/<memo_id>.md  ◄── or write this file yourself,
        │                  skipping Stage 1 entirely
        ▼
  You review / edit it
        │
        ▼                                Source PDFs
  STAGE 2: build                              │
  Read the claims file                        ▼
        │                              Cut into
        │                              overlapping snippets
        └────────────────┬───────────────────┘
                          ▼
               For every claim...
                          │
                          ▼
       Keyword search for candidate snippets
                          │
                          ▼
             AI checks each snippet
                          │
                          ▼
        How many snippets really matched?
                          │
        ┌─────────┬───────────────┬────────────────────┐
        ▼         ▼               ▼                     ▼
      none    one match    same passage twice    different sources
        │         │               │                     │
        ▼         ▼               ▼                     ▼
   Row: NOT   Row: FOUND    Row: FOUND, not      Rows: FOUND,
   FOUND                    ambiguous            AMBIGUOUS
        │         │               │                     │
        └─────────┴───────────────┴─────────────────────┘
                          ▼
                   Results table
                          │
                          ▼
                  Export to Excel
                          │
                          ▼
              You review & correct
                          │
                          ▼
             Corrections merged in
                          │
                          ▼
                     Golden set
```

A memo and its source PDFs go in; a reviewed, sourced answer key comes
out. A found match becomes one evidence row; ambiguous evidence (found in
genuinely different places, not just overlapping snippets echoing the
same passage) becomes multiple rows for the same claim, flagged for
review.

## Two stages, one file between them

The journey above is really two separate runs of the program, and the
only thing that connects them is a plain Markdown file you can open,
read, and edit — `claims/<memo_id>.md`.

- **`python golden_set_pipeline.py extract`** reads `memos.yaml`, splits
  each section's memo text into single-fact claims (step 02 below), and
  writes one `claims/<memo_id>.md` file per memo. It makes no attempt to
  match claims against the source PDFs — that costs nothing to run beyond
  the claim-splitting itself.
- **You review that file.** Read it like any text file, fix a bad split,
  delete a claim you don't want checked, add one the split missed, or
  leave it as-is.
- **`python golden_set_pipeline.py build`** (or running the program with
  no argument at all) reads every `.md` file directly in `claims/` (not a
  subdirectory, and not e.g. `.markdown`), never `memos.yaml` again, and
  runs the rest of the journey above (steps 03–07) against whatever
  claims are in those files.

Because the two stages only talk to each other through that file, you can
also skip Stage 1 completely and **write a claims file by hand** — useful
if you'd rather compose the claims yourself, or want to use a different
tool to split them. `build` can't tell the difference between a file
`extract` wrote and one a person typed.

### How to write a claims file

`claims.example.md` (repo root) is a small, valid example to copy from.
The shape:

```markdown
---
memo_id: MEMO-001
source_folder: sources/acme
relative_threshold: 0.45
---

## Business Profile

1. Acme is the largest listed widget maker in Europe.
2. Acme's plants were valued at EUR 20bn at 31 December 2024.

## Ownership

1. Acme is listed in London and in Frankfurt.
```

In plain sentences:

- **The top block, between the two `---` lines, is required setup.** It
  needs `memo_id` (must match the filename — `MEMO-001.md` must say
  `memo_id: MEMO-001`, and may only contain letters, digits, `.`, `_` and
  `-`) and `source_folder` (the folder of source PDFs to match claims
  against). You can also add any of the three tuning knobs described in
  `memos.yaml.example` (`relative_threshold`, `min_candidates`,
  `batch_size`) — only include the ones you actually want to change; any
  other key is an error, not a typo silently ignored. `filing_entity` (the
  company the memo is about, e.g. `filing_entity: Acme Ltd`) is optional
  for `extract` and `build`, but `tag_pipeline.py draft` refuses a memo
  without it — its prompt tells the model that "we" and "our" mean that
  company.
- **Each `## ` line starts a new section** — exactly two `#` characters,
  one space, then a non-blank name, matching a section name from
  `memos.yaml` if the file came from `extract`. Section names must be
  unique within the file, and nothing before the first `## ` line may be
  a claim.
- **Each claim is one line starting with a number, a period, and a
  space** — `1. `, `2. `, and so on. The number is just for reading
  along; it's cosmetic, so deleting claim 3 and leaving 1, 2 and 4 as-is
  is fine, no renumbering needed. A claim must be a single line — if it
  needs a second line, that's an error, not a silent continuation.
  Every section needs at least one claim.
- **An indented number is never a claim**, on purpose — it's genuinely
  ambiguous whether it's a mis-indented claim, a nested sub-point, or a
  continuation of the claim above, so the file format doesn't guess. If
  it directly follows a claim line with no blank line between them,
  that's an error asking you to add a blank line; separated by a blank
  line, it's just ignored (with a warning logged, in case it was meant to
  be a claim).
- **Blank lines are always ignored, silently.** A one-line
  `<!-- ... -->` comment is too — anywhere in the file, this is **the**
  way to leave yourself a silent note about a review decision (see the
  example above `## Ownership` in `claims.example.md`). A comment that spans more than
  one line is an error; so is any other stray line of text sitting
  directly against a claim line with no blank line between them (it
  looks like it might be meant as a continuation of that claim, which the
  format doesn't allow).
- **Bare prose — not a comment, not a claim — is still ignored, but logs
  a WARNING quoting the line**, instead of being silent. This is
  deliberate: a plain sentence that's missing its leading `1. ` looks
  exactly like a note, and there's no way to tell "an intentional
  comment" from "a claim whose number got forgotten" except by the
  `<!-- ... -->` wrapper. If you want a note that stays silent, wrap it
  in `<!-- ... -->`; if you see this warning and the line was meant to be
  a claim, add the missing number.

  This holds **wherever the line sits** — under a section (the warning
  names the section) or above the first `## ` heading (it names the
  file). Note that a claim only counts under a heading, so a numbered
  claim stranded above the first one is an error rather than a warning.
- **An `extract`-written file carries one extra marker line**,
  `<!-- written by 'extract'; edit freely, then run 'build' -->`, right
  after the closing `---`. It's just a comment — edit around it freely.

If anything about the file's shape is wrong, `build` refuses to run and
tells you which file — and, for most body errors (though not every
frontmatter error), which line too — a silently misread or silently
dropped claim would poison every retrieval and hallucination metric this
golden set is later used for, so nothing here fails quietly.

## Step by step, in plain English

Each entry below is one stage of the journey above, numbered in the order
they're explained here rather than the order the two-stage split runs them
in. Step 02 (splitting the memo into claims) is Stage 1, `extract`; it
touches only `memos.yaml`, never the source PDFs. Steps 01 and 03–07 are
all Stage 2, `build` — it reads the claims file written (or hand-authored)
in step 02, and only then reads the source PDFs and matches evidence
against them. The code reference names the actual function, in case you
want to find it while reading.

### 01 — Read the PDFs into plain text

Before anything else can happen, each source PDF gets converted from "a
PDF file" into plain text the rest of the pipeline can search through.
There's a fast option and a slower, more careful option for documents
that are mostly tables (financial statements often are) — tables are
where PDF text extraction most often scrambles rows and columns. **No
scanned/image PDFs** — those need a separate step (OCR) not built here.

`load_pdf_text` / `load_pdf_text_pdfplumber`

### 02 — Split the memo into single-fact claims

A memo sentence often bundles more than one fact — "revenue grew and
margins expanded," or "net income was $774M in 2025 **compared to** $805M
in 2024." The AI splits these into separate, independently-checkable
claims, so each one can be verified on its own instead of being accepted
or rejected as a bundle. See the worked example below. This is what
`extract` writes into `claims/<memo_id>.md` — the file you review before
any of the steps below run (see "Two stages, one file between them"
above).

`extract_atomic_claims` (called by `run_extract`)

### 03 — Cut the source documents into overlapping snippets

Source PDFs can run dozens of pages — far too much to feed an AI all at
once for every single claim. So each document gets sliced into small
overlapping snippets (about 1,000 characters, roughly two paragraphs), a bit
like cutting a document into index cards. The slices overlap on purpose
so a fact sitting right on a cut line doesn't get lost.

`chunk_document` / `build_chunk_index`

### 04 — Narrow down the snippets worth checking

Checking every single snippet against every single claim with the AI
would be slow and expensive. So first, a cheap keyword-ranking search (no
AI involved) scores every snippet by how relevant it looks, and keeps
whichever ones score close to the best match — not a fixed number, but
however many genuinely look promising. A claim whose evidence is
scattered across many places can legitimately keep dozens of snippets;
nothing gets silently dropped just to keep the list short.

`bm25_threshold_shortlist`

### 05 — Ask the AI to actually check each snippet

This is the one step that costs real AI usage. For each claim, the AI
reads its shortlisted snippets and reports back every one that genuinely
supports the claim, quoting the exact supporting text. If the shortlist
is very large, it's split into smaller batches so one oversized request
doesn't fail outright — and if one batch happens to fail, the evidence
any other batch already found is kept, not thrown away.

`propose_evidence_from_chunks_batched`

### 06 — Tell real ambiguity apart from duplicate echoes

Because snippets overlap, the same sentence can genuinely get "found"
twice — once in two neighboring snippets that both happened to catch it.
That's not a real conflict, so it's **not** flagged. It's only marked
**ambiguous** when a claim's evidence truly turns up in different
documents or unrelated places — the case that actually deserves a
human's attention.

`_rows_for_claim`

### 07 — Build the results table, then hand it to a person

Everything above produces one row per claim (or per match, if there's
more than one) in a table: the claim, where the evidence was found, a
confidence level, and whether the quote was word-for-word or paraphrased.
That table gets exported to an Excel file for you to review, correct, or
delete rows from — and re-importing merges your edits back in. Deleting a
row on purpose counts as rejecting it, not skipping it.

`export_for_review` / `import_reviewed`

### 08 — Tag the evidence: draft, review, finalize

The golden set says *where* evidence was found. The eval also needs to
know *how* each piece supports its claim. That takes two commands, with
your review in between.

**Draft — `python tag_pipeline.py draft [memo_id ...]`.** For each claim,
an AI reads every piece of source text search found for it, all together,
plus a little of the text just before and after each piece (it often holds
the table headings pypdf drops). It drafts one answer for the claim —
*stated directly* (a piece says it outright), *needs combining* (true only
by putting pieces together, calculating, or describing numbers with a word
the text doesn't use) or *not supported* — and marks the pieces it relied
on. A claim search found nothing for is pre-filled *not supported* without
asking the AI. The drafts go into one spreadsheet per memo,
`review/<memo_id>.xlsx`, which has a "how to" tab. An existing sheet is
never overwritten.

**Review — in Excel or Numbers.** Each blue row is a claim; the rows under
it are its pieces. Change any verdict or mark you disagree with, and set
CHECKED to `yes` on every claim — also when you agree. If a piece is
missing, paste a quote of 25 or more characters from the PDF into a `+` row
under the claim and set it to `yes`. If a claim says `draft failed`, the AI
couldn't answer: pick the verdict yourself and mark the pieces. Save under
the same file name.

**Finalize — `python tag_pipeline.py finalize <memo_id>`.** No AI. It reads
your sheet, the claims file and the PDFs, and either lists every problem
with its row number — nothing is written until they are all fixed — or
writes `reviewed/<memo_id>.xlsx`, the file the eval reads. Each piece's tag
comes from your answer: marked under *stated directly* → extractive, marked
under *needs combining* → synthesized, anything else → unverifiable. The
AI's own answer is kept beside it as `tag_draft`, and your notes go into
`tag_rationale`. Re-run it as often as you like; it rebuilds `reviewed/`
from the sheet each time and never changes the sheet.

It refuses a few things on purpose:

- **A claim that changed after drafting.** Rewording a claim gives it a new
  id, so the sheet no longer matches the claims file. Reword claims
  *before* `build` and `draft`. To recover afterwards: delete the sheet,
  re-run `build` and `draft`, and review again.
- **A second copy such as `<memo_id> 2.xlsx`** next to the sheet — Excel or
  Numbers saved your answers to a new file. Keep the one with your answers
  under the original name and delete the other.
- **A quote it can't place** — too short, not in the PDF text (a quote that
  runs past the end of one piece into the next, or a PDF viewer that copies
  ligatures or hyphenation differently), or found in several unrelated
  places. Paste a shorter or longer part. A quote that sits in the stretch
  two neighbouring pieces of the same PDF share is fine: both pieces are
  marked, and `finalize`'s summary says so.

`tag_pipeline.py` `draft` / `finalize` — design decision 18

## Worked example: splitting one real sentence

> "Net income was $774 million in 2025 compared to $805 million in 2024."

↓ step 02 splits this into two independently-checkable claims

- **claim 1** — Net income was $774 million in 2025.
- **claim 2** — Net income was $805 million in 2024.

↓ steps 03–06 run independently for each claim, each ending in its own row(s)

## A few terms worth knowing

| Term | Meaning |
|---|---|
| **claim** | One single fact pulled out of the memo — small enough that it's either true or false on its own. |
| **chunk / snippet** | A small overlapping slice of a source PDF's text — the unit the AI actually checks evidence against. |
| **evidence span** | The exact quote the AI points to as proof of a claim. |
| **confidence** | How sure the AI says it is about a match — high, medium, or low. Not the same as "found=False" (no match at all) or "confidence=error" (the pipeline itself broke, not a real answer). |
| **verbatim match** | True if the quoted evidence is word-for-word from the source. False means the AI paraphrased instead of quoting exactly — worth a closer look during review. |
| **ambiguous match** | True only when a claim's evidence genuinely shows up in more than one distinct place — not when overlapping snippets just echoed the same sentence twice. |

## Why the fussiness matters

**This spreadsheet becomes an answer key.** Later, a different AI tool
will write these financial memos automatically, and this golden set is
what will be used to grade it — did it find the right evidence, did it
make anything up? If this pipeline silently dropped a real piece of
evidence or mislabeled genuine ambiguity as "duplicate," that mistake
would quietly poison every grade that comes after it. That's why so much
care goes into things like "don't cap the shortlist," "don't let one
failed batch erase evidence another batch already found," and "a deleted
row means rejected, not skipped."

---

## Retrieval prototype (separate from the golden-set pipeline)

`retrieval_pipeline.py` is a sibling experiment, not part of the two-stage
`extract` / `build` flow. It exists to have a retrieval system to measure
once the golden set is built.

- **`python retrieval_pipeline.py embed [memo_id ...]`** — reads
  `retrieval/<memo_id>.yaml` (copy `retrieval.example.yaml`), chunks that
  memo's `source_folder` PDFs *with the exact same chunker the golden set
  uses*, embeds every chunk once, and caches the vectors to
  `retrieval_index/<memo_id>.parquet`. Re-running only re-embeds chunks
  whose text changed.
- **`python retrieval_pipeline.py retrieve`** — for every phrase file,
  searches with each section's phrases three ways: by meaning (`dense`: the
  chunks whose embedding is most similar to the phrase's), by keyword
  (`keyword`: BM25 word overlap) and both combined (`both`: the two rankings
  merged). It writes `retrieval_results.parquet` (+ a `.xlsx` to eyeball):
  one row per `(memo_id, section, phrase, method, chunk_id)` with the
  retrieval `rank` and `score`, where `method` says which search found it.
- **`python retrieval_pipeline.py recheck`** — searches by meaning with
  each claim's own text from `claims/<memo_id>.md` and writes the top
  chunks per claim to `claim_queries.parquet`. The eval uses it to list
  passages worth a second look for claims nobody could verify.

It uses its own `EMBED_*` environment variables (OpenRouter now, an on-prem
embedding model later — only the env values change). `retrieve` is
decoupled from `claims/` (only `recheck` reads it): it runs fine with no
claims files at all, though it will log a soft WARNING if a
`claims/<memo_id>.md` exists and its section names don't line up with the
phrase file's (they need to match for the eval, which refuses a section
with claims but no search results).

It does **not** score itself: `eval_pipeline.py` (below) measures its
results against the reviewed evidence sheets in `reviewed/`. Full design:
`docs/superpowers/specs/2026-09-07-local-retrieval-design.md`
— in the maintainer's private notes repo cloned at `docs/superpowers/` (see
`CONTRIBUTING.md`, section 7); not present on a fresh clone of this repo alone.

### Measuring it: `eval_pipeline.py`

`python eval_pipeline.py score` compares what retrieval found with what a
person confirmed as evidence, claim by claim, and `report` turns the result
into one web page. The headline is **claim coverage**: of the claims that
can be supported from the sources, how many had a supporting passage among
the passages retrieved. It is shown for three kinds of search (meaning,
keyword, both) and for 1 to 20 passages per search phrase, so you can see
what each setting buys. Claims nobody could verify are counted separately,
with a list of passages worth a second look — never as proof the sources
lack them.

---

*`golden_set_pipeline.py` · Phase 1 of an eval framework for an internal
financial-memo generator.*
