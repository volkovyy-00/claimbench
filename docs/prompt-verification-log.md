# Prompt-verification log

Run-by-run evidence for the `extract_atomic_claims` prompt-text edits
(design decisions 5's ticket-007 addendum, 15, and 16 in `CLAUDE.md`).

**Why this file exists.** Prompt-text changes can't be verified by a
mocked test — they need real live calls, each case repeated 3x to catch
run-to-run non-determinism (see `CLAUDE.md` → "Testing convention"). The
resulting transcripts are proof-of-work for a *past* edit: useful when you
revisit that edit, but not something every session needs in context. So
`CLAUDE.md` keeps the **rule**, the **why**, the **motivating
observation**, and the **current known residual** for each decision; the
blow-by-blow `0/3 → 3/3` results live here.

**Names and figures.** Company names, people and figures quoted from real
memos and filings have been replaced with fictional ones (Acme, Borealis,
Gantry, Globex …) throughout, including in the worked examples, which are
therefore no longer verbatim copies of the prompt text they describe.

**Precedence.** `CLAUDE.md` wins on anything about the *current* state of
the code or the prompt. This file is a dated record of what was measured
when each edit landed; it is not kept in sync with later changes.

---

## Embedded worked examples — the regression baseline

Extending decision 5's standing convention: the worked examples embedded
in the `extract_atomic_claims` prompt are the regression baseline for any
future edit. Referred to by content, not line number (line numbers drift).
**Any future edit re-verifies all 14, in both directions** (over- and
under-splitting are both real failure modes, and it's easy to overfit).

The 8 pre-existing:

1. "Net income … $774m vs $805m" → **2 claims**
2. "Acme … market cap of EUR 5bn" → **2 claims**. "Acme has a market cap
   of EUR 5bn." is itself an "X has Y" form whose predicate *is* the
   assertion — a built-in guard against decision 16 over-firing.
3. "Acme is the 2nd largest retailer … after Globex (revenue $50bn …)" →
   **1 claim** (comparator-entity exception).
4. "Acme is a listed German logistics operator" → **1 claim** (adjective
   counterweight — don't shred a description into separate adjectives).
5. "…wth assets of $40bn" → reworded to "The issuer has assets of $40bn."
   Fires ~2/3 runs (decision 5); also an "X has Y" over-firing guard.
6. "Acme uses this process …" → "this process" correctly left
   **unresolved**, because the isolated fixture names nothing to resolve
   it to (the prompt rule replaces it with a name only when the text
   gives one).
7. "…the portfolio was under pressure … However, it recovered …" → **3
   claims**, second claim says "the portfolio", not "it".
8. "Change-of-Control Clause" heading + "The clause … It survives …" →
   **2 claims**, both borrowing the heading name.

Added by decision 15 (ticket 008):

9. Audit-committee **closed/open contrast pair**. Closed ("role is
   limited to X and Y, such as …") → **1 claim**, "such as" items kept
   inside. Open (three uses joined plainly) → **splits**, one claim per
   member.
10. Hedged ratings-upgrade **forecast** contingent on several conditions
    jointly → **1 claim**.
11. Fleet-supplier **partial modifier** ("mainly by Globex, and to a
    lesser extent by Initech and Umbrella") → "to a lesser extent" rides
    onto the Initech/Umbrella claims only, **never** the Globex claim.

Added by decision 16 (ticket 009):

12. Causal attribution ("Acme's margins improved on lower costs and
    better pricing.") → either the 2-way attributed split ("Acme's
    margins improved on lower costs." / "… on better pricing.") **or**
    kept whole with attribution intact. **NOT** the bare "Acme has lower
    costs." / "Acme has better pricing." forms.
13. Attached-attribute carve-out ("Acme is well positioned, with $2bn of
    committed liquidity.") → "Acme is well positioned." standalone, "$2bn"
    splits off into its own claim.
14. Leave illustration ("Acme is well run.") → **unchanged** (a whole
    sentence the memo asserts outright that happens to be unfalsifiable
    is left alone).

---

## Decision 5 / ticket 007 — pointer resolution beyond the subject

Live, `openai/gpt-oss-120b`, 3 runs/case (cases 3/5/6 fresh out of
sample; all but the embedded-example sweep and the identical-text case
also run against the unmodified prompt as baseline).

Stranded-pronoun fix (the earlier, subject-only version), verified 3/3 on
the original case (→ "The asset has stabilized as of Dec'24 versus
Dec'23." every run) and 3/3 on six more: no regression on the four
motivating examples; a same-sentence pronoun with no antecedent anywhere
stayed unresolved 3/3 (no hallucinated subject); an ambiguous
two-candidate case ("The group acquired Beta Logistics… It reported
revenue…") resolved to Beta Logistics 3/3; the comparator-entity
exception held with the pronoun resolved to the section's subject, not
the competitor; the minimal-replacement guard held ("Acme", never "Acme,
a listed German logistics operator…"); two pronouns with different
referents in one sentence resolved independently and correctly 3/3.

The comparator-entity exception itself was verified 3/3 (a
Globex-reports-its-own-revenue sentence still split into separate Globex
claims), plus 3/3 on the original shape and no regression on the
compound-attribute and numeric-comparison examples.

Ticket 007 widening (heading/label lines borrowable; pointer anywhere in
the claim, not only the subject):

- **Non-subject pointer** ("…uses this process…"): 2/3 pre → **3/3**
  post. The primary new capability.
- **Separate-line heading borrowing** (paragraph under the second of two
  `## ` headings, pointed at with "the facility"/"it"): 1/3 pre → **3/3**
  post. Pre-edit, 2/3 runs left "The facility is undrawn as of year-end
  2024." bare.
- **Referent named nowhere** ("This ratio has improved…" with no ratio
  named): unchanged, no invented subject, 3/3 — identical pre/post (guard
  already firing).
- **Over-resolution direction** (single-entity section, healthy "the
  company"/"the portfolio"): 0/3 re-inflation pre and post. Incidental
  over-splitting of "…through two acquisitions during the year…" in both
  arms, slightly worse pre-edit — that's the atomic-splitter, not this
  edit.
- **Embedded prompt examples**, isolated: all 7 (as they were then) still
  split (or stay one) as written, 3/3 each; the garbled-input example
  reworded cleanly 3/3 this run.
- **Identical-text WARNING**: two sentences whose pointers resolve to the
  same heading name → byte-identical claims, WARNING fired, both kept,
  3/3 — but only with no redundancy cue in the input. Add an "as noted
  above" phrase and the model self-deduplicates to one claim (runs 1–2)
  or rewords the second copy (run 3), so the WARNING correctly never
  fires (0/3).

**Residual — the motivating case is not fixed** (kept in `CLAUDE.md`
decision 5 as the current known residual): the Borealis
cross-acceleration clause, named once by an inline "Cross-acceleration
Clause:" label at the head of a multi-claim paragraph — 3 of that
paragraph's claims read a bare "The clause …" per run, identical before
and after this edit (3/3/3 → 3/3/3).

---

## Decision 15 / ticket 008 — closure + modifier survival

Live, `openai/gpt-oss-120b`, 3 runs/case; pre-edit baseline captured
against `bf31154` before the edit; extraction-level eyeball is the
pass/fail instrument.

- **Closed set** (`G1-closed`, audit committee "limited to"): 0/3 →
  **3/3** (one claim, "such as" items inside). **Open counterweight**
  (`G1-open`): 3/3 → 3/3, unchanged; the out-of-sample open list
  (`OOS-B`, three uses joined plainly): 3/3 → 3/3, still splits.
- **Hedged forecast** (`G1-forecast`): 0/3 → **3/3**. Out-of-sample bare
  conjunction, no "if" scaffold (`OOS-C`): 0/3 → **3/3**.
- **Partial modifier** (`G1-modifier`): count 2/3 → **3/3**, "to a lesser
  extent" on Initech and Umbrella separately and never Globex, 3/3.
  Out-of-sample (`OOS-E`): count 0/3 → 0/3 — "the two regional units"
  stays one claim (EV-7 economization); qualifier placement correct
  3/3 either way. Reverse case, qualifier true of all members (`OOS-F`):
  3/3 → 3/3.
- **Out-of-sample closed lists**: `OOS-A` ("may be used only for X and
  Y") 0/3 → **1/3** whole — measured residual, the "only … and …" shape
  still reads as two permissions 2 runs in 3. `OOS-G` ("mandate limited
  to hedging FX ($2bn notional) and …") 0/3 → **2/3** whole; the "$2bn
  notional" figure rides inside the one claim 2/3 and splits off as its
  own attached-attribute claim 1/3 (closure vs. the
  attached-measured-attribute rule) — and when it splits, the remaining
  pieces stay coherent, not the contradiction harm. Recorded, not forced
  either way.
- **Ranking-with-competitor-list** (`OOS-D`, checks the closure exception
  does not tangle with the adjacent comparator-entity exception): 3/3 →
  3/3, one claim about Acme, nothing about the competitors.
- **Ticket AC-1 guard — copula member-list must still split** (`CoP-1`,
  "Acme's largest customers are Globex, Initech and Umbrella." → one
  claim per member): 3/3 → **3/3**. The "plain enumeration is NOT a
  closed list" clause holds; no over-fire.
- **8 embedded examples, regression**: all no worse than the frozen
  pre-edit verdict. `EX1`–`EX5`, `EX7`, `EX8` unchanged at 3/3; `EX6`
  ("Acme uses this process …") *improved* 0/3 → 3/3 (one claim, no
  stranded-pointer warning).
- **Real `MEMO-002` section, in-context** (monitored, not a pass/fail bar
  — the same economization ceiling degrades it, per EV-7): the
  Gantry management-role sentence, run by run: fully clean (no contradiction *and* the "such as"
  examples kept inside) 0/3 → 0/3; free of the contradiction 1/3 → 1/3
  (pre-edit, one run was already coherent but lifted the examples out;
  post-edit, one run keeps "budgets and other approval actions" together
  — still lifting the examples — one run weakly contradicts, milder than
  pre-edit, and one run still fully contradicts). The in-context movement
  is mostly that fewer claims means accidentally fewer contradictions, so
  `G1-closed` carries the verdict, not this sentence. Fleet-supplier "but
  to a lesser extent" 0/3 → **2/3** preserved on the trailing suppliers. Upgrade
  rationale 0/3 → 0/3, unchanged — the source sentence is genuinely
  mangled ("… along with debt reduction, this will likely result in an
  upgrade", a detached conjunction and a vague "this"), a decision-5
  vague-subject problem the closure rule cannot reach; the isolation
  forecast cases prove the rule works on well-formed input. "Key
  customers …" 0/3 → 3/3 (see decision 15 "Scope").
- **Focused end-to-end** (the full-section `build_golden_set_draft` run
  is infeasible — at default tuning against the 5 MB annual report one
  generic claim pulls 600–1,900 BM25 candidates → 15–49 LLM batches,
  projecting to ~9 h per side; so `extract_atomic_claims` was
  monkeypatched to feed the exact pre-edit-harmful and post-edit-clean
  claim variants through the real evidence pipeline against the memo's
  annual report, evidence-step prompt unchanged by this edit): the Gantry
  contradiction does not manifest downstream on this corpus — every
  Gantry claim variant returns `found=False` (the annual report does not
  discuss Gantry's internal approval process), so the benefit there is 2 not-found rows becoming 1.
  The collapsed-forecast fragment "Debt reduction will likely result in
  an upgrade." pulls 2 `confidence=high` matches — the clearest instance
  of a degraded claim drawing misleading evidence, and unchanged pre→post
  because the collapse is in-context. The fleet-supplier tail claim (the last
  supplier, with and without "to a lesser extent") returns
  `found=False` both ways. The "five unrelated metrics" blow-up from
  decision 5 did not recur this run — but that harm is non-deterministic
  and this corpus is thin on the relevant content, so one focused run is
  a spot check, not proof of absence.

**Measured residuals** (summarized in `CLAUDE.md` decision 15): `OOS-A`
1/3 and `OOS-G` 2/3 (with the `$2bn notional` split-off, closure vs.
attached-measured-attribute) — logged with their failure mode, not chased
with re-runs. In-context, the mangled upgrade-rationale sentence and the
general granularity ceiling (EV-7) are unchanged; a minor
in-context pointer-resolution wobble ("This results in a strong downside
correlation …" stranded 2/3 post vs 0/3 pre) sits inside decision 5's
documented variance and the `_STRANDED_POINTING_WORD_RE` backstop fired
on it.

---

## Decision 16 / ticket 009 — predicate retention

Live, `openai/gpt-oss-120b`, 3 runs/case; pre-edit baseline captured
against `1b7019d` before the edit.

- **Verdict carriers.** Isolated driver sentence (`B-driver-real`, real,
  a many-item run-on): the bare-existence pieces ("Borealis has pricing
  initiatives." and its siblings) land at ≈1/3 in the
  exact bare "has" form, but the drivers are still severed ~2/3 into thin
  verbs ("practiced"/"implemented"), and the detached-driver
  decomposition itself is unchanged 3/3 → 3/3 (the §4.2 wording does not
  re-attach drivers in a many-item run-on read in isolation). Per spec
  §8.2 a B/C miss is a recorded measured residual with its failure mode,
  not chased — the isolated `C-driver-clean` fixture carries the verdict.
  Clean constructed variant (`C-driver-clean`, "Acme's margins improved
  on lower costs and better pricing."): the bad attribution-dropping
  shatter ["Acme has lower costs.", "Acme has better pricing."] is
  eliminated — 0/6 across the Task 7 run and a final-verification re-run
  (pre-edit 1/3). Post-edit the model either produces the desired
  attribution-preserving 2-way split ["Acme's margins improved on lower
  costs.", "Acme's margins improved on better pricing."] or keeps the
  sentence whole with its attribution intact; which of the two it does is
  non-deterministic (3/3 split in Task 7, 1/3 split + 2/3 whole on the
  re-run) and both are acceptable outcomes. This sentence is embedded
  worked example #12.
- **Over-firing guards** (the direction that matters). Pure-qualitative
  (`D-qualitative-real`): PASS, unchanged 3/3 — its first claim (a
  finance subsidiary's role for its parent's products) kept every run,
  sub-claim splitting only, durations kept as source-worded, no figure
  invented, nothing dropped. Borderline "has" phrasing
  (`E-has-phrasing-real`): PASS 3/3, byte-identical to pre-edit — e.g.
  "Borealis Finance has the largest market share in all Borealis engine
  types.", "has the largest market share" preserved as the assertion. Split-with-vague-half (`F-mixed-constructed`): PASS 3/3,
  byte-identical — ["Acme is well positioned.", "Acme has $2bn of
  committed liquidity."], "well positioned" not dropped, not concretized,
  not re-absorbed; "$2bn" splits off per the attached-attribute rule.
  Out-of-sample leave cases: `O-leave-1` PASS 3/3 byte-identical
  (["Management is widely regarded as capable.", "The board is
  experienced."], both stay evaluative); `O-leave-2` PASS against the bar
  "no worse than the pre-edit 2-claim shape" — pre-edit already splits
  3/3 via the attached-attribute rule (peeling the EUR 500m amount into
  its own claim), and post holds that 2-claim shape 3/3 (contents differ
  on 2 of 3 runs — see Residual), no 1-claim collapse.
- **Restore direction, out of sample.** `O-restore-1`: PASS — attribution
  ("was helped by X") kept on both driver claims 3/3, no bare "The
  company has pricing discipline." form (0/3); r2 adds one harmless
  near-event claim ("There was an earnings beat."), not a drop or a
  concretization. `O-restore-2`: PASS, improved — the full "Free cash
  flow rose on X" attribution kept 1/3 pre → 2/3 post, FCF attribution
  never dropped. `O-mixed-1`: PASS 3/3, no re-absorption — ["The group is
  well diversified.", "The group operates in 14 countries."] every run,
  the vague half stays its own bare claim and the factual half splits off
  (the second attached-attribute-boundary guard).
- **Boundary check** (`R5-forecast` / `R4a-closed`, that decision 16's
  attribution split and decision 15's closure rule don't tangle): Group R
  held 3/3 on both.
- **Embedded worked examples, regression** (the 11 pre-existing, run as
  12 input strings — the audit-committee closed/open pair is two): all no
  worse than each case's frozen pre-edit 3/3 verdict; the only
  differences vs pre-edit are cosmetic wording noise (R11 run 3 borrows
  the heading's casing, "The Change-of-Control Clause"; R10 wobbles "the
  portfolio" vs "Acme's portfolio"). Both "X has Y" over-firing guards
  hold 3/3 — "Acme has a market cap of EUR 5bn." (R2) and "The issuer has
  assets of $40bn." (R8); R8's "wth"→"has" rewording also landed 3/3 this
  run (decision 5 documents ~2/3, so a future 2/3 there is within the
  floor, not a regression).
- **Real `MEMO-002` section, in-context** (monitored, not a gate —
  EV-7's granularity ceiling degrades this section generally): the
  in-context result is **not reproducible**. Task 7 saw the driver
  sentence's bare "Borealis has pricing initiatives." form at 0/3 and the drivers attached 2/3; a
  final-verification re-run saw the bare form at **2/3** (run 1 detached
  into thin verbs "maintained"/"implemented", runs 2–3 the bare "has"
  form) and the drivers attached **0/3** — i.e. back to the pre-edit
  rate. In-context at 4,400 chars the fix does not reliably fire; this is
  EV-7's granularity ceiling (the spec scopes Group A as monitored,
  not a gate) and the isolated fixtures carry the verdict. The 4-claim
  healthy sample was all present and un-concretized across both runs,
  with no new in-context over-firing from this edit. The mangled
  upgrade-rationale sentence and the `_STRANDED_POINTING_WORD_RE` wobble
  on "This results in…" are pre-existing (decision 15), not counted here.

**Measured residuals** (summarized in `CLAUDE.md` decision 16):

- The isolated many-item run-on `B-driver-real`: the exact bare "has"
  form only fell to ≈1/3 (not 0/3) and the drivers are still severed ~2/3
  into thin verbs — §4.2's wording does not reach a long list of siblings
  read out of context; logged, not chased (`C-driver-clean` carries the
  verdict). Spec §10 pre-authorised a tightening for exactly this
  underfire ("said something checkable that this thing was part of") —
  **declined**, because that phrasing reintroduces the checkability
  framing the rule was built to avoid; the next editor should not reach
  for it.
- `O-leave-2`: 2/3 runs post emit *both* "Acme has a revolving credit
  facility." and the fuller "Acme has a EUR 500m revolving credit
  facility." — the first contained in the second, which the prompt
  forbids — so the new rule nudges a genuine "X has Y" toward a redundant
  pair (nothing dropped or concretized; a reviewer deletes one row,
  decision 3). This containment is silent — the ticket-007
  duplicate-claim WARNING fires only on byte-identical text — and it
  costs that claim a second `claim_id`, BM25 shortlist and evidence run
  downstream; a candidate for EV-6's always-on check.
- The abstract trigger is a loose fit for the causal case — "asserted
  something more about that thing" is imperfect when the sentence asserts
  about *margins*, not "about lower costs" — and in-context reliability
  rides on EV-7's granularity ceiling staying open.

---

## `tag_pipeline` prompt (design decision 18)

Run-by-run evidence behind `tag_pipeline.py`'s row-tagging prompt (design
decision 18 in `CLAUDE.md`). Unlike the sections above, this prompt tags
evidence *rows* (`extractive` / `synthesized` / `unverifiable` / `unsure`),
not claim splits — a different judgement task with its own worked
examples and its own model choice.

### 3-model probe (2026-09-10)

Sample: 16 claims / 60 `found=True` rows, stratified across the 3 memos ×
3 sections. `filing_entity` context in the prompt from the start in every
row below.

| model | prompt | dist. (E / S / U) | fail | verdict |
|---|---|---|---|---|
| `gpt-4.1-mini` | initial | 63 / 2 / 36 | 0 | **rejected** — over-calls `extractive`; violates the value-contradiction rule even after doing the arithmetic (it computes a share ~4 points off the claim's, then calls them equal); one fabricated denominator. ~40% of `extractive` wrong, all coverage-flattering. |
| `gemini-2.5-pro` | initial | 8 / 15 / 46 | **31%** | **rejected** — sharpest reasoning on completed rows, but hard JSON failures from reasoning-token budget exhaustion + a provider 504. |
| `claude-sonnet-4` | initial | 5 / 59 / 36 | 0 | viable but over-pedantic — demotes clean `extractive` (a named-customer claim, plain "we provide hardware") to `synthesized`. |
| **`claude-sonnet-4`** | **tuned** | **41 / 22 / 37** | **0** | value contradictions caught (a segment-share claim → `unverifiable` "computed share differs, contradicts"); role mismatch caught (4/4 `unverifiable`); substance-match fixed (the hardware claim → `extractive`). Residuals, all safe-direction: a rating-agency claim with a month still `synthesized` not `extractive`; "UK-based" pedantry; one derived-figure row not arithmetic-checked. `unsure` = 0. |

**Initial → tuned diff.** Both probe prompts already carried the
`filing_entity` context. The tuned prompt (the one shipped in
`tag_pipeline._TAG_PROMPT_PREAMBLE`) added the **peripheral vs.
substantive** distinction — a missing date, example, or wording variant
does not block `extractive`; a missing measure, value, or subject does —
tightened disqualifier 3 to name the compute-to-contradiction case
explicitly, and added worked examples 8–10 below. This is what took
`claude-sonnet-4` from over-pedantic (59% `synthesized`) to its shipped
distribution (22% `synthesized`) without reopening the value-contradiction
hole the tuning was meant to close.

`claude-sonnet-4` (tuned) is the model shipped as `_TAG_MODEL`.

### §12 pilot (2026-09-12) — first hand-labelled measurement: **NO-GO**

The 3-model probe above chose the model and tuned the prompt by inspecting
individual rows. It never compared the output to a human label set, so no
agreement rate existed. This pilot is that measurement, and it fails two of
the four go/no-go bars. **The full `run_tag` was not run.**

Sample: 11 claims / 72 `found=True` rows drawn by a fixed seed across all 9
`(memo_id, section)` cells plus both ends of the rows-per-claim range
(1 → 22 rows), deliberately not "the hardest shapes". Model `_TAG_MODEL`
(`anthropic/claude-sonnet-4`), 3 runs. Labels were made blind: the label
sheet carried no model output, and the two files are joined on
`(claim_id, chunk_id)` only at scoring time.

| bar | result | |
|---|---|---|
| `extractive` on hand-`unverifiable` = 0 | **0** | PASS — the unsafe direction is clean |
| `synthesized` on hand-`unverifiable` ≈ 0 | **8 / 17 (47%)** | **FAIL** |
| `unsure` rate < ~20% | **0%** | PASS, but trivially — see below |
| agreement ≥ ~80% | **41.7%** | **FAIL** |

Plumbing is sound: 3-run self-consistency 65/72 (90%), and **0 blank drafts**
across ~39 live calls — no JSON failure, no unknown tag, no
`synthesized`-without-`part` coercion ever fired on real output.

**Distribution, model vs. human** (72 rows, model = 3-run consensus):

| | extractive | synthesized | unverifiable | unsure | disagreed |
|---|---|---|---|---|---|
| model | 36% | 21% | 33% | **0%** | 10% |
| human | **61%** | 15% | 24% | — | — |
| §13 probe (60 rows, no labels) | 41% | 22% | 37% | 0% | — |

**This is not a regression.** The model reproduces the probe's accepted
distribution within sampling noise (36/21/33 vs 41/22/37). What the probe
could not see is that the distribution is wrong in one consistent
direction: the model **under-calls `extractive` by ~25 points**,
redistributing into `unverifiable` (+9) and `synthesized` (+6). The probe's
"over-pedantic" note against untuned `claude-sonnet-4` was the same defect;
tuning reduced it (59% → 22% `synthesized`) without eliminating it.

Confusion (rows = hand label, cols = 3-run consensus; `''` = runs disagreed):

```
consensus        ''  extractive  synthesized  unverifiable
hand_label
extractive        4          22            5            13
synthesized       0           4            2             5
unverifiable      3           0            8             6
```

**Two diagnosed causes.**

1. **Worked example 6 is reasoned from but its label is not applied.**
   Example 6 sends a segment-subject claim against a first-person
   whole-entity passage to `unsure`. The 22-row sampled claim is exactly
   that shape — a product-range claim whose subject is a segment of
   the filing entity. On 8 of its 19 hand-`extractive` rows the model
   returned `unverifiable` with a rationale naming the example's own
   reasoning — *"the claim's subject is '…' (a segment), not the whole
   filing entity, so the subject is unresolved"* — including on rows
   whose passage restates the claim almost word for word, differing only
   in saying "The Company" where the claim says the segment. It applied
   the analysis and then chose the wrong landing. `unsure` = 0 across all
   three runs is the visible symptom; the `unsure` bar cannot detect it,
   because "< ~20%" passes trivially at 0%. **The bar is one-sided and
   should be two-sided.**
   Inconsistency compounds it: the same claim's 19 near-identical
   hand-`extractive` rows split 11 `extractive` / 8 `unverifiable`.

2. **`synthesized` is used as a "partially related" catch-all**, which is
   the failing 47% bar. On a claim whose text is not self-contained (a
   "…operating through the below segments" claim — the segment list lives
   outside the claim), vague overview passages came back `synthesized`
   with a `part` conceding the claim's substance is absent — e.g.
   *"Establishes [the entity] as a leading company in its field…; the
   specific operating segments are not described in this passage."* That
   is the
   "do not promote the best of a bad set" failure the prompt explicitly
   forbids, landing on `synthesized` instead of `unverifiable`.

**Cohort breakdown — the failure is systemic, not confined to bad claims.**

| cohort | rows | agreement |
|---|---|---|
| claim not self-contained (stranded pointing phrase — decision 5's open residual / EV-6) | 24 | 25% |
| segment-as-subject claim (cause 1) | 22 | 50% |
| everything else | 26 | 50% |

Two of the 11 sampled claims carry a stranded pointing phrase ("the below
segments", "these plans") and so are unverifiable by construction — no
tagger can resolve them, and they are decision 5's known residual arriving
downstream. But the remaining 26 rows still agree only 50%, so claim
quality does not explain the result.

**Safety read.** The error is mostly in the conservative direction —
calling real evidence `unverifiable` wastes a reviewer's time but loses
nothing, and the hard-stop bar (`extractive` on hand-`unverifiable`) is
clean at 0. Cause 2 is the exception and is why the pilot is a NO-GO
rather than a "ship with caveats": a `synthesized` draft asserts a
derivation the passage does not support.

**Harness note.** The first scoring pass reported 44.8% agreement over
"96 rows" from a 72-row sample. All three throwaway pilot scripts keyed
rows by `chunk_id` alone; the row identity is the pair
`(claim_id, chunk_id)`, since one passage can be evidence for several
claims and is judged separately against each (4 such chunks here, across
2 claims). That fanned out both merges and, worse, let one claim's tag
overwrite another's in the run aggregator. Fixed on the pair key with
post-merge fan-out assertions, the affected claims re-tagged, and the
numbers above are from the corrected run. `tag_pipeline.py` itself is
unaffected — `propose_row_tags` keys by `chunk_id` within a single claim,
where it is unique.

### §12 confirmation round (2026-09-13) — fresh sample, two-pass labelling: **still NO-GO**

The pilot above drove six prompt versions (v1 → v6), each scored on the same
72 rows and each adjudicated by the same human. By v5 the reported result had
inverted: against *adjudicated* rulings, v5 scored 78.8% while the human's own
unaided labels scored 66.7–69.7%, i.e. "the tagger beats the reference it is
graded against". That number was measured on rows the adjudication loop had
already touched, so this round re-measures on rows it never saw.

Sample: **30 fresh `found=True` rows / 14 claims**, fixed seed, all 9
`(memo_id, section)` cells, drawn from the 317 rows not used in the pilot.
Prompt v5 as landed in `tag_pipeline.py`, `_TAG_MODEL`, 3 runs.

Protocol — two passes, deliberately ordered:

- **Pass A**: the human labels all 30 rows *blind*. No model output visible.
- **Pass B**: the same 30 rows, now showing the model's label and its stated
  reasoning; the human re-rules every row, including the ones already
  agreeing. Pass B is the adjudicated truth for this round.

Pass B is the reference, and it is anchored — the human sees the model's
argument before ruling. Pass A is the only unanchored number here.

**The four numbers.** The model's label is taken three ways, because a
production run calls the model **once**; "unanimous only" treats a row whose
3 runs disagreed as a miss, which is the strictest reading and not what
production would emit.

| | vs Pass A (blind) | vs Pass B (truth) |
|---|---|---|
| single run (= production) | 60.0% | 73.3% |
| majority of 3 | 63.3% | **83.3%** |
| unanimous of 3, else miss | 50.0% | 66.7% |

| | |
|---|---|
| **Pass A vs Pass B — unaided human accuracy** | **80.0%** (24/30) |
| rows the human changed after reading the reasoning | 6 |
| …that moved to the model's label | 5 |
| …that moved to a third label | 1 |
| rows where Pass A and the model already agreed | 15 |
| …that the reasoning then talked the human *out* of | **0** |

**The pilot's headline does not reproduce.** On rows the adjudication loop
never touched, unaided human accuracy is **80%**, not the 66.7–69.7% the
pilot measured — and a single production run scores **73.3%**, below it. The
pilot's finding that the tagger outperforms the human was an artefact of
grading against a reference that six rounds of adjudication had progressively
pulled toward the model. Anchoring is the mechanism: Pass B is 10–20 points
more generous to the model than Pass A on identical rows.

**Bars, against Pass B truth:**

| bar | single run | majority | |
|---|---|---|---|
| `extractive` on true `unverifiable` = 0 | **2** | **2** | **FAIL — hard stop, first time** |
| `synthesized` on true `unverifiable` ≈ 0 | 5/15 | 3/15 | **FAIL** |
| self-consistency ≥ ~90% | **83.3%** | | **FAIL** (was 90% on the pilot rows) |
| `unsure` reachable > 0 | **0** | **0** | **FAIL** — dead in all 7 versions |

The hard stop had been clean across v1–v6 on the pilot rows. Both breaking
rows were ruled `unverifiable` by the human in *both* passes — they are model
errors, not label errors, and both are checks the prompt already states:

- A share-count claim in billions against a balance-sheet line stating the
  count in millions. The model computed the passage's value, stated it
  alongside the claim's, and called a 15% gap "approximately" equal —
  disqualifier 3 written out and then ignored. This is the exact failure that
  disqualified `gpt-4.1-mini` in the §13 3-model probe ("fails the
  value-contradiction check even after doing the arithmetic"); it is now
  observed on `claude-sonnet-4`.
- A joint-venture holding claim against a subsidiary table whose row reads
  "(no equity held)" beside a number in a year-pair column with its header cut
  off. Disqualifier 4 (a table row severed from its header) applies, and the
  passage's own words contradict the claim.

**New failure mode, not present in the pilot: sector vs. entity.** Three of
the human's Pass B notes name it independently — claims taken from the memo's
*industry overview* section have an **industry or sector** as their subject,
and the model matched them to passages about the filing entity itself. The
prompt's FILING CONTEXT block resolves entity ↔ sub-entity and states that it
"runs in ONE direction only" (whole-entity supports a unit claim, never the
reverse). It says nothing about a claim whose subject is *broader* than the
filing entity, so nothing blocks a company passage from supporting a sector
claim. This is a gap in the rule set, not an instance of it misfiring.

**Instability is concentrated at one boundary.** 5 of 30 rows were not
unanimous across 3 runs, and **4 of those 5 flip only between `synthesized`
and `unverifiable`** — the same boundary that fails the second bar. Per
decision 9's pattern, run-to-run instability on identical input is the
signature of two rules with an unchosen precedence, not of a hard case.

**What Pass B does establish.** Showing the reasoning changed 6 of 30 human
labels, 5 of them toward the model, and cost **nothing**: of the 15 rows where
the blind human and the model already agreed, the reasoning talked the human
out of exactly zero. That is the first direct measurement of decision 18's
product claim — `tag_draft` + `tag_rationale` as a review aid rather than a
label source — and it is the one claim in this effort the evidence supports.
It is also not what the §12 bars measure.

### Canonical worked examples (11) — regression baseline

Embedded verbatim in `tag_pipeline._TAG_PROMPT_PREAMBLE`. Same convention
as the `extract_atomic_claims` worked examples above: any future prompt
edit re-verifies all 11 live, in both directions.

1. "Acme's revenue was EUR 4.2bn in 2024." / "...revenue for the year
   ended 31 December 2024 was EUR 4,203m..." → **extractive**
2. "Acme grew faster than its main competitor." / "...Acme revenue rose
   8%..." → **synthesized** | part: establishes Acme's growth; competitor
   comparison not here
3. "Acme's revenue was EUR 4.2bn in 2024." / "...Acme's order book stood
   at EUR 4.2bn..." → **unverifiable** (same number, different measure)
4. "Acme's plan owns 44% of total shares." / "...the plan beneficially
   owns approximately 47% of outstanding common stock..." →
   **unverifiable** (same measure, different value)
5. "Acme's Industrial segment was about a third of FY24 revenue." /
   "...Industrial segment, year ended 31 December 2024 ... Revenues:
   Products 110,000; Services 5,000..." → **synthesized** | part: gives
   Industrial FY24 revenue ~115,000; consolidated total to divide by not
   in this passage (no contradiction: a third is plausible)
6. "Globex Health serves more than 20m members." (Globex Health is a
   segment of the filing entity) / "...We serve more than 20 million
   people through ... products..." → **unsure** (first person, but
   claim's subject is a segment, not the whole filing entity)
7. "Dana Reyes was President of Globex Rx." / "...Dana Reyes, Executive
   Vice President of Globex Corporation and President of Retail
   Services..." → **unverifiable** (different named role — disqualifier 1)
8. "A rating agency reaffirmed a positive outlook on Acme in March 2024."
   / "...all three agencies now hold us at investment grade with a
   positive outlook..." → **extractive** (central assertion — positive
   outlook — is directly supported; "March 2024" and "reaffirmed" vs
   "hold" are peripheral)
9. "Acme provides hardware to government security departments." /
   "...We provide software, hardware and technical expertise; our
   customers include national defence and intelligence agencies..." →
   **extractive** (hardware + those customers IS "security departments"
   in the passage's own words)
10. "Acme's Industrial segment was 27% of FY24 revenue." /
    "...Industrial segment FY24 revenue 115,000; consolidated total
    revenue 350,000..." → **unverifiable** (115,000/350,000 = 32.9%,
    contradicts 27% — disqualifier 3)
11. "Acme's net debt fell to EUR 1.2bn in 2024." / "...We remain
    committed to a conservative funding profile and continued
    deleveraging..." → **unverifiable** (directionally agreeable,
    establishes nothing nameable)

## tag_pipeline v2 — bundle mode

Spec: `docs/superpowers/specs/2026-09-13-tag-pipeline-v2-bundle-design.md`
(gitignored, local). One verdict per claim over all its chunks; prompt frozen
as `tag_pipeline._BUNDLE_PROMPT`.

### §9.0 pre-flight (2026-09-13)

22-chunk MEMO-001 claim, plumbing only (verdict not inspected). Both passed at
`max_tokens` 16000 — no change.

| model | finish_reason | prompt tokens | completion tokens | reasoning tokens | seconds | pass |
|---|---|---|---|---|---|---|
| google/gemini-3.1-pro-preview | stop | 9898 | 2007 | 1907 (12% of budget) | 15.9 | yes |
| openai/gpt-5.6-luna | stop | 9277 | 105 | 0 | 2.5 | yes |

### §9 acceptance (2026-09-13) — **FAIL**

Pre-registered before any run: MEMO-004, test A (66 chunks the user judged)
and test B (55 pipeline chunks), 3 runs each, 2 models; primary decides.
Prompt sha256 `13a8ca519cc7e0d1098dc414ac97d902d348b2a6be980fc128606a5f0e108edc`;
max_tokens 16000; temperature 0; 500-char context. 192 answers, 0 retries,
0 `draft_failed`. Claims numbered C1–C16 in `memo004_final_claims.csv` order.

#### google/gemini-3.1-pro-preview (PRIMARY) — FAIL (1 bar of 8)

| rule | measure | value | bar met |
|---|---|---|---|
| 1 | never stated_directly on unsupported claims (B, 18 answers) | 0 | yes |
| 1b | rejected chunks marked under stated_directly (108) | 0 | yes |
| 1b | rejected chunks marked under needs_combining (108) | 3 | yes |
| 2 | needs_combining found, majority (of 9) | 6 | yes |
| 3 | same verdict in all 3 runs (of 32) | 32 | yes |
| 4 | verdict agreement, majority, test A (of 16) | 13 | yes |
| 5 | chunk precision, test A pooled | 95.5% | yes |
| 5 | chunk recall, test A pooled | 73.6% | **NO** |

Test B verdict agreement (reported, not gated): 14 of 16. draft_failed: 0.

- Rule 1b slips: C16 `_204`, 3 of 3 runs — one chunk repeatedly (a blind
  spot, not noise). Not the pre-noted arguable `_226`; it sits exactly at the
  ≤ 3 allowance.
- Rule 4 misses (all 3 runs identical): C7, C9, C15 answered `not_supported`
  where the key says `needs_combining`. Each also drives recall — the key
  chunks (C7 `_224`; C9 `_32`; C15 `_225`+`_226`) were never marked. C7 and C9
  repeat the same `not_supported` in test B.
- Other recall losses: C2 left out `_249`+`_250` (3 runs) and C3 left out
  `_250` (2 runs) while still answering `stated_directly` from another marked
  chunk; C16 left out `_19` (3 runs). Per spec §9.5, C16 did not answer
  `stated_directly`.
- Rules 1 and 3 clean: zero `stated_directly` on the 6 test-B unsupported
  claims, and every one of 32 claim-versions gave the same verdict 3 times.

#### openai/gpt-5.6-luna (secondary, not gating) — FAIL (4 bars of 8)

| rule | measure | value | bar met |
|---|---|---|---|
| 1 | never stated_directly on unsupported claims (B, 18 answers) | 0 | yes |
| 1b | rejected chunks marked under stated_directly (108) | 3 | **NO** |
| 1b | rejected chunks marked under needs_combining (108) | 0 | yes |
| 2 | needs_combining found, majority (of 9) | 5 | yes |
| 3 | same verdict in all 3 runs (of 32) | 27 | **NO** |
| 4 | verdict agreement, majority, test A (of 16) | 12 | **NO** |
| 5 | chunk precision, test A pooled | 93.4% | yes |
| 5 | chunk recall, test A pooled | 65.5% | **NO** |

Test B verdict agreement (reported, not gated): 15 of 16. draft_failed: 0.

- Rule 1b slips: C1 `_31`, 3 of 3 runs — one chunk repeatedly (blind spot).
- Rule 4 misses (majority): C9, C10, C15 `not_supported`; C12
  `stated_directly`. Luna got C7 right, where gemini did not.

Per spec §9.6: FAIL → stop; no prompt change, no re-run, no re-scoring. A
changed prompt needs fresh blind ground truth from a memo it has never seen.

#### Decision (2026-09-14) — gemini accepted despite the recall miss

The user looked at the result and explicitly accepted
`google/gemini-3.1-pro-preview` with the frozen prompt, building the rest
(spec §10 step 5). **The measured result above stays FAIL** — this is not a
re-score and no claim was dropped from the population. Reasoning: the only
failing bar is chunk recall, concentrated in table-division claims (a known
pypdf/retrieval limit on tables); the safety bars (rules 1, 1b, 3) are clean;
and every draft is advisory — no tag reaches the eval unless the user marks
the claim CHECKED — so a missed chunk costs review time, not a wrong tag.
Expect reviewers to add marks on table-division claims most often.
