# Design decision 16 — full narrative

Split out of `CLAUDE.md` to keep that file's per-session load down; the
condensed rule and current residual live there under design decision 16.
This file holds the full observed-failure narrative behind it. Live
verification run-by-run: `docs/prompt-verification-log.md` →
"Decision 16 / ticket 009" and "Embedded worked examples".

---

**A split piece must keep the predicate its source sentence gave it —
never decay to a bare "X has Y."** `extract_atomic_claims`'s prompt
gained one paragraph appended to the second self-test (the referent
test ticket 007 built); the docstring gained a matching paragraph. No
code-path change, no committed test — a prompt-text edit verified by
live runs (see "Testing convention").

**The rule is mechanical, not a checkability judgment.** It fires on a
shape — a split has reduced a claim to "X has Y." / "X is Y." while the
source sentence asserted more about that thing — and the remedy is to
carry the source-sentence predicate back onto the claim, copying
wording already there. It does NOT ask "is this claim vague"; a broad
"every claim must assert something checkable" bar was rejected in
brainstorming because it pushes the model to invent specificity
(decision 5's failure mode) and pulls in standalone evaluative
sentences, which are ticket 010's job. A whole sentence the memo
asserts outright that happens to be unfalsifiable ("Acme is well run.")
is left unchanged. The split is deliberate on cost asymmetry: a bare
evaluative claim that slips through is one `found=False` row a reviewer
discards, whereas a prompt edit aggressive enough to drop it will
sometimes also drop or concretize a real qualitative claim the memo
does make ("maintains a conservative funding profile") — a silent hole
in the eval. 009 only ever restores a predicate already in the source
sentence; it never deletes a claim or invents detail.

**New granularity precedent.** "Acme's margins improved on lower
costs." stays one claim — a causal attribution is not further split
into "margins improved" + "the improvement was due to lower costs".
The prompt had no causal-attribution worked example before this.

**Boundary against the attached-attribute rule.** What the new rule
restores is a *severed relationship* (the reason/basis/driver), never
a *separately measured attribute* (amount/date/holding/ranking) —
which still splits into its own claim as before. "Acme is well
positioned, with $2bn of committed liquidity." → "Acme is well
positioned." stays bare, "$2bn" splits off. Without the §4.2 clause
the two rules collide here — the decision-9 hazard applied to this
edit's own rule. Distinct from decision 15's closure rule, which pulls
the *opposite* way: an outcome contingent on several conditions holding
jointly ("an upgrade if A, B, and C") stays one claim, whereas an
outcome attributed to separable drivers ("margins improved on lower
costs and better pricing") splits, each piece keeping the attribution.
Verified not to tangle in practice (log → "Decision 16 / ticket 009",
boundary check).

**Why fold into self-test 2, not a third test.** "Every part has a
referent" and "a split piece keeps its predicate" are independent
necessary conditions — no input can fail one because it passes the
other — so the addition is additive, not the design-decision-9
contradiction hazard. It is phrased as a continuation ("a claim must
also stand on its own in what it asserts, not only in what it names")
because three visibly-separate self-tests is itself the shape decision
9 warns leads to coin-flipping. Decision 15's post-ticket-007 audit
note reserved this exact slot.

**The self-test-1 tension.** The new rule sits *after* self-test 1
(atomicity) in the prompt, and self-test 1 independently drives toward
re-splitting a severed attribution: "Acme's margins improved on lower
costs." → "Acme's margins improved." + "Acme has lower costs." — the
exact bare form the new rule forbids. Decision 15 handled its analogous
closure case with a carve-out *inside* self-test 1; ticket 009 relies
on the worked example alone — deliberate, per decision 9 (no reflexive
overlapping rule), and the isolated evidence supports it (0/6 bad
shatter on the clean constructed fixture). But the residual
split-vs-keep-whole non-determinism there, and the many-item-run-on /
in-context MEMO-002 underfiring (see Residual), are this unresolved
tension surfacing. A future editor weighing a self-test-1 carve-out
should know the worked example was tried first and holds the invariant.

**What was observed** (real, `MEMO-002` Borealis "Business Profile"
section, 2026-08-27): a sentence of the shape "…improving profitability
and cash generation due to [four operational drivers], good cost
management and pricing initiatives." split the trailing drivers off as
bare claims such as "Borealis has pricing initiatives." — claims no
source document would confirm or deny, guaranteed found=False.

**Regression baseline — the embedded worked examples.** This edit
added three: (a) the causal-attribution example ("Acme's margins
improved on lower costs and better pricing." → the 2-way attributed
split, NOT the bare "Acme has lower costs." forms); (b) the
attached-attribute carve-out ("Acme is well positioned, with $2bn of
committed liquidity." → "Acme is well positioned." standalone, "$2bn"
split off); (c) the leave illustration ("Acme is well run." →
unchanged). Example 2 ("Acme has a market cap of EUR 5bn.") and the
garbled "wth" example ("The issuer has assets of $40bn.") are
themselves "X has Y" forms whose predicate *is* the assertion — the
built-in over-firing guards. Canonical list of all 14, with expected
outputs: `docs/prompt-verification-log.md` → "Embedded worked
examples".

**Verification** (live, `openai/gpt-oss-120b`, 3 runs/case; pre-edit
baseline against `1b7019d`): the bad attribution-dropping shatter on
the clean constructed fixture went 1/3 → 0/6; every over-firing guard
(pure-qualitative, borderline-"has", split-with-vague-half, two
out-of-sample leave cases) PASSed unchanged; the restore direction
held or improved out of sample; the 11 embedded examples regressed
only in cosmetic wording. In-context on `MEMO-002` the fix does **not**
reliably fire (ticket 011's granularity ceiling — the isolated
fixtures carry the verdict). Full run-by-run results:
`docs/prompt-verification-log.md` → "Decision 16 / ticket 009".

**Residual.** The isolated many-item run-on is the recorded measured
residual: the exact bare "has" form only fell to ≈1/3 (not 0/3) and
the drivers are still severed ~2/3 into thin verbs — §4.2's wording
does not reach a long list of siblings read out of context; logged,
not chased. The spec pre-authorised a "said something checkable that
this thing was part of" tightening for this underfire — **declined**,
because that phrasing reintroduces the checkability framing the rule
was built to avoid; the next editor should not reach for it. A second:
the new rule nudges a genuine "X has Y" ("Acme has a revolving credit
facility.") toward a redundant pair with a fuller version of itself
("… a EUR 500m revolving credit facility.") 2/3 runs — one contained
in the other, which the prompt forbids. This containment is silent
(the ticket-007 WARNING fires only on byte-identical text) and costs a
second `claim_id`, shortlist and evidence run downstream; a candidate
for ticket 010's always-on check. In-context reliability rides on
ticket 011's granularity ceiling staying open.

**Companion nets.** Ticket 010 (open — an always-on unmatchable-claim
check that will own the standalone-evaluative sentence that keeps its
bare predicate); ticket 011 (long-section granularity ceiling).
Decision 16 is one of several related prompt edits, landed separately
so a regression stays
attributable to one.
