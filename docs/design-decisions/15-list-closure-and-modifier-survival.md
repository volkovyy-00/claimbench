# Design decision 15 — full narrative

Split out of `CLAUDE.md` to keep that file's per-session load down; the
condensed rule and current residual now live in `.claude/rules/golden-set-pipeline.md` under design decision 15.
This file holds the full observed-failure narrative behind it. Live
verification run-by-run: `docs/prompt-verification-log.md` →
"Decision 15 / ticket 008" and "Embedded worked examples".

---

**A sentence that closes over a list stays one claim; a list it does
not close over still splits; and a partial qualifier survives the
split.** `extract_atomic_claims`'s prompt gained one inline
`Exception:` clause (immediately after the comparator-entity
exception), a short modifier-survival paragraph (placed just after the
adjective counterweight), and a one-clause parenthetical on the first
self-test; the docstring gained a matching summary paragraph. No
code-path change, no committed test — a prompt-text edit verified by
live runs (see `.claude/rules/testing.md` → "Testing convention").

**The rule is a principle, not a trigger-word list.** A sentence closes
over a list when it asserts something true only of the whole set — a
role/right/restriction defined over a named set, or an outcome
contingent on several conditions holding together. "limited to",
"only", "solely", "jointly", "provided that" are illustrations in the
prompt; their absence does not make a closed list splittable, and the
prompt says so outright. This follows decision 5's "general principle,
not a fixed word list" convention. The exception carries an explicit
"a plain enumeration is NOT a closed list" carve-out ("Acme's largest
customers are Globex, Initech and Umbrella." splits, one claim per
member) so it does not swallow ordinary member-lists.

Separately — and this is a distinct concern, not part of closure — when
a list *does* split, every qualifier on an item rides onto that item's
claim: "mainly"/"primarily" onto the leading member's claim, and a
qualifier that ranks other members *below* that leading one ("but to a
lesser extent", "less commonly") onto those trailing members' claims
only, never the leading member's. Closure keeps a list whole;
modifier-survival is about not losing information when a list splits —
opposite situations, so they are two separate paragraphs, not one
merged "sometimes keep lists whole" rule. (The prompt's parenthetical
originally lumped "primarily" in with the trailing-qualifier examples
before the "never the leading member" clause — a latent
self-contradiction the `/final-verification` audit caught; corrected to
name the head- and tail-qualifier cases separately. Head-qualifier
retention is covered by the general rule and the worked example, not by
a dedicated live test.)

**What was observed** (real, on the `MEMO-002` Borealis "Business
Profile" section, 2026-08-27, reproduced in the pre-edit baseline
2026-08-28): the sentence "Gantry's management role is limited to
approval of budgets and other approval actions (such as asset
acquisitions and non-budgeted disposals)" split, 2 runs in 3, into a
claim asserting the role is limited to budgets alongside a claim
asserting it includes more — which, matched one at a time, contradict
each other. A ratings-upgrade forecast contingent on several conditions
jointly collapsed 3/3 to "Debt reduction will likely result in an
upgrade." And "but to a lesser extent" was dropped 3/3 from a
fleet-supplier list ("primarily by two suppliers but to a lesser extent
also by three others"), leaving the tail suppliers reading as equal to
the head.

**Why an `Exception:` clause and not a gate.** An ordered "run the
closure test, then the atomicity test, the first can short-circuit the
second" structure is two rules the model must reconcile — the
design-decision-9 hazard (that shape once returned `low`, `low`, `[]`
on three identical runs). Instead the closure rule sits inline beside
the comparator-entity exception, and the first self-test simply names
the closure case as out of its scope.

**Scope — AC-1 was split out to EV-7.** The originating ticket
also asked that a plain member-list return one claim per member. The
pre-edit baseline showed that already works *in isolation* ("Acme's
largest customers are Globex, Initech, and Umbrella." → 3 claims, 3/3)
but *not* inside a large section: the real three-member "Key customers
are …" sentence stayed a single claim 3/3 in
the 4,400-character MEMO-002 section. Diagnosis (8 context sizes, 3
runs each; the dose-response table is in EV-7): a clean
dose-response — the list splits at ≤ ~230 characters and stops by
~1,270 — i.e. output-volume granularity
economization, not a closure-semantics effect (the isolated cases
split), not folded-scalar heading glue (glued vs. blank-line
identical), not phrasing. That ceiling is a general property of
long-section extraction and is now EV-7; this edit is closure +
modifier only. (Incidentally, post-edit the "Key customers" sentence
*does* split 3/3 in section context — the closure exception's
plain-enumeration carve-out example raised copula-list salience enough
to beat the economization for that one shape. The ceiling itself
(EV-7) is unchanged, e.g. "the two regional units" still
collapses to one claim.)

**Regression baseline — the embedded worked examples.** This edit
added the audit-committee closed/open contrast pair, the hedged
ratings-upgrade forecast, and the fleet-supplier partial modifier to
decision 5's standing set. The canonical list (now 14 examples, with
expected outputs) and the rule that any future edit re-verifies all of
them in both directions live in `docs/prompt-verification-log.md` →
"Embedded worked examples".

**Verification** (live, `openai/gpt-oss-120b`, 3 runs/case; pre-edit
baseline against `bf31154`): the closed set, hedged forecast and
partial-modifier fixtures went 0/3 → 3/3; the open-list counterweight,
the comparator-entity interaction, the copula member-list guard and
the 8 embedded examples held with no regression; the real in-context
`MEMO-002` Gantry sentence improved only marginally (EV-7's
granularity ceiling degrades it — the isolated `G1-closed` fixture
carries the verdict). Full run-by-run results and the focused
end-to-end run: `docs/prompt-verification-log.md` → "Decision 15 /
ticket 008".

**Residual.** Two out-of-sample closed lists are the measured
residuals: an "only … and …" permission still reads as two permissions
2 runs in 3, and a "limited to hedging FX ($2bn notional) and …"
mandate splits the `$2bn notional` off as its own attached-attribute
claim 1 run in 3 (closure vs. the attached-measured-attribute rule) —
logged with their failure modes, not chased. In-context, the mangled
upgrade-rationale sentence and the granularity ceiling (EV-7)
are unchanged.

**Companion nets.** Ticket 009 adds a predicate-retention bar to the
*second* self-test — a split piece must keep the predicate its source
sentence gave it rather than decaying to a bare "X has Y" (see decision
16); this edit's parenthetical is on the *first* self-test — different
self-tests, different failure classes. EV-6
is the always-on unmatchable-claim check that catches list shapes the
prompt rule misses. EV-7 owns AC-1 and the long-section
granularity ceiling split out of this ticket. Decision 15 is one of
several related prompt edits, landed separately so a regression stays
attributable to one.
