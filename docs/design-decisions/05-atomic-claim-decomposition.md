# Design decision 5 — full narrative

Split out of `CLAUDE.md` to keep that file's per-session load down; the
condensed rule and current residual live there under design decision 5.
This file holds the full observed-failure narrative behind it. Live
verification run-by-run: `docs/prompt-verification-log.md` → the
"Embedded worked examples" and "Decision 5 / ticket 007" sections.

---

**`extract_atomic_claims`'s prompt states a general principle, not a
fixed word list** — a claim is atomic only if it can't decompose into
two independently verifiable facts, "regardless of what word or
punctuation joins them" (connector words are examples, not an
exhaustive trigger list; verified against phrasings not in the prompt).
It also covers a fact *attached inside a phrase* ("X is a listed
operator with a market cap of €5bn" → two claims), counterweighted so
the model doesn't shred a description into separate adjectives ("a
listed French real estate investment trust" stays one claim), plus a
self-test ("could a source confirm one part while saying nothing about
another?"). **Standing convention:** the worked examples embedded in
this prompt are the regression baseline for any future edit — re-verify
all of them, plus out-of-sample cases in both directions. Over- and
under-splitting are both real failure modes and it's easy to overfit.
The canonical list (14 examples, expected outputs) and every prompt
edit's run-by-run verification are in `docs/prompt-verification-log.md`.

**Garbled memo text is a separate failure mode with a separate fix.**
Measured, 3 runs each: a single typo anywhere ("...a top-5 regional
lender wit assets of $40bn.", or "iis" with a clean "with") splits
perfectly every time — the model silently rewrites the attachment to
"has". *Two* typos in one sentence ("iis" + "wit") flipped it into
literal-copy mode: it still found both facts but emitted the second as
the fragment "The bank wit assets of $40bn.", and on one run restated
the whole input sentence as claim 2 (the compound-claim bug returning).
The trigger was the second typo, not that "wit"/"win" are real words.
Cause: "Preserve the original wording as closely as possible" outranked
grammar once the text looked corrupted. **Fix, a product decision the
user made explicitly:** the prompt now permits supplying the few
connecting words that make a claim a sentence (leaving the text's own
misspellings alone), and forbids a claim that restates the whole input
or contains another returned claim. Chosen over word-for-word fidelity
because a fragment is harder to match and review — so `claim_text` is
NOT guaranteed to be a verbatim copy of the memo.

**Residual:** on heavily garbled input the light-rewording fires ~2
runs in 3 ("The operator iss a leading fibre provider serviing 3.4m
homes." → usually "The operator serves 3.4m homes.", occasionally the
raw "serviing"). Both facts are extracted either way; only one claim's
wording degrades. Model variance at a genuinely ambiguous input; the
harmful version (whole-sentence restatement) did not recur post-fix.

**Comparator entities inside a ranking/comparison clause are exempt
from the attached-attribute rule.** Observed live: "Acme ... is the 2nd
largest operator worldwide (after Globex (Grades: 2/4, TFL: £100m) out
in the US)." split Acme's own facts correctly but also spun off
standalone claims about Globex's grades and location — a competitor
Acme's source PDFs can never substantiate, so guaranteed `found=False`, pure review noise.
Not wrong per the attached-attribute rule as written (genuinely
separable facts), just facts about the wrong entity for this corpus.
**Product decision the user made explicitly:** when the other entity's
attributes appear only inside a comparison naming the section's actual
subject, keep them on one claim about the subject. If that entity is
genuinely the subject of its own sentence elsewhere, its attributes
still split normally.

**A split can strand a pronoun or bare pointing phrase from its
antecedent.** Observed live: "...the asset was under significant
pressure until Dec'23. However, it has stabilized as of Dec'24 vs
Dec'23." split into two claims, the second keeping "It" verbatim with
no source able to tell what it meant. Downstream, the evidence step
(decision 9) had no subject to test candidates against and fell back
to keyword-matching "stable"/"stabilised": five unrelated metrics
matched at medium/high confidence (net debt/EBITDA, IFRS net debt,
ICR, a portfolio yield, pipeline NRI) — five confidently wrong rows.
**Fix, a product decision the user made explicitly:** when a split
separates a pronoun or bare pointing phrase from what it refers to,
the model replaces it with the plainest naming the text itself already
uses — copying wording, not adding information. Minimal: name or
description only, never re-attaching a further fact (that would trip
the attached-attribute rule and manufacture an unasserted claim).
Resolves to its best guess when the text names more than one candidate
(a wrong guess is visible review noise, same reasoning as decision 9);
left unresolved only when the text names nothing it could refer to,
never an invented subject. Backstop: a cheap non-LLM diagnostic logs
(not drops) any claim starting with a bare pronoun/pointing word, so a
case the prompt rule misses stays visible.

**Residual, confirmed against a real 822-chunk corpus:** this
fixes the *claim's* missing subject, not a pre-existing *vague* one.
Re-running the fixed claim end-to-end surfaced a genuinely new correct
match invisible to the old bare-pronoun version ("combined with the
stabilisation of retail asset values...") — but the wrong "IFRS net
debt was stable" match still came back once at high confidence,
because the claim's own subject ("the asset") is itself a generic
unnamed noun phrase and decision 9's same-subject check is imperfect
against any generic subject, pronoun or not. Resolving further (to a
specific asset name) would invent information the text doesn't carry,
which this fix refuses to do — a separate, pre-existing problem (the
source claim is under-specified), needing its own product decision,
not a reflexive prompt tweak.

**Pointing words beyond the subject, and names carried only in a
heading/label line (ticket 007).** The fix above resolved a claim
whose *subject* was a bare pronoun; two narrower gaps remained, both
closed in the prompt (the regex backstop untouched):

1. A pointing word that is **not** the subject. "Acme uses this
   process to retain a competitive advantage" names its subject, so
   the old self-test ("would a reader know what this claim is
   *about*?") passed it while "this process" still pointed at nothing.
   The genuinely uncovered case.
2. A referent named only in a **heading or label line on its own line
   above the paragraph** — e.g. `## Revolving Credit Facility` then a
   paragraph of "The facility ..." sentences. Nothing told the model
   that line was borrowable text.

Fix: the pointer rule now (a) names a heading/label line as borrowable,
and (b) applies to a pointing word anywhere in the claim, not only its
subject — the self-test widened to "does every part of this claim have
a referent a reader could identify?". Added one fictional worked
example (a "Change-of-Control Clause" heading whose name both "The
clause" and "It" borrow); all prompt illustration entities stay
fictional (Acme/Globex), never a client name. Over-resolution
counterweights unchanged: replacement is name or short description
only, never a further fact, and a referent the text never names is
left as-is.

`_STRANDED_POINTING_WORD_RE` was deliberately **not** widened to bare
definite noun phrases — tried, and it flagged healthy claims. "The
clause was amended in 2024." is fine; "The clause caps liability."
needs to know which clause; syntactically identical, so a regex can't
tell them apart. Resolution rests on the prompt alone; the regex still
catches only claim-initial pronouns.

Resolving a pointer can also make two originally-distinct claims come
back with **identical text** — logged at WARNING (repeated string +
count), both copies kept, never collapsed. If the claims really are
redundant, the reviewer deletes one row (`tag="rejected"`, decision
3); if extraction collapsed two facts that should have stayed
separate, that's not fixable in the review spreadsheet (merge already
happened) — the fix is pre-run, with the WARNING and
`preview_claim_splits` (decision 14) surfacing it. That second branch
is why the check is warn-only and fires at extraction time.
Downstream, two identical claim strings still get separate `claim_id`s:
two BM25 shortlists, two evidence runs (double LLM cost), two identical
row-sets.

A section whose headings carry names a claim needs must reach
extraction with the heading intact: a folded YAML scalar (`>`) with no
blank line after a heading glues it onto the next sentence. Use a
literal block (`|`) or a blank line after each heading — see
`memos.yaml.example`.

Run-by-run verification (the primary new capability, the
over-resolution counterweights, the embedded-example sweep, the
identical-text WARNING behaviour): `docs/prompt-verification-log.md` →
"Decision 5 / ticket 007".

**Residual — the motivating case is not fixed.** The Borealis
section names its cross-acceleration clause once, by an inline
"Cross-acceleration Clause:" label at the head of a paragraph that
yields several claims. Measured: 3 of that paragraph's claims read a
bare "The clause ..." per run, identical before and after this edit
(3/3/3 → 3/3/3). An inline "Label:" is not an own-line heading, so the
heading clause doesn't reach it; and the "resolve from words earlier
in the text" rule doesn't, in practice, propagate a name to every
sibling claim of a multi-claim paragraph. No deterministic backstop
for a mid-claim "The clause" (unlike a claim-initial pronoun), so
these pass silently and can produce confidently-wrong evidence rows —
the five-wrong-metrics shape again. The forcing wording ("every claim
from a paragraph must carry the paragraph's name for the thing") is a
blunt instrument that reintroduces the over-resolution the
counterweights guard against and case 5 passes. Ticket 010 (an
always-on unmatchable-claim check in `build_golden_set_draft`) is the
designed net; `preview_claim_splits` (decision 14) the opt-in human
catch. Best-evidenced residual here, not a target for another prompt
round. Separately, `_STRANDED_POINTING_WORD_RE` still false-positives
on an expletive "it" ("It is the policy of Borealis to offer..."),
pre-existing, left alone.
