# Design decision 14 — full narrative

Split out of `CLAUDE.md` to keep that file's per-session load down; the
condensed rule and current residual now live in `.claude/rules/golden-set-pipeline.md` under design decision 14.
This file holds the full text of that entry as it stood before the split.

---

**`preview_claim_splits` is a separate, opt-in pass, never wired into
the real evidence-matching path.** The batch run can take hours
(decision 7); an `input()` in that loop would break an unattended run.
So the preview stays outside
`build_golden_set_draft`/`build_golden_set_batch` and **re-runs
`extract_atomic_claims` itself** rather than being handed the claims a
later `build_golden_set_draft` will produce — same reason, the real
path stays byte-for-byte as it was.

**What a clean preview buys is narrower than it looks.** It checks the
*extraction prompt against this section's text*, not the next run's
exact output. A systematically bad split (over/under-splitting, a
stranded pronoun, a wrong-entity claim) is a property of prompt +
text, recurs across runs, and the preview catches it. A one-run
wording fluke it won't: `temperature≈0` is not fully deterministic
(decisions 5 and 9). So ticket 005's "the same wording the real run
would produce" is approximate — inherent to "don't touch the real
path," not a defect.

**The `RUN_CLAIM_SPLIT_PREVIEW` cell is guarded by a plain module
flag, not `if __name__ == "__main__"`.** That guard is what `python
golden_set_pipeline.py` runs, and ticket 005 forbids the script
pausing; a bare top-level call is worse — it fires on `import`,
hanging the test suite on `input()` and possibly making a real LLM
call. The flag (`RUN_CLAIM_SPLIT_PREVIEW = False`, flipped by hand in
a notebook) is a toggle, not config surface — no `memos.yaml` key, no
env var. `tests/test_claim_split_preview.py` asserts the default is
`False` and parses the module source to assert every
`preview_claim_splits(...)` call sits inside `if RUN_CLAIM_SPLIT_PREVIEW:`
— structural, since a live "import and see if it prompts" test passes
for the wrong reason whenever the stray LLM call fails.

Contrast EV-6: an always-on, non-interactive check
*inside* `build_golden_set_draft`'s unguarded gap between extraction
and the evidence loop; decision 14 adds nothing to that function.
