"""
Tests for ticket 007's two deterministically-testable surfaces in
extract_atomic_claims: (a) the identical-claim-text warning (two or more
returned claims with the same text after stripping), and (b) the boundary
assertion that `_STRANDED_POINTING_WORD_RE` was NOT widened to bare
definite noun phrases. The prompt-wording half of ticket 007 is verified
by live runs, not here. See local ticket 007 (archived in the maintainer's
notes repo) and CLAUDE.md design
decision 5's "Pointing words beyond the subject" addendum.

Mocks golden_set_pipeline.call_llm so no real LLM provider is contacted,
consistent with this repo's testing convention (CLAUDE.md, "Testing
convention").
"""
import json
import logging

import pytest

import golden_set_pipeline as gsp

GOLDEN_SET_LOGGER = gsp.logger.name


def _mock_claims(monkeypatch, claims):
    monkeypatch.setattr(gsp, "call_llm", lambda *a, **k: json.dumps(claims))


def _duplicate_warnings(caplog):
    return [
        r
        for r in caplog.records
        if "identical text" in r.getMessage() and r.levelno == logging.WARNING
    ]


def _stranded_pointing_warnings(caplog):
    return [
        r
        for r in caplog.records
        if "unresolved pronoun/pointing" in r.getMessage()
        and r.levelno == logging.WARNING
    ]


def test_returns_both_duplicate_claims_uncollapsed(monkeypatch):
    _mock_claims(monkeypatch, ["A.", "A.", "B."])

    result = gsp.extract_atomic_claims("section text", object())

    assert result == ["A.", "A.", "B."]


def test_warns_once_per_duplicated_group_with_text_and_count(monkeypatch, caplog):
    _mock_claims(monkeypatch, ["A.", "A.", "B."])
    caplog.set_level(logging.WARNING, logger=GOLDEN_SET_LOGGER)

    gsp.extract_atomic_claims("section text", object())

    warnings = _duplicate_warnings(caplog)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "'A.'" in message
    assert "2 claims have identical text" in message


def test_warns_with_count_for_three_or_more_repeats(monkeypatch, caplog):
    _mock_claims(monkeypatch, ["A.", "A.", "A."])
    caplog.set_level(logging.WARNING, logger=GOLDEN_SET_LOGGER)

    result = gsp.extract_atomic_claims("section text", object())

    assert result == ["A.", "A.", "A."]
    warnings = _duplicate_warnings(caplog)
    assert len(warnings) == 1
    assert "3 claims have identical text" in warnings[0].getMessage()


def test_no_warning_when_all_claims_distinct(monkeypatch, caplog):
    _mock_claims(monkeypatch, ["A.", "B.", "C."])
    caplog.set_level(logging.WARNING, logger=GOLDEN_SET_LOGGER)

    gsp.extract_atomic_claims("section text", object())

    assert _duplicate_warnings(caplog) == []


def test_strip_is_applied_before_comparison(monkeypatch, caplog):
    _mock_claims(monkeypatch, [" A.", "A."])
    caplog.set_level(logging.WARNING, logger=GOLDEN_SET_LOGGER)

    result = gsp.extract_atomic_claims("section text", object())

    assert result == [" A.", "A."]  # returned verbatim, not stripped
    assert len(_duplicate_warnings(caplog)) == 1


@pytest.mark.parametrize("claims", [[], ["only one claim."]])
def test_empty_or_single_claim_list_does_not_warn_or_crash(monkeypatch, caplog, claims):
    _mock_claims(monkeypatch, claims)
    caplog.set_level(logging.WARNING, logger=GOLDEN_SET_LOGGER)

    result = gsp.extract_atomic_claims("section text", object())

    assert result == claims
    assert _duplicate_warnings(caplog) == []


def test_both_warning_loops_fire_together_for_identical_stranded_claims(monkeypatch, caplog):
    """
    The only point where the two ticket-007 commits' code paths meet: two
    identical claims that also each start with a bare pointing word. The
    duplicate-claim loop fires once (one duplicated group), and the
    stranded-pointing-word loop fires once per copy.
    """
    _mock_claims(monkeypatch, ["It rose.", "It rose."])
    caplog.set_level(logging.WARNING, logger=GOLDEN_SET_LOGGER)

    result = gsp.extract_atomic_claims("section text", object())

    assert result == ["It rose.", "It rose."]
    assert len(_duplicate_warnings(caplog)) == 1
    assert len(_stranded_pointing_warnings(caplog)) == 2


def test_stranded_pointing_word_regex_still_matches_only_claim_initial_pronouns():
    """
    Ticket 007 explicitly does NOT widen this regex to bare definite noun
    phrases (tried against the same claims, it flagged healthy ones too).
    Guard that decision so a later edit doesn't quietly broaden it.
    """
    assert gsp._STRANDED_POINTING_WORD_RE.match("It recovered by year-end.")
    assert gsp._STRANDED_POINTING_WORD_RE.match("The latter figure rose.")
    assert not gsp._STRANDED_POINTING_WORD_RE.match("The clause caps liability at $5m.")
    assert not gsp._STRANDED_POINTING_WORD_RE.match("Acme uses this process.")
