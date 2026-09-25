"""
The two regexes EV-16 rewrote to remove super-linear backtracking (Sonar
python:S8786): the fence pattern in _strip_to_json and _ATX_CLOSE_RE.

Each gets two checks. Equivalence: the rewrite gives exactly what the old
pattern gave, on every short string over an alphabet chosen to hit its edge
cases (the old code is kept below as the reference; short inputs keep its
cost negligible). Speed: the rewrite stays fast on the input that made the
old pattern slow, so re-adding the fence's \\s* or dropping the lookbehind
fails here instead of passing ruff, basedpyright and every other test.
"""
import itertools
import re
import time

import golden_set_pipeline as gsp

# The patterns as they were before EV-16.
_OLD_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_OLD_ATX_CLOSE_RE = re.compile(r"\s+#+$")

# One call on the slow input. The rewritten patterns take about a millisecond
# there and the old ones several seconds, so the budget is wide both ways.
_TIME_BUDGET_S = 0.5


def _old_strip_to_json(text: str) -> str:
    """_strip_to_json as it was before EV-16, unchanged but for the name."""
    text = text.strip()
    fence_match = _OLD_FENCE_RE.search(text)
    if fence_match:
        return fence_match.group(1).strip()

    start_candidates = [i for i in (text.find("["), text.find("{")) if i != -1]
    if not start_candidates:
        return text
    start = min(start_candidates)
    end = max(text.rfind("]"), text.rfind("}"))
    if end == -1 or end < start:
        return text
    return text[start : end + 1]


def _all_strings(pieces: list[str], max_len: int):
    for n in range(max_len + 1):
        for combo in itertools.product(pieces, repeat=n):
            yield "".join(combo)


def _seconds(fn, arg) -> float:
    start = time.perf_counter()
    fn(arg)
    return time.perf_counter() - start


def test_strip_to_json_matches_the_old_pattern():
    pieces = ["```", "`", "json", " ", "\n", "{", "}", "a"]
    for text in _all_strings(pieces, 5):
        assert gsp._strip_to_json(text) == _old_strip_to_json(text), repr(text)


def test_atx_close_matches_the_old_pattern():
    pieces = ["#", " ", "\t", "\n", "a", "C"]
    for name in _all_strings(pieces, 7):
        assert gsp._ATX_CLOSE_RE.sub("", name) == _OLD_ATX_CLOSE_RE.sub("", name), repr(name)


def test_strip_to_json_is_fast_on_an_unclosed_fence():
    # A reply cut off inside a fence: the old pattern took ~4 s here, and its
    # time grows with the cube of the whitespace run.
    text = "```" + " " * 2_000 + "x"
    assert _seconds(gsp._strip_to_json, text) < _TIME_BUDGET_S


def test_atx_close_is_fast_on_a_long_whitespace_run():
    # No '#' needed: the old pattern retried from every position inside the
    # run, ~5 s here, growing with the square of its length.
    name = "a" + " " * 20_000 + "b"
    assert _seconds(lambda s: gsp._ATX_CLOSE_RE.sub("", s), name) < _TIME_BUDGET_S
