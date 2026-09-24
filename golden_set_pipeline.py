# %% [markdown]
# # Golden Set Draft Builder — Claim ↔ Evidence Pipeline
#
# This notebook turns pairs of (human-written memo section, source
# documents) into a draft "golden set" of atomic claims matched to
# supporting evidence, ready for you to review and correct by hand.
#
# The pipeline is two stages joined only by a reviewable Markdown "claims
# file" (design decision 17 in CLAUDE.md):
#
# - **`extract`** reads `memos.yaml`, splits each section's human-written
#   text into atomic claims (section 4's `extract_atomic_claims`), and
#   writes one `claims/<memo_id>.md` per memo — nothing else. You then
#   review or hand-edit those files (or write one from scratch, skipping
#   `extract` entirely — a hand-authored claims file is a first-class
#   input, not a fallback).
# - **`build`** reads only `claims/*.md` (never `memos.yaml`), matches each
#   claim against the source PDFs, and produces the golden-set draft.
#
# `python golden_set_pipeline.py extract`, then `... build` (or no
# argument, which also runs `build`) — see section 8's demo cell and
# CLAUDE.md's "Commands".
#
# ## How to plug in your own data
#
# 1. **Set environment variables** before running (e.g. in a `.env` file
#    loaded with `python-dotenv`, or exported in your shell):
#    - `LLM_BASE_URL` — e.g. `https://api.your-onprem-endpoint.com/v1`
#    - `LLM_API_KEY`
#    - `LLM_MODEL` — the model name your endpoint expects, e.g. `gpt-4o` or
#      whatever your on-prem deployment calls itself
#
# 2. **Describe your memos in `memos.yaml`** (copy `memos.yaml.example`),
#    one folder of source PDFs and one or more named sections of
#    human-written text per memo, then run `extract` to turn each memo into
#    a `claims/<memo_id>.md` file. Review or edit the claims there — this is
#    the point to catch a bad split before any evidence-matching API spend.
#    (You can also write a claims file by hand, following the grammar in
#    `claims.example.md` and docs/pipeline-overview.md's "How to write a
#    claims file" section, and skip `extract` for that memo entirely.)
#
# 3. **Run `build`.** It reads every `.md` file directly in `claims/`,
#    matches each claim against that memo's source PDFs, exports for
#    review, and you correct the result in Excel and re-import — see the
#    demo cells at the bottom for the exact calls.
#
# ## Requirements
#
# `pip install pandas requests openpyxl pyarrow rank_bm25 pypdf pdfplumber pyyaml`
#
# ## Opening this as a notebook
#
# This file uses `# %%` cell markers (Jupytext "light" format). VS Code's
# Python extension and PyCharm both recognize these and let you run cell by
# cell directly. To get a real `.ipynb`: `pip install jupytext` then
# `jupytext --to notebook golden_set_pipeline.py`.

# %%
import json
import logging
import os
import re
import sys
import uuid
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()  # reads .env in the current directory, if present, into os.environ

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("golden_set")

# %% [markdown]
# ## 0. Loading real documents (PDF → text)
#
# The rest of this file works on plain text (`doc_text`), not PDF files
# directly. `load_pdf_text` is the one place that turns a PDF into text —
# use it to build `source_documents` from your actual files.

# %%
def load_pdf_text(pdf_path: str) -> str:
    """
    Extracts all text from one PDF file, page by page, joined with a blank
    line between pages.

    This does plain text extraction only, no OCR. If a PDF is a scanned
    image rather than a text-based PDF, this will return little or no
    text — scanned documents need an OCR step (e.g. pytesseract) first,
    which is out of scope here.

    pypdf can garble table-heavy layouts (columns running together, row
    order scrambling) — financial source documents are often mostly
    tables, which is where most of the actual figures live. If pypdf's
    output looks scrambled on your real documents, try
    load_pdf_text_pdfplumber instead (same signature, drop-in swap,
    generally better on tables but slower and needs the pdfplumber
    package). pypdf stays the default here for speed/light dependencies.

    Worked example — one section with 5 source PDFs:

        pdf_paths = [
            "sources/10-K_2025.pdf",
            "sources/investor_presentation.pdf",
            "sources/industry_report.pdf",
            "sources/credit_agreement.pdf",
            "sources/management_bio.pdf",
        ]
        source_documents = [
            (os.path.basename(p), load_pdf_text(p)) for p in pdf_paths
        ]
        # source_documents is now a list of (doc_id, doc_text) pairs, ready
        # to pass into build_golden_set_draft / build_golden_set_batch.
    """
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages_text)


def load_pdf_text_pdfplumber(pdf_path: str) -> str:
    """
    Extracts all text from one PDF file using pdfplumber instead of pypdf,
    page by page, joined with a blank line between pages. Same signature
    and behavior as load_pdf_text — a drop-in swap for it.

    Prefer this over load_pdf_text specifically when your source documents
    are table-heavy (financial statements, schedules, exhibits): pdfplumber
    does layout-aware extraction and generally preserves table structure
    much better than pypdf, which tends to run columns together or
    scramble row order on tabular content. It's slower and pulls in more
    dependencies (Pillow, pdfminer.six), which is why pypdf stays the
    default in load_pdf_text — reach for this one when pypdf's output on
    your real documents turns out to be unusable.

    Also plain text extraction only, no OCR — same scanned-PDF caveat as
    load_pdf_text.
    """
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        pages_text = [page.extract_text() or "" for page in pdf.pages]
    return "\n\n".join(pages_text)


def warn_if_text_suspiciously_short(doc_id: str, text: str, min_chars: int = 200) -> None:
    """
    Logs a warning if a document's extracted text looks suspiciously short
    to be a real source document (under min_chars characters).

    This deliberately does NOT try to diagnose a cause (scanned PDF, wrong
    file, extraction failure, or the document genuinely just being this
    short) — it only flags that the result is worth checking before you
    trust anything downstream. A near-empty extraction is otherwise silent:
    load_pdf_text / load_pdf_text_pdfplumber don't raise on it, chunking
    and BM25 just quietly produce nothing for that document, and the first
    visible symptom tends to be "this document never shows up in results" —
    by which point it's easy to mistake for a retrieval question rather
    than an extraction one. Call this right after building doc_texts, for
    every document, so a bad extraction gets flagged immediately.
    """
    if len(text) < min_chars:
        logger.warning(
            "doc_id=%r produced only %d character(s) of extracted text (expected at least %d "
            "for a real source document) — verify its content before trusting downstream results.",
            doc_id,
            len(text),
            min_chars,
        )


# %% [markdown]
# ## 1. LLM client — isolated behind one function
#
# `LLMClient` just holds config. `call_llm` is the *only* function that
# knows the HTTP shape of the LLM API. It currently speaks the
# OpenAI-compatible chat-completions dialect, which is what most on-prem /
# self-hosted model servers (vLLM, TGI, LiteLLM proxies, etc.) expose. To
# point this at a different API shape later, edit the inside of
# `call_llm` only — every other function calls through this one.

# %%
@dataclass
class LLMClient:
    """Holds connection config for the LLM endpoint. Build with LLMClient.from_env()."""

    base_url: str
    api_key: str
    model: str

    @classmethod
    def from_env(cls) -> "LLMClient":
        """Reads LLM_BASE_URL, LLM_API_KEY, LLM_MODEL from the environment."""
        base_url = os.environ.get("LLM_BASE_URL")
        api_key = os.environ.get("LLM_API_KEY")
        model = os.environ.get("LLM_MODEL")
        if not base_url or not api_key or not model:
            missing = [
                name
                for name, val in [
                    ("LLM_BASE_URL", base_url),
                    ("LLM_API_KEY", api_key),
                    ("LLM_MODEL", model),
                ]
                if not val
            ]
            raise EnvironmentError(
                f"Missing required environment variable(s): {', '.join(missing)}"
            )
        return cls(base_url=base_url, api_key=api_key, model=model)


def call_llm(prompt: str, llm_client: LLMClient, temperature: float = 0.0, max_tokens: int = 4096) -> str:
    """
    Sends one user-role prompt to the configured LLM endpoint and returns the
    raw text of the response.

    This is the single point of contact with the LLM provider's HTTP API.
    Swapping providers (e.g. moving to an on-prem endpoint with a different
    request/response shape) should mean editing only the body of this
    function — nothing else in this file constructs an HTTP request.

    max_tokens defaults to a generous 4096 rather than being left unset.
    This matters specifically for "reasoning" models (e.g. gpt-oss,
    o-series, some Claude/Gemini configurations): observed in practice
    against OpenRouter's openai/gpt-oss-120b, an unset/too-low max_tokens
    lets hidden reasoning tokens consume the entire completion budget,
    producing either JSON truncated mid-object or an empty
    message.content — the latter isn't a JSON problem at all and retrying
    the same request with the same budget doesn't reliably fix it, whereas
    raising the budget does.

    Empty message.content has a second, unrelated cause worth telling apart:
    on OpenRouter, finish_reason="error" (as opposed to "length") means the
    upstream provider itself failed — rate limit, mid-stream disconnect,
    context length exceeded, etc. — and OpenRouter attaches the real cause
    in a choices[0].error object (code/message/metadata.error_type). That's
    a transient/provider problem, not a token-budget one, so this function
    surfaces that error detail directly rather than defaulting to the
    max_tokens explanation when it's present.
    """
    url = f"{llm_client.base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {llm_client.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": llm_client.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    choice = data["choices"][0]
    content = choice["message"]["content"]
    if content is None:
        finish_reason = choice.get("finish_reason")
        error_info = choice.get("error")
        if error_info:
            # OpenRouter-specific: finish_reason="error" means the upstream
            # provider itself failed (rate limit, mid-stream disconnect,
            # context length exceeded, etc.) — a transient/provider problem,
            # not a token-budget one. The actual cause is in this "error"
            # object, not diagnosable by raising max_tokens.
            raise ValueError(
                f"LLM response had empty content due to a provider-side error "
                f"(finish_reason={finish_reason!r}): {error_info.get('message')!r} "
                f"(code={error_info.get('code')!r}, "
                f"error_type={error_info.get('metadata', {}).get('error_type')!r}). "
                "This is a provider/rate-limit/upstream failure, not a max_tokens issue — "
                "retrying (or backing off) is more likely to help than raising max_tokens."
            )
        raise ValueError(
            f"LLM response had empty content (finish_reason={finish_reason!r}) with no "
            "provider error attached. This usually means the completion was cut off "
            "before producing visible output — often hidden reasoning tokens exhausting "
            "max_tokens on a reasoning model. Try raising max_tokens."
        )
    return content


# %% [markdown]
# ## 2. JSON robustness helpers
#
# Models often wrap JSON in ```` ```json ... ``` ```` fences or add a
# sentence of preamble/postamble even when told not to. These helpers strip
# that noise and retry once or twice before giving up.

# %%
def _strip_to_json(text: str) -> str:
    """
    Extracts a JSON payload from a raw LLM response that may be wrapped in
    markdown code fences or surrounded by prose.

    Tries, in order: a fenced ```json ... ``` or ``` ... ``` block, then
    falls back to slicing from the first '[' or '{' to the matching last
    ']' or '}'. If neither pattern is found, returns the text unchanged
    (json.loads will then fail loudly, which is what we want).
    """
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
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


def _call_llm_with_json_retry(
    prompt: str, llm_client: LLMClient, context: str, max_attempts: int = 2,
    max_tokens: int = 4096,
):
    """
    Calls the LLM and parses its response as JSON, retrying up to
    max_attempts times if the call fails or the response isn't valid JSON
    (models occasionally return malformed JSON or extra prose on a first
    try).

    Returns the parsed JSON value (list or dict, depending on the prompt).
    Raises the last exception if every attempt fails — callers catch this
    and record an error row instead of letting one bad claim crash the
    whole batch.

    `context` is a short label (e.g. "extract_atomic_claims") used only in
    log messages, to make failures traceable back to which step produced
    them.

    `max_tokens` is passed straight through to call_llm. It defaults to the
    same 4096 as call_llm; a caller whose response can legitimately be long
    (extract_atomic_claims on a big memo section returns 60+ claims) raises
    it, since a reasoning model that runs out of budget returns truncated
    JSON or empty content rather than an error (see call_llm's docstring).
    """
    last_error: Optional[Exception] = None
    raw_text: Optional[str] = None
    for attempt in range(1, max_attempts + 1):
        try:
            raw_text = call_llm(prompt, llm_client, max_tokens=max_tokens)
            cleaned = _strip_to_json(raw_text)
            return json.loads(cleaned)
        except Exception as exc:
            last_error = exc
            logger.warning(
                "[%s] attempt %d/%d failed: %s | raw response: %r",
                context,
                attempt,
                max_attempts,
                exc,
                raw_text,
            )
    if last_error is None:
        # Only reachable when the loop never ran, i.e. max_attempts < 1.
        # Raising the ValueError beats `raise None`'s opaque TypeError.
        raise ValueError(f"[{context}] max_attempts must be >= 1, got {max_attempts}")
    raise last_error


# %% [markdown]
# ## 3. Chunking and BM25 shortlisting
#
# Source documents run tens of pages each and there can be up to ~5 per
# section, so a full document can't be sent through the LLM per claim on
# cost or context grounds. Instead: chunk every source document once per
# section, then for each claim use BM25 (cheap, no LLM call) to shortlist
# only the most lexically relevant chunks before the one LLM call that
# actually checks for evidence.

# %%
def chunk_document(
    doc_id: str, doc_text: str, chunk_size: int = 1000, overlap: int = 200
) -> list[dict]:
    """
    Splits one source document into overlapping chunks.

    chunk_size and overlap are character counts, not tokens/words — this
    keeps chunking dependency-light (no tokenizer needed). If you'd rather
    think in words, scale both numbers up by roughly 5-6x as a rule of
    thumb.

    Returns a list of dicts: {chunk_id, doc_id, chunk_text, start_offset}.
    chunk_id is deterministic (f"{doc_id}_{index}"), so re-chunking the
    same document always produces the same ids.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap, or chunking never advances")

    chunks = []
    step = chunk_size - overlap
    text_len = len(doc_text)
    start = 0
    index = 0
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(
            {
                "chunk_id": f"{doc_id}_{index}",
                "doc_id": doc_id,
                "chunk_text": doc_text[start:end],
                "start_offset": start,
            }
        )
        index += 1
        if end == text_len:
            break
        start += step
    return chunks


def build_chunk_index(
    documents: list[tuple[str, str]], chunk_size: int = 1000, overlap: int = 200
) -> list[dict]:
    """
    Chunks every (doc_id, doc_text) pair in `documents` — one section's
    source PDFs — and returns the combined flat list of chunk dicts across
    all of them, ready for BM25 shortlisting.
    """
    chunk_index = []
    for doc_id, doc_text in documents:
        chunk_index.extend(chunk_document(doc_id, doc_text, chunk_size=chunk_size, overlap=overlap))
    return chunk_index


_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word-token split for BM25 — no stemming or stopword removal, kept simple and transparent."""
    return _TOKEN_RE.findall(text.lower())


def bm25_shortlist(claim: str, chunk_index: list[dict], top_n: int = 30) -> list[dict]:
    """
    Scores every chunk in chunk_index against claim using BM25 and returns
    the top_n highest-scoring chunk dicts.

    This is the cost/context control: instead of sending every chunk from
    every source document to the LLM for every claim, only the top_n most
    lexically relevant chunks go into the evidence-lookup prompt.
    """
    if not chunk_index:
        return []
    tokenized_corpus = [_tokenize(chunk["chunk_text"]) for chunk in chunk_index]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(_tokenize(claim))
    ranked_indices = sorted(range(len(chunk_index)), key=lambda i: scores[i], reverse=True)
    return [chunk_index[i] for i in ranked_indices[:top_n]]


# %% [markdown]
# ### 3b. Threshold-based shortlist (default) instead of a fixed top_n
#
# A fixed `top_n` silently drops real evidence once a claim's supporting
# chunks are more numerous than `top_n` — e.g. the same figure restated
# across several source documents. `bm25_threshold_shortlist` scales the
# shortlist size to how obvious the match actually is (score distribution)
# instead of a fixed count, and is what `build_golden_set_draft` uses by
# default. `bm25_shortlist` above is kept as-is for an optional cheaper/
# fixed-count pass.


# %%
def bm25_threshold_shortlist(
    claim: str,
    chunk_index: list[dict],
    relative_threshold: float = 0.3,
    min_candidates: int = 5,
) -> list[dict]:
    """
    Scores every chunk in chunk_index against claim using BM25 and returns
    every chunk whose score is at least relative_threshold * the top score
    for this claim — no fixed cap. This is the fix for bm25_shortlist's
    fixed top_n silently truncating claims whose evidence is scattered
    across many near-equally-relevant chunks.

    If the top score is at/near zero (<= 1e-9), a relative threshold is
    meaningless against ~nothing: BM25Okapi floors negative per-term idf at
    epsilon * average_idf specifically to prevent common-term-driven
    negative scores, so a top score this low in practice means either
    genuinely no lexical overlap with the claim, or (rarer) a degenerate
    small corpus producing a negative average_idf. Either way, this falls
    back to the min_candidates top-scoring chunks instead of applying the
    threshold, and logs a warning naming the claim.

    If fewer than min_candidates chunks clear the threshold, the result is
    topped up to min_candidates using the next-highest scorers, so a sharp
    score dropoff never returns a dangerously small shortlist.

    Logs both the count that passed the threshold and the final count
    returned after any top-up — the two differ whenever top-up fires, and
    losing that distinction would hide how marginal a claim's best match
    actually was.

    Each returned dict is a shallow copy of its chunk_index entry with a
    "bm25_score" field added (that claim's BM25 score for that chunk) —
    chunk_index itself is never mutated. This matters because chunk_index is
    built once per section and re-scored fresh by every claim's shortlist
    call; a score is only meaningful for the claim that produced it, so it
    must never be written back onto the shared chunk_index entries.
    """
    if min_candidates < 0:
        raise ValueError(f"min_candidates must be >= 0, got {min_candidates}")
    if not chunk_index:
        return []

    tokenized_corpus = [_tokenize(chunk["chunk_text"]) for chunk in chunk_index]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(_tokenize(claim))
    ranked_indices = sorted(range(len(chunk_index)), key=lambda i: scores[i], reverse=True)
    top_score = scores[ranked_indices[0]]

    if top_score <= 1e-9:
        logger.warning(
            "[bm25_threshold_shortlist] top BM25 score for claim %r is at/near zero — "
            "falling back to top %d candidate(s) by score instead of a relative threshold",
            claim,
            min_candidates,
        )
        selected_indices = ranked_indices[:min_candidates]
        passed_threshold = 0
    else:
        cutoff = relative_threshold * top_score
        selected_indices = [i for i in ranked_indices if scores[i] >= cutoff]
        passed_threshold = len(selected_indices)
        if passed_threshold < min_candidates:
            selected_indices = ranked_indices[:min_candidates]

    logger.info(
        "[bm25_threshold_shortlist] claim %r: %d chunk(s) passed threshold, %d returned after top-up",
        claim,
        passed_threshold,
        len(selected_indices),
    )
    return [{**chunk_index[i], "bm25_score": scores[i]} for i in selected_indices]


# %% [markdown]
# ## 4. Single-purpose LLM steps
#
# Each of these is independently testable: pass it a string (or two) and a
# client, get a plain Python value back. No DataFrame, no batching, no I/O.

# A claim starting with one of these is likely a pronoun/pointing word
# stranded by a split (see extract_atomic_claims's docstring) — used only
# for a diagnostic warning, not to drop or alter the claim.
_STRANDED_POINTING_WORD_RE = re.compile(
    r"^(it|its|they|them|their|this|that|these|those|the (former|latter))\b", re.IGNORECASE
)


# %%
def extract_atomic_claims(section_text: str, llm_client: LLMClient) -> list[str]:
    """
    Breaks a human-written memo section into atomic factual claims.

    "Atomic" means each returned claim states exactly one fact — compound
    sentences ("Revenue grew 12% and margins expanded") get split into two
    claims. This includes a fact attached inside a phrase rather than
    joined between clauses ("X is a listed operator with a market cap of
    EUR 5bn" is two facts), but not a single description broken into its
    separate adjectives. Wording is kept as close to the original as
    possible; the model is told not to add information that isn't in the
    text.

    Exception to the attached-attribute rule: when the subject is ranked
    or compared against a different, named entity, and that other
    entity's own attributes appear only inside the comparison (e.g. "the
    2nd largest operator worldwide, after Acme Corp (revenue $10bn)"),
    those attributes are NOT split into standalone claims about the other
    entity — they stay attached to the one claim about the section's
    actual subject. This was added after a live run split a memo's
    ranking claim against a competitor into several claims entirely about
    the competitor, which the source documents (about the memo's own
    subject) could never substantiate — pure review noise, not a missed
    fact. If the other entity is genuinely the subject of its own
    sentence elsewhere in the text, its attributes still split normally.

    A further exception to splitting: when a sentence asserts something
    only about a list taken as a whole — a role or restriction defined
    over a named set, or a prediction contingent on several conditions
    holding jointly — splitting it makes each piece overstate what the
    sentence said, and two pieces can directly contradict each other.
    Such a sentence stays one claim, and any items the set names as
    examples ("such as ...") stay inside it. A list the sentence does not
    close over still splits, one claim per member. Separately, when a
    list does split, every qualifier on an item — including a partial one
    like "but to a lesser extent" — is carried onto that item's claim;
    a qualifier that ranks some members below a leading one goes only on
    those trailing members' claims, never the leading member's. See
    design decision 15 in CLAUDE.md for the observed failures this fixes.

    One exception to "keep the original wording": when the memo text is
    garbled and a split would otherwise leave an ungrammatical fragment
    ("The bank wit assets of $40bn."), the model is allowed to supply the
    small connecting words that make the claim a sentence ("The bank has
    assets of $40bn.") while leaving the text's own misspellings of names,
    terms and figures alone. So a claim is not guaranteed to be a
    word-for-word copy of the memo. This was a deliberate call (see design
    decision 5 in CLAUDE.md): a fragment is harder both to match against
    source documents and to read during review. The model is also told
    never to return a claim that restates the whole input sentence, or that
    fully contains another returned claim — both are signs of an unfinished
    split, and the whole-sentence restatement was an observed failure on
    garbled input.

    A related exception: when a split strands a pronoun or bare pointing
    phrase ("it", "the latter", "the asset") away from the words earlier
    in the text that say what it refers to, the model replaces it with the
    plainest naming the text itself already uses for that thing — copying
    existing wording, not adding information. It resolves to its best
    guess when the text names more than one candidate, and leaves the
    pointing word unchanged only when the text names nothing it could
    refer to (never inventing a subject). The substitution is deliberately
    minimal — name only, never pulling in a further attached fact — because
    doing otherwise would trip the attached-attribute rule above and
    manufacture a claim the memo never separately asserted. See design
    decision 5's "stranded pronoun" addendum in CLAUDE.md for the observed
    failure this fixes (a split-off claim reading "It has stabilized as of
    Dec'24." matched five unrelated metrics in the source documents because
    it carried no identifiable subject) and the residual risk on ambiguous
    antecedents. This now also covers a pointing word that is not the
    claim's subject ("Acme uses this process ...") and a referent carried
    only by a heading or a label on its own line above the paragraph — see
    design decision 5's "Pointing words beyond the subject" addendum in
    CLAUDE.md, including the residual it does not fix (an inline "Label:" at
    a paragraph head).

    A cheap, non-LLM diagnostic backs this up: any returned claim starting
    with a bare pronoun or pointing word is logged as a warning (not
    dropped) so a case the prompt rule missed is still visible rather than
    silently degrading.

    A related bar on the split pieces themselves: if splitting a sentence
    leaves a claim merely naming a thing — a bare "X has Y." / "X is Y." —
    when the source sentence asserted something more about it (typically a
    list of drivers or attributions, "earnings rose on X and Y"), the
    model restores that predicate from the source sentence rather than
    emitting the bare form ("The company has X."). What it restores is a
    severed relationship (the reason/basis/driver), never a separately
    measured attribute — an amount, date or holding still splits into its
    own claim under the attached-attribute rule. It does not do this
    when the source sentence asserted nothing more than that bare naming
    ("The company is well run.") — that claim is left unchanged; flagging
    it is EV-6's job (an always-on per-section check, not yet built).
    As with the pointer
    and connecting-word substitutions, this copies wording already in the
    section; it never invents a detail to make a vague claim concrete. See
    design decision 16 in CLAUDE.md for the observed failure this fixes
    ("Borealis has pricing initiatives." carved off a driver list).

    Separately, if extraction returns two or more claims with identical
    text (after stripping surrounding whitespace), each duplicated string
    is logged at WARNING with the repeated text and the count (the
    whitespace strip is for the comparison only; the returned claim strings
    are unchanged); all copies are kept, not collapsed — identical text can
    also mean the split failed to separate two distinct facts, and dropping
    one would lose a fact.

    Returns a plain list of claim strings.

    Raises on unrecoverable failure (bad JSON after retries, or JSON that
    isn't a list of strings) — callers are expected to catch this and log
    an error row rather than losing the whole batch.
    """
    prompt = f"""You will be given one section of a human-written financial memo.

Break it into atomic factual claims. A claim is atomic only if it cannot be
decomposed into two independently verifiable facts. If a sentence asserts
two separate facts, split it into two claims — regardless of what word or
punctuation joins them. This is not limited to "and" or "which also"; it
also covers comparisons and contrasts joined by words like "compared to",
"versus", "up from", "down from", "while", "whereas", or a semicolon, since
each side of a comparison is usually its own independently verifiable fact
(e.g. a current-period figure and a prior-period figure).

For example, "Net income was $774 million in 2025 compared to $805 million
in 2024." asserts two separate facts — the 2025 figure and the 2024 figure —
so it must split into:
["Net income was $774 million in 2025.", "Net income was $805 million in 2024."]

Facts attached inside a phrase count the same as facts joined between
clauses. When a sentence describes something and also attaches a separate
measured attribute to it — a figure, amount, percentage, date, rate,
ranking, holding, or similar — that attribute is its own independently
verifiable fact and must become its own claim. For example, "Acme is a
listed German logistics operator with a market cap of EUR 5bn." splits
into:
["Acme is a listed German logistics operator.", "Acme has a market cap of EUR 5bn."]
Attributes attached by parentheses, commas, or apposition work the same
way. Do not decide by which connector you see: the connecting word may be
missing entirely, or misspelled. Decide by what is being asserted.

Exception: when a sentence ranks or compares its subject against a
different, named entity, and that other entity's own attributes are
mentioned only inside the comparison, do not split those attributes into
separate claims about the other entity. Keep the whole comparison as one
claim about the sentence's actual subject. For example, "Acme is the 2nd
largest retailer worldwide, after Globex (revenue $50bn, based in the
US)." stays one claim:
["Acme is the 2nd largest retailer worldwide, after Globex (revenue $50bn, based in the US)."]
Do not emit "Globex has revenue of $50bn." or "Globex is based in the US."
as their own claims — those are Globex's attributes, cited only as context
for Acme's ranking, not facts this section is separately asserting. This
is narrower than it looks: if the other entity is the actual subject of
its own sentence elsewhere in the text (not just named inside a
comparison), split its attributes normally like any other subject.

Exception: some sentences assert something that holds only for a list
taken as a whole, not for its members one at a time — a role, right, or
restriction defined over a named set, or an outcome that depends on
several conditions holding together. Splitting such a sentence makes each
piece assert something the sentence never said, and two pieces can end up
contradicting each other. Keep it as one claim, and keep any items the
set names as examples ("such as ...") inside that one claim. Decide this
by what the sentence asserts, not by particular words: "limited to",
"only", "solely", "jointly", "provided that" are typical of this shape
but are not a checklist, and their absence does not make a closed list
splittable. This is narrow: a plain enumeration whose members are each
independently true — "Acme's largest customers are Globex, Initech and
Umbrella." — is NOT a closed list in this sense. It splits, one claim per
member, exactly as it would without this exception. The exception is for
a sentence whose point is the boundary of the set, not for one that
simply lists what is in it.

For example, "Acme's audit committee is limited to reviewing the annual
accounts and approving certain senior appointments (such as the external
auditor and the chief risk officer)." stays one claim:
["Acme's audit committee is limited to reviewing the annual accounts and approving certain senior appointments (such as the external auditor and the chief risk officer)."]
The two duties cannot be split — the first claim would then say the
committee does nothing else. The appointments named after "such as" also
stay inside that one claim: they illustrate the closed set, they are not
separate facts to lift out.

Without the closure phrasing, the same list splits normally. "Acme's
audit committee reviews the annual accounts and approves senior
appointments." asserts two independently checkable facts:
["Acme's audit committee reviews the annual accounts.", "Acme's audit committee approves senior appointments."]

An outcome contingent on several conditions together is the same shape.
"We expect a ratings upgrade if Acme sustains its margin recovery, keeps
cutting net debt, and holds its order book above 2x." stays one claim:
["We expect a ratings upgrade if Acme sustains its margin recovery, keeps cutting net debt, and holds its order book above 2x."]
Pulling out "Acme keeps cutting net debt." asserts a prediction the
sentence never made. This is a contingent forecast, not a rating agency's
published list of upgrade criteria — if the sentence instead said "the
agency's stated upgrade triggers for Acme are X, Y and Z", each trigger
is a documented fact and splits normally.

Do not go the other way and split a single description into its separate
adjectives. "Acme is a listed German logistics operator." stays one claim,
because "listed", "German" and "logistics operator" jointly characterize
what Acme is rather than stating separately measured facts.

When a list does split, carry every qualifier attached to an item onto
that item's claim rather than dropping it — "mainly"/"primarily" onto the
leading member's claim, "but to a lesser extent"/"less commonly" onto the
trailing ones. A qualifier that ranks some members below a leading member
("but to a lesser extent", "less commonly") goes only on those trailing
members' claims, never on the leading member it is measured against.
"Acme's fleet is supplied mainly by Globex, and to a lesser extent by
Initech and Umbrella." splits into:
["Acme's fleet is supplied mainly by Globex.", "Acme's fleet is supplied to a lesser extent by Initech.", "Acme's fleet is supplied to a lesser extent by Umbrella."]

Before returning each claim, test it: could a source document confirm one
part of this claim while saying nothing at all about another part of it? If
so, the claim is not yet atomic — split it and test the pieces again. (A
sentence that closes over a list — a limit defined over the whole set, or
an outcome contingent on every condition together — is the exception
above, not a failure of this test: keep it whole.)

Preserve the original wording as closely as possible. Do not add any
information that is not present in the text.

Preserving the wording does not mean copying out a broken fragment. Each
claim must read as a complete sentence on its own, so where the text is
garbled, or a connecting word is missing or misspelled, supply the few
ordinary connecting words needed to make the claim a sentence — an
attached "... wth assets of $40bn" becomes "The bank has assets of $40bn."
Leave the text's own spelling of names, terms and figures as it is; only
the connecting words may be adjusted.

A claim must also stand on its own about WHAT it describes, not only read
as a grammatical sentence — claims are matched against source documents
one at a time, without the rest of the section beside them. When a split
separates a pronoun ("it", "they", "its", "the latter", etc.) or a bare
pointing phrase ("the asset", "that figure", "the same period") from the
words earlier in the text that say what it refers to — including a heading
or a label line above the paragraph, such as a "## "-style heading or a
line like "Termination Clause:" — replace it with the plainest naming the
text itself already uses for that thing. This is copying wording that
already exists in the section, not adding new information, and it does not
change the fact being asserted. Keep the replacement minimal — the name or
short description only, never pulling in another fact attached to it
elsewhere in the text (that fact stays its own separate claim, per the
rule above).

This applies to a pointing word anywhere in the claim, not only its
subject. "Acme uses this process to retain a competitive advantage."
leaves "this process" pointing at nothing when the claim is read alone,
even though "Acme" is named — replace "this process" with the name the
text gives it.

If the text names more than one thing the pointing word could mean,
resolve it to whichever one the sentence you are splitting is itself
about. If the text does not clearly name what it refers to at all, leave
the pointing word unchanged rather than invent a subject. And if what it
refers to is already named within the same claim ("Acme said its
recurring revenue rose 12% in 2024."), leave it as is — there is nothing
to resolve.

For example, "Acme's exposure is mostly to logistics assets; the
portfolio was under pressure through Q3. However, it recovered by
year-end." splits into:
["Acme's exposure is mostly to logistics assets.", "The portfolio was under pressure through Q3.", "The portfolio recovered by year-end."]
Not ["It recovered by year-end."] — read on its own, that claim names
nothing a source document could be matched against.

As another example, a section headed "Change-of-Control Clause" followed
by "The clause lets lenders demand immediate repayment. It survives any
refinancing." splits into:
["The change-of-control clause lets lenders demand immediate repayment.", "The change-of-control clause survives any refinancing."]
The name comes from the heading; both "The clause" and "It" resolve to it.

Before returning each claim, test it this way too: if this claim were
shown on its own, with nothing else from the section visible, would a
reader know what every part of it refers to — its subject, and any word
or phrase that points at something named only elsewhere in the section
(for instance "this process", "those provisions", "that figure")? If the
section names that thing — including in a heading or a label line — use
the name instead of leaving the claim pointing at nothing.

A claim must also stand on its own in what it asserts, not only in what
it names: if splitting a sentence has left this claim merely naming a
thing — a bare "X has Y." or "X is Y." — when the sentence it was taken
from asserted something more about that thing, carry that back onto the
claim. What you carry back is a relationship the split severed — the
reason, basis, or driver the sentence gave — not a separately measured
attribute (an amount, date, holding, ranking), which still splits into
its own claim under the attached-attribute rule above. For example,
"Acme's margins improved on lower costs and better pricing." must not
split into ["Acme has lower costs.", "Acme has better pricing."] — those
drop what the sentence asserted, the reason margins improved — but into
["Acme's margins improved on lower costs.", "Acme's margins improved on
better pricing."], each keeping the attribution the sentence made.
("Acme is well positioned, with $2bn of committed liquidity." is not this
case: "$2bn of committed liquidity" is a separately measured attribute,
so it splits off as its own claim and "Acme is well positioned." is left
alone.) If the sentence a bare claim came from asserts nothing more than
the bare naming ("Acme is well run."), also leave it as it is — never add
a detail to make it checkable.

Never return a claim that restates the whole input sentence once you have
already split a part of it off, and never return one claim that contains
another claim you are returning. Either means the split is unfinished.

Return ONLY a JSON array of strings, nothing else. No markdown fences, no
explanation, no leading or trailing text.

Section text:
\"\"\"
{section_text}
\"\"\"
"""
    parsed = _call_llm_with_json_retry(
        prompt, llm_client, context="extract_atomic_claims", max_tokens=16384
    )
    if not isinstance(parsed, list) or not all(isinstance(c, str) for c in parsed):
        raise ValueError(f"Expected a JSON list of strings, got: {parsed!r}")
    for claim in parsed:
        if _STRANDED_POINTING_WORD_RE.match(claim):
            logger.warning(
                "extract_atomic_claims: claim starts with an unresolved pronoun/pointing "
                "word, likely stranded by a split — %r",
                claim,
            )
    # Exact match after .strip() only — claim text is model-generated prose,
    # never PDF-table extraction, so _normalize_span's dot-leader collapsing
    # is the wrong tool here. Internal whitespace differences ("a  b" vs
    # "a b") are NOT collapsed; revisit only if a real run produces a
    # near-miss. Identical claims are kept, not collapsed: identical text
    # can also mean the split failed to separate two distinct facts, and
    # dropping one would lose a fact (see CLAUDE.md design decision 5).
    for claim_text, occurrences in Counter(c.strip() for c in parsed).items():
        if occurrences > 1:
            logger.warning(
                "extract_atomic_claims: %d claims have identical text %r — keeping all; "
                "identical text can also mean the split failed to separate two facts",
                occurrences,
                claim_text,
            )
    return parsed


def propose_evidence_from_chunks(
    claim: str, candidate_chunks: list[dict], llm_client: LLMClient, counts: Optional[dict] = None
) -> list[dict]:
    """
    Given one atomic claim and a shortlist of candidate chunks (from
    bm25_shortlist), asks the LLM to identify EVERY candidate chunk that
    supports the claim. Evidence can legitimately appear in more than one
    source document, so the model is explicitly told not to pick just one.

    Returns a list of dicts, one per matching chunk:
        {"chunk_id": str, "chunk_text": str, "bm25_score": float | None,
         "evidence_span": str, "confidence": "high"|"medium"|"low", "verbatim_match": bool}
    Returns an empty list if no candidate supports the claim.

    chunk_text and bm25_score are looked up from candidate_chunks by that
    match's own chunk_id (not copied from another candidate), so they always
    describe the actual chunk the match came from. bm25_score is None if the
    candidate came from bm25_shortlist (the legacy fixed-top_n path), which
    doesn't attach a score.

    The prompt requires a chunk to be about the same thing the claim is
    about — same subject, same quantity measured — not merely to share an
    entity name, a keyword, or a similar-looking number, since these
    documents report many different measures for the same entity. Exactly
    three things disqualify a candidate: it is about a different measure,
    period or population; you can't tell what its figure refers to at all;
    or it contradicts the claim.

    Saying LESS than the claim deliberately does not disqualify a
    candidate. A passage that states the claim's fact but omits one of the
    claim's qualifiers (claim says "recurring revenue", passage says only
    "revenue") is returned with confidence "low" rather than dropped —
    the user's explicit preference is that a near-miss surfaced for review
    beats real evidence silently going unlinked (see design decision 9 in
    CLAUDE.md). "low" is reserved for that case and must not be used to
    pass along a different-measure lookalike.

    confidence is therefore: "high" when the passage plainly states the
    claim's fact on the claim's own terms, "medium" when wording/period/
    scope has to be interpreted to line up, "low" when the passage is about
    the claim's measure but leaves one of the claim's qualifiers unsaid. It
    is a review-priority hint, not a stable score — the same candidate can
    come back "high" on one run and "medium"/"low" on the next.

    verbatim_match is True when evidence_span (whitespace-normalized) is a
    literal substring of that chunk's chunk_text (also normalized) — i.e.
    the model actually quoted rather than paraphrased. A False here isn't
    dropped like a missing span is; it's kept so these rows can be
    prioritized during manual review, since a non-verbatim quote is more
    likely to be a paraphrase, a misattribution, or otherwise worth
    double-checking.

    Any entry the model returns with a chunk_id that isn't among the
    candidates it was actually given (a hallucinated id) can't be trusted
    as given — this can't be validated by _call_llm_with_json_retry itself
    since it doesn't know what a valid id looks like for this call. Before
    dropping it, its evidence_span (normalized) is checked as a substring
    against every candidate's chunk_text (also normalized) in this batch:
    if exactly one candidate's chunk_text contains it, the entry is
    recovered under that candidate's real chunk_id instead of lost
    (verbatim_match then comes out True by that same check, and this is
    logged as a recovery). If the span is missing/empty, or matches zero
    or more than one candidate's chunk_text, the entry is dropped with a
    warning, same as before.

    counts, if given, is a dict with integer "dropped" and "recovered" keys
    that this function increments in place at every site above that drops
    or recovers an entry: a malformed entry, an unrecoverable hallucinated
    chunk_id, and a missing evidence_span each increment "dropped" (this is
    one undifferentiated total — the WARNING log at each site is what still
    distinguishes them); a successful evidence_span-based chunk_id recovery
    increments "recovered". Omit it (the default) to call this function
    standalone for a manual pass exactly as before — the returned match
    list is identical either way. propose_evidence_from_chunks_batched
    passes one in so it can report per-claim totals.

    Raises on unrecoverable failure — callers catch this and log an error
    row.
    """
    if not candidate_chunks:
        return []

    # Presented in document order (doc_id, then start_offset), not BM25
    # score order — overlapping/adjacent chunks (the ones _rows_for_claim
    # later needs to recognize as the same evidence restated) end up next
    # to each other in the prompt instead of interleaved with unrelated
    # candidates. Doesn't change which candidates are evaluated or how
    # they're batched, only how they're presented within one prompt.
    candidate_chunks = sorted(candidate_chunks, key=lambda c: (c["doc_id"], c["start_offset"]))

    # chunk_id/doc_id are quoted and on their own labeled lines rather than
    # packed into one bracketed "[chunk_id=... doc_id=...]" header — when
    # doc_id contains spaces (e.g. "Overlap demo doc"), an unquoted inline
    # header is ambiguous enough that the model can concatenate the two
    # fields into one bogus chunk_id, which then gets correctly rejected as
    # hallucinated by the validation below but loses a real match. This was
    # observed in practice against a live endpoint, not just theorized.
    candidates_block = "\n\n".join(
        f'Candidate chunk_id: "{c["chunk_id"]}"\n'
        f'Candidate doc_id: "{c["doc_id"]}"\n'
        f'Text:\n"""\n{c["chunk_text"]}\n"""'
        for c in candidate_chunks
    )
    prompt = f"""You will be given one factual claim and a set of candidate text chunks
pulled from several source documents.

Identify EVERY chunk among the candidates that contains a passage
supporting the claim. The same evidence can legitimately appear in more
than one document or chunk — do not limit yourself to a single match if
more than one candidate genuinely supports the claim. If no candidate
supports the claim, return an empty list.

A chunk supports the claim only if it is about the same thing the claim is
about: the same subject, and the same quantity or property being measured.
Sharing an entity name, a keyword, or a number that merely looks similar is
not support. These documents report many different measures for the same
entity, and a figure belonging to a different measure can look almost
identical to the one the claim states. So for each candidate you are
tempted to accept, first work out what the passage's number or statement is
actually measuring — read its column header, row label, or surrounding
sentence — and check that this is the thing the claim asserts.

Exactly three things disqualify a passage:
1. It is about a different measure, a different period, or a different
   population than the claim — a figure that resembles the claim's figure
   but belongs to another measure is a coincidence, not evidence.
2. You cannot tell what its figure or statement refers to at all. An
   unlabelled number that happens to resemble the claim's number is not
   evidence either.
3. It contradicts the claim.

Nothing else disqualifies a passage — in particular, saying LESS than the
claim does not. A passage that states the claim's fact but leaves out a
qualifier the claim carries (the claim says "recurring revenue", the
passage says only "revenue"; the claim says "net debt", the passage says
only "debt") is probably, though not certainly, the same fact. Return it,
with confidence "low". Leaving real evidence unreturned is a worse outcome
here than returning a near-miss for a person to check, so when a passage is
genuinely about the claim's subject and measure, err towards returning it.

Judge each candidate on its own merits, as if it were the only candidate
you had been given. That another candidate in this batch states the claim
more fully, or is wrong, or contradicts the claim, tells you nothing about
the candidate in front of you — it must not lower your confidence in a
different candidate or make you return fewer of them. Return every
candidate that qualifies, not only the best one.

For each matching chunk, quote the supporting passage verbatim from that
chunk — do not paraphrase or summarize it. Quote enough of the surrounding
label, header, or sentence for the quote to show what it measures.

Use confidence "high" when the passage plainly states the claim's fact on
the claim's own terms, "medium" when it states the fact but the wording,
period, or scope has to be interpreted to line up, and "low" when it is
about the claim's subject and measure but leaves one of the claim's
qualifiers unsaid, so it is probably rather than certainly the same fact.

"low" is for a passage that is about the claim's measure and says less
about it. It is never a way to pass along a passage about a different
measure: if the measure, period or population is a different one, return
nothing for that candidate, however closely its number resembles the
claim's.

Return ONLY a JSON array, nothing else, of objects shaped exactly like:
{{"chunk_id": "<chunk_id of a candidate below>", "evidence_span": "<verbatim quote from that chunk>", "confidence": "high" or "medium" or "low"}}
Return an empty array [] if nothing matches. No markdown fences, no
explanation, no leading or trailing text.

Claim:
\"\"\"
{claim}
\"\"\"

Candidate chunks:
{candidates_block}
"""
    parsed = _call_llm_with_json_retry(prompt, llm_client, context="propose_evidence_from_chunks")
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON list, got: {parsed!r}")

    valid_chunk_ids = {c["chunk_id"] for c in candidate_chunks}
    candidate_by_id = {c["chunk_id"]: c for c in candidate_chunks}
    if counts is None:
        counts = {"dropped": 0, "recovered": 0}
    matches = []
    for entry in parsed:
        if not isinstance(entry, dict) or "chunk_id" not in entry:
            logger.warning("[propose_evidence_from_chunks] skipping malformed entry: %r", entry)
            counts["dropped"] += 1
            continue
        if entry.get("evidence_span") is not None and not isinstance(entry["evidence_span"], str):
            # _normalize_span assumes a string; a model that returns evidence_span
            # as a bare number (e.g. 774.1 instead of "774.1") would otherwise
            # crash here and abort the whole batch, discarding matches already
            # found for other, valid entries earlier in this same call.
            logger.warning("[propose_evidence_from_chunks] skipping malformed entry: %r", entry)
            counts["dropped"] += 1
            continue
        if entry["chunk_id"] not in valid_chunk_ids:
            normalized_span = _normalize_span(entry.get("evidence_span"))
            recovered_candidates = [
                c for c in candidate_chunks if _span_matches_chunk_text(normalized_span, c["chunk_text"])
            ]

            if len(recovered_candidates) == 1:
                candidate = recovered_candidates[0]
                logger.warning(
                    "[propose_evidence_from_chunks] model returned chunk_id %r not among the %d candidates given — "
                    "recovered via evidence_span match to chunk_id %r",
                    entry["chunk_id"],
                    len(candidate_chunks),
                    candidate["chunk_id"],
                )
                counts["recovered"] += 1
                # Self-correct and fall through to the normal match-construction
                # path below, so a recovered entry is built exactly like one the
                # model got right the first time (verbatim_match included).
                entry["chunk_id"] = candidate["chunk_id"]
            else:
                logger.warning(
                    "[propose_evidence_from_chunks] model returned chunk_id %r not among the %d candidates given — dropping",
                    entry["chunk_id"],
                    len(candidate_chunks),
                )
                counts["dropped"] += 1
                continue
        if not entry.get("evidence_span"):
            logger.warning(
                "[propose_evidence_from_chunks] entry for chunk_id %r has no evidence_span — dropping: %r",
                entry["chunk_id"],
                entry,
            )
            counts["dropped"] += 1
            continue
        candidate = candidate_by_id[entry["chunk_id"]]
        normalized_span = _normalize_span(entry.get("evidence_span"))
        verbatim_match = _span_matches_chunk_text(normalized_span, candidate["chunk_text"])
        matches.append(
            {
                "chunk_id": entry["chunk_id"],
                "chunk_text": candidate["chunk_text"],
                "bm25_score": candidate.get("bm25_score"),
                "evidence_span": entry.get("evidence_span"),
                # A malformed answer's list/dict confidence is kept as "not rated"
                # (None): as a golden-set value it would fail the parquet write.
                "confidence": entry.get("confidence") if isinstance(entry.get("confidence"), str) else None,
                "verbatim_match": verbatim_match,
            }
        )
    return matches


# %% [markdown]
# ### 4b. Batched evidence lookup (default) for oversized shortlists
#
# `bm25_threshold_shortlist` has no upper bound on candidate count, so a
# claim with widely-scattered evidence can produce more candidates than
# comfortably fits in one prompt. `propose_evidence_from_chunks_batched`
# splits the candidates into batches and calls `propose_evidence_from_chunks`
# once per batch, unioning the results — none are dropped for exceeding one
# call's context. A batch that fails is logged and skipped rather than
# discarding the evidence already found by other, successful batches.


# %%
_BATCH_COUNT_WARNING_THRESHOLD = 10


def propose_evidence_from_chunks_batched(
    claim: str,
    candidate_chunks: list[dict],
    llm_client: LLMClient,
    batch_size: int = 40,
) -> tuple[list[dict], dict]:
    """
    Like propose_evidence_from_chunks, but splits candidate_chunks into
    batches of at most batch_size before calling the LLM, so a threshold
    shortlist that grows past what fits comfortably in one prompt doesn't
    have to be truncated.

    Each batch's call is wrapped in its own try/except: if one batch fails
    (LLM/JSON error, after propose_evidence_from_chunks's own retries are
    exhausted), that failure is logged — naming the claim, the batch's
    position, and how many candidate chunks in that batch were never
    evaluated — and lookup continues with the remaining batches rather than
    raising immediately. The point of removing the fixed top_n cap is to
    stop silently losing real evidence; letting one bad batch wipe out
    evidence already confirmed by other batches would just be a bigger,
    worse-shaped version of the same bug. Only if every batch fails does
    this raise, so build_golden_set_draft's per-claim error-row handling is
    still reached for a claim with no usable evidence at all.

    Returns a (matches, counts) tuple. matches is the union (concatenation)
    of whatever batches succeeded. No separate dedup is needed: chunk_ids
    are unique across batches (each candidate chunk appears in exactly one
    batch), and any duplicate or overlapping evidence across chunks is
    already handled downstream by _rows_for_claim's union-find grouping.

    counts is {"dropped": int, "recovered": int}, aggregated across every
    batch for this claim via the counts accumulator propose_evidence_from_chunks
    accepts (see that function's docstring for what "dropped" vs.
    "recovered" means). {"dropped": 0, "recovered": 0} if candidate_chunks
    was empty (no batches ran). Logged once as an INFO line — claim %r: %d
    dropped, %d recovered, plus how many of the claim's batches actually
    succeeded — after all batches complete; a claim for which every batch
    fails raises instead of reaching this line, so it never logs a
    misleading "0 dropped, 0 recovered" for a claim where nothing actually
    completed.

    The pre-loop candidate/batch-count line above is logged at WARNING
    instead of INFO when a claim splits into more than
    _BATCH_COUNT_WARNING_THRESHOLD batches. Batch count, not candidate
    count, is the trigger: hundreds of candidates are routine (see
    bm25_threshold_shortlist's docstring), but batch count tracks this
    claim's actual LLM cost and its exposure to the out-of-batch chunk_id
    failure (recovered, if possible, above) — one independent chance per
    batch. This makes an unusually expensive claim visible while a long run
    is still in progress, not just after the fact from batch-failure
    warnings.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    counts = {"dropped": 0, "recovered": 0}
    if not candidate_chunks:
        return [], counts

    batches = [candidate_chunks[i : i + batch_size] for i in range(0, len(candidate_chunks), batch_size)]
    log_level = logging.WARNING if len(batches) > _BATCH_COUNT_WARNING_THRESHOLD else logging.INFO
    logger.log(
        log_level,
        "[propose_evidence_from_chunks_batched] claim %r: %d candidate(s) split into %d batch(es) of up to %d",
        claim,
        len(candidate_chunks),
        len(batches),
        batch_size,
    )

    matches = []
    succeeded_batches = 0
    for batch_num, batch in enumerate(batches, start=1):
        try:
            matches.extend(propose_evidence_from_chunks(claim, batch, llm_client, counts=counts))
            succeeded_batches += 1
        except Exception as exc:
            logger.error(
                "[propose_evidence_from_chunks_batched] claim %r: batch %d/%d failed (%d candidate(s) "
                "never evaluated): %s",
                claim,
                batch_num,
                len(batches),
                len(batch),
                exc,
            )

    if succeeded_batches == 0 and batches:
        raise RuntimeError(
            f"propose_evidence_from_chunks_batched: all {len(batches)} batch(es) failed for claim {claim!r} "
            "— no evidence could be evaluated for this claim"
        )

    logger.info(
        "[propose_evidence_from_chunks_batched] claim %r: %d dropped, %d recovered (%d/%d batch(es) succeeded)",
        claim,
        counts["dropped"],
        counts["recovered"],
        succeeded_batches,
        len(batches),
    )
    return matches, counts


# %% [markdown]
# ## 5. Per-section pipeline
#
# Each claim gets a stable `claim_id` (a UUID) assigned here, at creation
# time — it stays constant across every row that claim produces, including
# when it has multiple evidence matches. Review/re-import matches on this
# id (plus `chunk_id`, see section 7) rather than on `claim_text`, so
# correcting a typo in a claim's wording during manual review doesn't
# break the match back to its row(s).

# %%
def _error_row(memo_id: str, section_name: str, claim_text: Optional[str]) -> dict:
    """Builds one error-sentinel row: found=None, confidence='error' — reserved for pipeline failures, never for a legitimate not-found result."""
    return {
        "claim_id": str(uuid.uuid4()),
        "memo_id": memo_id,
        "section": section_name,
        "claim_text": claim_text,
        "doc_id": None,
        "chunk_id": None,
        "chunk_text": None,
        "bm25_score": None,
        "evidence_span": None,
        "found": None,
        "confidence": "error",
        "ambiguous_match": False,
        "verbatim_match": None,
        "human_reviewed": False,
        "tag": None,
    }


# The DataFrame schema, in column order -- the keys _error_row builds. Used
# wherever an empty frame must still carry the schema (e.g.
# build_golden_set_draft with an empty claims list) so pandas' empty/all-NA
# concat path stays clean.
_SCHEMA_COLUMNS = list(_error_row("", "", None).keys())


_LEADER_RUN_RE = re.compile(r"[.\-_]{2,}")


@lru_cache(maxsize=8192)
def _normalize_span(text: Optional[str]) -> Optional[str]:
    """
    Normalizes an evidence span so that two quotes of the same underlying
    passage compare equal even when a chunk boundary split them
    differently.

    Two things get collapsed, in this order:
      1. Runs of 2+ periods, dashes, or underscores -> a single '.'
         placeholder. PDF table extraction commonly produces dot-leader
         artifacts ("Net income .......... $774.1"), and the leader's
         length isn't stable — the SAME physical table row can come out
         with a different number of dots depending on exactly where the
         extractor happened to split it across adjacent chunks. Left
         uncollapsed, two quotes of the same row would differ only in dot
         count, so neither is a substring of the other and the
         adjacent-chunk merge in _rows_for_claim (and the verbatim_match
         check in propose_evidence_from_chunks) would wrongly treat them
         as unrelated.
      2. Runs of whitespace -> a single space (pre-existing behavior).

    Cached (a pure function of its string): eval_pipeline compares the same
    few hundred chunk texts tens of thousands of times, and normalizing
    them afresh on every comparison made scoring ~5x slower.
    """
    if text is None:
        return None
    collapsed = _LEADER_RUN_RE.sub(".", text.strip())
    return re.sub(r"\s+", " ", collapsed)


def _span_matches_chunk_text(normalized_span: Optional[str], chunk_text: str) -> bool:
    """
    True if normalized_span (already run through _normalize_span) is a
    non-empty substring of chunk_text's own normalized form. Shared by
    propose_evidence_from_chunks for two purposes: computing verbatim_match
    for a chunk_id the model got right, and searching for which candidate a
    hallucinated chunk_id's quoted span actually belongs to.
    """
    normalized_chunk_text = _normalize_span(chunk_text)
    if not normalized_span or not normalized_chunk_text:
        return False
    return normalized_span in normalized_chunk_text


def _chunk_index_of(chunk_id: Optional[str]) -> Optional[int]:
    """
    Extracts the numeric chunk index from a chunk_id built by chunk_document
    as f"{doc_id}_{index}" (rsplit from the right, so doc_ids containing
    underscores still parse correctly — the index is always the final
    segment). Used to confirm two matches come from *physically adjacent*
    chunks, not just any two chunks that happen to share a doc_id. Returns
    None on unparseable input, which callers treat as "not adjacent" — the
    conservative choice. That includes None (_same_evidence passes a missing
    chunk_id straight through) and a non-string such as a NaN read back from
    a DataFrame, which the AttributeError catches.
    """
    if chunk_id is None:
        return None
    try:
        return int(chunk_id.rsplit("_", 1)[-1])
    except (ValueError, AttributeError):
        return None


class _UnionFind:
    """Minimal disjoint-set structure for grouping matches that chain together transitively (e.g. the same passage echoed across 3 overlapping chunks)."""

    def __init__(self, n: int):
        self._parent = list(range(n))

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


def _same_evidence(
    doc_a: Optional[str],
    chunk_id_a: Optional[str],
    text_a: Optional[str],
    doc_b: Optional[str],
    chunk_id_b: Optional[str],
    text_b: Optional[str],
) -> bool:
    """
    Design decision 1's rule — the one definition of "the same evidence",
    shared by _rows_for_claim (through _group_equivalent_chunks) and
    eval_pipeline.py. Two items are the same evidence when they share a
    doc_id AND either
      (a) their texts are identical after _normalize_span, or
      (b) their chunk indices (_chunk_index_of) differ by exactly 1 and one
          normalized text contains the other.
    Texts are passed raw and normalized here, so design decision 2's
    leader-run collapsing cannot be skipped by a caller.

    A chunk is NOT the same evidence as itself under this rule: its index
    delta is 0, and a quote is rarely identical to a whole chunk.
    _rows_for_claim collapses duplicate chunk_ids before grouping, and
    eval_pipeline tests chunk_id equality before calling this. A missing
    doc_id never matches; an unparseable chunk_id can only match by (a),
    identical text (the conservative choice).
    """
    if doc_a is None or doc_a != doc_b:
        return False
    span_a, span_b = _normalize_span(text_a), _normalize_span(text_b)
    if span_a is not None and span_a == span_b:
        return True
    index_a, index_b = _chunk_index_of(chunk_id_a), _chunk_index_of(chunk_id_b)
    return bool(
        index_a is not None
        and index_b is not None
        and abs(index_a - index_b) == 1
        and span_a
        and span_b
        and (span_a in span_b or span_b in span_a)
    )


def _group_equivalent_chunks(
    doc_ids: list[Optional[str]],
    chunk_ids: list[Optional[str]],
    texts: list[Optional[str]],
) -> list[list[int]]:
    """
    Partitions item indices 0..n-1 (one per entry of the three parallel
    lists) into groups of the same evidence under _same_evidence,
    transitively via union-find, so a passage echoed across chunks 0-1-2 is
    one group although chunks 0 and 2 are not adjacent. Groups come in order
    of their first member; members ascending.
    """
    n = len(chunk_ids)
    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if _same_evidence(doc_ids[i], chunk_ids[i], texts[i], doc_ids[j], chunk_ids[j], texts[j]):
                uf.union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    return list(groups.values())


# How tag_pipeline.py finalize writes a chunk a person added in review: its
# tag_rationale starts with _HUMAN_ADDED_PREFIX, and its evidence_span joins
# the quotes they pasted with _QUOTE_SEPARATOR. eval_pipeline.py reads both
# back to test each quote on its own; defined here, where both import from,
# so writer and reader cannot drift apart.
_HUMAN_ADDED_PREFIX = "human-added"
_QUOTE_SEPARATOR = " | "


# Ordering for "keep the strongest of several matches for one chunk_id"
# (_rows_for_claim's dedupe). `confidence` is the string the evidence LLM
# returns; a bare max() would sort it alphabetically ("low" > "high").
# An absent/unknown value ranks below "low".
_CONFIDENCE_RANK = {"high": 2, "medium": 1, "low": 0}


def _rows_for_claim(
    memo_id: str,
    section_name: str,
    claim_id: str,
    claim_text: str,
    matches: list[dict],
    chunk_id_to_doc_id: dict,
) -> list[dict]:
    """
    Turns one claim's chunk matches into one or more DataFrame rows.

    Before grouping, multiple matches for the same chunk_id are collapsed to
    the single strongest by confidence (tie -> first seen); each collapse is
    logged WARNING. This can happen when a chunk appears in two BM25
    batches, is repeated in one evidence-LLM response, or a hallucinated-
    chunk_id recovery lands on an already-matched chunk.

    Ambiguity handling: matches are grouped so that two matches land in the
    same group — and are treated as the SAME underlying evidence rather
    than real ambiguity — when they share a doc_id AND either:
      (a) their evidence_span text is identical after whitespace
          normalization, or
      (b) they come from chunks whose chunk_id indices differ by exactly 1
          (i.e. physically adjacent, per _chunk_index_of) AND one
          normalized span is a substring of the other.
    (b) exists because overlapping chunks can get truncated differently at
    the shared boundary, so the LLM's verbatim quote from each side may be
    a slightly different substring of the same passage rather than
    byte-identical text — checking chunk_id adjacency (not just text
    similarity) is what confirms these are boundary-truncation variants of
    one passage rather than two coincidentally similar quotes from
    unrelated parts of the same document.

    The rule itself is _same_evidence; _group_equivalent_chunks partitions by it.

    Grouping uses union-find rather than a single grouping key so a
    passage echoed across three or more overlapping chunks (0-1-2) still
    collapses into one group even though chunk 0 and chunk 2 aren't
    directly adjacent.

    ambiguous_match is only set True when more than one group remains,
    i.e. the claim's evidence genuinely appears in different documents or
    different (non-overlapping) locations — the case that actually needs
    human review. A chunk-overlap duplicate must NOT be flagged ambiguous:
    a later retrieval-metric phase will use these chunk_ids as the gold
    set, and arbitrarily treating only one of several functionally
    equivalent overlapping chunks as "the" answer would penalize a
    retrieval system that legitimately surfaces a different one of them.

    chunk_text/bm25_score on each row come from that row's own match dict
    (chunk_text and bm25_score keys, set by propose_evidence_from_chunks
    from its own per-claim candidate_chunks) — never from a shared,
    section-level lookup, so one claim's rows can never carry another
    claim's score.
    """
    if not matches:
        return [
            {
                "claim_id": claim_id,
                "memo_id": memo_id,
                "section": section_name,
                "claim_text": claim_text,
                "doc_id": None,
                "chunk_id": None,
                "chunk_text": None,
                "bm25_score": None,
                "evidence_span": None,
                "found": False,
                "confidence": None,
                "ambiguous_match": False,
                "verbatim_match": None,
                "human_reviewed": False,
                "tag": None,
            }
        ]

    # Collapse multiple matches for one chunk_id to the single strongest,
    # BEFORE the union-find grouping below. Two matches with the same
    # chunk_id but different quoted spans would otherwise land in separate
    # groups (the adjacency rule needs a chunk-index delta of 1, not 0) and
    # flip ambiguous_match on spuriously. Also the one place a hallucinated-
    # chunk_id recovery that duplicated an existing match becomes visible.
    if len({m["chunk_id"] for m in matches}) != len(matches):
        best_by_chunk: dict[str, dict] = {}
        for m in matches:
            cid = m["chunk_id"]
            kept = best_by_chunk.get(cid)
            if kept is None:
                best_by_chunk[cid] = m
                continue
            # A non-string confidence (a malformed answer's list or dict) is
            # unhashable: rank it like a missing one rather than raising,
            # which would turn the whole claim into an error row.
            new_conf, kept_conf = m.get("confidence"), kept.get("confidence")
            new_rank = _CONFIDENCE_RANK.get(new_conf, -1) if isinstance(new_conf, str) else -1
            kept_rank = _CONFIDENCE_RANK.get(kept_conf, -1) if isinstance(kept_conf, str) else -1
            winner, loser = (m, kept) if new_rank > kept_rank else (kept, m)
            best_by_chunk[cid] = winner
            logger.warning(
                "[_rows_for_claim] claim %s: collapsed duplicate chunk_id %s "
                "(kept confidence=%r, dropped confidence=%r)",
                claim_id, cid, winner.get("confidence"), loser.get("confidence"),
            )
        matches = list(best_by_chunk.values())

    doc_ids = [chunk_id_to_doc_id.get(m["chunk_id"]) for m in matches]
    groups = _group_equivalent_chunks(
        doc_ids, [m["chunk_id"] for m in matches], [m["evidence_span"] for m in matches]
    )

    ambiguous = len(groups) > 1
    rows = []
    for group_indices in groups:
        for i in group_indices:
            match = matches[i]
            rows.append(
                {
                    "claim_id": claim_id,
                    "memo_id": memo_id,
                    "section": section_name,
                    "claim_text": claim_text,
                    "doc_id": doc_ids[i],
                    "chunk_id": match["chunk_id"],
                    "chunk_text": match.get("chunk_text"),
                    "bm25_score": match.get("bm25_score"),
                    "evidence_span": match["evidence_span"],
                    "found": True,
                    "confidence": match["confidence"],
                    "ambiguous_match": ambiguous,
                    "verbatim_match": match.get("verbatim_match"),
                    "human_reviewed": False,
                    "tag": None,
                }
            )
    return rows


# Deterministic claim_id namespace (design decision 17). Generated once with
# `python -c "import uuid; print(uuid.uuid4())"`. NEVER change this: it would
# rotate every claim_id in every golden set ever built.
_CLAIM_ID_NAMESPACE = uuid.UUID("ee12e9dc-1de2-4125-aabc-244988261980")


def _valid_section_name(name: str) -> bool:
    """
    The single section-name predicate, enforced identically in
    _read_memo_config, write_claims_file, and parse_claims_file: a non-blank
    string equal to its own .strip(), with no newline/tab/CR, no control
    character, not '#'-initial, and unchanged by the parser's ATX-close strip.

    - 'equals its own strip' keeps the drift check honest: a memos.yaml key
      '  Business Profile  ' would otherwise round-trip to a stripped '## '
      heading and read as drift on every subsequent extract.
    - '\\t' and '\\r' are checked here explicitly -- they sit in the gaps of
      _CONTROL_CHARS_RE (\\x09, \\x0d), and .strip() doesn't touch an interior
      tab. _CONTROL_CHARS_RE is deliberately NOT widened: it is shared with
      claim-text validation, and a claim pasted from a table may contain a tab.
    - 'ATX-close-strip is a no-op on it': parse_claims_file removes a
      trailing ' #+' run (ATX close, '## Foo ##') from a heading. A name that
      would be altered by that ('Foo ##', 'Bar #') can't round-trip, so it is
      rejected here. 'C#' has no space before the '#' -> not an ATX close ->
      allowed.
    """
    return _section_name_issue(name) is None


def _section_name_issue(name) -> str | None:
    """
    The clause list behind _valid_section_name: returns None if `name` is a
    valid section name, or a phrase naming the FIRST clause it fails, for an
    error message that gives the actual cause instead of restating the whole
    rule (or, worse, no reason at all).

    One function rather than a predicate and a parallel diagnostic: those
    were two copies of the same seven-clause chain, in the same order, kept
    in step only by hand -- and every caller walked both, once to test and
    once to find out why. _valid_section_name is now a thin wrapper, so the
    two answers cannot disagree.
    """
    if not isinstance(name, str):
        return f"must be a string, got {type(name).__name__}"
    if name == "":
        return "must not be blank"
    if name != name.strip():
        return "must not have leading or trailing whitespace"
    if name.startswith("#"):
        return "must not start with '#' (would be misread as a heading marker)"
    if any(c in name for c in "\n\t\r"):
        return "must not contain a newline, tab, or carriage return"
    if _CONTROL_CHARS_RE.search(name) is not None:
        return "must not contain a control character"
    if _ATX_CLOSE_RE.sub("", name) != name:
        return (
            "must not end in whitespace followed by one or more '#' -- looks like an "
            "ATX heading close (e.g. 'Foo ##') and wouldn't round-trip"
        )
    return None


def _claims_with_occurrence(claims: list[str]) -> list[tuple[str, int]]:
    """
    Pairs each claim with how many byte-identical claims preceded it in the
    list (0 for all but a genuine in-section duplicate). Used by
    build_golden_set_draft to give two word-for-word identical claims in one
    section distinct, stable claim_ids.
    """
    seen: dict[str, int] = {}
    out: list[tuple[str, int]] = []
    for c in claims:
        n = seen.get(c, 0)
        out.append((c, n))
        seen[c] = n + 1
    return out


def _derive_claim_id(memo_id: str, section_name: str, claim_text: str, occurrence_index: int) -> str:
    """
    Deterministic claim_id: uuid5 of (memo_id, section_name, canonical claim
    text, occurrence_index) joined by \\x1f (a control char the parser
    rejects in every input, so the join is collision-free). Replaces
    str(uuid.uuid4()) -- see design decision 17 for the consequences
    (import_reviewed still one-build-only; a re-run build is now idempotent).
    Returns the str form; the schema calls claim_id a UUID string.
    """
    key = "\x1f".join((memo_id, section_name, claim_text, str(occurrence_index)))
    return str(uuid.uuid5(_CLAIM_ID_NAMESPACE, key))


def _is_deterministic_claim_id(claim_id) -> bool:
    """
    True only if `claim_id` is an RFC-4122 version-5 UUID string -- i.e. one
    minted by `_derive_claim_id` (design decision 17), not a legacy
    `str(uuid.uuid4())`. `== 5` rather than `!= 4` so a non-UUID string or a
    non-string refuses rather than slips through.

    Consumer: `tag_pipeline.prepare_draft` refuses a memo whose checkpoint
    ids are not deterministic -- a `uuid4`-keyed checkpoint is not mergeable
    or re-runnable. Second consumer: `eval_pipeline.load_ground_truth`
    (Phase 3), for the same reason — a uuid4 id can never be recomputed from
    the claims file.
    """
    try:
        return isinstance(claim_id, str) and uuid.UUID(claim_id).version == 5
    except (ValueError, AttributeError, TypeError):
        return False


def build_golden_set_draft(
    memo_id: str,
    section_name: str,
    claims: list[str],
    source_documents: list[tuple[str, str]],
    llm_client: LLMClient,
    relative_threshold: float = 0.3,
    min_candidates: int = 5,
    batch_size: int = 40,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> pd.DataFrame:
    """
    Runs the full single-section pipeline: chunk this section's source
    documents once, then for each claim get a threshold-based BM25
    shortlist of candidate chunks (scaled to how obvious the match is, not
    capped at a fixed count) and ask the LLM, batching the candidates if
    needed, which of them support it.

    source_documents is a list of (doc_id, doc_text) pairs — this
    section's source PDFs (up to ~5), each passed as its full extracted
    text. chunk_size/overlap are passed straight through to
    build_chunk_index — the defaults match chunk_document's; override them
    per call if a section's documents need a different chunking granularity
    (e.g. a smaller chunk_size for short/dense text). relative_threshold/
    min_candidates are passed straight through to bm25_threshold_shortlist,
    and batch_size straight through to propose_evidence_from_chunks_batched
    — see those functions' docstrings for what each controls.

    Returns one row per claim-evidence match, with columns:
    claim_id, memo_id, section, claim_text, doc_id, chunk_id, chunk_text,
    bm25_score, evidence_span, found, confidence, ambiguous_match,
    verbatim_match, human_reviewed, tag

    claim_id stays constant across every row belonging to the same claim.
    Zero matches → one row (found=False, doc_id/chunk_id=None). One match
    → one row. Multiple matches → one row per match; ambiguous_match is
    True only when the matches point at genuinely distinct evidence (see
    _rows_for_claim) — not when they're the same passage duplicated across
    overlapping chunks.

    `claims` is the list of canonical (`.strip()`ed) claim strings for this
    section, as produced by `parse_claims_file`; this function does not
    re-strip them. An empty list yields an empty frame carrying the schema
    columns.

    If evidence lookup fails for an individual claim, that claim still gets
    one error row (confidence="error") so one bad claim doesn't drop the
    rest of the section.

    After the claim loop completes, logs one INFO line with the
    section-wide totals of dropped and recovered evidence entries, summed
    across every claim's propose_evidence_from_chunks_batched call (see
    that function's and propose_evidence_from_chunks's docstrings for what
    "dropped"/"recovered" mean). A claim whose evidence lookup raises
    entirely (caught above, turned into an error row) contributes no
    counts to this total, the same way it contributes no evidence rows.
    """
    if not isinstance(claims, list):
        raise TypeError(
            "build_golden_set_draft takes a list of claim strings, not section "
            "text — extraction is now Stage 1 (design decision 17)"
        )

    chunk_index = build_chunk_index(source_documents, chunk_size=chunk_size, overlap=overlap)
    chunk_id_to_doc_id = {c["chunk_id"]: c["doc_id"] for c in chunk_index}

    rows = []
    section_counts = {"dropped": 0, "recovered": 0}
    for claim, occ in _claims_with_occurrence(claims):
        claim_id = _derive_claim_id(memo_id, section_name, claim, occ)
        try:
            shortlist = bm25_threshold_shortlist(
                claim, chunk_index, relative_threshold=relative_threshold, min_candidates=min_candidates
            )
            matches, claim_counts = propose_evidence_from_chunks_batched(
                claim, shortlist, llm_client, batch_size=batch_size
            )
            section_counts["dropped"] += claim_counts["dropped"]
            section_counts["recovered"] += claim_counts["recovered"]
            rows.extend(_rows_for_claim(memo_id, section_name, claim_id, claim, matches, chunk_id_to_doc_id))
        except Exception as exc:
            logger.error(
                "[%s/%s] evidence lookup failed for claim %r: %s",
                memo_id,
                section_name,
                claim,
                exc,
            )
            error_row = _error_row(memo_id, section_name, claim_text=claim)
            error_row["claim_id"] = claim_id
            rows.append(error_row)

    logger.info(
        "[%s/%s]: %d claim(s), %d dropped, %d recovered evidence entries total",
        memo_id,
        section_name,
        len(claims),
        section_counts["dropped"],
        section_counts["recovered"],
    )
    return pd.DataFrame(rows, columns=_SCHEMA_COLUMNS)


# %% [markdown]
# ## 6. Batch runner with checkpointing
#
# Loops `build_golden_set_draft` over a list of sections and saves a
# `.parquet` checkpoint every `checkpoint_every` sections, so a crash partway
# through a long batch doesn't lose completed work.

# %%
# The three tuning knobs a memo may override. Also the basis for _MEMO_FIELDS
# in section 6b, so the config loader and the check below can't drift apart.
_OVERRIDE_FIELDS = {"relative_threshold", "min_candidates", "batch_size"}

# The keys a claims-file frontmatter block may carry: the two required
# identity keys, the optional filing_entity (the company the memo is about;
# only tag_pipeline's `draft` reads it, via read_filing_entity), plus the
# same three optional tuning overrides memos.yaml allows (design decision
# 12). Anything else is rejected, not ignored.
_CLAIMS_FRONTMATTER_FIELDS = {"memo_id", "source_folder", "filing_entity"} | _OVERRIDE_FIELDS

# C0 control characters except tab (\t); \n and \r are already gone after
# line splitting. Rejected in claim text and section names so the \x1f
# claim_id delimiter is collision-free and review text stays clean.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# An ATX heading close: whitespace then a run of '#' at end of line, as in
# '## Foo ##'. Compiled once and shared by everything that strips it or asks
# whether stripping would change a name -- the parser (_classify_claims_body),
# the drift scanner (_scan_claims_file_sections) and the section-name rule
# (_section_name_issue). Those must apply the SAME transformation or a name
# can pass validation yet parse back differently than it was written, breaking
# the write_claims_file -> parse_claims_file round-trip theorem. Note 'C#'
# has no whitespace before the '#', so it is not a close and survives.
_ATX_CLOSE_RE = re.compile(r"\s+#+$")

# A memo id, which doubles as the claims-file name (claims/<memo_id>.md).
# Shared by _read_memo_config (memos.yaml), _parse_frontmatter (reading a
# claims file) and write_claims_file (writing one) -- the same three call
# sites _valid_section_name centralises the section-name rule for, and for
# the same reason: if _read_memo_config's copy drifts looser than
# write_claims_file's, run_extract accepts an id, pays for every section's
# extraction, and only then fails at write time -- losing the cheap
# validate-before-spend guard design decision 17 exists to provide.
_MEMO_ID_RE = re.compile(r"[A-Za-z0-9._-]+")


def _valid_memo_id(memo_id) -> bool:
    """True if `memo_id` is a string usable as both a memo id and a filename."""
    return isinstance(memo_id, str) and _MEMO_ID_RE.fullmatch(memo_id) is not None


def _validate_memo_sections_shape(memo_sections: list[tuple]) -> None:
    """
    Checks the shape of every memo_sections entry, raising ValueError
    naming the offending entry's position (and memo, once known).

    An entry is either a 4-tuple (hand-written, no overrides) or a
    5-tuple whose 5th element is a dict holding any of _OVERRIDE_FIELDS.
    Every other shape raises rather than being quietly read as "no
    overrides": a 6-tuple, a non-dict 5th element, or a misspelled
    override key would each otherwise run that memo on the batch
    defaults with nothing raised and nothing logged. The misspelled key
    is the worst of the three -- the run would go on to log "per-memo
    overrides in effect" while reporting the defaults, so the log would
    actively assert the memo ran tuned when it did not.

    The 3rd element must be a list of claims. build_golden_set_draft
    enforces this too, but only once that entry is reached -- so a list
    whose entry 40 still carries raw section text (a hand-built list, or
    one concatenated from an older caller) would raise only after 39
    sections of real spend. Same reasoning as the rest of this pre-pass.

    Runs as a pre-pass over the whole list rather than per-entry inside
    the run loop. These are pure structural checks that touch no files
    and cost no API calls, so there's no reason to discover a bad entry
    at position 40 after 39 sections of real spend -- work the periodic
    checkpoint may not have saved yet.
    """
    for i, entry in enumerate(memo_sections):
        if len(entry) not in (4, 5):
            raise ValueError(f"memo_sections[{i}]: expected a 4- or 5-tuple, got {len(entry)}")
        if not isinstance(entry[2], list):
            raise ValueError(
                f"memo_sections[{i}] ({entry[0]}): the 3rd element must be a list of claims, "
                f"got {type(entry[2]).__name__} — since the two-stage split, this slot holds "
                f"already-split claims (from a claims file), not raw section text; "
                f"load_memo_sections_from_claims produces the right shape"
            )
        if len(entry) == 4:
            continue
        memo_id, overrides = entry[0], entry[4]
        if not isinstance(overrides, dict):
            raise ValueError(
                f"memo_sections[{i}] ({memo_id}): the 5th element must be a dict of "
                f"per-memo overrides, got {type(overrides).__name__}"
            )
        # sorted(..., key=repr) because a hand-built dict may mix key types,
        # which a bare sorted() would reject with a TypeError naming no memo.
        unrecognized = set(overrides) - _OVERRIDE_FIELDS
        if unrecognized:
            raise ValueError(
                f"memo_sections[{i}] ({memo_id}): unrecognized override key(s) "
                f"{sorted(unrecognized, key=repr)} — expected any of {sorted(_OVERRIDE_FIELDS)} "
                f"(an unrecognized key would otherwise be ignored and this memo would "
                f"run on the batch defaults)"
            )


def build_golden_set_batch(
    memo_sections: list[tuple],
    llm_client: LLMClient,
    checkpoint_path: str,
    checkpoint_every: int = 3,
    relative_threshold: float = 0.3,
    min_candidates: int = 5,
    batch_size: int = 40,
) -> pd.DataFrame:
    """
    Runs build_golden_set_draft over a list of
    (memo_id, section_name, claims, source_documents) tuples — claims
    being that section's list of claim strings, source_documents being
    that section's list of (doc_id, doc_text) source PDFs — concatenating
    the results into one DataFrame.

    An entry may instead be a 5-tuple, with a dict of per-memo overrides
    appended: {"relative_threshold": ..., "min_candidates": ...,
    "batch_size": ...}, holding only the keys that memo actually
    overrides. That's the shape load_memo_sections_from_claims returns
    (load_memo_sections_from_config's 5-tuples don't compose here directly
    -- their slot 3 is section text, not a claims list; that loader's only
    remaining production caller is preview_claim_splits, while Stage 1 --
    run_extract -- reads memos.yaml directly via _read_memo_config
    instead, see design decision 17); a hand-written list of plain
    4-tuples is equally valid and behaves exactly as it did before
    overrides existed. Any other tuple length, a
    5th element that isn't a dict, or an override key that isn't one of
    the three above raises ValueError naming the offending entry rather
    than being quietly ignored — checked for the whole list before the
    first section runs, so a bad entry costs nothing.

    relative_threshold/min_candidates/batch_size are passed straight
    through to each section's build_golden_set_draft call (see its
    docstring) — real cost/latency tuning knobs on a large corpus, not
    just theoretical parameters, which is why a memo can override them.
    A memo's override applies to that memo's sections only; a knob it
    doesn't override falls back to the value passed here. Any section
    running with an override logs one INFO line naming the effective
    values, so a long run's log shows what each memo actually ran with.

    Saves a parquet checkpoint to `checkpoint_path` every `checkpoint_every`
    sections (and once more at the end). If the process crashes partway
    through, the checkpoint file holds every section completed up to that
    point — rerun with the remaining tuples and pd.concat the two results.
    """
    # Checked up front, before any section runs: a malformed entry late in the
    # list would otherwise surface only after the sections before it had already
    # been paid for.
    _validate_memo_sections_shape(memo_sections)

    completed_frames: list[pd.DataFrame] = []
    for i, entry in enumerate(memo_sections, start=1):
        # A 4-tuple comes from a hand-written list (no overrides); a 5-tuple
        # from load_memo_sections_from_claims, with that memo's override dict
        # appended (load_memo_sections_from_config's tuples don't compose
        # here -- their slot 3 is section text, not a claims list; that
        # loader's only remaining production caller is preview_claim_splits,
        # while Stage 1 -- run_extract -- reads memos.yaml directly via
        # _read_memo_config instead, see design decision 17).
        memo_id, section_name, claims, source_documents = entry[:4]
        overrides = entry[4] if len(entry) == 5 else {}

        section_relative_threshold = overrides.get("relative_threshold", relative_threshold)
        section_min_candidates = overrides.get("min_candidates", min_candidates)
        section_batch_size = overrides.get("batch_size", batch_size)

        logger.info("Processing %s/%s (%d/%d)", memo_id, section_name, i, len(memo_sections))
        if overrides:
            logger.info(
                "[%s/%s] per-memo overrides in effect: relative_threshold=%s, min_candidates=%s, batch_size=%s",
                memo_id,
                section_name,
                section_relative_threshold,
                section_min_candidates,
                section_batch_size,
            )
        section_df = build_golden_set_draft(
            memo_id,
            section_name,
            claims,
            source_documents,
            llm_client,
            relative_threshold=section_relative_threshold,
            min_candidates=section_min_candidates,
            batch_size=section_batch_size,
        )
        completed_frames.append(section_df)

        if i % checkpoint_every == 0:
            pd.concat(completed_frames, ignore_index=True).to_parquet(checkpoint_path)
            logger.info("Checkpoint saved to %s after %d section(s)", checkpoint_path, i)

    result = (
        pd.concat(completed_frames, ignore_index=True)
        if completed_frames
        else pd.DataFrame(columns=_SCHEMA_COLUMNS)
    )
    result.to_parquet(checkpoint_path)
    logger.info("Final checkpoint saved to %s (%d rows)", checkpoint_path, len(result))
    return result


# %% [markdown]
# ## 6b. Config-file loader for batch memo_sections
#
# Builds the `memo_sections` list `build_golden_set_batch` expects, from a
# YAML file instead of hand-typed tuples — so running a new memo means
# editing config, not code.

# Every key a memo entry in the config may carry. Anything else is rejected
# rather than ignored: a misspelled knob (batchsize: 25) would otherwise run
# the memo on the default and say nothing, which is the silent fallback this
# loader exists to prevent.
_MEMO_FIELDS = {"id", "source_folder", "sections"} | _OVERRIDE_FIELDS


def _parse_memo_overrides(memo: dict, label: str) -> dict:
    """
    Pulls the optional per-memo tuning overrides out of one memo's config
    entry, returning only the keys actually present — an omitted key is
    left out entirely so build_golden_set_batch's own default fills in,
    keeping each default defined in exactly one place.

    Validates each value against what the knob's own consumer requires:
    relative_threshold a number in 0.0-1.0 (stored as float),
    min_candidates an int >= 0 (bm25_threshold_shortlist's own floor),
    batch_size an int >= 1 (propose_evidence_from_chunks_batched's own
    floor). Checking here rather than leaving it to those runtime checks
    means a bad value fails before the run starts, not 30 minutes and
    real API spend into a batch — and relative_threshold has no runtime
    check at all, so out of range would otherwise just quietly produce a
    useless shortlist.

    Raises ValueError naming `label` (the memo) and the offending field.
    Note two YAML traps this handles deliberately: bool is a subclass of
    int, so `min_candidates: yes` would slip past a plain isinstance
    check, and a key written with no value (`batch_size:`) parses as None
    — treated as malformed, not as absent.
    """
    overrides: dict = {}

    if "relative_threshold" in memo:
        value = memo["relative_threshold"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{label}: 'relative_threshold' must be a number between 0.0 and 1.0 "
                f"(got {type(value).__name__}: {value!r})"
            )
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{label}: 'relative_threshold' must be between 0.0 and 1.0, got {value!r}")
        overrides["relative_threshold"] = float(value)

    if "min_candidates" in memo:
        value = memo["min_candidates"]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"{label}: 'min_candidates' must be an integer >= 0 (got {type(value).__name__}: {value!r})"
            )
        if value < 0:
            raise ValueError(f"{label}: 'min_candidates' must be >= 0, got {value!r}")
        overrides["min_candidates"] = value

    if "batch_size" in memo:
        value = memo["batch_size"]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"{label}: 'batch_size' must be an integer >= 1 (got {type(value).__name__}: {value!r})"
            )
        if value < 1:
            raise ValueError(f"{label}: 'batch_size' must be >= 1, got {value!r}")
        overrides["batch_size"] = value

    return overrides


# %%
def _load_source_documents(source_folder: str, label: str) -> list[tuple[str, str]]:
    """
    Scans a folder for files whose extension is '.pdf' (matched
    case-insensitively, so 'x.PDF' counts), sorts the paths with a plain
    str sort for a deterministic read order, reads each once via
    load_pdf_text, checks it with warn_if_text_suspiciously_short, and
    returns (doc_id, doc_text) pairs with doc_id = os.path.basename(path).
    Raises ValueError prefixed with `label` (a memo id or a claims-file
    name) if the folder is missing or holds no PDFs. Shared by
    load_memo_sections_from_config and load_memo_sections_from_claims;
    lifted verbatim from the former's old inline block.
    """
    if not os.path.isdir(source_folder):
        raise ValueError(f"{label}: source_folder {source_folder!r} is not a directory")
    pdf_paths = sorted(
        os.path.join(source_folder, n)
        for n in os.listdir(source_folder)
        if n.lower().endswith(".pdf") and os.path.isfile(os.path.join(source_folder, n))
    )
    if not pdf_paths:
        raise ValueError(f"{label}: source_folder {source_folder!r} contains no PDF files")
    docs: list[tuple[str, str]] = []
    for pdf_path in pdf_paths:
        doc_id = os.path.basename(pdf_path)
        doc_text = load_pdf_text(pdf_path)
        warn_if_text_suspiciously_short(doc_id, doc_text)
        docs.append((doc_id, doc_text))
    return docs


def _read_memo_config(config_path: str) -> list[dict]:
    """
    Reads and validates memos.yaml's shape — YAML parse, then per-memo
    id/source_folder/sections/section-name/override validation, as one
    pre-pass over every memo — without reading any PDF. Returns one dict
    per memo: {"id": str, "source_folder": str, "sections": dict[str, str]
    (name -> text, insertion-ordered as written), "overrides": dict}.

    Raises ValueError naming the offending memo (by id, or its position in
    the list if id is itself missing or not a string — a bare, unquoted
    numeric-looking id like `001` parses as the integer 1 under YAML's
    implicit typing, not the string "001", so it's rejected rather than
    silently carried through as the wrong value), a memo id that isn't
    [A-Za-z0-9._-]+ (it is also used as a claims-file name in the
    two-stage flow), or section if a required field is missing, sections
    is empty, a section's text isn't a non-blank string (e.g. a YAML list
    instead of a block-scalar string — an easy slip that YAML itself
    won't flag), a section name that isn't already stripped or contains a
    control character, a memo id is reused by an earlier entry in the
    same config, an override value is malformed, or a memo carries a key
    this loader doesn't recognize (a misspelled `batchsize: 25` is a
    malformed override in effect: it would silently yield the default
    instead of the value that was written) — never a silent skip.

    Shared by load_memo_sections_from_config (which follows this with a
    per-memo _load_source_documents call and its "Loaded memo …" log
    line) and the extract stage, which validates memos.yaml without
    touching any PDF.
    """
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    memos = config.get("memos") if isinstance(config, dict) else None
    if not memos:
        raise ValueError(f"{config_path}: no 'memos' list found (or it's empty)")

    result: list[dict] = []
    seen_ids: set[str] = set()
    for i, memo in enumerate(memos):
        if not isinstance(memo, dict):
            raise ValueError(f"memos[{i}]: memo entry must be a mapping, got {type(memo).__name__}")

        memo_id = memo.get("id")
        label = memo_id if isinstance(memo_id, str) and memo_id.strip() else f"memos[{i}]"
        if not isinstance(memo_id, str) or not memo_id.strip():
            raise ValueError(
                f"{label}: 'id' must be a non-blank string (got {type(memo_id).__name__}: "
                f"{memo_id!r} — if this looks like a number, quote it in the YAML, e.g. id: \"001\")"
            )
        if not _valid_memo_id(memo_id):
            raise ValueError(
                f"{label}: 'id' must match [A-Za-z0-9._-]+ (it is also used as a claims-file "
                f"name in the two-stage flow), got {memo_id!r}"
            )
        # Case-folded, because the id is also the claims-file name: on a
        # case-insensitive filesystem (APFS, HFS+, NTFS) 'Memo' and 'memo'
        # are one file, so run_extract's never-overwrite guard would count
        # the second memo as *skipped* -- exit status 0, no failure -- and
        # build would then silently produce a golden set missing that memo
        # entirely. An exact-match check cannot see that collision coming.
        if memo_id.casefold() in seen_ids:
            raise ValueError(
                f"{label}: 'id' is reused by an earlier entry in {config_path} "
                f"(compared case-insensitively: each id becomes a claims-file name, "
                f"and two ids differing only in case would collide on one file)"
            )
        seen_ids.add(memo_id.casefold())

        source_folder = memo.get("source_folder")
        if not isinstance(source_folder, str) or not source_folder.strip():
            raise ValueError(
                f"{label}: 'source_folder' must be a non-blank string "
                f"(got {type(source_folder).__name__}: {source_folder!r})"
            )

        sections = memo.get("sections")
        if not sections:
            raise ValueError(f"{label}: missing required field 'sections' (or it's empty)")
        if not isinstance(sections, dict):
            raise ValueError(f"{label}: 'sections' must be a mapping of section name to text")

        for section_name, section_text in sections.items():
            if not isinstance(section_text, str) or not section_text.strip():
                raise ValueError(
                    f"{label} / section '{section_name}': text must be a non-blank string "
                    f"(got {type(section_text).__name__} — check the YAML uses a block-scalar "
                    f"value, not a bulleted list, for this section)"
                )
            issue = _section_name_issue(section_name)
            if issue:
                raise ValueError(
                    f"{label} / section {section_name!r}: invalid section name -- "
                    f"{issue}"
                )

        unrecognized = set(memo) - _MEMO_FIELDS
        if unrecognized:
            raise ValueError(
                f"{label}: unrecognized field(s) {sorted(unrecognized, key=repr)} — expected one of "
                f"{sorted(_MEMO_FIELDS)} (check for a typo; an unrecognized key would "
                f"otherwise be ignored and this memo would run on the defaults)"
            )
        overrides = _parse_memo_overrides(memo, label)

        result.append(
            {"id": memo_id, "source_folder": source_folder, "sections": sections, "overrides": overrides}
        )

    return result


def load_memo_sections_from_config(
    config_path: str,
) -> list[tuple[str, str, str, list[tuple[str, str]], dict]]:
    """
    Reads a YAML config describing one or more memos and returns a
    memo_sections list -- but NOT one ready to pass into
    build_golden_set_batch: slot 3 of each tuple is that section's raw
    text, and build_golden_set_batch now requires a claims list there
    (design decision 17), so passing this straight through raises
    TypeError. Its only remaining production caller is
    preview_claim_splits, which splits that text into claims itself for
    its own preview purposes; Stage 1 (run_extract) does not call this
    function at all -- it reads memos.yaml directly via _read_memo_config
    instead. load_memo_sections_from_claims is the loader whose output
    actually composes with build_golden_set_batch.

    Expected shape:

        memos:
          - id: MEMO-001
            source_folder: sources/acme
            relative_threshold: 0.45     # optional
            min_candidates: 8            # optional
            batch_size: 25               # optional
            sections:
              Business Profile: >-
                Acme is the largest...
              Ownership: >-
                Acme is listed...

    Each memo's source_folder is scanned for *.pdf files (sorted by name
    for a deterministic read order); every PDF is read once via
    load_pdf_text and checked with warn_if_text_suspiciously_short, then
    shared as that memo's source_documents across every one of its
    sections — (doc_id, doc_text) pairs with
    doc_id = os.path.basename(pdf_path), matching load_pdf_text's own
    docstring convention.

    relative_threshold/min_candidates/batch_size are optional per-memo
    tuning overrides (see _parse_memo_overrides for the accepted ranges).
    They're returned as a 5th element on each of that memo's tuples — a
    dict holding only the keys that memo actually set — and applied by
    build_golden_set_batch to that memo's sections alone; a knob a memo
    doesn't set falls back to the value passed to that batch run. They
    exist because observed corpora range from roughly 1,000 to 6-7,000
    chunks per memo, too wide a spread for one setting to serve well (see
    CLAUDE.md design decision 7). Nothing auto-tunes: an override is only
    ever what someone wrote in this file.

    Raises ValueError naming the offending memo (by id, or its position
    in the list if id is itself missing or not a string — a bare, unquoted
    numeric-looking id like `001` parses as the integer 1 under YAML's
    implicit typing, not the string "001", so it's rejected rather than
    silently carried through as the wrong value), a memo id that isn't
    [A-Za-z0-9._-]+ (it is also used as a claims-file name in the
    two-stage flow), or section if a required field is missing, sections
    is empty, a section's text isn't a non-blank string (e.g. a YAML list
    instead of a block-scalar string — an easy slip that YAML itself
    won't flag), a section name that isn't already stripped or contains a
    control character, a memo id is reused by an earlier entry in the
    same config, an override value is malformed, a memo carries a key
    this loader doesn't recognize (a misspelled `batchsize: 25` is a
    malformed override in effect: it would silently yield the default
    instead of the value that was written), or source_folder doesn't
    exist or contains no PDFs (matched case-insensitively, so a `.PDF`
    file isn't silently skipped) — never a silent skip.
    """
    memo_sections: list[tuple[str, str, str, list[tuple[str, str]], dict]] = []
    for memo in _read_memo_config(config_path):
        source_documents = _load_source_documents(memo["source_folder"], memo["id"])
        logger.info(
            "Loaded memo %s: %d source PDF(s), %d section(s)%s",
            memo["id"], len(source_documents), len(memo["sections"]),
            f", overrides: {memo['overrides']}" if memo["overrides"] else "",
        )
        for section_name, section_text in memo["sections"].items():
            memo_sections.append(
                (memo["id"], section_name, section_text, source_documents, memo["overrides"])
            )
    return memo_sections


# %% [markdown]
# ## 6c. Claims file: parse, write, scan
#
# The two-stage pipeline hands claims between `extract` and `build` as a
# human-editable Markdown file — one per memo, `---`-fenced YAML frontmatter
# (memo_id, source_folder, optional filing_entity and tuning overrides) followed by `## `
# sections of numbered claims. `parse_claims_file` is the strict reader
# `build` uses; `write_claims_file` is its exact inverse, the writer
# `extract` uses; `_scan_claims_file_sections` is a lenient, never-raising
# heading scan used only to warn `extract` about section drift against
# memos.yaml — not for parsing claims themselves.

# %%
def _scan_claims_file_sections(path: str) -> list[str]:
    """
    Best-effort list of a claims file's '## ' heading names, for extract's
    drift check only. Never raises on a malformed body. .splitlines() (not
    design spec section 5's strict \\n-only normalisation) is deliberate: the worst an
    exotic separator inside a heading can do here is a spurious drift
    WARNING -- it can never skip a section or block a run. A malformed
    existing file (e.g. '### ' headings) returns [], so drift then reports
    every memos.yaml section as "added"; misleading but harmless (the
    sections are there, just malformed, and build will give the real error).
    """
    with open(path, encoding="utf-8-sig") as f:
        text = f.read()
    return [
        _ATX_CLOSE_RE.sub("", ln[3:].rstrip())   # drop an ATX close ('## Foo ##'), keep 'C#'
        for ln in text.splitlines()
        if ln.startswith("## ") and ln[3:].strip()
    ]


def _parse_frontmatter(lines: list[str], name: str) -> tuple[str, str, str | None, dict, int]:
    """
    Validates the '---'-fenced YAML block at the top of a claims file.
    Returns (memo_id, source_folder, filing_entity or None when absent,
    overrides-only-when-set, index of the first body line). Raises
    ValueError naming `name`. See design spec section 5 Phase A.
    """
    i = 0
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    if i >= len(lines) or lines[i].strip() != "---":
        raise ValueError(f"{name}: must start with a '---' frontmatter fence")
    open_at = i
    i += 1
    start = i
    while i < len(lines) and lines[i].strip() != "---":
        i += 1
    if i >= len(lines):
        raise ValueError(f"{name}: frontmatter fence opened at line {open_at + 1} is never closed")
    block = "\n".join(lines[start:i])
    first_body_lineno = i + 1  # 0-based index of the line after the closing fence

    if block.strip() == "":
        raise ValueError(f"{name}: frontmatter is empty — it needs at least memo_id and source_folder")
    try:
        fm = yaml.safe_load(block)
    except yaml.YAMLError as exc:
        raise ValueError(f"{name}: frontmatter is not valid YAML: {exc}")
    if not isinstance(fm, dict):
        raise ValueError(f"{name}: frontmatter must be a mapping, got {type(fm).__name__}")

    unknown = set(fm) - _CLAIMS_FRONTMATTER_FIELDS
    if unknown:
        raise ValueError(
            f"{name}: unrecognized frontmatter key(s) {sorted(unknown, key=repr)} — "
            f"expected {sorted(_CLAIMS_FRONTMATTER_FIELDS)}"
        )

    memo_id = fm.get("memo_id")
    stem = os.path.splitext(name)[0]
    if not isinstance(memo_id, str) or not memo_id.strip():
        raise ValueError(f"{name}: 'memo_id' must be a non-blank string (got {memo_id!r})")
    if not _valid_memo_id(memo_id):
        raise ValueError(f"{name}: 'memo_id' {memo_id!r} must match [A-Za-z0-9._-]+ (it is also the filename stem)")
    if memo_id != stem:
        raise ValueError(f"{name}: 'memo_id' {memo_id!r} must equal the filename stem {stem!r}")

    source_folder = fm.get("source_folder")
    if not isinstance(source_folder, str) or not source_folder.strip():
        raise ValueError(f"{name}: 'source_folder' must be a non-blank string (got {source_folder!r})")

    # Optional, but a key written with no value parses as None -- treated as
    # malformed, not as absent (the same YAML trap _parse_memo_overrides handles).
    filing_entity = fm.get("filing_entity")
    if "filing_entity" in fm and (not isinstance(filing_entity, str) or not filing_entity.strip()
                                  or "\n" in filing_entity):
        raise ValueError(f"{name}: 'filing_entity' must be a non-blank one-line string (got {filing_entity!r})")

    overrides = _parse_memo_overrides(fm, name)
    return memo_id, source_folder, filing_entity, overrides, first_body_lineno


def parse_claims_file(path: str) -> tuple[str, str, dict, list[tuple[str, list[str]]]]:
    """
    Parses one per-memo claims file into
    (memo_id, source_folder, overrides, [(section_name, [claim, ...]), ...]).

    The frontmatter is '---'-fenced YAML (memo_id + source_folder required,
    filing_entity and the three tuning-override keys optional, anything else
    rejected; filing_entity is validated here but not returned -- see
    read_filing_entity). The
    body is '## ' sections each holding one-line 'N.' claims, each starting
    at column 0 -- an indented 'N.' is not a claim. Every ambiguous shape is
    a ValueError naming the file. A complete single-line '<!-- ... -->'
    comment is a silent note -- the only kind. Stray prose under a section
    is also ignored, not turned into a claim, but logs a WARNING naming the
    section and quoting the line, since prose is indistinguishable from a
    claim whose leading 'N. ' was simply forgotten; use '<!-- ... -->' for
    a note you want silent. Touches only its own file -- source_folder
    existence is checked later, by _load_source_documents. See design spec
    section 5 for the full grammar.
    """
    name = os.path.basename(path)
    with open(path, encoding="utf-8-sig") as f:
        raw = f.read()
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    memo_id, source_folder, _, overrides, first_body = _parse_frontmatter(lines, name)
    sections = _classify_claims_body(lines[first_body:], first_body, name)
    return memo_id, source_folder, overrides, sections


def read_filing_entity(path: str) -> str | None:
    """
    The claims file's optional `filing_entity` -- the company the memo is
    about -- or None when the frontmatter does not set it. Reads the
    frontmatter only; raises ValueError for a malformed one exactly as
    parse_claims_file would. Only tag_pipeline's `draft` needs it (its
    prompt tells the model that "we", "our", "the Group" mean this
    company); `build` ignores it. Kept out of parse_claims_file's return
    value so that function's three other consumers are unaffected.
    """
    name = os.path.basename(path)
    with open(path, encoding="utf-8-sig") as f:
        raw = f.read()
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    _, _, filing_entity, _, _ = _parse_frontmatter(lines, name)
    return filing_entity


# Anchored at column 0 -- an indented 'N. ' is deliberately NOT a claim
# marker (see _classify_claims_body): it's ambiguous between a mis-indented
# claim, a nested sub-point, and a continuation of the claim above, and
# silently promoting it to a top-level claim would manufacture a claim with
# its own claim_id, BM25 shortlist and evidence LLM call.
_CLAIM_MARKER_RE = re.compile(r"^\d+\.\s+")
_LOOSE_LIST_RE = re.compile(r"^\s*(?:\d+[.)](?:\s|$)|[-*]\s)")
_COMPLETE_COMMENT_RE = re.compile(r"^<!--.*-->$")


def _trips_comment_guard(line: str) -> bool:
    """
    True if `line` (already stripped) looks like one end of a multi-line HTML
    comment, which this format doesn't support: it opens one ('<!--...'),
    closes one ('-->'), or trails an opener at end of line ('...<!--').

    THE definition of that guard, not a copy of it: _classify_claims_body
    raises on it while parsing, and write_claims_file refuses to write a
    heading or claim whose rendered line would hit it, so the writer's accept
    set matches the parser's by construction rather than by two expressions
    being kept identical by hand.

    Callers apply their own preceding checks. _classify_claims_body tests
    _COMPLETE_COMMENT_RE ('^<!--.*-->$') first, so a complete single-line
    comment never reaches here; for write_claims_file that bypass is
    unreachable anyway, since every line it checks begins '## ' or 'N. '.
    """
    return line.startswith("<!--") or line == "-->" or line.endswith("<!--")



def _classify_claims_body(
    body_lines: list[str], first_body_lineno: int, name: str
) -> list[tuple[str, list[str]]]:
    """
    Classifies each body line (design spec section 5, Phase B). A claim
    marker must start at column 0 -- an indented 'N.' is never a claim, only
    ever a loose-list candidate or (abutting the line above) a continuation
    error; see _CLAIM_MARKER_RE. Checks are tried in order -- the '#'-typo
    check precedes the claim-continuation check, so a heading typo right
    after a claim gets the heading message. Returns
    [(section_name, [claim, ...]), ...] in file order. Phase C post-checks
    (empty section, no sections, ignored-line WARNING) run here too. The
    ignored-line WARNING fires for every non-blank line under a section that
    isn't blank, a comment, a heading, or a claim -- not just lines shaped
    like a loose list -- since a forgotten claim number is otherwise
    indistinguishable from deliberate prose and would be dropped silently.
    """
    sections: list[tuple[str, list[str]]] = []
    by_name: dict[str, list[str]] = {}
    loose_under: dict[str, list[str]] = {}
    loose_preamble: list[str] = []
    current: str | None = None
    prev_was_claim = False

    for offset, line in enumerate(body_lines):
        lineno = first_body_lineno + offset + 1
        stripped = line.strip()

        if stripped == "":
            prev_was_claim = False
            continue
        if _COMPLETE_COMMENT_RE.match(stripped):
            prev_was_claim = False
            continue
        if _trips_comment_guard(stripped):
            raise ValueError(
                f"{name}: line {lineno}: multi-line HTML comments aren't supported — "
                f"comment each line, or use a blank-line-separated note"
            )
        if line.startswith("## "):
            heading = _ATX_CLOSE_RE.sub("", line[3:].rstrip())  # ATX close only; 'C#' survives
            if heading == "":
                raise ValueError(f"{name}: line {lineno}: section heading has no name")
            issue = _section_name_issue(heading)
            if issue:
                raise ValueError(
                    f"{name}: line {lineno}: invalid section heading {heading!r} -- "
                    f"{issue}"
                )
            if heading in by_name:
                raise ValueError(f"{name}: line {lineno}: duplicate section heading {heading!r}")
            by_name[heading] = []
            loose_under[heading] = []
            sections.append((heading, by_name[heading]))
            current = heading
            prev_was_claim = False
            continue
        if re.match(r"^\s*#", line):
            if re.match(r"^#{1,6}[^#\s]", line):
                raise ValueError(f"{name}: line {lineno}: no space after '#' in a heading")
            if line != line.lstrip():
                raise ValueError(f"{name}: line {lineno}: section heading must not be indented")
            raise ValueError(f"{name}: line {lineno}: section headings use exactly '## ' (two hashes, one space)")
        if _CLAIM_MARKER_RE.match(line):
            if current is None:
                raise ValueError(f"{name}: line {lineno}: claim before the first '## ' section heading: {stripped!r}")
            text = _CLAIM_MARKER_RE.sub("", line, count=1).strip()
            if text == "":
                raise ValueError(f"{name}: line {lineno}: claim has no text")
            if _CONTROL_CHARS_RE.search(text):
                raise ValueError(f"{name}: line {lineno}: claim contains a control character")
            by_name[current].append(text)
            prev_was_claim = True
            continue
        # any other non-blank line
        if prev_was_claim:
            raise ValueError(
                f"{name}: line {lineno} looks like it continues the claim above; claims start "
                f"at column 0 and are one line each — put a blank line before an indented list "
                f"item or a note"
            )
        if current is not None:
            # Record EVERY ignored non-blank line under a section, not just ones
            # that already look list-item-shaped (_LOOSE_LIST_RE). A forgotten
            # "N. " on an otherwise ordinary sentence renders identically to
            # deliberate prose -- there is no syntactic way to tell "author meant
            # this as a silent note" from "author dropped a claim by mistake". If
            # this were only recorded when it already looked like a list item, a
            # forgotten number on a plain sentence would be dropped with zero log
            # output (the defect this branch exists to catch). Deliberate trade:
            # a bare prose note now WARNs where it used to be silent; a
            # single-line '<!-- ... -->' comment (handled earlier, by
            # _COMPLETE_COMMENT_RE) is still the silent way to leave a note.
            loose_under[current].append(stripped)
        else:
            # Before the first '## ' heading. A line that still *has* its
            # 'N. ' marker errors in the claim-marker branch above, but one
            # whose number was forgotten (or deleted during review) is
            # indistinguishable from prose -- the same silent-claim-loss
            # shape as under a section, so it gets the same WARNING rather
            # than being dropped with no output at all.
            loose_preamble.append(stripped)

    if not sections:
        raise ValueError(f"{name}: no '## ' section headings found")
    if loose_preamble:
        logger.warning(
            "%s: %d line(s) before the first '## ' heading were not parsed as claims and are "
            "being ignored (not blank, not a '<!-- ... -->' comment). A claim only counts "
            "under a '## ' heading — if one of these was meant to be a claim, move it under "
            "a heading and add the missing 'N. '; if it's an intentional note, wrap it in "
            "'<!-- ... -->' so it's ignored silently instead: %s",
            name, len(loose_preamble), loose_preamble,
        )
    for sec_name, claims in sections:
        if not claims:
            hint = ""
            if loose_under[sec_name]:
                first = loose_under[sec_name][0]
                if _LOOSE_LIST_RE.match(first):
                    hint = (f" (found lines that look like list items but don't match 'N. ' — "
                            f"digit, dot, space: {first!r})")
                else:
                    hint = f" (found an ignored non-claim line: {first!r})"
            raise ValueError(f"{name}: section {sec_name!r} has no claims{hint}")
        if loose_under[sec_name]:
            logger.warning(
                "%s: section %r: %d line(s) under this section were not parsed as claims "
                "and are being ignored (not blank, not a '<!-- ... -->' comment, not an "
                "'N. ' claim). If one of these was meant to be a claim, add the missing "
                "'N. '; if it's an intentional note, wrap it in '<!-- ... -->' so it's "
                "ignored silently instead: %s",
                name, sec_name, len(loose_under[sec_name]), loose_under[sec_name],
            )
    return sections


_EXTRACT_MARKER = "<!-- written by 'extract'; edit freely, then run 'build' -->"


def write_claims_file(
    path: str,
    memo_id: str,
    source_folder: str,
    overrides: dict,
    sections: list[tuple[str, list[str]]],
) -> None:
    """
    Serialises one per-memo claims file, the inverse of parse_claims_file:
    bare '---'-fenced frontmatter (memo_id, source_folder, then only the
    override keys present -- no filled-in defaults), the fixed body marker,
    then '## ' sections with sequentially-numbered 'N.' claims.

    Validates its inputs and raises ValueError on anything parse_claims_file
    would reject, so the round-trip theorem holds and a direct caller can't
    write a corrupt file. run_extract catches that as a per-memo failure.
    Holds for canonical (already-`.strip()`ed) claim text, which is what the
    pipeline produces: a claim passed in with leading/trailing whitespace is
    still accepted, but comes back `.strip()`ed by the round trip, since the
    writer itself does `f"{n}. {c.strip()}"`.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    if not _valid_memo_id(memo_id):
        raise ValueError(f"memo_id {memo_id!r} must match [A-Za-z0-9._-]+")
    if memo_id != stem:
        raise ValueError(f"memo_id {memo_id!r} must equal the filename stem {stem!r}")
    if not isinstance(source_folder, str) or not source_folder.strip():
        raise ValueError("source_folder must be a non-blank string")
    if "\n" in source_folder or "\r" in source_folder:
        raise ValueError(f"source_folder must not contain a newline or carriage return: {source_folder!r}")
    if set(overrides) - _OVERRIDE_FIELDS:
        raise ValueError(f"unrecognized override key(s): {sorted(set(overrides) - _OVERRIDE_FIELDS)}")
    _parse_memo_overrides(overrides, os.path.basename(path))  # range check on the values

    if not sections:
        raise ValueError(f"{os.path.basename(path)}: no sections to write")

    seen: set[str] = set()
    for sec_name, claims in sections:
        issue = _section_name_issue(sec_name)
        if issue:
            raise ValueError(f"invalid section name {sec_name!r} -- {issue}")
        if _trips_comment_guard(f"## {sec_name}"):
            raise ValueError(
                f"section name {sec_name!r} would render as a line the parser reads as an "
                f"unterminated HTML comment (ends in '<!--'); rename it"
            )
        if sec_name in seen:
            raise ValueError(f"duplicate section name {sec_name!r}")
        seen.add(sec_name)
        if not claims:
            raise ValueError(f"section {sec_name!r} has no claims")
        for n, c in enumerate(claims, start=1):
            if not isinstance(c, str) or c.strip() == "":
                raise ValueError(f"section {sec_name!r}: a claim is empty")
            if "\n" in c or "\r" in c or _CONTROL_CHARS_RE.search(c):
                raise ValueError(
                    f"section {sec_name!r}: claim has a newline, carriage return, or control character: {c!r}"
                )
            if _trips_comment_guard(f"{n}. {c.strip()}"):
                raise ValueError(
                    f"section {sec_name!r}: claim {c!r} would render as a line the parser reads as an "
                    f"unterminated HTML comment (ends in '<!--'); rephrase it"
                )

    fm: dict = {"memo_id": memo_id, "source_folder": source_folder}
    for key in ("relative_threshold", "min_candidates", "batch_size"):
        if key in overrides:
            fm[key] = overrides[key]

    parts = ["---\n", yaml.safe_dump(fm, sort_keys=False), "---\n", "\n", _EXTRACT_MARKER + "\n"]
    for sec_name, claims in sections:
        parts.append(f"\n## {sec_name}\n\n")
        for n, c in enumerate(claims, start=1):
            parts.append(f"{n}. {c.strip()}\n")

    # Write to a sibling temp path first, then os.replace() onto the real
    # target -- os.replace is atomic on POSIX and Windows, so a mid-write
    # failure (a full disk, a KeyboardInterrupt) can never leave a
    # truncated file at `path`. The temp path is removed on any failure so
    # a partial file never lingers either.
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write("".join(parts))
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


# %% [markdown]
# ## 6d. Interactive claim-split preview (opt-in)
#
# Claim splitting is one fast LLM call per section, but the split it
# produces isn't visible until the slow per-claim evidence loop has
# finished for the whole batch — potentially hours later, after a bad
# split has already been paid for. `preview_claim_splits` runs just the
# extraction step for each section and prints the claims, pausing between
# sections, so a bad split can be caught and the `extract_atomic_claims`
# prompt fixed before a real run starts. It is opt-in (see the
# `RUN_CLAIM_SPLIT_PREVIEW` cell below), for interactive/notebook use,
# and is never called by `build_golden_set_draft`,
# `build_golden_set_batch`, or the demo at the bottom of this file.

# %%
def preview_claim_splits(memo_sections: list[tuple], llm_client: LLMClient) -> None:
    """
    Walks a memo_sections list, printing each section's atomic-claim
    split one section at a time and pausing for a yes/no between them —
    an interactive sanity check to run BEFORE a real batch run, while a
    bad split still costs only one cheap LLM call to fix.

    memo_sections is the same list build_golden_set_batch takes (4- or
    5-tuples); only the first three elements — memo_id, section_name and
    the section's human text — are used. Sections are processed in the
    given order. For each one this calls extract_atomic_claims once,
    prints the returned claims numbered, then (unless it is the last
    section) waits for input on stdin: "y"/"yes" (any case) continues to
    the next section, anything else stops. A claim-extraction failure
    also stops the preview, with the error printed in place of a claim
    list. When the preview stops early the remaining sections are not
    shown.

    This re-runs extraction rather than reusing what a later
    build_golden_set_draft call will produce, so the expensive path stays
    completely untouched. Because the model is not perfectly
    deterministic even at temperature 0 (see design decisions 5 and 9), a
    real run's split may differ slightly in wording from what the preview
    showed. A systematically bad split, though — over- or under-
    splitting, a stranded pronoun, a claim about the wrong entity — is a
    property of the prompt and the section text, so it recurs and the
    preview catches it. That is what this is for; it is not a guarantee
    about the next run's exact output.

    Prints to stdout and returns None: it changes nothing and writes no
    files. Interactive use only — it reads stdin via input(), so a closed
    or non-interactive stdin raises EOFError (deliberately not caught; the
    RUN_CLAIM_SPLIT_PREVIEW guard keeps this path out of plain script and
    test runs).
    """
    n = len(memo_sections)
    for i, entry in enumerate(memo_sections, start=1):
        memo_id, section_name, human_text = entry[:3]
        print(f"\n=== {memo_id}/{section_name} ({i}/{n}) ===")
        try:
            claims = extract_atomic_claims(human_text, llm_client)
        except Exception as exc:
            print(f"  claim extraction failed: {exc}")
            print("Stopping preview.")
            return
        for j, claim in enumerate(claims, start=1):
            print(f"  {j}. {claim}")
        if not claims:
            print("  (no claims extracted)")
        if i < n:  # nothing to gate after the last section
            answer = input("\nShow next section? [y/N] ").strip().lower()
            if answer not in {"y", "yes"}:
                print("Stopping preview.")
                return
    print("\nPreview complete.")


# %%
# Set to True and run this cell in a notebook to preview each section's
# claim split before a real batch run. Left False so `python
# golden_set_pipeline.py` and the test suite never pause on the input()
# prompt below. Never wired into build_golden_set_batch.
RUN_CLAIM_SPLIT_PREVIEW = False

if RUN_CLAIM_SPLIT_PREVIEW:
    preview_claim_splits(
        load_memo_sections_from_config("memos.yaml"),
        LLMClient.from_env(),
    )


# %% [markdown]
# ## 6e. Stage-2 loader: claims files → memo_sections
#
# The build stage's counterpart to `load_memo_sections_from_config` (§6b):
# instead of reading section text out of `memos.yaml`, it reads every
# claims file (§6c's grammar) under a directory and returns the same
# `memo_sections` shape `build_golden_set_batch` expects, with each
# section's already-split claim list in slot 3 instead of raw section
# text. Placed here, after `parse_claims_file`/`write_claims_file` (§6c)
# and the preview (§6d), rather than immediately beside §6b, since it
# depends on `parse_claims_file`.

# %%
def load_memo_sections_from_claims(
    claims_dir: str = "claims",
) -> list[tuple[str, str, list[str], list[tuple[str, str]], dict]]:
    """
    Builds build_golden_set_batch's memo_sections list from the per-memo
    claims files under `claims_dir` (Stage 2 of the two-stage split). Returns
    5-tuples (memo_id, section_name, claims, source_documents, overrides) --
    the same shape load_memo_sections_from_config returns, except slot 3 is a
    list of claim strings instead of section text.

    Every *.md file is parsed and structurally validated first (frontmatter,
    grammar, filename == memo_id, no cross-file memo_id collision); only then
    are the source PDFs read. A malformed file therefore fails before any PDF
    is loaded. Raises ValueError naming the offending file -- never a silent
    skip.
    """
    if not os.path.isdir(claims_dir):
        raise ValueError(f"{claims_dir}: not a directory — run 'python golden_set_pipeline.py extract' first")
    paths = sorted(
        os.path.join(claims_dir, n)
        for n in os.listdir(claims_dir)
        if n.lower().endswith(".md") and os.path.isfile(os.path.join(claims_dir, n))
    )
    if not paths:
        raise ValueError(
            f"{claims_dir}: no .md claims files found — run "
            f"'python golden_set_pipeline.py extract' first"
        )

    parsed: list[tuple] = []
    by_memo_id: dict[str, str] = {}
    for path in paths:
        memo_id, source_folder, overrides, sections = parse_claims_file(path)
        if memo_id in by_memo_id:
            raise ValueError(
                f"memo_id {memo_id!r} is declared in both "
                f"{os.path.basename(by_memo_id[memo_id])} and {os.path.basename(path)}"
            )
        by_memo_id[memo_id] = path
        parsed.append((memo_id, source_folder, overrides, sections, os.path.basename(path)))

    memo_sections: list[tuple] = []
    for memo_id, source_folder, overrides, sections, fname in parsed:
        source_documents = _load_source_documents(source_folder, fname)
        for section_name, claims in sections:
            memo_sections.append((memo_id, section_name, claims, source_documents, overrides))
    return memo_sections


# %% [markdown]
# ## 6f. Stage 1 orchestrator: `run_extract`
#
# Reads `memos.yaml`, runs `extract_atomic_claims` per section, and writes
# `claims/<memo_id>.md` (§6c) for a human to review before any
# evidence-matching API spend happens. Complete-or-absent per memo: a
# memo's file is written only if its `source_folder` exists AND every
# section extracted to at least one claim — either failure aborts that
# memo's file, never the whole run. Makes no evidence-matching calls.
# `run_build` (Stage 2's counterpart, reading `claims/` and running the
# existing evidence pipeline) and the `__main__` argv dispatch belong
# beside this, in §6g.

# %%
def run_extract(config_path: str = "memos.yaml", claims_dir: str = "claims", *, llm_client) -> dict:
    """
    Stage 1: read memos.yaml, run extract_atomic_claims on each section, and
    write claims/<memo_id>.md -- but only when the memo's source_folder
    exists and every section extracted to at least one claim (either failure
    aborts that memo's file, not the run). Never overwrites an existing
    file; reports it as skipped, and warns if memos.yaml has since gained
    sections the file lacks. Makes no evidence-matching LLM calls -- and the
    source_folder check happens before them, so a mistyped folder costs no
    API spend (design decision 17's "review before API spend"). Returns
    {"written","skipped","drift","failed"} counts (drift is a subset of
    skipped). See design decision 17.
    """
    memos = _read_memo_config(config_path)
    os.makedirs(claims_dir, exist_ok=True)
    counts = {"written": 0, "skipped": 0, "drift": 0, "failed": 0}

    for memo in memos:
        memo_id = memo["id"]
        target = os.path.join(claims_dir, f"{memo_id}.md")

        if os.path.exists(target):
            counts["skipped"] += 1
            # _scan_claims_file_sections only promises never to raise on a
            # malformed *body* -- an unreadable file (re-saved as latin-1 by
            # Word/Excel, or permission-denied) is a different failure class
            # and must not abort the whole run. The file already exists, so
            # it's still skipped (never overwritten); it just can't be
            # checked for drift, since we can't know what sections it has.
            try:
                have = set(_scan_claims_file_sections(target))
            except (OSError, UnicodeDecodeError) as exc:
                logger.warning("skipped %s — could not read existing %s to check for drift: %s", memo_id, target, exc)
                continue
            missing = [s for s in memo["sections"] if s not in have]
            if missing:
                counts["drift"] += 1
                logger.warning("skipped %s — file exists; memos.yaml adds section(s): %s", memo_id, missing)
            else:
                logger.info("skipped %s — file exists", memo_id)
            continue

        # Cheap guard before any extraction LLM call: a mistyped source_folder
        # would otherwise cost API spend and write a claims file that fails at
        # build (design decision 17: review before API spend). The no-PDF case
        # is still build's to catch -- this only checks the folder exists.
        if not os.path.isdir(memo["source_folder"]):
            counts["failed"] += 1
            logger.error(
                "failed %s — source_folder %r is not a directory; no file written",
                memo_id, memo["source_folder"],
            )
            continue

        sections: list[tuple[str, list[str]]] = []
        failed = False
        for section_name, section_text in memo["sections"].items():
            try:
                claims = extract_atomic_claims(section_text, llm_client)
            except Exception as exc:
                counts["failed"] += 1
                failed = True
                logger.error(
                    "failed %s — section %r: %s; no file written — run extract again",
                    memo_id, section_name, exc,
                )
                break
            claims = [c.strip() for c in claims]
            if not claims:
                counts["failed"] += 1
                failed = True
                logger.warning(
                    "failed %s — section %r produced no atomic claims; no file written — "
                    "remove it from memos.yaml or hand-author %s",
                    memo_id, section_name, target,
                )
                break
            sections.append((section_name, claims))
        if failed:
            continue

        # write_claims_file validates its input and raises on anything
        # parse_claims_file would reject (a claim with an interior newline or
        # control char that extract_atomic_claims' JSON happened to contain,
        # a bad section name from memos.yaml, ...). That is a per-memo
        # failure -- it must not abort the run (AC-1, design decision 17).
        try:
            write_claims_file(target, memo_id, memo["source_folder"], memo["overrides"], sections)
        except (ValueError, OSError) as exc:
            counts["failed"] += 1
            logger.error("failed %s — could not write %s: %s; no file written", memo_id, target, exc)
            continue
        counts["written"] += 1
        logger.info("wrote %s (%d sections, %d claims)", memo_id, len(sections), sum(len(c) for _, c in sections))

    logger.info(
        "extract: %d written, %d skipped (%d with drift), %d failed",
        counts["written"], counts["skipped"], counts["drift"], counts["failed"],
    )
    return counts


# %% [markdown]
# ## 6g. Stage 2 orchestrator: `run_build`; `__main__` argv dispatch
#
# Reads the claims files under `claims_dir` (§6e), runs the BM25 +
# evidence pipeline over them, checkpoints to parquet, and writes the
# review spreadsheet. This is the old single-stage demo body (formerly
# under `if __name__ == "__main__":`) with `load_memo_sections_from_claims`
# in place of `load_memo_sections_from_config`, and its hardcoded artifact
# names turned into parameters.

# %%
def run_build(
    claims_dir: str = "claims",
    *,
    llm_client,
    checkpoint_path: str = "golden_set_checkpoint.parquet",
    checkpoint_every: int = 1,
    relative_threshold: float = 0.3,
    min_candidates: int = 5,
    batch_size: int = 40,
    review_path: str = "review.xlsx",
) -> pd.DataFrame:
    """
    Stage 2: read the per-memo claims files under `claims_dir` (via
    load_memo_sections_from_claims), run the BM25 + evidence pipeline over
    them (build_golden_set_batch), checkpoint to parquet, and write the
    review spreadsheet (export_for_review). Returns the resulting DataFrame.

    claims_dir: directory of `<memo_id>.md` claims files (§6c/§6e), the
    output of run_extract. llm_client: passed straight through to
    build_golden_set_batch for the evidence-matching calls. checkpoint_path
    / checkpoint_every: where and how often build_golden_set_batch saves
    progress. review_path: where export_for_review writes the reviewable
    spreadsheet (.csv or .xlsx).

    relative_threshold / min_candidates / batch_size are batch-wide
    defaults, passed straight through to build_golden_set_batch — each one
    is only a default here, since any memo's claims-file frontmatter can
    override it per memo (design decisions 7 and 12; see
    build_golden_set_batch's and write_claims_file's docstrings).
    """
    memo_sections = load_memo_sections_from_claims(claims_dir)

    # Sanity check before trusting anything downstream: how big is the chunk
    # pool bm25_threshold_shortlist is selecting from, for each memo? (One
    # build_chunk_index call per memo, not per section — every section of a
    # memo shares the same source_documents, so the pool is identical.)
    printed: set[str] = set()
    for entry in memo_sections:
        memo_id, _, _, source_documents = entry[:4]
        if memo_id in printed:
            continue
        printed.add(memo_id)
        print(f"{memo_id} chunk pool ({len(source_documents)} doc(s)): "
              f"{len(build_chunk_index(source_documents))} chunks")

    df = build_golden_set_batch(
        memo_sections, llm_client,
        checkpoint_path=checkpoint_path, checkpoint_every=checkpoint_every,
        relative_threshold=relative_threshold, min_candidates=min_candidates, batch_size=batch_size,
    )
    print(df)
    export_for_review(df, review_path)
    return df


def _main(argv: list[str]) -> None:
    """
    The `python golden_set_pipeline.py [extract|build]` dispatch body,
    factored out of `if __name__ == "__main__":` so it is unit-testable
    without `runpy`/`subprocess`. No argument means `build`. An unrecognised
    command raises SystemExit with a usage message naming it, BEFORE
    LLMClient.from_env() runs, so a typo doesn't require credentials to
    diagnose.

    Exit code contract for `extract`: SystemExit(1) if run_extract reports
    any failed memo, so a cron job or shell script wrapping this can branch
    on the process exit code without parsing log output. `build` has no
    analogous partial-failure count to check, so it never raises on its own.
    """
    command = argv[1] if len(argv) > 1 else "build"
    if command not in ("extract", "build"):
        raise SystemExit(f"usage: python golden_set_pipeline.py [extract|build] (got {command!r})")
    client = LLMClient.from_env()
    if command == "extract":
        if run_extract("memos.yaml", "claims", llm_client=client)["failed"]:
            raise SystemExit(1)
    else:
        run_build("claims", llm_client=client)


# %% [markdown]
# ## 7. Export for manual review / re-import corrections
#
# Export writes CSV or Excel depending on the file extension you pass in.
# Excel exports get widened `claim_text`/`chunk_text`/`evidence_span`
# columns and wrapped text, since `DataFrame.to_excel` doesn't expose
# column-width controls itself. Re-import matches rows on
# `(claim_id, chunk_id)`.

# %%
def export_for_review(df: pd.DataFrame, path: str) -> None:
    """
    Writes the draft DataFrame to `path` for manual review. Format is
    chosen by file extension: `.csv` or `.xlsx`.

    For `.xlsx`, the claim_text, chunk_text, evidence_span and
    tag_rationale columns are widened and wrap-text is turned on so long
    strings are readable without you having to resize columns by hand
    first.
    """
    if path.endswith(".xlsx"):
        df.to_excel(path, index=False, sheet_name="golden_set_draft")
        _widen_review_columns(path, df.columns)
    elif path.endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        raise ValueError("path must end in .csv or .xlsx")


def _widen_review_columns(path: str, columns) -> None:
    """
    Post-processes an already-written .xlsx: widens claim_text, chunk_text,
    evidence_span and tag_rationale, sets a modest width on everything else,
    and turns on wrap-text for all data rows.
    """
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment
    from openpyxl.utils import get_column_letter

    wb = load_workbook(path)
    ws = wb.active
    if ws is None:
        # openpyxl returns None only for a workbook with no sheets at all,
        # which the DataFrame.to_excel call above can't produce. Guarding
        # beats an AttributeError raised deep inside the loop below.
        raise ValueError(f"{path} has no active worksheet to widen")

    wide_columns = {"claim_text", "chunk_text", "evidence_span", "tag_rationale"}
    wrap = Alignment(wrap_text=True, vertical="top")
    last_row = ws.max_row

    for idx, col_name in enumerate(columns, start=1):
        letter = get_column_letter(idx)
        ws.column_dimensions[letter].width = 60 if col_name in wide_columns else 18
        for row in range(2, last_row + 1):
            ws[f"{letter}{row}"].alignment = wrap

    wb.save(path)


_NO_CHUNK_SENTINEL = "__NO_CHUNK__"


def import_reviewed(original_df: pd.DataFrame, reviewed_path: str) -> pd.DataFrame:
    """
    Reads a manually reviewed CSV/Excel file back in and merges corrections
    into original_df.

    Matches rows on (claim_id, chunk_id) rather than claim_id alone: a
    claim with ambiguous_match=True produces multiple rows sharing one
    claim_id, so claim_id alone is no longer a unique key. Each row —
    including tag/human_reviewed — is merged independently rather than
    shared across a claim's ambiguous rows, since you may want to confirm
    one match and reject a sibling ambiguous match individually during
    review.

    chunk_id is None for the single "not found" row a claim gets when no
    candidate chunk matched. None/NaN don't reliably compare equal as a
    merge key once a file has been saved and reloaded through Excel/CSV,
    so chunk_id is temporarily replaced with a sentinel string on both
    sides before merging and restored to None afterward.

    Columns overwritten from the reviewed file (where present and
    non-empty): claim_text, doc_id, evidence_span, found, confidence,
    ambiguous_match, verbatim_match, human_reviewed, tag.

    chunk_text and bm25_score are deliberately NOT in that list — they're
    pipeline-sourced (the actual chunk text and that claim's own BM25
    score), not reviewer-editable, so original_df's values for those two
    columns always survive a re-import untouched, even if a reviewer edits
    those cells in the reviewed file.

    Deleting a row is a supported way to reject a match: any (claim_id,
    chunk_id) present in original_df but ABSENT from reviewed_path (because
    you deleted that row in Excel) is NOT left unchanged. DataFrame.update
    only touches rows it finds a matching index for, so a silently deleted
    row would otherwise come back through this function exactly as it went
    in, with no trace you ever reviewed it. Instead, every such row gets
    tag="rejected" and human_reviewed=True set automatically. If you want
    to leave a row untouched rather than reject it, keep it in the
    reviewed file (even with no edits) instead of deleting it.
    """
    if reviewed_path.endswith(".xlsx"):
        reviewed_df = pd.read_excel(reviewed_path)
    elif reviewed_path.endswith(".csv"):
        reviewed_df = pd.read_csv(reviewed_path)
    else:
        raise ValueError("reviewed_path must end in .csv or .xlsx")

    for col in ("claim_id", "chunk_id"):
        if col not in reviewed_df.columns:
            raise ValueError(
                f"reviewed_path is missing a '{col}' column — did you export it with "
                "export_for_review() and keep that column intact while editing?"
            )

    update_columns = [
        "claim_text",
        "doc_id",
        "evidence_span",
        "found",
        "confidence",
        "ambiguous_match",
        "verbatim_match",
        "human_reviewed",
        "tag",
    ]

    original = original_df.copy()
    reviewed = reviewed_df.copy()
    for frame in (original, reviewed):
        frame["chunk_id"] = frame["chunk_id"].where(frame["chunk_id"].notna(), _NO_CHUNK_SENTINEL)

    merged = original.set_index(["claim_id", "chunk_id"])
    reviewed = reviewed.set_index(["claim_id", "chunk_id"])
    merged.update(reviewed[update_columns])

    # Rows deleted from the reviewed file (rejected matches) are not left
    # untouched by update() above -- it only overwrites rows it finds a
    # matching index for. Detect the gap explicitly and mark it as a
    # reviewed rejection rather than silently returning the row unchanged.
    deleted_keys = merged.index.difference(reviewed.index)
    merged.loc[deleted_keys, "tag"] = "rejected"
    merged.loc[deleted_keys, "human_reviewed"] = True

    merged = merged.reset_index()
    merged["chunk_id"] = merged["chunk_id"].replace(_NO_CHUNK_SENTINEL, None)
    return merged


# %% [markdown]
# ## 8. Demo
#
# `python golden_set_pipeline.py extract` then `... build`, dispatched by
# `_main` (section 6g) below, on the memo(s) described in `memos.yaml` — two
# made-up short sections for the example memo MEMO-001, both sharing the same
# source documents, so the multi-document chunking, BM25 shortlisting, and
# ambiguous cross-document matching all get exercised end to end. `extract`
# writes `claims/MEMO-001.md`; review or edit it, then run `build`, which
# reads only that file (never `memos.yaml` again) and produces the golden
# set. Both commands require LLM_BASE_URL / LLM_API_KEY / LLM_MODEL to be
# set (`build`'s evidence matching, not `extract`'s claim splitting, is
# where the real API spend happens). To run a different or additional memo,
# edit `memos.yaml` — no code changes needed.
#
# Before running `extract` for real you can also eyeball each section's
# claim split with the opt-in preview in section 6d — a bad split is far
# cheaper to fix there than after `build`'s evidence loop has run.

# %%
if __name__ == "__main__":
    _main(sys.argv)
