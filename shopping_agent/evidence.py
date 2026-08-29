"""P5-T3: score a candidate against what the customer actually disclosed.

P4 established that asking questions is the phase's whole value, but it spent
the answers cheaply -- a disclosure was appended to the BM25 query as loose
terms and never used again. That undersells what a disclosure *is* on this
task.

The evaluator builds its hidden intent card from the target product's own
`features` and `details` (`local_evaluator.py:52`), whitespace-normalized and
truncated but otherwise **verbatim**. So when the customer answers a question,
they are quoting a sentence out of the one product we are trying to find. A
candidate whose own text covers that sentence is not merely a good lexical
match; it is very probably the target.

BM25 cannot express this. It scores a bag of terms, so a long disclosure is
diluted across many common words and a candidate matching *all* of it scores
much like one matching *most* of it. Coverage of the whole phrase is the
discriminating question, and this module asks it directly.

Measured on the public set: the lexical pool holds the target in 92% of
sessions but only 83% of sessions ever show it, so roughly nine points of
HitRate are lost to ranking a retrieved target out of the top ten. This
feature exists to close that gap.
"""

from __future__ import annotations

import re

from .catalog import ProductRecord
from .text import tokenize

# Phrase normalization keeps every token -- stopwords and single letters
# included -- because contiguity is the signal: "wash in cold" and "wash cold"
# are different phrases even though they filter to the same content tokens.
_PHRASE_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Words that carry no evidence: they appear in most product copy, so counting
# them inflates coverage for candidates that match nothing specific.
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "the", "this", "to", "with", "your", "you", "that",
    "will", "can", "has", "have", "not", "all", "any", "each", "our", "we",
})

# Every disclosure with at least one content token counts.
#
# This was 3, on the reasoning that a bare label -- "cotton", "color: black" --
# matches too much of the catalogue to discriminate and would reward the wrong
# candidates. MEASURED (E6): that reasoning was wrong, and expensively so.
# The Buying opener discloses the card's first constraint, which is usually
# exactly such a label, and discarding it silenced the table's dominant
# feature for the whole session in the sessions with the least other text.
# Counting short labels, down-weighted by their own token count so they can
# never outvote a quoted sentence, is worth +4.0 points of HitRate and
# +0.043 composite. It also closes the review finding that semicolons inside
# a single quoted constraint ("... inches; 7.8 Ounces") split off fragments
# that a 3-token floor then discarded.
MIN_EVIDENCE_TOKENS = 1

# Cache of product token sets, keyed by the text itself.
#
# Scoring re-tokenizes the same fifty-odd pooled products every turn of every
# session, and product text does not change within a run.
#
# Keyed on the text and deliberately not on parent_asin: an id is only unique
# within one catalogue, and two catalogues in one process -- which is what the
# tests are -- would then serve each other's tokens. Hashing the string is
# cheap because ProductRecord is frozen, so the same str object is reused and
# Python caches its hash after the first lookup.
#
# Bounded so a long-lived process cannot accrete a tokenized copy of every
# catalogue it ever loads: past the limit the cache is dropped wholesale and
# rebuilt from the pool traffic that actually recurs. One 50k catalogue fits
# with room to spare, so the submission never trips this.
_TOKEN_CACHE: dict[str, frozenset[str]] = {}
_TOKEN_CACHE_LIMIT = 120_000


def _content_tokens(text: str) -> list[str]:
    return tokenize(text, _STOPWORDS)


def product_tokens(product: ProductRecord) -> frozenset[str]:
    text = product.searchable_text
    cached = _TOKEN_CACHE.get(text)
    if cached is None:
        if len(_TOKEN_CACHE) >= _TOKEN_CACHE_LIMIT:
            _TOKEN_CACHE.clear()
        cached = frozenset(_content_tokens(text))
        _TOKEN_CACHE[text] = cached
    return cached


def split_disclosures(message: str) -> list[str]:
    """Recover the individual constraints from one reply.

    The simulator joins them with "; " (`customer_reply`), and each is a
    separate quoted sentence from the target. Scored as one blob they would
    average together; scored apart, covering either one is evidence.

    A semicolon *inside* one quoted constraint (22% of catalogue rows produce
    one) is indistinguishable from the joiner, so such a constraint arrives
    fragmented. That is tolerated rather than special-cased: coverage()
    weights each piece by its own token count, so the fragments together score
    within dedup noise of the whole."""
    return [part.strip() for part in message.split(";") if part.strip()]


def disclosure_token_sets(disclosures: list[str]) -> list[frozenset[str]]:
    """Tokenize disclosures once, for scoring against many candidates.

    Reranking scores a fifty-candidate pool against the same disclosures every
    turn; tokenizing per candidate did that work fifty times over (review
    finding, P5)."""
    sets = []
    for disclosure in disclosures:
        wanted = frozenset(_content_tokens(disclosure))
        if len(wanted) >= MIN_EVIDENCE_TOKENS:
            sets.append(wanted)
    return sets


def coverage_from_sets(tokens: frozenset[str], wanted_sets: list[frozenset[str]]) -> float:
    """How completely a token set accounts for the disclosed constraints.

    Each disclosure contributes the share of its content tokens the candidate
    carries, weighted by its own length -- a fourteen word feature sentence is
    far stronger evidence than a one word label, and averaging them flat would
    let a trivial match outvote a specific one.

    Returns 0-1. No usable disclosure yet is 0.0, the same neutral value as a
    candidate that matches nothing, because at that point the feature has
    nothing to say and must not reorder anything.
    """
    matched = 0
    total = 0
    for wanted in wanted_sets:
        matched += len(wanted & tokens)
        total += len(wanted)
    if total == 0:
        return 0.0
    return matched / total


def coverage(product: ProductRecord, disclosures: list[str]) -> float:
    """How completely this candidate's text accounts for what was disclosed."""
    return coverage_from_sets(product_tokens(product), disclosure_token_sets(disclosures))


# ---------------------------------------------------------------------------
# Phrase containment (E7). Token coverage ties at ~1.0 across near-duplicate
# catalogue copy -- any candidate sharing the vocabulary scores like the
# target. But the card quotes each constraint *verbatim*, so the true target
# contains the disclosure as a contiguous phrase, and a candidate with the
# same bag of tokens scattered does not. Contiguity is the tie-breaker.

# Same bounding rationale as _TOKEN_CACHE above; kept separate because these
# hold normalized strings, not token sets.
_PHRASE_CACHE: dict[str, str] = {}


def _normalize_phrase(text: str) -> str:
    return " ".join(_PHRASE_TOKEN_RE.findall(text.lower()))


def phrase_text(product: ProductRecord) -> str:
    """The candidate's own text as one normalized, space-padded token stream.

    Normalizing both sides to bare tokens is what lets a details-derived
    disclosure ("Material: 100% Cotton") match the colon-free "Material 100%
    Cotton" flattening in searchable_text. Padded with spaces so containment
    checks are whole-token."""
    text = product.searchable_text
    cached = _PHRASE_CACHE.get(text)
    if cached is None:
        if len(_PHRASE_CACHE) >= _TOKEN_CACHE_LIMIT:
            _PHRASE_CACHE.clear()
        # Normalize each "|"-bounded segment separately and keep the "|"
        # between them: a phrase is pure alphanumerics and spaces, so it can
        # never span a segment boundary (catalog.flatten_field).
        cached = " " + " | ".join(
            _normalize_phrase(segment) for segment in text.split("|")
        ) + " "
        _PHRASE_CACHE[text] = cached
    return cached


# The card clips each constraint at 180 characters (`_clean_constraint`), so
# only a disclosure whose text runs close to that limit can have lost part of
# its final word. Shorter disclosures arrived whole, and granting them the
# dropped-last-token retry below would let "machine wash warm" match a
# "machine wash cold" product by quietly discarding the one token that
# separates them (review finding, P5).
_TRUNCATION_LENGTH = 170


def disclosure_phrases(disclosures: list[str]) -> list[tuple[str, int, bool]]:
    """Normalize disclosures once per rerank pool: (phrase, content weight,
    may-be-truncated).

    The weight is the disclosure's count of *distinct* content tokens -- the
    same `len(wanted)` that coverage_from_sets uses, so the two evidence
    features agree on how much each disclosure is worth -- and the same
    MIN_EVIDENCE_TOKENS floor governs both, so the documented knob means one
    thing (both points were review findings, P5)."""
    phrases = []
    for disclosure in disclosures:
        weight = len(frozenset(_content_tokens(disclosure)))
        if weight < MIN_EVIDENCE_TOKENS:
            continue
        phrases.append(
            (_normalize_phrase(disclosure), weight, len(disclosure) >= _TRUNCATION_LENGTH)
        )
    return phrases


def phrase_coverage_from(text: str, phrases: list[tuple[str, int, bool]]) -> float:
    """Length-weighted share of disclosures the text contains contiguously.

    A disclosure long enough to have hit the card's 180-character clip gets
    one retry without its last token, which may have been cut mid-word; a
    shorter one arrived whole and must match whole. Returns 0-1; no usable
    disclosure is a neutral 0.0, exactly as coverage_from_sets."""
    matched = 0
    total = 0
    for phrase, weight, may_be_truncated in phrases:
        total += weight
        contained = f" {phrase} " in text
        if not contained and may_be_truncated and phrase.count(" ") >= 2:
            contained = f" {phrase.rsplit(' ', 1)[0]} " in text
        if contained:
            matched += weight
    if total == 0:
        return 0.0
    return matched / total
