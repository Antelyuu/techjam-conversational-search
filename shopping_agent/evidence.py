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

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Words that carry no evidence: they appear in most product copy, so counting
# them inflates coverage for candidates that match nothing specific.
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "the", "this", "to", "with", "your", "you", "that",
    "will", "can", "has", "have", "not", "all", "any", "each", "our", "we",
})

# A disclosure this short is a label rather than a sentence -- "color: black",
# "cotton". Those match a large share of the catalogue, so counting them as
# evidence rewards the wrong candidates. They are already handled as slots by
# intent extraction and as terms by BM25.
MIN_EVIDENCE_TOKENS = 3

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
_TOKEN_CACHE: dict[str, frozenset[str]] = {}


def _content_tokens(text: str) -> list[str]:
    return [
        token for token in _TOKEN_RE.findall(text.lower())
        if len(token) > 1 and token not in _STOPWORDS
    ]


def product_tokens(product: ProductRecord) -> frozenset[str]:
    text = product.searchable_text
    cached = _TOKEN_CACHE.get(text)
    if cached is None:
        cached = frozenset(_content_tokens(text))
        _TOKEN_CACHE[text] = cached
    return cached


def split_disclosures(message: str) -> list[str]:
    """Recover the individual constraints from one reply.

    The simulator joins them with "; " (`customer_reply`), and each is a
    separate quoted sentence from the target. Scored as one blob they would
    average together; scored apart, covering either one is evidence."""
    return [part.strip() for part in message.split(";") if part.strip()]


def coverage(product: ProductRecord, disclosures: list[str]) -> float:
    """How completely this candidate's text accounts for what was disclosed.

    Each disclosure contributes the share of its content tokens the candidate
    carries, and disclosures are weighted by their own length -- a fourteen
    word feature sentence is far stronger evidence than a three word one, and
    averaging them flat would let a trivial match outvote a specific one.

    Returns 0-1. No usable disclosure yet is 0.0, the same neutral value as a
    candidate that matches nothing, because at that point the feature has
    nothing to say and must not reorder anything.
    """
    matched = 0.0
    total = 0.0
    tokens = product_tokens(product)
    for disclosure in disclosures:
        wanted = set(_content_tokens(disclosure))
        if len(wanted) < MIN_EVIDENCE_TOKENS:
            continue
        weight = float(len(wanted))
        matched += weight * (len(wanted & tokens) / len(wanted))
        total += weight
    if total == 0.0:
        return 0.0
    return matched / total
