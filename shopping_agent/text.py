"""The one tokenizer every scorer shares.

Query building (starter/agent.py) and evidence scoring (evidence.py) each
filter through their own stopword list -- conversational filler is noise in a
BM25 query, product-copy filler is noise in evidence coverage, and the two
sets are deliberately different. The token definition underneath them is not:
a word must split identically everywhere, or the same disclosure tokenizes
one way going into the query and another way being scored against it.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str, stopwords: frozenset[str] | set[str]) -> list[str]:
    """Lowercased alphanumeric tokens, minus one-letter tokens and stopwords."""
    return [
        token for token in _TOKEN_RE.findall(text.lower())
        if len(token) > 1 and token not in stopwords
    ]
