from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from pathlib import Path

from shopping_agent import clarification
from shopping_agent.catalog import ProductRecord, flatten_field, normalize_product
from shopping_agent.dense_retrieval import load_dense_retriever
from shopping_agent.orchestrator import ConversationOrchestrator
from shopping_agent.reranking import rerank
from shopping_agent.retrieval import DEFAULT_FUSION, FUSION_METHODS, retrieve


# The evaluator matches on ask_attribute alone and never reads these, but a
# recommendation list paired with a bare attribute name is not a conversation.
_QUESTION_TEXT = {
    "feature": "Which features matter most to you?",
    "material": "Is there a material you prefer?",
    "color": "Do you have a colour in mind?",
    "style": "What style are you going for?",
    "size": "What size do you need?",
    "use_case": "What will you mainly use it for?",
    "other": "Is there anything else that matters to you?",
}
_DEFAULT_QUESTION = "Could you tell me a little more about what you need?"

# How deep a pool the reranker reorders before the top ten are shown. Capped
# by the pool retrieval actually built (see retrieve()).
RERANK_POOL = 50

# P5-T1: the dense route is off by default, reversing P3's decision on
# measurement rather than on preference.
#
# P3 adopted dense fusion because it was worth +0.0355 over lexical alone
# (0.1156 -> 0.1511) when every query was a single short opening message. P4
# changed what a query is: the customer now answers questions by quoting
# constraint sentences out of the target product's own features, and those
# accumulate. BM25 sharpens on that -- its recall of the target in a
# 50-candidate pool climbs from 0.38 at turn 1 to 0.74 by turn 3 -- while the
# dense route stays flat near 0.30, because a fixed-width sentence embedding
# averages a growing paragraph toward the corpus mean.
#
# MEASURED (E5): turning the route off is worth +0.050935, and it wins or
# ties every scenario. Down-weighting it instead does not work; there is a
# cliff rather than a slope, because retrieval takes the *union* of both
# routes' candidates and min-max normalization floors the lexical tail at
# 0.0, so dense-only candidates displace real ones even at 5% weight.
#
# Still switchable with SHOPPING_AGENT_DENSE=1. The artifact and the code
# stay in the repository: the finding is about this query distribution, and a
# private set that asks fewer questions would move the balance back.
DENSE_BY_DEFAULT = False

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

# How many distinct query terms reach BM25.
#
# This looked like a silent-truncation bug of the kind P4 found twice, and it
# is not. 8.2% of turns do exceed it, dropping a median of 11 terms and up to
# 42 -- but raising it to 60, 80, 120 or 400 reproduces the composite to six
# decimal places (E5). The reason is that build_query_text() ends with the
# latest raw message, which for an answered question is the same disclosure
# already carried earlier in the query, so the terms a cap discards are
# duplicates. Left at 40, now with a measurement behind it rather than a
# guess.
QUERY_TERM_LIMIT = 40

# BM25 column weights, in the order the FTS table declares them:
# parent_asin, title, categories, features, details, store, description.
# parent_asin is unindexed and always 0.0.
FIELD_WEIGHTS = "0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0"
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _resolve_fusion(value: str | None) -> str:
    """Accept a fusion name case-insensitively, or fall back to the default.

    retrieve() treats anything that is not "weighted" as RRF, so an unnoticed
    typo would quietly select the configuration that measured 0.145170 instead
    of the documented 0.151089. Say so rather than silently downgrading."""
    if not value or not value.strip():
        return DEFAULT_FUSION
    normalized = value.strip().lower()
    if normalized not in FUSION_METHODS:
        print(
            f"[shopping_agent] unknown fusion {value!r}; "
            f"expected one of {sorted(FUSION_METHODS)}. Using {DEFAULT_FUSION}.",
            file=sys.stderr,
        )
        return DEFAULT_FUSION
    return normalized


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


class Agent:
    """Stateful multi-turn agent: BM25 retrieval over a cumulative,
    session-aware query built from accumulated conversation state, reordered
    by a deterministic scorer that leads on how completely a candidate
    accounts for what the customer has actually disclosed.

    Each turn also returns one clarifying question, which is where most of
    this system's score comes from -- the simulator discloses a hidden
    constraint only when asked.

    The dense semantic route is **off** by default as of P5, on measurement:
    it helped when queries were single short messages and hurts now that they
    carry quoted product text (see DENSE_BY_DEFAULT). Re-enable it with
    SHOPPING_AGENT_DENSE=1, and pick the blend with
    SHOPPING_AGENT_FUSION=rrf|weighted. When enabled it still engages only if
    its artifact and dependencies are present, and otherwise falls back to
    BM25 and says so on stderr."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        enable_dense: bool | None = None,
        fusion_method: str | None = None,
        allow_wildcard: bool | None = None,
        use_disagreement: bool | None = None,
        enable_clarification: bool | None = None,
        enable_reranker: bool | None = None,
        block_soft_slots: bool | None = None,
        allow_repeats: bool | None = None,
        route_weights: dict[str, float] | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.orchestrator = ConversationOrchestrator()
        self.products: dict[str, ProductRecord] = {}
        self._build_index()

        if allow_wildcard is None:
            allow_wildcard = _env_flag("SHOPPING_AGENT_WILDCARD", default=True)
        if use_disagreement is None:
            use_disagreement = _env_flag("SHOPPING_AGENT_DISAGREEMENT", default=True)
        if enable_clarification is None:
            enable_clarification = _env_flag("SHOPPING_AGENT_CLARIFY", default=True)
        if enable_reranker is None:
            enable_reranker = _env_flag("SHOPPING_AGENT_RERANK", default=True)
        if block_soft_slots is None:
            block_soft_slots = _env_flag("SHOPPING_AGENT_BLOCK_SOFT", default=False)
        if allow_repeats is None:
            # MEASURED and rejected (E5). Re-asking a high-yield attribute
            # until the card is drained sounds strictly more informative, and
            # costs 0.0078 composite -- all of it in Intent Override
            # (0.800 -> 0.767), where hits cannot register before the override
            # fires and a repeated question crowds out the diversification
            # that finds the new intent. E3 left this open; it is closed now.
            allow_repeats = _env_flag("SHOPPING_AGENT_REPEAT", default=False)
        self.allow_repeats = allow_repeats
        self.allow_wildcard = allow_wildcard
        self.use_disagreement = use_disagreement
        self.enable_clarification = enable_clarification
        self.enable_reranker = enable_reranker
        self.block_soft_slots = block_soft_slots
        self.route_weights = route_weights
        self._warned: set[str] = set()

        if enable_dense is None:
            enable_dense = _env_flag("SHOPPING_AGENT_DENSE", default=DENSE_BY_DEFAULT)
        self.fusion_method = _resolve_fusion(
            fusion_method or os.environ.get("SHOPPING_AGENT_FUSION")
        )
        # None means the route is unavailable (no artifact, no deps, or an
        # artifact built from a different catalogue); the agent then serves
        # BM25 lexical results instead of failing.
        self.dense_search = (
            load_dense_retriever(expected_ids=sorted(self.products)) if enable_dense else None
        )

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                raw = json.loads(line)
                record = normalize_product(raw)
                self.products[record.parent_asin] = record
                batch.append(
                    (
                        record.parent_asin,
                        flatten_field(raw.get("title")),
                        flatten_field(raw.get("categories")),
                        flatten_field(raw.get("features")),
                        flatten_field(raw.get("details")),
                        flatten_field(raw.get("store")),
                        flatten_field(raw.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def close(self) -> None:
        """Release the in-memory FTS index.

        The official harness builds one Agent for a whole run and never needs
        this, but tests and the ablation scripts build many; without it each
        leaks a SQLite connection until garbage collection, which surfaces as
        ResourceWarning. Safe to call more than once.
        """
        connection = getattr(self, "connection", None)
        if connection is not None:
            connection.close()
            self.connection = None

    def __enter__(self) -> "Agent":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.orchestrator.reset(session_id, user_profile)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        request = self.orchestrator.process_turn(session_id, user_message, turn, top_k)
        candidates = retrieve(
            request,
            request.top_k,
            self._lexical_search,
            self.products,
            dense_search=self.dense_search,
            fusion_method=self.fusion_method,
            route_weights=self.route_weights,
            candidate_limit=RERANK_POOL if self.enable_reranker else None,
        )
        if self.enable_reranker:
            # A reranker failure must not cost the turn: the fused order is
            # already a valid ranking, so fall back to it rather than raising
            # into respond() and returning nothing (P4-T4).
            try:
                reranked = rerank(
                    candidates,
                    self.products,
                    request.state.constraints,
                    top_k,
                    request.state.disclosed_text,
                )
                recommendations = [
                    {"parent_asin": item.parent_asin, "score": item.score} for item in reranked
                ]
            except Exception as error:
                self._warn_once(f"reranker failed, using fused order: {error}")
                recommendations = self._fused_recommendations(candidates[:top_k])
        else:
            recommendations = self._fused_recommendations(candidates)
        # Asking is free: the evaluator scores recommendations first and then
        # handles ask_attribute separately, so a question never displaces a
        # recommendation. Always return both.
        # Not asking costs a turn; raising costs the whole session, because the
        # evaluator turns anything escaping respond() into zero recommendations.
        # Guarded for the same reason the dense route and reranker are (P4-T4).
        ask_attribute = None
        if self.enable_clarification:
            try:
                ask_attribute = clarification.choose_attribute(
                    request.state,
                    [self.products[c.parent_asin] for c in candidates],
                    allow_wildcard=self.allow_wildcard,
                    use_disagreement=self.use_disagreement,
                    block_soft_slots=self.block_soft_slots,
                    allow_repeats=self.allow_repeats,
                )
            except Exception as error:
                self._warn_once(f"clarification policy failed, asking nothing: {error}")
        self.orchestrator.record_question(request.state, ask_attribute)
        return {
            "message": self._build_message(request.state, ask_attribute),
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    @staticmethod
    def _fused_recommendations(candidates) -> list[dict]:
        return [
            {"parent_asin": candidate.parent_asin, "score": candidate.route_scores["combined"]}
            for candidate in candidates
        ]

    def _warn_once(self, message: str) -> None:
        if message in self._warned:
            return
        self._warned.add(message)
        print(f"[shopping_agent] {message}", file=sys.stderr)

    def _lexical_search(self, query_text: str, limit: int) -> list[tuple[str, float]]:
        unique_terms = list(dict.fromkeys(_terms(query_text)))[:QUERY_TERM_LIMIT]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        if not expression:
            return []
        rows = self.connection.execute(
            f"SELECT parent_asin, bm25(products, {FIELD_WEIGHTS}) AS cost "
            "FROM products WHERE products MATCH ? ORDER BY cost LIMIT ?",
            (expression, limit),
        ).fetchall()
        # FTS5 bm25() is a cost (lower is better); negate so higher is better,
        # matching the category boost sign used downstream.
        return [(str(row[0]), -float(row[1])) for row in rows]

    @staticmethod
    def _build_message(state, ask_attribute: str | None = None) -> str:
        if state.constraints:
            summary = ", ".join(f"{c.attribute}={c.value}" for c in state.constraints.values())
            opening = f"Here are the closest matches based on {summary}."
        else:
            opening = "Here are the closest matches I found."
        if ask_attribute is None:
            return opening
        return f"{opening} {_QUESTION_TEXT.get(ask_attribute, _DEFAULT_QUESTION)}"
