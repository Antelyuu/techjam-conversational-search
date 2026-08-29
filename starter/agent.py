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

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
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
    session-aware query built from accumulated conversation state,
    optionally fused with a dense semantic route.

    The dense route is on by default, because the official harness
    constructs the agent as Agent(catalog_path) with no arguments and no
    environment variables -- anything opt-in would never run there. It
    still engages only when its bundled artifact and dependencies are
    present; otherwise the agent falls back to BM25 lexical search and
    says so on stderr. Disable it with SHOPPING_AGENT_DENSE=0, and pick
    the blend with SHOPPING_AGENT_FUSION=rrf|weighted."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        enable_dense: bool | None = None,
        fusion_method: str | None = None,
        allow_wildcard: bool | None = None,
        use_disagreement: bool | None = None,
        enable_clarification: bool | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.orchestrator = ConversationOrchestrator()
        self.products: dict[str, ProductRecord] = {}
        self._build_index()

        if allow_wildcard is None:
            allow_wildcard = _env_flag("SHOPPING_AGENT_WILDCARD", default=False)
        if use_disagreement is None:
            use_disagreement = _env_flag("SHOPPING_AGENT_DISAGREEMENT", default=True)
        if enable_clarification is None:
            enable_clarification = _env_flag("SHOPPING_AGENT_CLARIFY", default=True)
        self.allow_wildcard = allow_wildcard
        self.use_disagreement = use_disagreement
        self.enable_clarification = enable_clarification

        if enable_dense is None:
            enable_dense = _env_flag("SHOPPING_AGENT_DENSE", default=True)
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
        )
        recommendations = [
            {"parent_asin": candidate.parent_asin, "score": candidate.route_scores["combined"]}
            for candidate in candidates
        ]
        # Asking is free: the evaluator scores recommendations first and then
        # handles ask_attribute separately, so a question never displaces a
        # recommendation. Always return both.
        ask_attribute = (
            clarification.choose_attribute(
                request.state,
                [self.products[c.parent_asin] for c in candidates],
                allow_wildcard=self.allow_wildcard,
                use_disagreement=self.use_disagreement,
            )
            if self.enable_clarification
            else None
        )
        self.orchestrator.record_question(request.state, ask_attribute)
        return {
            "message": self._build_message(request.state, ask_attribute),
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    def _lexical_search(self, query_text: str, limit: int) -> list[tuple[str, float]]:
        unique_terms = list(dict.fromkeys(_terms(query_text)))[:40]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        if not expression:
            return []
        rows = self.connection.execute(
            "SELECT parent_asin, bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) AS cost "
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
