from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from shopping_agent.catalog import ProductRecord, flatten_field, normalize_product
from shopping_agent.orchestrator import ConversationOrchestrator
from shopping_agent.retrieval import retrieve


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


class Agent:
    """Stateful multi-turn agent: BM25 retrieval over a cumulative,
    session-aware query built from accumulated conversation state."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.orchestrator = ConversationOrchestrator()
        self.products: dict[str, ProductRecord] = {}
        self._build_index()

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
        candidates = retrieve(request, request.top_k, self._lexical_search, self.products)
        recommendations = [
            {"parent_asin": candidate.parent_asin, "score": candidate.route_scores["combined"]}
            for candidate in candidates
        ]
        return {
            "message": self._build_message(request.state),
            "ask_attribute": None,
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
    def _build_message(state) -> str:
        if state.constraints:
            summary = ", ".join(f"{c.attribute}={c.value}" for c in state.constraints.values())
            return f"Here are the closest matches based on {summary}."
        return "Here are the closest matches I found."
