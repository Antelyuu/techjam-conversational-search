"""P3-T3 dense (semantic) retrieval route.

Finds products whose meaning is close to the query, so paraphrases and
scenario-style Browsing messages still retrieve sensible candidates when
the customer's words never appear in the listing.

Exposes the same shape as the lexical route in retrieval.py --
(query_text, pool_size) -> ranked [(parent_asin, score), ...] -- so
retrieve() treats the two identically and never sees vectors, models, or
file paths.

Similarity is an exact flat dot product over the whole catalog matrix
(vectors are L2-normalized at build time, so dot product == cosine). At
50k products that is one small matrix multiply, well inside the latency
budget, and it avoids the extra dependency and approximation error an ANN
index would bring at this scale.

The route is optional by construction: if the artifact has not been built,
or numpy/sentence-transformers are not installed, load_dense_retriever()
returns None and the caller falls back to lexical-only retrieval.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

from .embedding_config import (
    EMBEDDING_DIR,
    MODEL_ID,
    QUERY_PREFIX,
    ids_path,
    vectors_path,
)

# (query_text, pool_size) -> ranked [(parent_asin, cosine_score), ...].
DenseSearchFn = Callable[[str, int], list[tuple[str, float]]]


class DenseRouteUnavailable(RuntimeError):
    """The dense route cannot run: missing artifact, missing dependency, or
    an artifact that does not line up with its id list."""


class DenseRetriever:
    """Loads the embedding artifact and the query encoder once, then answers
    nearest-neighbour queries against the catalog."""

    def __init__(
        self,
        embedding_dir: str | Path = EMBEDDING_DIR,
        model_id: str = MODEL_ID,
        query_prefix: str = QUERY_PREFIX,
        expected_ids: list[str] | None = None,
    ) -> None:
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as error:  # pragma: no cover - depends on install
            raise DenseRouteUnavailable(f"dense dependencies missing: {error}") from error

        vectors_file = vectors_path(embedding_dir)
        ids_file = ids_path(embedding_dir)
        if not vectors_file.exists() or not ids_file.exists():
            raise DenseRouteUnavailable(
                f"embedding artifact not found ({vectors_file}); "
                "run `python3 -m scripts.build_embeddings` first"
            )

        self._np = np
        self._ids: list[str] = json.loads(ids_file.read_text(encoding="utf-8"))
        matrix = np.load(vectors_file)
        if matrix.shape[0] != len(self._ids):
            raise DenseRouteUnavailable(
                f"artifact mismatch: {matrix.shape[0]} vectors vs {len(self._ids)} ids; rebuild the artifact"
            )

        # An artifact built from a different catalogue would map row i to the
        # wrong product and quietly corrupt every dense result, which is worse
        # than not running the route at all.
        if expected_ids is not None and self._ids != expected_ids:
            if len(self._ids) != len(expected_ids):
                detail = f"{len(self._ids)} vectors vs {len(expected_ids)} catalogue products"
            else:
                differing = next(
                    (a for a, b in zip(self._ids, expected_ids) if a != b), "unknown"
                )
                detail = f"same count but different products (first difference: {differing})"
            raise DenseRouteUnavailable(
                f"artifact was built from a different catalogue -- {detail}; "
                "rebuild with `python3 -m scripts.build_embeddings`"
            )

        # The build script normalizes, but re-normalizing defensively keeps the
        # dot-product-as-cosine assumption true even for a hand-made artifact.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self._matrix = (matrix / np.maximum(norms, 1e-12)).astype(np.float32)

        self._query_prefix = query_prefix
        self._model = SentenceTransformer(model_id)

    def search(self, query_text: str, limit: int) -> list[tuple[str, float]]:
        np = self._np
        total = self._matrix.shape[0]
        top_n = min(limit, total)
        if top_n <= 0 or not query_text.strip():
            return []

        query_vector = self._model.encode(
            [self._query_prefix + query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0].astype(np.float32)

        similarities = self._matrix @ query_vector
        if top_n < total:
            # Partial selection first: cheaper than fully sorting 50k scores.
            candidate_idx = np.argpartition(-similarities, top_n - 1)[:top_n]
        else:
            candidate_idx = np.arange(total)
        ordered_idx = candidate_idx[np.argsort(-similarities[candidate_idx])]
        return [(self._ids[int(i)], float(similarities[int(i)])) for i in ordered_idx]


def load_dense_retriever(
    embedding_dir: str | Path = EMBEDDING_DIR,
    model_id: str = MODEL_ID,
    query_prefix: str = QUERY_PREFIX,
    strict: bool = False,
    expected_ids: list[str] | None = None,
) -> DenseSearchFn | None:
    """Build the dense route, or return None if it cannot run.

    Returning None is the supported fallback path: the agent then serves
    BM25 lexical results instead of failing. The reason is written to stderr
    rather than swallowed, because a silently missing dense route looks
    exactly like a working agent that simply scores worse. Pass strict=True
    to raise instead of falling back.
    """
    try:
        return DenseRetriever(embedding_dir, model_id, query_prefix, expected_ids).search
    except Exception as error:
        if strict:
            raise
        print(
            f"[shopping_agent] dense route unavailable, falling back to BM25 lexical search: {error}",
            file=sys.stderr,
        )
        return None
