from __future__ import annotations

from typing import Callable

from .catalog import ProductRecord
from .contracts import Candidate, SearchRequest
from .filtering import evaluate_candidate, matched_constraints

# (query_text, pool_size) -> ranked [(parent_asin, lexical_score), ...].
# Retrieval never sees SQL/FTS specifics, so a dense or metadata route can
# be swapped in later without changing this module.
LexicalSearchFn = Callable[[str, int], list[tuple[str, float]]]

# How much larger a pool to pull than top_k, so the hard price filter has
# room to drop over-budget items without starving the final list.
POOL_MULTIPLIER = 5
MAX_POOL_SIZE = 200


def retrieve(
    request: SearchRequest,
    limit: int,
    lexical_search: LexicalSearchFn,
    products: dict[str, ProductRecord],
) -> list[Candidate]:
    pool_size = min(max(limit * POOL_MULTIPLIER, limit), MAX_POOL_SIZE)
    hits = lexical_search(request.query_text, pool_size)

    scored: list[tuple[float, float, str]] = []  # (final_score, lexical_score, parent_asin)
    for parent_asin, lexical_score in hits:
        product = products.get(parent_asin)
        if product is None:
            continue
        outcome = evaluate_candidate(product, request.state.constraints)
        if not outcome.retained:
            continue
        final_score = lexical_score + outcome.category_boost
        scored.append((final_score, lexical_score, parent_asin))

    scored.sort(key=lambda item: item[0], reverse=True)

    candidates: list[Candidate] = []
    for rank, (final_score, lexical_score, parent_asin) in enumerate(scored[:limit], start=1):
        product = products[parent_asin]
        hard, soft = matched_constraints(product, request.state.constraints)
        candidates.append(
            Candidate(
                parent_asin=parent_asin,
                route_ranks={"lexical": rank},
                route_scores={"lexical": lexical_score, "combined": final_score},
                matched_hard_constraints=hard,
                matched_soft_preferences=soft,
            )
        )
    return candidates
