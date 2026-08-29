from __future__ import annotations

import sys
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
# Safety cap on any single fetch, NOT a tuning knob -- the tuned depth is
# RERANK_POOL (starter/agent.py), which widens the fetch through
# candidate_limit. Kept far above it deliberately: when the two were equal, a
# depth experiment above the cap silently reproduced the capped depth while
# presenting itself as deeper (review finding, P5) -- the silent-truncation
# pattern this project has now hit three times. E7's 300-800 sweep rows were
# measured with this cap raised to match each depth.
MAX_POOL_SIZE = 1000

FUSION_RRF = "rrf"
FUSION_WEIGHTED = "weighted"
# Weighted is the default: it scored 0.151089 against RRF's 0.145170 on the
# public set (E2), the gain sitting mostly in MRR. The margin is about one
# session on HitRate, so re-run scripts/fusion_ablation.py if retrieval inputs
# change materially -- notably once P4 clarification alters the query mix.
DEFAULT_FUSION = FUSION_WEIGHTED
FUSION_METHODS = frozenset({FUSION_RRF, FUSION_WEIGHTED})

# Standard RRF constant: damps the gap between the very top ranks so one
# route cannot dominate purely by being confident about its #1.
RRF_K = 60

# An even split, chosen as a neutral prior and never swept -- not a finding.
DEFAULT_ROUTE_WEIGHTS = {"lexical": 0.5, "dense": 0.5}

# Scales the P2 category/budget boosts onto the 0-1 fused score, so a category
# match cannot simply outrank the retrieval signal.
#
# MEASURED, and this value does not match it: over 80 real opening queries the
# BM25 pool span is median 5.12 (p10 2.60, p90 12.78), so P2's 2.0 category
# boost carried ~39% of the span. Preserving that influence would need ~0.39;
# at 0.1 the boosts carry ~20%, i.e. roughly half the weight P2 gave them.
# 0.1 is retained only because it is the value the recorded 0.151089 result was
# measured with -- it is not a tuned or justified choice.
#
# Deliberately not tuned here: P4-T1's reranker takes over hard-constraint and
# category compatibility, so this constant likely moves or disappears. Whether
# boosts want more weight is worth measuring as input to that design.
FUSED_BOOST_SCALE = 0.1


# Failures here repeat every turn of every session, so each distinct reason is
# reported once. Silent degradation is the specific failure this project has
# already been bitten by: a "model comparison" that was really a lexical run.
#
# Deliberately once per *process*, not per Agent: retrieve() is a module
# function with no instance to hang state on, and the point is that a human
# sees each distinct failure at least once per run. Agent._warned is per
# instance for the same reason in reverse -- it has an instance.
_WARNED: set[str] = set()


def _warn_once(message: str) -> None:
    if message in _WARNED:
        return
    _WARNED.add(message)
    print(f"[shopping_agent] {message}", file=sys.stderr)


def _minmax_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Map scores onto 0-1.

    A route with no spread (one candidate, or several tied) carries no
    ranking signal, but it did still return these candidates. Collapsing to
    1.0 rather than 0.0 keeps that endorsement distinguishable from the 0.0
    that _weighted_fusion gives a candidate the route never returned at all.
    Otherwise a hard filter that leaves a route one strong survivor would
    give that survivor no credit and tie it with the worst hit of the other
    route."""
    if not scores:
        return {}
    lowest = min(scores.values())
    highest = max(scores.values())
    span = highest - lowest
    if span <= 0:
        return {key: 1.0 for key in scores}
    return {key: (value - lowest) / span for key, value in scores.items()}


def _rrf_fusion(
    parent_asins: list[str],
    route_ranks: dict[str, dict[str, int]],
) -> dict[str, float]:
    """Reciprocal Rank Fusion: combine routes by rank only, so BM25 and
    cosine scores never have to be made comparable."""
    return {
        parent_asin: sum(1.0 / (RRF_K + rank) for rank in route_ranks[parent_asin].values())
        for parent_asin in parent_asins
    }


def _weighted_fusion(
    parent_asins: list[str],
    route_scores: dict[str, dict[str, float]],
    route_names: list[str],
    weights: dict[str, float],
) -> dict[str, float]:
    """Normalize each route's raw scores independently, then blend them with
    per-route weights. A candidate a route never returned contributes
    nothing for that route, so agreement between routes raises a candidate."""
    normalized_by_route: dict[str, dict[str, float]] = {}
    for route_name in route_names:
        route_view = {
            parent_asin: route_scores[parent_asin][route_name]
            for parent_asin in parent_asins
            if route_name in route_scores[parent_asin]
        }
        normalized_by_route[route_name] = _minmax_normalize(route_view)

    return {
        parent_asin: sum(
            weights.get(route_name, 0.0) * normalized_by_route[route_name].get(parent_asin, 0.0)
            for route_name in route_names
        )
        for parent_asin in parent_asins
    }


def retrieve(
    request: SearchRequest,
    limit: int,
    lexical_search: LexicalSearchFn,
    products: dict[str, ProductRecord],
    dense_search: Callable[[str, int], list[tuple[str, float]]] | None = None,
    fusion_method: str = DEFAULT_FUSION,
    route_weights: dict[str, float] | None = None,
    candidate_limit: int | None = None,
) -> list[Candidate]:
    """Rank the candidate pool and return the best `limit` of it.

    `candidate_limit` returns more than `limit`, so P4's reranker can reorder
    a pool deeper than the ten rows that will actually be shown -- and it
    widens the route fetch to match (capped at MAX_POOL_SIZE), because a
    reranker that can only reorder what a shallower fetch surfaced cannot
    rescue anything the fetch missed. It never exceeds the pool retrieval
    actually built.
    """
    pool_size = min(max(limit * POOL_MULTIPLIER, candidate_limit or 0), MAX_POOL_SIZE)
    result_limit = min(candidate_limit or limit, pool_size)

    routes: dict[str, list[tuple[str, float]]] = {
        "lexical": lexical_search(request.query_text, pool_size)
    }
    if dense_search is not None:
        # load_dense_retriever() already guards the route it builds, but
        # retrieve() accepts any callable -- the ablation scripts assign one
        # directly. A route that raises must cost its own contribution, not
        # the whole turn, and it must say so rather than degrade in silence
        # (P4-T4).
        try:
            routes["dense"] = dense_search(request.query_text, pool_size)
        except Exception as error:
            _warn_once(f"dense route failed, using lexical results only: {error}")

    # Candidate union: one entry per parent_asin, carrying each route's rank
    # and score so nothing about where a candidate came from is lost.
    route_ranks: dict[str, dict[str, int]] = {}
    route_scores: dict[str, dict[str, float]] = {}
    for route_name, hits in routes.items():
        for rank, (parent_asin, score) in enumerate(hits, start=1):
            candidate_ranks = route_ranks.setdefault(parent_asin, {})
            candidate_scores = route_scores.setdefault(parent_asin, {})
            # A route should be ranked by its first occurrence. Duplicate
            # hits must not overwrite the better rank/score already recorded.
            if route_name not in candidate_ranks:
                candidate_ranks[route_name] = rank
                candidate_scores[route_name] = score

    retained: list[str] = []
    outcomes = {}
    for parent_asin in route_ranks:
        product = products.get(parent_asin)
        if product is None:
            continue
        outcome = evaluate_candidate(product, request.state.constraints)
        if not outcome.retained:
            continue
        retained.append(parent_asin)
        outcomes[parent_asin] = outcome

    route_names = list(routes)
    if len(route_names) == 1:
        # Single route: use its raw score, exactly as P2 did. The lexical-only
        # path is the control in the model comparison, so it must not shift.
        only_route = route_names[0]
        base_scores = {
            parent_asin: route_scores[parent_asin][only_route] for parent_asin in retained
        }
        boost_scale = 1.0
    else:
        if fusion_method == FUSION_WEIGHTED:
            weights = route_weights or DEFAULT_ROUTE_WEIGHTS
            fused = _weighted_fusion(retained, route_scores, route_names, weights)
        else:
            fused = _rrf_fusion(retained, route_ranks)
        # Both methods land on a common 0-1 scale so the boosts below mean the
        # same thing whichever fusion is selected.
        base_scores = _minmax_normalize(fused)
        boost_scale = FUSED_BOOST_SCALE

    scored: list[tuple[float, str]] = []
    for parent_asin in retained:
        outcome = outcomes[parent_asin]
        final_score = base_scores[parent_asin] + boost_scale * (
            outcome.category_boost + outcome.budget_boost
        )
        scored.append((final_score, parent_asin))

    # Stable sort on score alone: ties keep union order, which puts the
    # lexical route's own ranking first rather than breaking ties arbitrarily.
    scored.sort(key=lambda item: item[0], reverse=True)

    candidates: list[Candidate] = []
    for final_score, parent_asin in scored[:result_limit]:
        product = products[parent_asin]
        hard, soft = matched_constraints(product, request.state.constraints)
        candidates.append(
            Candidate(
                parent_asin=parent_asin,
                route_ranks=dict(route_ranks[parent_asin]),
                route_scores={**route_scores[parent_asin], "combined": final_score},
                matched_hard_constraints=hard,
                matched_soft_preferences=soft,
            )
        )
    return candidates
