"""P4-T1: the deterministic final scorer.

Retrieval hands over a pool of candidates carrying each route's rank and
score. This module puts them in final order using an explicit, fixed feature
checklist, in the order P4 specifies:

    1. hard-constraint satisfaction
    2. category compatibility
    3. lexical route rank
    4. dense route rank
    5. metadata compatibility
    6. soft-preference matches

Every feature contributes ``weight * value`` with ``value`` in 0-1, and every
contribution is recorded on the result. That is the point of the phase's
acceptance criterion -- any candidate's placement can be read off rather than
guessed at, so a bad ranking can be traced to the feature that caused it.

Weights decrease down the checklist, which is how the specified feature order
is expressed: a hard constraint outweighs a category match, which outweighs
any amount of retrieval score. No model, no network, no learned parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

from .catalog import ProductRecord
from .contracts import Candidate, Constraint
from .filtering import evaluate_price, score_category

# Ordered highest-priority first, matching the P4 feature checklist. The gaps
# are deliberate: no accumulation of lower features can overturn a higher one,
# which is what makes the order meaningful rather than decorative.
FEATURE_WEIGHTS: dict[str, float] = {
    "hard_constraints": 4.0,
    "category": 2.0,
    "lexical_rank": 1.0,
    "dense_rank": 1.0,
    "metadata": 0.5,
    "soft_preferences": 0.25,
}

# Converts a 1-based route rank to a 0-1 value. Matches the shape of the RRF
# constant used in fusion, so a rank difference means about the same thing
# here as it does there.
RANK_DECAY = 60.0


@dataclass(frozen=True)
class ScoreContribution:
    """One feature's input to a candidate's final score."""

    feature: str
    weight: float
    value: float

    @property
    def contribution(self) -> float:
        return self.weight * self.value


@dataclass(frozen=True)
class RerankedCandidate:
    """A scored candidate whose score can be taken apart."""

    parent_asin: str
    score: float
    contributions: tuple[ScoreContribution, ...]

    def explain(self) -> str:
        parts = ", ".join(
            f"{c.feature}={c.value:.3f}x{c.weight:g}={c.contribution:+.3f}"
            for c in self.contributions
        )
        return f"{self.parent_asin} score={self.score:.4f} [{parts}]"


def _rank_value(candidate: Candidate, route: str) -> float:
    """A route that never returned this candidate scores 0, not a poor rank --
    absence is not a weak endorsement."""
    rank = candidate.route_ranks.get(route)
    if rank is None:
        return 0.0
    return RANK_DECAY / (RANK_DECAY + rank - 1.0)


def _share(matched: tuple[str, ...], constraints: dict[str, Constraint], strength: str) -> float:
    """Share of the constraints of this strength that the candidate matches.
    No constraints of that strength is neutral (0.0), not a free win."""
    total = sum(
        1 for a, c in constraints.items() if c.strength == strength and a != "budget"
    )
    if total == 0:
        return 0.0
    return len(matched) / total


def score_candidate(
    candidate: Candidate,
    product: ProductRecord,
    constraints: dict[str, Constraint],
) -> RerankedCandidate:
    """Score one candidate against the feature checklist, in order."""
    category = constraints.get("category")
    category_boost, _reason = score_category(
        product, str(category.value) if category is not None else None
    )
    # score_category returns +2.0 match / 0.0 unverified / -0.5 mismatch;
    # rescale onto 0-1 so every feature value means the same thing.
    category_value = (category_boost + 0.5) / 2.5

    budget = constraints.get("budget")
    if budget is None:
        metadata_value = 0.0
    elif product.price is None:
        # Unverified, not compliant: an unpriced item must not outrank one
        # known to be within budget.
        metadata_value = 0.25
    else:
        retained, _ = evaluate_price(product, float(budget.value))
        metadata_value = 1.0 if retained else 0.0

    values = {
        "hard_constraints": _share(candidate.matched_hard_constraints, constraints, "hard"),
        "category": category_value,
        "lexical_rank": _rank_value(candidate, "lexical"),
        "dense_rank": _rank_value(candidate, "dense"),
        "metadata": metadata_value,
        "soft_preferences": _share(candidate.matched_soft_preferences, constraints, "soft"),
    }

    contributions = tuple(
        ScoreContribution(feature=feature, weight=FEATURE_WEIGHTS[feature], value=values[feature])
        for feature in FEATURE_WEIGHTS
    )
    return RerankedCandidate(
        parent_asin=candidate.parent_asin,
        score=sum(c.contribution for c in contributions),
        contributions=contributions,
    )


def rerank(
    candidates: list[Candidate],
    products: dict[str, ProductRecord],
    constraints: dict[str, Constraint],
    limit: int,
) -> list[RerankedCandidate]:
    """Order the pool by the deterministic scorer and keep the top `limit`.

    The sort is stable on score alone, so candidates the scorer cannot
    separate keep the order retrieval gave them rather than being permuted
    arbitrarily.
    """
    scored = [
        score_candidate(candidate, products[candidate.parent_asin], constraints)
        for candidate in candidates
        if candidate.parent_asin in products
    ]
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:limit]
