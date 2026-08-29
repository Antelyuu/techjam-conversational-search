"""P4-T1: the deterministic final scorer.

Retrieval hands over a pool of candidates carrying each route's rank and
score. This module puts them in final order using an explicit, fixed feature
checklist. P4 specifies these six features, in this stated priority order:

    1. hard-constraint satisfaction
    2. category compatibility
    3. lexical route rank
    4. dense route rank
    5. metadata compatibility
    6. soft-preference matches

All six are scored. **The stated priority is not the priority used**, and that
is deliberate: weighting them in this order measured 0.047 composite *worse*
than not reranking at all (E4). Retrieval rank now leads and the constraint
features act as adjustments beneath it. See FEATURE_WEIGHTS for the numbers
and the reasoning.

Every feature contributes ``weight * value`` with ``value`` in 0-1, and every
contribution is recorded on the result. That is the point of the phase's
acceptance criterion -- any candidate's placement can be read off rather than
guessed at, so a bad ranking can be traced to the feature that caused it. It
is also how the weights below were diagnosed: the first version ranked worse
than no reranking at all, and the breakdown said which feature was doing it.

No model, no network, no learned parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

from .catalog import ProductRecord
from .contracts import Candidate, Constraint
from .filtering import evaluate_price, score_category

# MEASURED, and not what the checklist order suggests. Weighting the features
# in P4's stated priority order -- hard_constraints 4.0 down to
# soft_preferences 0.25 -- *lost* 0.047 composite against the fused ordering it
# replaces, and lost all of it in MRR (0.329 against 0.475). It found the same
# products and ranked them worse, because matched_constraints() is a coarse
# word-containment check and at weight 4.0 it swamped the retrieval score
# margin that weighted fusion exists to preserve.
#
# These weights lead with retrieval instead and win by +0.0059 (E4). The
# checklist order still holds among the *adjustment* features -- hard
# constraints outrank category, which outranks metadata, which outranks soft
# preferences -- but retrieval rank outranks all of them, which the specified
# order does not say and the measurement does.
FEATURE_WEIGHTS: dict[str, float] = {
    "hard_constraints": 1.0,
    "category": 0.5,
    "lexical_rank": 2.0,
    "dense_rank": 2.0,
    "metadata": 0.25,
    "soft_preferences": 0.1,
}

# Converts a 1-based route rank to a 0-1 value.
#
# MEASURED: this was 60, mirroring the RRF constant, and that was the larger
# half of the mistake above. Over a fifty-candidate pool it spans only 1.00 to
# 0.55, so the rank features could not discriminate against features spanning
# the full 0-1. At 5 the span is 1.00 to 0.09. Sharpening this alone recovered
# most of the lost MRR (0.441) even under the original weights.
RANK_DECAY = 5.0


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


def _budget_value(product: ProductRecord, budget: Constraint | None) -> float:
    """Score price against the budget, honouring what kind of budget it is.

    A *hard* budget ("under $100") is a ceiling: within it or not.

    A *soft* budget ("around $100") is a target, and must be ranked by
    closeness the way P2 did. Collapsing it to a ceiling loses that ordering
    twice over -- $50 ties with $99 despite being half the asking price, and
    $101 falls off a cliff to zero despite being what the customer asked for.
    Closeness decays linearly with relative distance, so the target scores
    1.0, twice the target scores 0.0, and the boundary is smooth.
    """
    if budget is None:
        return 0.0
    if product.price is None:
        # Unverified, not compliant: an unpriced item must not outrank one
        # known to fit.
        return 0.25

    target = float(budget.value)
    if budget.strength == "hard":
        retained, _ = evaluate_price(product, target)
        return 1.0 if retained else 0.0

    distance = abs(product.price - target)
    return max(0.0, 1.0 - distance / max(target, 1.0))


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
    if category is None:
        # A constraint the customer never gave earns no credit. Constant
        # across candidates either way, so this cannot reorder anything -- but
        # scoring it 0.2 made every explanation claim partial category credit
        # in sessions where category was never mentioned.
        category_value = 0.0
    else:
        category_boost, _reason = score_category(product, str(category.value))
        # score_category returns +2.0 match / 0.0 unverified / -0.5 mismatch;
        # rescale onto 0-1 so every feature value means the same thing.
        category_value = (category_boost + 0.5) / 2.5

    metadata_value = _budget_value(product, constraints.get("budget"))

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
