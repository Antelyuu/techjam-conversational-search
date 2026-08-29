from __future__ import annotations

import re
from dataclasses import dataclass

from .catalog import ProductRecord
from .contracts import Constraint

# The catalog's umbrella category appears on almost every row and carries
# no discriminating signal; presence of only this value must not count as
# a category match.
BROAD_CATEGORY_TERMS = {"clothing", "clothing, shoes & jewelry", "clothing shoes & jewelry"}

# Attributes that already have a dedicated scoring feature and must not also
# be counted through generic text containment.
#
# `budget` is numeric -- evaluate_price() decides it, and its value ("100")
# would never appear as a word in product text anyway.
#
# `category` is scored by score_category(), which knows about the umbrella
# term and about mismatch. Leaving it in here double-counted it: category is
# the only DEFAULT_HARD_ATTRIBUTE, so in the common session whose one hard
# constraint is the category, a title containing the category word scored a
# full hard_constraints share *and* a full category feature -- up to 1.5 of
# the reranker's scale where the weights intend 0.5.
SCORED_SEPARATELY = frozenset({"budget", "category"})


def _contains_word(text: str, value: str) -> bool:
    """Word-boundary containment check, so "bag" does not match inside
    "Baggy" and "tan" does not match inside "instant"."""
    return re.search(r"\b" + re.escape(value) + r"\b", text, re.IGNORECASE) is not None

CATEGORY_MATCH_BOOST = 2.0
CATEGORY_MISMATCH_PENALTY = -0.5


@dataclass(frozen=True)
class FilterOutcome:
    """One candidate's filter/boost decision, with an audited reason."""

    parent_asin: str
    retained: bool
    reason: str
    category_boost: float
    budget_boost: float = 0.0


def evaluate_price(product: ProductRecord, max_price: float | None) -> tuple[bool, str]:
    """Tri-state price rule: known price over budget excludes; known price
    within budget retains; missing price always retains (unverified, not
    assumed compliant or zero). A "from $X" lower-bound price still
    excludes if even that lower bound is over budget (the real price can
    only be higher), but is never confirmed "within_budget" — the actual
    variant price is unknown."""
    if max_price is None:
        return True, "no_budget_constraint"
    if product.price is None:
        return True, "budget_unverified"
    if product.price > max_price:
        return False, "over_budget"
    if product.price_is_lower_bound:
        return True, "budget_unverified"
    return True, "within_budget"


def soft_budget_closeness(price: float, target: float) -> float:
    """Linear closeness to a soft budget target: 1.0 at the target, falling
    by relative distance, so twice the target reaches 0.0 and beyond it goes
    negative. Callers clamp and scale to their own range -- the filter turns
    this into a small pool boost, the reranker into a 0-1 feature value --
    but the shape is defined once (P4 review: it was duplicated)."""
    return 1.0 - abs(price - target) / max(target, 1.0)


def score_category(product: ProductRecord, requested_category: str | None) -> tuple[float, str]:
    """Conservative category signal: exact match boosts; a product whose
    only category information is the broad umbrella term is neutral
    (unverified, not penalized); anything else with specific categories
    that fail to match gets a soft ranking penalty, never exclusion."""
    if requested_category is None:
        return 0.0, "no_category_constraint"
    specific_categories = [c for c in product.categories if c not in BROAD_CATEGORY_TERMS]
    if requested_category in specific_categories or _contains_word(product.title.lower(), requested_category):
        return CATEGORY_MATCH_BOOST, "category_match"
    if not specific_categories:
        return 0.0, "category_unverified"
    return CATEGORY_MISMATCH_PENALTY, "category_soft_mismatch"


def evaluate_candidate(product: ProductRecord, constraints: dict[str, Constraint]) -> FilterOutcome:
    budget = constraints.get("budget")
    budget_boost = 0.0
    if budget is None:
        retained, price_reason = evaluate_price(product, None)
    elif budget.strength == "hard":
        retained, price_reason = evaluate_price(product, float(budget.value))
    else:
        retained = True
        if product.price is None:
            price_reason = "budget_preference_unverified"
        else:
            closeness = soft_budget_closeness(product.price, float(budget.value))
            budget_boost = max(-0.25, 0.25 * closeness)
            price_reason = "budget_preference"

    category = constraints.get("category")
    requested_category = str(category.value) if category is not None else None
    boost, category_reason = score_category(product, requested_category)

    reason = price_reason if not retained else f"{price_reason}+{category_reason}"
    return FilterOutcome(
        parent_asin=product.parent_asin,
        retained=retained,
        reason=reason,
        category_boost=boost,
        budget_boost=budget_boost,
    )


def matched_constraints(product: ProductRecord, constraints: dict[str, Constraint]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Lexical-only match check: does the constraint's value text appear
    in the product's searchable text? Used to report which hard/soft
    constraints a candidate actually satisfies, not to filter.

    Attributes in SCORED_SEPARATELY are skipped: they carry their own
    feature downstream and counting them here as well scores them twice."""
    text = product.searchable_text.lower()
    hard: list[str] = []
    soft: list[str] = []
    for attribute, constraint in constraints.items():
        if attribute in SCORED_SEPARATELY:
            continue
        target = (hard if constraint.strength == "hard" else soft)
        if _contains_word(text, str(constraint.value).lower()):
            target.append(attribute)
    return tuple(hard), tuple(soft)
