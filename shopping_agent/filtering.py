from __future__ import annotations

from dataclasses import dataclass

from .catalog import ProductRecord
from .contracts import Constraint

# The catalog's umbrella category appears on almost every row and carries
# no discriminating signal; presence of only this value must not count as
# a category match.
BROAD_CATEGORY_TERMS = {"clothing", "clothing, shoes & jewelry", "clothing shoes & jewelry"}

CATEGORY_MATCH_BOOST = 2.0
CATEGORY_MISMATCH_PENALTY = -0.5


@dataclass(frozen=True)
class FilterOutcome:
    """One candidate's filter/boost decision, with an audited reason."""

    parent_asin: str
    retained: bool
    reason: str
    category_boost: float


def evaluate_price(product: ProductRecord, max_price: float | None) -> tuple[bool, str]:
    """Tri-state price rule: known price over budget excludes; known price
    within budget retains; missing price always retains (unverified, not
    assumed compliant or zero)."""
    if max_price is None:
        return True, "no_budget_constraint"
    if product.price is None:
        return True, "budget_unverified"
    if product.price <= max_price:
        return True, "within_budget"
    return False, "over_budget"


def score_category(product: ProductRecord, requested_category: str | None) -> tuple[float, str]:
    """Conservative category signal: exact match boosts; a product whose
    only category information is the broad umbrella term is neutral
    (unverified, not penalized); anything else with specific categories
    that fail to match gets a soft ranking penalty, never exclusion."""
    if requested_category is None:
        return 0.0, "no_category_constraint"
    specific_categories = [c for c in product.categories if c not in BROAD_CATEGORY_TERMS]
    if requested_category in specific_categories or requested_category in product.title.lower():
        return CATEGORY_MATCH_BOOST, "category_match"
    if not specific_categories:
        return 0.0, "category_unverified"
    return CATEGORY_MISMATCH_PENALTY, "category_soft_mismatch"


def evaluate_candidate(product: ProductRecord, constraints: dict[str, Constraint]) -> FilterOutcome:
    budget = constraints.get("budget")
    max_price = float(budget.value) if budget is not None else None
    retained, price_reason = evaluate_price(product, max_price)

    category = constraints.get("category")
    requested_category = str(category.value) if category is not None else None
    boost, category_reason = score_category(product, requested_category)

    reason = price_reason if not retained else f"{price_reason}+{category_reason}"
    return FilterOutcome(
        parent_asin=product.parent_asin,
        retained=retained,
        reason=reason,
        category_boost=boost,
    )


def matched_constraints(product: ProductRecord, constraints: dict[str, Constraint]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Lexical-only match check: does the constraint's value text appear
    in the product's searchable text? Used to report which hard/soft
    constraints a candidate actually satisfies, not to filter."""
    text = product.searchable_text.lower()
    hard: list[str] = []
    soft: list[str] = []
    for attribute, constraint in constraints.items():
        if attribute == "budget":
            continue
        target = (hard if constraint.strength == "hard" else soft)
        if str(constraint.value).lower() in text:
            target.append(attribute)
    return tuple(hard), tuple(soft)
