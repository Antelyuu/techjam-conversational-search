from __future__ import annotations

import re
from dataclasses import dataclass

from .catalog import ProductRecord
from .contracts import Constraint

# The catalog's umbrella category appears on almost every row and carries
# no discriminating signal; presence of only this value must not count as
# a category match.
BROAD_CATEGORY_TERMS = {"clothing", "clothing, shoes & jewelry", "clothing shoes & jewelry"}


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
            distance = abs(product.price - float(budget.value))
            budget_boost = max(-0.25, 0.25 - (distance / max(float(budget.value), 1.0)) * 0.25)
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
    constraints a candidate actually satisfies, not to filter."""
    text = product.searchable_text.lower()
    hard: list[str] = []
    soft: list[str] = []
    for attribute, constraint in constraints.items():
        if attribute == "budget":
            continue
        target = (hard if constraint.strength == "hard" else soft)
        if _contains_word(text, str(constraint.value).lower()):
            target.append(attribute)
    return tuple(hard), tuple(soft)
