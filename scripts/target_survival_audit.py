"""P2-T5 target-survival audit.

Read-only analysis, not part of the Agent runtime or the official
evaluator: checks whether the hard price filter in
shopping_agent.filtering would ever exclude the actual ground-truth
target on the public set, and how often the category vocabulary in
shopping_agent.intent fails to extract any category from the
customer's first message.

Run from the repository root:

    python3 -m scripts.target_survival_audit
"""

from __future__ import annotations

import json

from evaluator.local_evaluator import (
    catalog_index,
    coarse_category,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)
from shopping_agent.catalog import load_catalog
from shopping_agent.filtering import evaluate_price, score_category
from shopping_agent.intent import extract_candidate_slots


def audit(catalog_path: str = "data/catalog.jsonl", dataset_path: str = "data/public_set.jsonl") -> dict:
    catalog_ids, categories, raw_products = catalog_index(catalog_path)
    products = load_catalog(catalog_path)
    samples = load_jsonl(dataset_path)

    sessions_checked = 0
    targets_with_known_price = 0
    targets_a_hard_budget_filter_would_exclude = 0
    sessions_with_no_category_extracted = 0
    hard_budgets_evaluated = 0
    targets_within_budget = 0
    targets_with_unverified_budget = 0
    category_constraints_evaluated = 0
    category_matches = 0
    category_unverified = 0
    category_soft_mismatches = 0

    for sample in samples:
        target = str(sample["ground_truth"]["parent_asin"])
        product = products.get(target)
        if product is None or target not in catalog_ids:
            continue
        sessions_checked += 1
        if product.price is not None:
            targets_with_known_price += 1

        card, behavior = materialize_hidden_fields(sample, raw_products)
        effective_sample = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        message = initial_message(effective_sample, coarse_category(categories.get(target, [])), disclosed)
        candidates = extract_candidate_slots(message)

        hard_budget = next((value for attribute, value, strength in candidates if attribute == "budget" and strength == "hard"), None)
        if hard_budget is not None:
            hard_budgets_evaluated += 1
            retained, price_reason = evaluate_price(product, float(hard_budget))
            if not retained:
                targets_a_hard_budget_filter_would_exclude += 1
            elif price_reason == "within_budget":
                targets_within_budget += 1
            elif price_reason == "budget_unverified":
                targets_with_unverified_budget += 1

        requested_category = next((value for attribute, value, _strength in candidates if attribute == "category"), None)
        if requested_category is None:
            sessions_with_no_category_extracted += 1
        else:
            category_constraints_evaluated += 1
            _boost, category_reason = score_category(product, str(requested_category))
            if category_reason == "category_match":
                category_matches += 1
            elif category_reason == "category_unverified":
                category_unverified += 1
            elif category_reason == "category_soft_mismatch":
                category_soft_mismatches += 1

    targets_survived = sessions_checked - targets_a_hard_budget_filter_would_exclude

    return {
        "sessions_checked": sessions_checked,
        "targets_survived": targets_survived,
        "targets_removed_by_hard_filters": targets_a_hard_budget_filter_would_exclude,
        "targets_with_known_price": targets_with_known_price,
        "targets_a_hard_budget_filter_would_exclude": targets_a_hard_budget_filter_would_exclude,
        "sessions_with_no_category_extracted_from_turn_1": sessions_with_no_category_extracted,
        "price_filter": {
            "hard_constraints_evaluated": hard_budgets_evaluated,
            "targets_excluded": targets_a_hard_budget_filter_would_exclude,
            "within_budget": targets_within_budget,
            "budget_unverified": targets_with_unverified_budget,
            "no_hard_budget_constraint": sessions_checked - hard_budgets_evaluated,
        },
        "category_signal": {
            "constraints_evaluated": category_constraints_evaluated,
            "matches": category_matches,
            "unverified": category_unverified,
            "soft_mismatches": category_soft_mismatches,
            "hard_exclusions": 0,
            "no_category_constraint": sessions_with_no_category_extracted,
        },
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
