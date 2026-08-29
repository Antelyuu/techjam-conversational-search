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

from evaluator.local_evaluator import catalog_index, coarse_category, initial_message, intent_card, load_jsonl
from shopping_agent.catalog import load_catalog
from shopping_agent.filtering import evaluate_price
from shopping_agent.intent import extract_candidate_slots


def audit(catalog_path: str = "data/catalog.jsonl", dataset_path: str = "data/public_set.jsonl") -> dict:
    catalog_ids, categories, raw_products = catalog_index(catalog_path)
    products = load_catalog(catalog_path)
    samples = load_jsonl(dataset_path)

    sessions_checked = 0
    targets_with_known_price = 0
    targets_a_hard_budget_filter_would_exclude = 0
    sessions_with_no_category_extracted = 0

    for sample in samples:
        target = str(sample["ground_truth"]["parent_asin"])
        product = products.get(target)
        if product is None or target not in catalog_ids:
            continue
        sessions_checked += 1
        if product.price is not None:
            targets_with_known_price += 1

        card = sample.get("intent_card") or intent_card(raw_products[target])
        effective_sample = {**sample, "intent_card": card}
        disclosed: set[str] = set()
        message = initial_message(effective_sample, coarse_category(categories.get(target, [])), disclosed)
        candidates = extract_candidate_slots(message)

        hard_budget = next((value for attribute, value, strength in candidates if attribute == "budget" and strength == "hard"), None)
        if hard_budget is not None:
            retained, _reason = evaluate_price(product, float(hard_budget))
            if not retained:
                targets_a_hard_budget_filter_would_exclude += 1

        if not any(attribute == "category" for attribute, _, _ in candidates):
            sessions_with_no_category_extracted += 1

    return {
        "sessions_checked": sessions_checked,
        "targets_with_known_price": targets_with_known_price,
        "targets_a_hard_budget_filter_would_exclude": targets_a_hard_budget_filter_would_exclude,
        "sessions_with_no_category_extracted_from_turn_1": sessions_with_no_category_extracted,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
