"""What each askable attribute is actually worth, measured not assumed.

Produces the per-attribute prior in shopping_agent/clarification.py. For every
sample it materializes the hidden intent card exactly as the evaluator does,
replays the opening message (which discloses one constraint in Buying), and
then counts, per attribute, the sessions still holding at least one
*undisclosed* constraint that the evaluator's classify_constraint() maps to
that attribute.

That count is precisely the share of sessions where asking returns content
rather than "I don't have an additional preference for X", so it is the right
prior for a question-value estimate.

Run from the repository root:

    python3 -m scripts.ask_value_analysis
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator.local_evaluator import (
    catalog_index,
    classify_constraint,
    coarse_category,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)


def analyze(
    catalog_path: str = "data/catalog.jsonl",
    dataset_path: str = "data/public_set.jsonl",
) -> dict:
    samples = load_jsonl(dataset_path)
    _ids, categories, products = catalog_index(catalog_path)

    useful: Counter[str] = Counter()
    all_classes: Counter[str] = Counter()
    by_scenario: dict[str, Counter[str]] = defaultdict(Counter)
    scenario_n: Counter[str] = Counter()
    remaining: Counter[int] = Counter()

    for sample in samples:
        scenario = sample["scenario_type"]
        scenario_n[scenario] += 1
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        target = str(sample["ground_truth"]["parent_asin"])

        constraints = [
            *[str(value) for value in card.get("hard_constraints", [])],
            *[str(value) for value in card.get("soft_preferences", [])],
        ]
        for value in constraints:
            all_classes[classify_constraint(value)] += 1

        # The opening mutates `disclosed` for Buying, so replay it rather than
        # counting constraints the customer has already volunteered.
        disclosed: set[str] = set()
        initial_message(effective, coarse_category(categories.get(target, [])), disclosed)

        undisclosed = [value for value in constraints if value not in disclosed]
        remaining[len(undisclosed)] += 1
        for attribute in {classify_constraint(value) for value in undisclosed}:
            useful[attribute] += 1
            by_scenario[scenario][attribute] += 1

    return {
        "samples": len(samples),
        "useful": useful,
        "all_classes": all_classes,
        "by_scenario": by_scenario,
        "scenario_n": scenario_n,
        "remaining": remaining,
    }


def _report(result: dict) -> str:
    total = result["samples"]
    lines = [f"samples: {total}", "", "sessions where asking X yields content (after the opening)"]
    for attribute, count in result["useful"].most_common():
        lines.append(f"  {attribute:10s} {count:4d}  {count / total:6.1%}")

    lines += ["", "class of every constraint string (not per session)"]
    for attribute, count in result["all_classes"].most_common():
        lines.append(f"  {attribute:10s} {count:4d}")

    attributes = [attribute for attribute, _ in result["useful"].most_common()]
    lines += ["", "per scenario (share of that scenario's sessions)"]
    lines.append("  scenario         n  " + " ".join(f"{a[:8]:>8s}" for a in attributes))
    for scenario in sorted(result["scenario_n"]):
        n = result["scenario_n"][scenario]
        row = " ".join(f"{result['by_scenario'][scenario][a] / n:7.0%} " for a in attributes)
        lines.append(f"  {scenario:15s} {n:3d}  {row}")

    lines += ["", "undisclosed constraints remaining after the opening"]
    for count in sorted(result["remaining"]):
        lines.append(f"  {count} left: {result['remaining'][count]:4d} sessions")
    return "\n".join(lines)


if __name__ == "__main__":
    print(_report(analyze()))
