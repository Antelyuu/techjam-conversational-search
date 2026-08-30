"""Can the deterministic scorer be made to earn its place?

The first reranker measurement (E4) lost 0.047 composite against the fused
order it replaced, and lost it entirely in MRR: it found the same products and
ranked them worse. The diagnosis is that the top of P4's feature checklist
swamps the retrieval signal -- `hard_constraints` at weight 4.0 spans the full
0-1 range, while the rank features at RANK_DECAY=60 span only 1.0 to 0.55 over
a fifty-candidate pool. Coarse word-containment matching therefore overrides
the score margin that weighted fusion exists to preserve.

This sweeps that diagnosis rather than assuming it, over one agent instance:

    spec_order      the weights as first written (the E4 control)
    sharp_ranks     same weights, ranks discriminating far more sharply
    retrieval_led   ranks weighted above the constraint features
    both            sharper ranks and retrieval-led weights

Run from the repository root:

    python3 -m scripts.rerank_weight_sweep
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# These four scripts study the dense route, which P5 turned off by default on
# measurement (starter/agent.py, DENSE_BY_DEFAULT). A bare Agent() therefore has
# no dense route to ablate and the script exits saying so. Ask for it explicitly
# rather than making the caller remember an environment variable to run a script
# whose entire subject is that route.
os.environ.setdefault("SHOPPING_AGENT_DENSE", "1")

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from shopping_agent import reranking
from starter.agent import Agent

# These reproduce the E4-era comparison and deliberately predate P5: the
# evidence features are pinned to 0.0 EXPLICITLY, because FEATURE_WEIGHTS is
# replaced wholesale below and a silently missing key would remove the
# scorer's dominant feature while the output presented itself as a comparison
# of the current configuration (review finding, P5). Re-running this script
# measures the historical E4 question, not today's reranker.
_P5_FEATURES_ZEROED = {
    "constraint_evidence": 0.0,
    "phrase_evidence": 0.0,
}
SPEC_ORDER = {
    "hard_constraints": 4.0,
    "category": 2.0,
    "lexical_rank": 1.0,
    "dense_rank": 1.0,
    "metadata": 0.5,
    "soft_preferences": 0.25,
    **_P5_FEATURES_ZEROED,
}
RETRIEVAL_LED = {
    "hard_constraints": 1.0,
    "category": 0.5,
    "lexical_rank": 2.0,
    "dense_rank": 2.0,
    "metadata": 0.25,
    "soft_preferences": 0.1,
    **_P5_FEATURES_ZEROED,
}

CONFIGURATIONS = (
    {"name": "fused_control", "rerank": False, "weights": SPEC_ORDER, "decay": 60.0},
    {"name": "spec_order", "rerank": True, "weights": SPEC_ORDER, "decay": 60.0},
    {"name": "sharp_ranks", "rerank": True, "weights": SPEC_ORDER, "decay": 5.0},
    {"name": "retrieval_led", "rerank": True, "weights": RETRIEVAL_LED, "decay": 60.0},
    {"name": "both", "rerank": True, "weights": RETRIEVAL_LED, "decay": 5.0},
)

METRIC_KEYS = ("hit_rate_at_10", "mrr", "mttc", "efficiency", "recommended_technical_score")


def run_sweep(
    catalog_path: str = "data/catalog.jsonl",
    dataset_path: str = "data/public_set.jsonl",
) -> dict:
    samples = load_jsonl(dataset_path)
    catalog_ids, categories, raw_products = catalog_index(catalog_path)

    agent = Agent(catalog_path)
    if agent.dense_search is None:
        raise SystemExit("dense route unavailable; the dense_rank feature would be dead")

    original_weights = dict(reranking.FEATURE_WEIGHTS)
    original_decay = reranking.RANK_DECAY
    results: dict[str, dict] = {}
    try:
        for configuration in CONFIGURATIONS:
            agent.enable_reranker = configuration["rerank"]
            reranking.FEATURE_WEIGHTS.clear()
            reranking.FEATURE_WEIGHTS.update(configuration["weights"])
            reranking.RANK_DECAY = configuration["decay"]
            outcome = evaluate(agent, samples, catalog_ids, categories, raw_products)
            results[configuration["name"]] = {
                key: value for key, value in outcome.items() if key != "sessions"
            }
    finally:
        reranking.FEATURE_WEIGHTS.clear()
        reranking.FEATURE_WEIGHTS.update(original_weights)
        reranking.RANK_DECAY = original_decay
    return results


def _format_table(results: dict) -> str:
    header = f"{'config':<16}" + "".join(
        f"{key.replace('recommended_', ''):>22}" for key in METRIC_KEYS
    )
    lines = [header, "-" * len(header)]
    for name, metrics in results.items():
        lines.append(f"{name:<16}" + "".join(f"{metrics[key]:>22.6f}" for key in METRIC_KEYS))
    return "\n".join(lines)


if __name__ == "__main__":
    outcome = run_sweep()
    print(_format_table(outcome))
    Path("rerank_weight_sweep.json").write_text(
        json.dumps(outcome, indent=2) + "\n", encoding="utf-8"
    )
    print("\nwrote rerank_weight_sweep.json")
