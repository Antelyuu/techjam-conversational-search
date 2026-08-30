"""P3-T5 fusion ablation.

Evaluates the same agent under each retrieval configuration so the fusion
choice rests on measured scores rather than preference:

    lexical_only  dense route disabled -- the control
    rrf           reciprocal rank fusion of lexical + dense
    weighted      normalized weighted fusion of lexical + dense

This satisfies P3-T5's acceptance criterion that both fusion methods be
"configuration-selectable and independently evaluated", and supplies the
fusion ablation named in P3's exit criteria.

Run from the repository root, after `pip install -r requirements.txt`:

    python3 -m scripts.fusion_ablation

One Agent is constructed and its retrieval configuration is switched between
runs, so the FTS index and the embedding matrix are each loaded once rather
than three times. Sessions are isolated per run by the evaluator's own
reset() call, so no state carries across configurations.
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
from shopping_agent.retrieval import FUSION_RRF, FUSION_WEIGHTED
from starter.agent import Agent

CONFIGURATIONS = (
    {"name": "lexical_only", "dense": False, "fusion": FUSION_RRF},
    {"name": "rrf", "dense": True, "fusion": FUSION_RRF},
    {"name": "weighted", "dense": True, "fusion": FUSION_WEIGHTED},
)

METRIC_KEYS = ("hit_rate_at_10", "mrr", "mttc", "efficiency", "recommended_technical_score")


def run_ablation(
    catalog_path: str = "data/catalog.jsonl",
    dataset_path: str = "data/public_set.jsonl",
) -> dict:
    samples = load_jsonl(dataset_path)
    catalog_ids, categories, raw_products = catalog_index(catalog_path)

    agent = Agent(catalog_path)
    if agent.dense_search is None:
        raise SystemExit(
            "dense route unavailable, so there is nothing to ablate; "
            "check the stderr warning above and install dependencies / rebuild the artifact"
        )
    dense_search = agent.dense_search

    results: dict[str, dict] = {}
    for configuration in CONFIGURATIONS:
        agent.dense_search = dense_search if configuration["dense"] else None
        agent.fusion_method = configuration["fusion"]
        outcome = evaluate(agent, samples, catalog_ids, categories, raw_products)
        # Drop per-session rows; only aggregates matter for the comparison.
        results[configuration["name"]] = {
            key: value for key, value in outcome.items() if key != "sessions"
        }
    agent.dense_search = dense_search
    return results


def _format_table(results: dict) -> str:
    header = f"{'config':<14}" + "".join(f"{key.replace('recommended_', ''):>22}" for key in METRIC_KEYS)
    lines = [header, "-" * len(header)]
    for name, metrics in results.items():
        row = f"{name:<14}" + "".join(f"{metrics[key]:>22.6f}" for key in METRIC_KEYS)
        lines.append(row)

    lines.append("")
    lines.append(f"{'per-scenario hit_rate_at_10':<14}")
    scenarios = sorted(next(iter(results.values()))["scenario_metrics"])
    lines.append(f"{'config':<14}" + "".join(f"{name:>18}" for name in scenarios))
    for name, metrics in results.items():
        row = f"{name:<14}" + "".join(
            f"{metrics['scenario_metrics'][scenario]['hit_rate_at_10']:>18.4f}"
            for scenario in scenarios
        )
        lines.append(row)
    return "\n".join(lines)


if __name__ == "__main__":
    outcome = run_ablation()
    print(_format_table(outcome))
    Path("fusion_ablation.json").write_text(json.dumps(outcome, indent=2) + "\n", encoding="utf-8")
    print("\nwrote fusion_ablation.json")
