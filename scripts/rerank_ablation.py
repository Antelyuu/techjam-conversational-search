"""P4 reranker ablation, crossed with the P3 fusion choice.

Two questions, one run, because they interact:

  * Does the deterministic final scorer (P4-T1) beat the fused order it
    replaces? P4's exit criteria only allow keeping it if it does.
  * Does `weighted` still beat `rrf`? E2 chose weighted by ~one session and
    warned that RRF was better at Browsing, so the choice should be re-tested
    once clarification changes what the queries look like -- which it now has.

The clarification policy is held at its default throughout; only the two
variables under test move.

Run from the repository root:

    python3 -m scripts.rerank_ablation
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

CONFIGURATIONS = tuple(
    {"name": f"{fusion}_{'rerank' if rerank else 'fused'}", "fusion": fusion, "rerank": rerank}
    for fusion in (FUSION_WEIGHTED, FUSION_RRF)
    for rerank in (False, True)
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
            "dense route unavailable, so the fusion arm of this ablation is "
            "meaningless; check the stderr warning above"
        )

    results: dict[str, dict] = {}
    for configuration in CONFIGURATIONS:
        agent.fusion_method = configuration["fusion"]
        agent.enable_reranker = configuration["rerank"]
        outcome = evaluate(agent, samples, catalog_ids, categories, raw_products)
        results[configuration["name"]] = {
            key: value for key, value in outcome.items() if key != "sessions"
        }
    return results


def _format_table(results: dict) -> str:
    header = f"{'config':<20}" + "".join(
        f"{key.replace('recommended_', ''):>22}" for key in METRIC_KEYS
    )
    lines = [header, "-" * len(header)]
    for name, metrics in results.items():
        lines.append(f"{name:<20}" + "".join(f"{metrics[key]:>22.6f}" for key in METRIC_KEYS))

    scenarios = sorted(next(iter(results.values()))["scenario_metrics"])
    lines.append("")
    lines.append("per-scenario hit_rate_at_10")
    lines.append(f"{'config':<20}" + "".join(f"{name:>18}" for name in scenarios))
    for name, metrics in results.items():
        lines.append(
            f"{name:<20}"
            + "".join(
                f"{metrics['scenario_metrics'][s]['hit_rate_at_10']:>18.4f}" for s in scenarios
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":
    outcome = run_ablation()
    print(_format_table(outcome))
    Path("rerank_ablation.json").write_text(json.dumps(outcome, indent=2) + "\n", encoding="utf-8")
    print("\nwrote rerank_ablation.json")
