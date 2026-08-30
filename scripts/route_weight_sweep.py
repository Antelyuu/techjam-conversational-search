"""Is the dense route still worth having, and at what weight?

P3 adopted a 0.5/0.5 lexical/dense split as an unswept neutral prior, and E2
flagged tuning it as the obvious untried lever. P4 then changed the query
distribution completely: the customer now discloses constraint strings taken
from the target product's own features and details, so queries carry
near-verbatim product text. BM25 is very strong on verbatim overlap, and dense
embeddings blur exactly that signal -- which predicts the dense route is now
diluting a better lexical one rather than complementing a weak one.

This sweeps the split from all-lexical to the current even blend, so the
decision rests on measurement rather than on P3's finding, which was made
under a query distribution that no longer exists.

Run from the repository root:

    python3 -m scripts.route_weight_sweep
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
from starter.agent import Agent

CONFIGURATIONS = (
    {"name": "dense_off", "dense": False, "weights": None},
    {"name": "lex_0.95", "dense": True, "weights": {"lexical": 0.95, "dense": 0.05}},
    {"name": "lex_0.9", "dense": True, "weights": {"lexical": 0.9, "dense": 0.1}},
    {"name": "lex_0.8", "dense": True, "weights": {"lexical": 0.8, "dense": 0.2}},
    {"name": "lex_0.7", "dense": True, "weights": {"lexical": 0.7, "dense": 0.3}},
    {"name": "lex_0.6", "dense": True, "weights": {"lexical": 0.6, "dense": 0.4}},
    {"name": "even_0.5", "dense": True, "weights": {"lexical": 0.5, "dense": 0.5}},
)

METRIC_KEYS = ("hit_rate_at_10", "mrr", "mttc", "efficiency", "recommended_technical_score")


def run_sweep(
    catalog_path: str = "data/catalog.jsonl",
    dataset_path: str = "data/public_set.jsonl",
) -> dict:
    samples = load_jsonl(dataset_path)
    catalog_ids, categories, raw_products = catalog_index(catalog_path)

    agent = Agent(catalog_path)
    try:
        if agent.dense_search is None:
            raise SystemExit(
                "dense route unavailable, so every row would be the dense_off "
                "control; check the stderr warning above"
            )
        dense_search = agent.dense_search

        results: dict[str, dict] = {}
        for configuration in CONFIGURATIONS:
            agent.dense_search = dense_search if configuration["dense"] else None
            agent.route_weights = configuration["weights"]
            outcome = evaluate(agent, samples, catalog_ids, categories, raw_products)
            results[configuration["name"]] = {
                key: value for key, value in outcome.items() if key != "sessions"
            }
        return results
    finally:
        agent.close()


def _format_table(results: dict) -> str:
    header = f"{'config':<12}" + "".join(
        f"{key.replace('recommended_', ''):>22}" for key in METRIC_KEYS
    )
    lines = [header, "-" * len(header)]
    for name, metrics in results.items():
        lines.append(f"{name:<12}" + "".join(f"{metrics[key]:>22.6f}" for key in METRIC_KEYS))

    scenarios = sorted(next(iter(results.values()))["scenario_metrics"])
    lines += ["", "per-scenario hit_rate_at_10"]
    lines.append(f"{'config':<12}" + "".join(f"{name:>18}" for name in scenarios))
    for name, metrics in results.items():
        lines.append(
            f"{name:<12}"
            + "".join(
                f"{metrics['scenario_metrics'][s]['hit_rate_at_10']:>18.4f}" for s in scenarios
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":
    outcome = run_sweep()
    print(_format_table(outcome))
    Path("route_weight_sweep.json").write_text(
        json.dumps(outcome, indent=2) + "\n", encoding="utf-8"
    )
    print("\nwrote route_weight_sweep.json")
