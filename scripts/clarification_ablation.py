"""P4 clarification ablation.

Evaluates the same agent under each dialogue-policy configuration so the two
open P4 decisions rest on measured scores rather than preference:

    no_questions        ask_attribute always None -- the P3 control
    prior_only          ask by measured per-attribute yield alone
    prior_disagreement  yield modulated by candidate disagreement (P4-T3)
    plus_wildcard       the above, with "other" as a last-resort fallback

The first two answer "does asking help at all"; the third answers "does
candidate-diversity analysis earn its place"; the fourth prices the degenerate
"other" wildcard, which matches any undisclosed constraint in the simulator.

Run from the repository root, after `pip install -r requirements.txt`:

    python3 -m scripts.clarification_ablation

One Agent is constructed and its policy flags are switched between runs, so
the FTS index and the embedding matrix are each loaded once rather than four
times. Sessions are isolated per run by the evaluator's own reset() call, so
no state carries across configurations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent

CONFIGURATIONS = (
    {"name": "no_questions", "ask": False, "disagreement": False, "wildcard": False, "block_soft": True},
    {"name": "prior_only", "ask": True, "disagreement": False, "wildcard": False, "block_soft": True},
    {"name": "prior_disagreement", "ask": True, "disagreement": True, "wildcard": False, "block_soft": True},
    {"name": "soft_askable", "ask": True, "disagreement": True, "wildcard": False, "block_soft": False},
    {"name": "soft_plus_wildcard", "ask": True, "disagreement": True, "wildcard": True, "block_soft": False},
    {"name": "wildcard_only", "ask": True, "disagreement": True, "wildcard": True, "block_soft": True},
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
        print(
            "warning: dense route unavailable, so this ablation runs lexical-only "
            "and its numbers are not comparable to the recorded results",
            file=sys.stderr,
        )

    # Hold the reranker out so this ablation moves one variable. Its own
    # effect is measured separately; mixing the two would leave neither
    # attributable.
    agent.enable_reranker = False

    results: dict[str, dict] = {}
    for configuration in CONFIGURATIONS:
        agent.enable_clarification = configuration["ask"]
        agent.use_disagreement = configuration["disagreement"]
        agent.allow_wildcard = configuration["wildcard"]
        agent.block_soft_slots = configuration["block_soft"]
        outcome = evaluate(agent, samples, catalog_ids, categories, raw_products)
        # Drop per-session rows; only aggregates matter for the comparison.
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

    lines.append("")
    lines.append("per-scenario hit_rate_at_10")
    scenarios = sorted(next(iter(results.values()))["scenario_metrics"])
    lines.append(f"{'config':<20}" + "".join(f"{name:>18}" for name in scenarios))
    for name, metrics in results.items():
        lines.append(
            f"{name:<20}"
            + "".join(
                f"{metrics['scenario_metrics'][scenario]['hit_rate_at_10']:>18.4f}"
                for scenario in scenarios
            )
        )

    lines.append("")
    lines.append("per-scenario mttc")
    lines.append(f"{'config':<20}" + "".join(f"{name:>18}" for name in scenarios))
    for name, metrics in results.items():
        lines.append(
            f"{name:<20}"
            + "".join(
                f"{metrics['scenario_metrics'][scenario]['mttc']:>18.4f}"
                for scenario in scenarios
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":
    outcome = run_ablation()
    print(_format_table(outcome))
    Path("clarification_ablation.json").write_text(
        json.dumps(outcome, indent=2) + "\n", encoding="utf-8"
    )
    print("\nwrote clarification_ablation.json")
