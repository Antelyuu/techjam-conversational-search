"""P7-T2: sweep the paraphrase weight profile without re-reading the catalogue.

`scripts.paraphrase_eval` runs one configuration per process, which is right
for a headline number and wasteful for a sweep -- the catalogue parse is paid
again for every point on the curve. This script pays it once and re-runs the
replay per configuration.

Two hazards it exists to handle, both of which silently corrupt a sweep rather
than failing it:

* `install_paraphrasing_customer` captures `ev.customer_reply` at call time.
  Called twice without restoring, the second call wraps the *patched* function
  and the customer is paraphrased twice over. The originals are therefore
  saved once here and restored before every install.
* the agent carries per-session state, so a configuration must get a fresh
  `ev.Agent` rather than a reused one.

Usage:
    python3 -m scripts.paraphrase_profile_sweep --feature lexical_rank \
        --values 1 2 4 6 8 --level 2
"""

from __future__ import annotations

import argparse
import json

from evaluator import local_evaluator as ev
from scripts import paraphrase_eval as pe
from shopping_agent import reranking

# Captured before anything is patched; every install starts from these.
_REAL_REPLY = ev.customer_reply
_REAL_INITIAL = ev.initial_message


def run_one(level: int, seed: int, keep_opener: bool, reword_cat: bool,
            paraphrase_cat: bool, samples, catalog_ids, categories, products,
            catalog: str) -> dict:
    ev.customer_reply = _REAL_REPLY
    ev.initial_message = _REAL_INITIAL
    pe.install_paraphrasing_customer(
        level, paraphrase_cat, seed, reword_cat, keep_opener
    )
    result = ev.evaluate(
        ev.Agent(catalog), samples, catalog_ids, categories, products
    )
    ev.customer_reply = _REAL_REPLY
    ev.initial_message = _REAL_INITIAL
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--level", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--keep-opener", action="store_true")
    parser.add_argument("--reword-category", action="store_true")
    parser.add_argument("--paraphrase-category", action="store_true")
    parser.add_argument(
        "--feature", default="lexical_rank",
        help="which PARAPHRASE_WEIGHTS entry to sweep",
    )
    parser.add_argument("--values", type=float, nargs="+", required=True)
    parser.add_argument(
        "--threshold", type=float, nargs="*", default=None,
        help="instead sweep reranking.RELAXED_OWNERSHIP_THRESHOLD",
    )
    args = parser.parse_args()

    samples = ev.load_jsonl(args.dataset)
    catalog_ids, categories, products = ev.catalog_index(args.catalog)

    baseline = dict(reranking.PARAPHRASE_WEIGHTS)
    sweep = args.threshold if args.threshold is not None else args.values
    for value in sweep:
        if args.threshold is not None:
            reranking.RELAXED_OWNERSHIP_THRESHOLD = value
        else:
            reranking.PARAPHRASE_WEIGHTS.clear()
            reranking.PARAPHRASE_WEIGHTS.update(baseline)
            reranking.PARAPHRASE_WEIGHTS[args.feature] = value
        result = run_one(
            args.level, args.seed, args.keep_opener, args.reword_category,
            args.paraphrase_category, samples, catalog_ids, categories,
            products, args.catalog,
        )
        print(json.dumps({
            "swept": "threshold" if args.threshold is not None else args.feature,
            "value": value,
            "level": args.level,
            "score": result["recommended_technical_score"],
            "hit_rate_at_10": result["hit_rate_at_10"],
            "mrr": result["mrr"],
            "mttc": result["mttc"],
        }), flush=True)


if __name__ == "__main__":
    main()
