"""P7-T2: count how often the paraphrase regime actually fires.

The conditional weight profile's whole cost argument is that it is applied on
zero public turns, so it cannot move the public score. That is a claim about
a count, and this script counts it rather than inferring it from a score that
did not move -- an unchanged score is consistent with the profile never
firing AND with it firing and cancelling out, and those are very different
things to carry to an unseen split.

Wraps `reranking.prepare_evidence` rather than editing it, so the agent under
audit is byte-for-byte the submitted one.

Usage:
    python3 -m scripts.paraphrase_regime_audit --level 0
    python3 -m scripts.paraphrase_regime_audit --level 2
"""

from __future__ import annotations

import argparse
import json

from evaluator import local_evaluator as ev
from scripts import paraphrase_eval as pe
from shopping_agent import reranking

_REAL_REPLY = ev.customer_reply
_REAL_INITIAL = ev.initial_message


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--level", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--keep-opener", action="store_true")
    parser.add_argument("--paraphrase-category", action="store_true")
    args = parser.parse_args()

    stats = {
        "turns": 0,
        "turns_with_disclosures": 0,
        "regime_fired": 0,
        "relaxed_engaged": 0,
    }
    real_prepare = reranking.prepare_evidence

    def counting_prepare(disclosures, candidates, products):
        found = real_prepare(disclosures, candidates, products)
        stats["turns"] += 1
        if found.disclosed > 0:
            stats["turns_with_disclosures"] += 1
        if found.paraphrase_regime:
            stats["regime_fired"] += 1
        if found.relaxed:
            stats["relaxed_engaged"] += 1
        return found

    reranking.prepare_evidence = counting_prepare
    # starter.agent imported the name directly, so rebind it there too.
    import starter.agent as agent_module
    agent_module.prepare_evidence = counting_prepare

    if args.level > 0 or args.paraphrase_category:
        ev.customer_reply = _REAL_REPLY
        ev.initial_message = _REAL_INITIAL
        pe.install_paraphrasing_customer(
            args.level, args.paraphrase_category, args.seed, False,
            args.keep_opener,
        )

    samples = ev.load_jsonl(args.dataset)
    catalog_ids, categories, products = ev.catalog_index(args.catalog)
    result = ev.evaluate(
        ev.Agent(args.catalog), samples, catalog_ids, categories, products
    )

    print(json.dumps({
        "level": args.level,
        "paraphrase_category": args.paraphrase_category,
        **stats,
        "regime_share_of_disclosing_turns": (
            round(stats["regime_fired"] / stats["turns_with_disclosures"], 4)
            if stats["turns_with_disclosures"] else None
        ),
        "score": result["recommended_technical_score"],
    }))


if __name__ == "__main__":
    main()
