"""How much of the paraphrase gap is reachable, and by what.

scripts/paraphrase_eval.py says what the agent scores when the customer stops
quoting. It does not say how much of the shortfall is *recoverable*, and those
are different questions: a target that never enters the candidate pool cannot
be recovered by any amount of reranking, while one that is pooled and ranked
eleventh is pure ranking loss.

This splits the gap into the two, by instrumenting the same replay:

  * **pool recall** -- the share of sessions where the ground-truth target
    enters the rerank pool on at least one turn. This is a hard ceiling on
    HitRate: nothing downstream of retrieval can beat it.

  * **the ranking oracle** -- the same run with the target hoisted to rank 1
    on every turn it is pooled. This is the score a perfect reranker would
    get against today's retrieval, so the distance from the real score to it
    is the headroom available to ranking work, and the distance from the
    oracle to 1.0 is what only retrieval can buy.

Both are diagnostics and neither is reachable in practice -- the oracle knows
the answer. They bound the search rather than predicting it.

The composite is 0.50*HitRate + 0.30*MRR + 0.20*(11-MTTC)/10, so each row also
reports where its gap sits, because the three are not worth the same and MTTC
saturates.

Usage:
    python3 -m scripts.paraphrase_headroom --level 2
    python3 -m scripts.paraphrase_headroom --level 0        # verbatim control
"""

from __future__ import annotations

import argparse
import json

from evaluator import local_evaluator as ev
from scripts.paraphrase_eval import install_paraphrasing_customer
from starter import agent as agent_module

# The session currently being replayed. `evaluate` runs one session at a time,
# and initial_message is called exactly once at the top of each, which makes it
# the reset point.
CURRENT: dict = {"target": None, "pooled": False, "turns_pooled": 0, "turns": 0}
SESSIONS: list[dict] = []
# Every rank the target held, over every turn it was pooled, when --rank is on.
RANKS: list[int] = []


def _close_session() -> None:
    if CURRENT["target"] is not None:
        SESSIONS.append(
            {
                "target": CURRENT["target"],
                "pooled": CURRENT["pooled"],
                "turns_pooled": CURRENT["turns_pooled"],
                "turns": CURRENT["turns"],
            }
        )


def install_instrumentation(oracle: bool, record_rank: bool = False) -> None:
    """Record pool membership, and optionally hoist the target to rank 1.

    Wraps the names `starter.agent` actually calls, so nothing in the agent
    changes and the run is otherwise the one paraphrase_eval measures.
    """
    real_initial = ev.initial_message
    real_rerank = agent_module.rerank

    def instrumented_initial(sample, category, disclosed):
        _close_session()
        gt = sample["ground_truth"]
        CURRENT.update(
            {
                "target": gt["parent_asin"] if isinstance(gt, dict) else gt,
                "pooled": False,
                "turns_pooled": 0,
                "turns": 0,
            }
        )
        return real_initial(sample, category, disclosed)

    def instrumented_rerank(
        candidates, products, constraints, limit,
        disclosures=None, prepared=None, stated_category=None,
    ):
        target = CURRENT["target"]
        in_pool = any(c.parent_asin == target for c in candidates)
        CURRENT["turns"] += 1
        if in_pool:
            CURRENT["pooled"] = True
            CURRENT["turns_pooled"] += 1
        if not (oracle and in_pool):
            if record_rank and in_pool:
                # The rank the real scorer gives the target, over the whole
                # pool rather than the returned slice -- a target at 40 and a
                # target at 400 are the same miss in the output and very
                # different problems.
                full = real_rerank(
                    candidates, products, constraints, len(candidates),
                    disclosures, prepared, stated_category,
                )
                for position, item in enumerate(full, start=1):
                    if item.parent_asin == target:
                        RANKS.append(position)
                        break
            return real_rerank(
                candidates, products, constraints, limit,
                disclosures, prepared, stated_category,
            )
        # Rank the whole pool so the target is present however badly it
        # scored, then hoist it. Truncating first would hide exactly the
        # sessions this is measuring.
        full = real_rerank(
            candidates, products, constraints, len(candidates),
            disclosures, prepared, stated_category,
        )
        hoisted = [item for item in full if item.parent_asin == target]
        rest = [item for item in full if item.parent_asin != target]
        return (hoisted + rest)[:limit]

    ev.initial_message = instrumented_initial
    agent_module.rerank = instrumented_rerank


def _composite(hit_rate: float, mrr: float, mttc: float) -> dict:
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    return {
        "hit_share": round(0.50 * hit_rate, 6),
        "mrr_share": round(0.30 * mrr, 6),
        "efficiency_share": round(0.20 * efficiency, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--level", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--keep-opener", action="store_true")
    parser.add_argument("--reword-category", action="store_true")
    parser.add_argument("--oracle", action="store_true", help="hoist the target to rank 1")
    parser.add_argument(
        "--rank", action="store_true",
        help="also report where the real scorer puts the target in the pool",
    )
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    # Order matters: the paraphrasing customer rebinds initial_message, and the
    # instrumentation must wrap the rebound one so both take effect.
    install_paraphrasing_customer(
        args.level, False, args.seed, args.reword_category, args.keep_opener
    )
    install_instrumentation(args.oracle, args.rank)

    samples = ev.load_jsonl(args.dataset)
    catalog_ids, categories, products = ev.catalog_index(args.catalog)
    result = ev.evaluate(ev.Agent(args.catalog), samples, catalog_ids, categories, products)
    _close_session()

    rank_summary = {}
    if RANKS:
        ordered = sorted(RANKS)
        def share(limit):
            return round(sum(1 for r in ordered if r <= limit) / len(ordered), 4)
        rank_summary = {
            "target_turns_ranked": len(ordered),
            "rank_median": ordered[len(ordered) // 2],
            "rank_p90": ordered[int(len(ordered) * 0.9) - 1],
            "share_rank_1": share(1),
            "share_top_3": share(3),
            "share_top_10": share(10),
            "share_top_50": share(50),
        }

    pooled = sum(1 for s in SESSIONS if s["pooled"])
    turns = sum(s["turns"] for s in SESSIONS)
    turns_pooled = sum(s["turns_pooled"] for s in SESSIONS)
    print(json.dumps({
        "level": args.level,
        "oracle": args.oracle,
        "keep_opener": args.keep_opener,
        "sessions": len(SESSIONS),
        "pool_recall": round(pooled / len(SESSIONS), 4) if SESSIONS else None,
        "turn_pool_recall": round(turns_pooled / turns, 4) if turns else None,
        "score": result["recommended_technical_score"],
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        **_composite(result["hit_rate_at_10"], result["mrr"], result["mttc"]),
        **rank_summary,
    }))


if __name__ == "__main__":
    main()
