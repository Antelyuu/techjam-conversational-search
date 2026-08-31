"""Which ranking features still earn their weight when nobody is quoting.

The evidence features are documented to "fail quiet": if the customer
paraphrases, nothing matches, every candidate scores 0.0 and the ordering falls
through to whatever is beneath. The claim is load-bearing -- it is why three
features carry 34 of the table's ~48 weight -- and it had never been tested.
Measured at level 2 it is true of one of them, false of another, and not quite
the right question for the third:

  * `phrase_evidence` fails quiet exactly as documented. Zeroing it changes the
    paraphrased score by nothing at all, to six decimals.
  * `constraint_evidence` does not. It scores *partial* token coverage, so it
    does not go to zero under paraphrase -- it goes to a small number driven by
    whichever words happened to survive the rewording. Zeroing it makes the
    paraphrased score go **up**. At weight 12.0 that is not silence, it is
    noise with a loud voice.
  * `slot_evidence` is not silent either, in the other direction: zeroing it
    *costs* 0.010 paraphrased. Whole-value ownership still fires, because a
    constraint containing none of the substituted words survives rewording
    intact. Quieter, not quiet.

This probe re-runs the paraphrase evaluation under alternative weight profiles,
patching reranking.FEATURE_WEIGHTS in place. Nothing here ships -- it exists to
say whether a paraphrase-triggered reweighting (the cheapest structural fix
available, since shortlist.py already computes the detector) has anything to
win, before anyone writes it.

Usage:
    python3 -m scripts.paraphrase_weight_probe --level 2
"""

from __future__ import annotations

import argparse
import json

from evaluator import local_evaluator as ev
from scripts.paraphrase_eval import install_paraphrasing_customer
from shopping_agent import reranking


# Each profile is an override applied on top of the shipped table.
PROFILES: dict[str, dict[str, float]] = {
    "shipped": {},
    # Is partial token coverage signal or noise once nothing matches whole?
    "no_constraint_evidence": {"constraint_evidence": 0.0},
    "no_phrase_evidence": {"phrase_evidence": 0.0},
    "no_partial_evidence": {"constraint_evidence": 0.0, "phrase_evidence": 0.0},
    # Included as the control for the two above, on the expectation that
    # whole-value ownership is dead under paraphrase and dropping it would
    # change nothing. It costs 0.010, which is how the "some constraints
    # survive rewording intact" effect was found.
    "no_slot_evidence": {"slot_evidence": 0.0},
    # If the exact features are dead, does what is underneath deserve more say?
    "lexical_rank_4": {"lexical_rank": 4.0},
    "lexical_rank_8": {"lexical_rank": 8.0},
    # The combination the two probes above point at: drop the feature that is
    # scoring noise, keep the one that is still occasionally right, and let
    # retrieval rank have more say.
    "no_constraint_plus_lexical_4": {
        "constraint_evidence": 0.0,
        "lexical_rank": 4.0,
    },
    "fallback_profile": {
        "constraint_evidence": 0.0,
        "phrase_evidence": 0.0,
        "slot_evidence": 0.0,
        "lexical_rank": 4.0,
    },
}


def run(profile: str, level: int, seed: int, catalog: str, dataset: str) -> dict:
    original = dict(reranking.FEATURE_WEIGHTS)
    try:
        reranking.FEATURE_WEIGHTS.update(PROFILES[profile])
        install_paraphrasing_customer(level, False, seed)
        samples = ev.load_jsonl(dataset)
        catalog_ids, categories, products = ev.catalog_index(catalog)
        result = ev.evaluate(
            ev.Agent(catalog), samples, catalog_ids, categories, products
        )
        return {
            "profile": profile,
            "level": level,
            "score": result["recommended_technical_score"],
            "hit_rate_at_10": result["hit_rate_at_10"],
            "mrr": result["mrr"],
            "mttc": result["mttc"],
        }
    finally:
        reranking.FEATURE_WEIGHTS.clear()
        reranking.FEATURE_WEIGHTS.update(original)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--level", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--profile", default="shipped", choices=sorted(PROFILES))
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    print(json.dumps(run(args.profile, args.level, args.seed, args.catalog, args.dataset)))


if __name__ == "__main__":
    main()
