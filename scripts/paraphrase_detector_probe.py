"""Does the agent actually KNOW when the customer has stopped quoting?

shopping_agent/shortlist.py already carries a paraphrase-regime detector, and
it is one line:

    if disclosed > 0 and live_disclosures <= 0:
        return top_k          # stop withholding, the ranking is unreadable

`live_disclosures` comes from `reranking.prepare_evidence()` and counts the
disclosed constraints that at least one POOLED candidate owns as a whole card
value. `disclosed` is `len(state.disclosed_text)`. So the condition reads "the
customer has said something and nothing in the pool owns any of it", which is
what a paraphrasing customer looks like from the inside -- and it is the only
signal the agent has that the regime has changed at all.

Before more ranking work is hung off that condition it has to be shown to
fire, and to fire under paraphrasers that were not used to design it. That is
what this measures, for each `--paraphraser` mode of scripts/paraphrase_eval.py:

  * `detector_rate` -- share of ALL measured turns where the condition holds.
  * `detector_rate_given_disclosed` -- the same, over the turns where the
    customer had actually said something (`disclosed > 0`). The first number
    is diluted by opening turns where nothing has been disclosed yet and no
    detector could fire; this one is the conditional the shortlist policy is
    really asking about.
  * `live_rate_given_disclosed` -- the complement's mirror: share of disclosed
    turns where at least one disclosure IS still owned. Verbatim quoting
    should sit near 1.0 here, which is the control that says the instrument
    works.

WHY IT MONKEYPATCHES INSTEAD OF EDITING THE AGENT. Nothing in this file may
ship, and `shopping_agent/` must not learn that it is being measured.
`starter/agent.py` imports `prepare_evidence` by name at module scope and
calls it as a global, so rebinding `starter.agent.prepare_evidence` wraps the
real one without touching the package -- the same trick, for the same reason,
that scripts/paraphrase_headroom.py uses on `rerank`.

The wrapper is pure observation: it calls the real function and returns its
result unchanged, so the run it measures is the run paraphrase_eval scores.
Every row below reprints the composite, and every one of them matches the
paraphrase_eval run at the same settings -- which is the check that the
instrument did not perturb what it was instrumenting.

MEASURED (seed 20260831, BM25 route, 200 public sessions):

  run                                    turns  disclosed  fired  fired|disc
  --level 0             (verbatim)         561        461      0      0.0000
  --level 2             (synonym)          845        635    537      0.8457
  --paraphraser structural --level 2       624        414    356      0.8599
  ... structural --level 2 +kc +ko         501        401    355      0.8853
  ... synonym    --level 2 +kc +ko         797        697    565      0.8106

  (+kc +ko = --keep-carrier --keep-opener, which leaves the simulator's own
   parseable wrapper in place so the ONLY thing paraphrased is the value.)

Three things follow.

FIRST, the control is clean. Verbatim quoting fires the detector 0 times in
461 disclosed turns, so the condition has no false-positive rate to speak of
on the regime it was not built for. It is a detector, not a coin.

SECOND, it fires under both paraphrasers, at 0.85 and 0.86 of disclosed turns
-- and the two mechanisms are different enough that this is not one result
counted twice. So a paraphrase-regime response hung off this condition would
actually run.

THIRD, and this is the caveat that matters: in the DEFAULT configuration a
large part of the firing is not the paraphrase at all. At level 1 and above
the customer stops using the simulator's colon-framed wrapper, the
orchestrator cannot strip a framing clause it cannot find, and whole
utterances land in `disclosed_text` that no product could ever own. The
+kc +ko rows are the honest measurement of the value rewriting alone, and
they are HIGHER (0.8853, 0.8106), not lower -- so the detector is genuinely
responding to the paraphrase and not merely to a broken wrapper. Worth
stating explicitly, because the default numbers alone could not distinguish
the two and the difference decides whether the detector is real.

Structural fires MORE than synonym in the isolated condition while costing
much less score (0.853733 against 0.729474). That is the expected shape: the
structural mode is a pure attack on whole-value ownership and defeats it on
100% of values, whereas the synonym lexicon misses about 9% of them
outright -- but it takes the words away too, so retrieval degrades and the
score falls further. The detector sees ownership, not retrieval, which is
precisely why it must be measured under a paraphraser that separates them.

Usage:
    python3 -m scripts.paraphrase_detector_probe --paraphraser structural --level 2
    python3 -m scripts.paraphrase_detector_probe --paraphraser synonym --level 2
    python3 -m scripts.paraphrase_detector_probe --level 0     # verbatim control
"""

from __future__ import annotations

import argparse
import json

from evaluator import local_evaluator as ev
from scripts.paraphrase_eval import PARAPHRASERS, install_paraphrasing_customer
from starter import agent as agent_module

TURNS: list[tuple[int, int]] = []


def install_detector_probe() -> None:
    """Record (disclosed, live_disclosures) for every turn that measures them.

    Turns where the reranker is disabled or raises never call this at all, so
    the denominator here is exactly the set of turns on which the shortlist
    policy could consult the detector -- which is the set the question is
    about. An unmeasured turn is not a measured zero; see the `measured` flag
    in starter/agent.py.
    """
    real_prepare = agent_module.prepare_evidence

    def instrumented_prepare(disclosures, candidates, products):
        prepared = real_prepare(disclosures, candidates, products)
        TURNS.append((len(disclosures), prepared.live_disclosures))
        return prepared

    agent_module.prepare_evidence = instrumented_prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--level", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--paraphraser", default="synonym", choices=sorted(PARAPHRASERS))
    parser.add_argument("--keep-opener", action="store_true")
    parser.add_argument("--keep-carrier", action="store_true")
    parser.add_argument("--reword-category", action="store_true")
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    install_paraphrasing_customer(
        args.level,
        False,
        args.seed,
        args.reword_category,
        args.keep_opener,
        paraphraser=args.paraphraser,
        keep_carrier=args.keep_carrier,
    )
    install_detector_probe()

    samples = ev.load_jsonl(args.dataset)
    catalog_ids, categories, products = ev.catalog_index(args.catalog)
    result = ev.evaluate(ev.Agent(args.catalog), samples, catalog_ids, categories, products)

    turns = len(TURNS)
    disclosed_turns = sum(1 for disclosed, _ in TURNS if disclosed > 0)
    fired = sum(1 for disclosed, live in TURNS if disclosed > 0 and live == 0)
    live = disclosed_turns - fired

    def share(count: int, total: int) -> float | None:
        return round(count / total, 4) if total else None

    print(json.dumps({
        "paraphraser": args.paraphraser,
        "level": args.level,
        "keep_opener": args.keep_opener,
        "keep_carrier": args.keep_carrier,
        "turns_measured": turns,
        "turns_with_disclosure": disclosed_turns,
        "detector_fired": fired,
        "detector_rate": share(fired, turns),
        "detector_rate_given_disclosed": share(fired, disclosed_turns),
        "live_rate_given_disclosed": share(live, disclosed_turns),
        "score": result["recommended_technical_score"],
    }))


if __name__ == "__main__":
    main()
