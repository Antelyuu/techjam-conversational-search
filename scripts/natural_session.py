"""Replay a hand-written, human-sounding conversation through the real agent.

`scripts/demo_session.py` prints a *benchmark* session: the customer is the
evaluator's simulator, and every message it sends is either quoted verbatim out
of the target's own description or a mechanical rewording of it. That is the
right instrument for measuring, and the wrong one for showing what talking to
this agent is like, because no person writes

    For that, what matters is: Material: 100% Cotton; Machine Wash Cold.

This script takes messages a human actually wrote and runs them through the
same `Agent` the harness uses -- same retrieval, same ranking, same shortlist
policy. Nothing is simulated and nothing is scored: it reports where the named
target sits in the returned list on each turn, so the transcript in README.md
can be reproduced rather than taken on trust.

**This is an illustration, not evidence.** One hand-written conversation says
nothing about the distribution; the measured claims come from
`scripts/paraphrase_eval.py`, which replays all 200 sessions through a
paraphrasing customer. Both belong in the README and they are not
interchangeable.

Run with no arguments to reproduce the README's transcript exactly:

    python3 -m scripts.natural_session

Or drive your own:

    python3 -m scripts.natural_session --target B08G4WVYLJ \
        --message "Hi, I'm after some slip-ons" --message "brown leather please"
"""

from __future__ import annotations

import argparse

from starter.agent import Agent

# The README's example. The target is a real catalogue product and these are
# ordinary English sentences -- no constraint is quoted from its description.
DEFAULT_TARGET = "B08G4WVYLJ"
DEFAULT_MESSAGES = [
    "Hi, I'm looking for some loafers or slip-ons - shoes I can just step into.",
    "Brown leather, ideally. Nothing that looks cheap.",
    "They should have a microfibre leather upper and a soft rubber sole - "
    "I want something I can wear outdoors in any season.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument(
        "--message", action="append", default=None,
        help="one customer turn; repeat for a longer conversation",
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    messages = args.message or DEFAULT_MESSAGES
    agent = Agent(args.catalog)
    try:
        agent.reset("natural", {})
        product = agent.products.get(args.target)
        print("=" * 78)
        print(f" target    {args.target}  {product.title[:52] if product else '(not in catalogue)'}")
        if product:
            print(f" category  {product.coarse_category}")
        print("=" * 78)
        print(" Hand-written messages. The agent sees nothing else.\n")

        for turn, message in enumerate(messages, start=1):
            response = agent.respond("natural", message, turn, args.top_k)
            returned = [item["parent_asin"] for item in response["recommendations"]]
            rank = returned.index(args.target) + 1 if args.target in returned else None
            print(f"--- turn {turn} " + "-" * 60)
            print(f"customer  {message}")
            print(f"agent     {response['message']}")
            if rank:
                print(f"          >>> target at rank {rank} of {len(returned)} returned")
            elif returned:
                top = agent.products.get(returned[0])
                print(f"          top result: {top.title[:56] if top else returned[0]}")
            print()
    finally:
        agent.close()


if __name__ == "__main__":
    main()
