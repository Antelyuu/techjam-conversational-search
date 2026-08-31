"""Print one real multi-turn session as a readable transcript.

The competition's final deliverables ask for "one demonstrated multi-turn
session". `local_evaluator` plays 200 of them and prints only the aggregate, so
this script plays exactly one and shows the conversation.

The loop below **mirrors `evaluator.local_evaluator.evaluate()`** -- the same
opener, the same customer reply policy, the same override schedule, and the
same break-on-first-hit rule. Every evaluator-side function is imported rather
than reimplemented, so a transcript cannot drift from what the official scorer
would have seen. What is printed is a scored session, not a re-enactment.

    python3 -m scripts.demo_session                        # first buying session
    python3 -m scripts.demo_session --scenario browsing
    python3 -m scripts.demo_session --scenario intent_override
    python3 -m scripts.demo_session --sample-id public_0002
    python3 -m scripts.demo_session --all-turns            # keep going past the hit

`--all-turns` is a diagnostic, not a scored run: it removes the evaluator's
early break so the later turns become visible. That is legitimate because the
simulated customer's replies depend only on `ask_attribute` and never on
`recommendations` (the same property `scripts/replay_ranks.py` relies on), but
the score shown is still the one taken at the first hit.
"""

from __future__ import annotations

import argparse
import textwrap
import uuid

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from starter.agent import Agent

WIDTH = 88
INDENT = " " * 10


def rule(char: str = "-") -> str:
    return char * WIDTH


def say(speaker: str, text: str) -> None:
    """One wrapped, left-labelled line of dialogue."""
    body = textwrap.fill(
        text.strip() or "(nothing)",
        width=WIDTH,
        initial_indent=f"{speaker:<10}",
        subsequent_indent=INDENT,
    )
    print(body)


def title_of(products: dict[str, dict], parent_asin: str, limit: int = 62) -> str:
    title = str(products.get(parent_asin, {}).get("title") or "?")
    title = " ".join(title.split())
    return title if len(title) <= limit else title[: limit - 1] + "…"


def pick_sample(samples: list[dict], sample_id: str | None, scenario: str | None) -> dict:
    if sample_id:
        for sample in samples:
            if sample["sample_id"] == sample_id:
                return sample
        raise SystemExit(f"no sample with sample_id {sample_id!r}")
    if scenario:
        for sample in samples:
            if sample["scenario_type"] == scenario:
                return sample
        raise SystemExit(f"no sample with scenario_type {scenario!r}")
    return samples[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Print one multi-turn session transcript")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--sample-id", default=None, help="e.g. public_0002")
    parser.add_argument(
        "--scenario",
        default="buying",
        choices=["buying", "browsing", "intent_override", "boundary"],
        help="first session of this type; ignored when --sample-id is given",
    )
    parser.add_argument(
        "--all-turns",
        action="store_true",
        help="do not stop at the first hit (diagnostic; the score still reflects it)",
    )
    parser.add_argument(
        "--paraphrase",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="replace the quoting customer with a paraphrasing one at this level "
             "(scripts/paraphrase_eval); 0 is the benchmark's own verbatim customer",
    )
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    if args.paraphrase:
        # Rebind rather than rely on the patch reaching us: this module imports
        # `customer_reply` and `initial_message` by name at import time, so
        # patching evaluator.local_evaluator alone would leave those two names
        # pointing at the original quoting customer and the transcript would
        # silently print a verbatim session.
        global customer_reply, initial_message
        from scripts.paraphrase_eval import install_paraphrasing_customer
        import evaluator.local_evaluator as _ev

        install_paraphrasing_customer(args.paraphrase, False, args.seed)
        customer_reply = _ev.customer_reply
        initial_message = _ev.initial_message

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    sample = pick_sample(samples, args.sample_id, args.scenario)

    # Exactly what the evaluator sets up before the first turn.
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}
    category = coarse_category(categories.get(target, []))
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    user_message = initial_message(effective, category, disclosed)

    print(rule("="))
    print(f" {sample['sample_id']}   scenario: {sample['scenario_type']}")
    print(rule("="))
    print(f" target     {target}  {title_of(products, target)}")
    print(f" category   {category}")
    print(" hidden card — the only strings this customer can ever disclose:")
    for label, key in (("hard", "hard_constraints"), ("soft", "soft_preferences")):
        for value in card.get(key, []):
            print(f"   {label:<5} {value}")
    if not override_applied:
        override = behavior.get("override") or {}
        print(f"   override on turn {override.get('turn')}: {override.get('new_value')!r}")
    print(rule("="))
    print(" The agent sees none of the above — only the messages marked 'customer'.")

    agent = Agent(args.catalog)
    session_id = f"public_{uuid.uuid4().hex}"
    agent.reset(session_id, sample["user_profile"])

    hit_turn: int | None = None
    best_rank: int | None = None
    for turn in range(1, MAX_TURNS + 1):
        print()
        print(f"--- turn {turn} " + rule()[len(f"--- turn {turn} "):])
        say("customer", user_message)

        response = agent.respond(session_id, user_message, turn, TOP_K)
        say("agent", str(response.get("message", "")))
        print(f"{'ask':<10}{response.get('ask_attribute')!r}")

        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        label = f"{len(ranked)} recommendation" + ("" if len(ranked) == 1 else "s")
        print(f"{'returns':<10}{label}")
        for position, parent_asin in enumerate(ranked, start=1):
            marker = ">>" if parent_asin == target else "  "
            print(f"{INDENT}{marker} {position:>2}. {parent_asin}  {title_of(products, parent_asin)}")

        # The evaluator's scoring rule: the first turn the target appears is the
        # one that counts, and the rank is frozen there.
        if override_applied and target in ranked and hit_turn is None:
            best_rank = ranked.index(target) + 1
            hit_turn = turn
            print(f"{INDENT}** target found at rank {best_rank} — the evaluator stops here **")
            if not args.all_turns:
                break
        if turn == MAX_TURNS:
            break

        # The customer's next message: an override if one is due, otherwise a
        # reply to whatever attribute the agent asked about.
        override = effective.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
        else:
            user_message, boundary_used = customer_reply(
                effective, response.get("ask_attribute"), disclosed, boundary_used
            )

    agent.close()

    print()
    print(rule("="))
    if hit_turn is None:
        print(" MISS — the target never entered the returned list (scores 0, MTTC 11)")
    else:
        print(
            f" HIT on turn {hit_turn} at rank {best_rank}"
            f"   reciprocal rank {1.0 / best_rank:.3f}"
        )
    print(rule("="))


if __name__ == "__main__":
    main()
