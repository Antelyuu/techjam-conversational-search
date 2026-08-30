"""Record the target's rank at every turn, with the evaluator's early break removed.

E8 found the shortlist policy with a throwaway version of this and recommended
recreating it. This is that recipe, kept.

The evaluator ends a session the moment the target appears in the returned list
and freezes the rank it appeared at. That makes every "when should the agent
return something" question expensive to ask: each variant is a full 40-second
run. But the simulator's next message depends only on `ask_attribute` and never
on `recommendations`, so removing the break leaves the conversation trajectory
byte-identical. One run then records, for every session and every turn, the
full ranked ten plus the two evidence counts the shortlist policy reads -- and
any policy over those inputs can be scored offline in milliseconds.

Validate before trusting: `python3 -m scripts.replay_score` replays the shipped
policy over the trace and checks it reproduces the live composite to six
decimals.

Run from the repository root:

    python3 -m scripts.replay_ranks              # -> replay_trace.json
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator import local_evaluator as evaluator
from shopping_agent import shortlist
from starter.agent import Agent

# Filled by the shim below on every call, then read back after respond()
# returns. The shortlist policy is the only place the agent computes these two
# counts, and it is handed them rather than returning them, so intercepting the
# call is the cheapest way to observe them without changing the agent.
_CAPTURE: dict[str, int] = {}


def _capture_size(
    turn: int, top_k: int, live_disclosures: int, consistent: int,
    disclosed: int = 0, enabled: bool = True,
) -> int:
    """Stand in for shortlist_size: record its inputs, then withhold nothing.

    Returning top_k unconditionally is what makes the trace policy-neutral --
    the recorded ranking is the one the agent would have shown with the policy
    off, and every policy is applied to it afterwards."""
    _CAPTURE["live"] = live_disclosures
    _CAPTURE["consistent"] = consistent
    return top_k


def record(catalog_path: str = "data/catalog.jsonl", samples_path: str = "data/public_set.jsonl") -> list[dict]:
    catalog_ids, categories, products = evaluator.catalog_index(catalog_path)
    samples = evaluator.load_jsonl(samples_path)

    # Patched on the module the agent imported from, which is the same object
    # starter.agent holds a reference to, so the call site sees the shim.
    original_size = shortlist.shortlist_size
    shortlist.shortlist_size = _capture_size
    agent = Agent(catalog_path)

    trace: list[dict] = []
    try:
        for sample in samples:
            session_id = f"public_{uuid.uuid4().hex}"
            agent.reset(session_id, sample["user_profile"])
            target = str(sample["ground_truth"]["parent_asin"])
            card, behavior = evaluator.materialize_hidden_fields(sample, products)
            effective = {**sample, "intent_card": card, "behavior": behavior}
            disclosed: set[str] = set()
            boundary_used = False
            # An Intent Override session cannot register a hit before its
            # override fires; the evaluator gates the `target in ranked` check
            # on this exact flag. Recording it per turn keeps the offline
            # scorer honest about which turns can score at all.
            override_applied = sample["scenario_type"] != "intent_override"
            override = effective.get("behavior", {}).get("override") or {}
            override_turn = 0 if override_applied else int(override.get("turn", 3))
            user_message = evaluator.initial_message(
                effective, evaluator.coarse_category(categories.get(target, [])), disclosed
            )

            turns: list[dict] = []
            for turn in range(1, evaluator.MAX_TURNS + 1):
                _CAPTURE.clear()
                try:
                    response = agent.respond(session_id, user_message, turn, evaluator.TOP_K)
                except Exception:
                    response = {"message": "", "ask_attribute": None, "recommendations": []}
                ranked = evaluator.normalize_recommendations(response.get("recommendations"), catalog_ids)
                turns.append({
                    "turn": turn,
                    "ranked": ranked,
                    "rank": ranked.index(target) + 1 if target in ranked else None,
                    "live": _CAPTURE.get("live", 0),
                    "consistent": _CAPTURE.get("consistent", 0),
                    # How many constraints the customer has disclosed so far.
                    # `live` can only be 0 when this is 0, so the two together
                    # separate "the customer has said nothing yet" from "the
                    # customer said something no candidate owns".
                    "disclosed": len(agent.orchestrator.store.get(session_id).disclosed_text),
                    "ask": response.get("ask_attribute"),
                    "scorable": override_applied,
                })
                if turn == evaluator.MAX_TURNS:
                    break
                if not override_applied and turn + 1 == int(override.get("turn", 3)):
                    override_applied = True
                    new_value = str(override.get("new_value", ""))
                    if new_value:
                        disclosed.add(new_value)
                    user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
                else:
                    user_message, boundary_used = evaluator.customer_reply(
                        effective, response.get("ask_attribute"), disclosed, boundary_used
                    )

            trace.append({
                "sample_id": sample["sample_id"],
                "scenario": sample["scenario_type"],
                "target": target,
                "override_turn": override_turn,
                "turns": turns,
            })
            print(f"\r{len(trace)}/{len(samples)}", end="", file=sys.stderr, flush=True)
    finally:
        print(file=sys.stderr)
        agent.close()
        shortlist.shortlist_size = original_size
    return trace


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "replay_trace.json"
    Path(out).write_text(json.dumps(record()), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
