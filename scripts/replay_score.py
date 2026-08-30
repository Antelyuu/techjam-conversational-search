"""Score a shortlist policy against a recorded rank trace, offline.

Reads `replay_trace.json` from scripts/replay_ranks.py and applies a policy
function of (turn, top_k, live, consistent) -> size, reproducing exactly what
the evaluator would have measured. Validating this against the live evaluator
is not optional: run `python3 -m scripts.replay_score` with no arguments and it
checks the shipped policy and the always-ten baseline against the two numbers
E8 recorded, before printing anything else.

Run from the repository root:

    python3 -m scripts.replay_score [trace.json]
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shopping_agent import shortlist

MAX_TURNS = 10
TOP_K = 10

Policy = Callable[[int, int, int, int, int], int]


def score(trace: list[dict], policy: Policy) -> dict:
    """Replay every session under `policy` and return the evaluator's metrics."""
    hits = 0
    reciprocal = []
    turns_to_first = []
    for session in trace:
        first_turn = None
        rank = None
        for record in session["turns"]:
            if not record["scorable"]:
                # Before the override fires the evaluator does not check the
                # list at all, so whatever the agent returned cannot score.
                continue
            size = policy(
                record["turn"], TOP_K, record["live"], record["consistent"], record["disclosed"]
            )
            shown = record["ranked"][:size]
            if session["target"] in shown:
                first_turn = record["turn"]
                rank = shown.index(session["target"]) + 1
                break
        hits += int(first_turn is not None)
        reciprocal.append(0.0 if rank is None else 1.0 / rank)
        turns_to_first.append(first_turn if first_turn is not None else MAX_TURNS + 1)

    hit_rate = hits / len(trace)
    mrr = statistics.fmean(reciprocal)
    mttc = statistics.fmean(turns_to_first)
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    return {
        "hit_rate": round(hit_rate, 6),
        "mrr": round(mrr, 6),
        "mttc": round(mttc, 6),
        "score": round(0.50 * round(hit_rate, 6) + 0.30 * round(mrr, 6) + 0.20 * round(efficiency, 6), 6),
    }


SHIPPED: Policy = lambda turn, top_k, live, consistent, disclosed: shortlist.shortlist_size(
    turn, top_k, live, consistent, disclosed, enabled=True
)
ALWAYS_TEN: Policy = lambda turn, top_k, live, consistent, disclosed: top_k


def load(path: str = "replay_trace.json") -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate(trace: list[dict]) -> None:
    """The trace is worthless unless it reproduces the live evaluator."""
    for name, policy, expected in (
        ("shipped", SHIPPED, 0.945497),
        ("always-ten", ALWAYS_TEN, 0.885293),
    ):
        got = score(trace, policy)
        status = "OK " if got["score"] == expected else "MISMATCH"
        print(f"{status} {name:12s} {got} expected {expected}")


def main() -> None:
    trace = load(sys.argv[1] if len(sys.argv) > 1 else "replay_trace.json")
    validate(trace)


if __name__ == "__main__":
    main()
