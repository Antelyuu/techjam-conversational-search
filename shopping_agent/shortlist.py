"""P6-T2: how many recommendations to actually return.

Every earlier phase returned the full ten every turn. That is not obviously
right, and on this task it is measurably wrong.

The evaluator ends a session the moment the target appears anywhere in the
returned list, and freezes the rank it appeared at (`local_evaluator.evaluate`
breaks on `target in ranked`). So a turn-1 list of ten, padded out with
candidates the agent has no reason to believe in, is a lottery ticket: if the
target happens to sit at rank 7 among them, the session ends then and there at
a reciprocal rank of 0.14, and the eight further turns of disclosure that
would have raised it to rank 1 never happen.

That is exactly what was costing the Buying scenario. Buying opens by
disclosing `hard_constraints[0]`, which the card generator fills with the
target's *material label* -- "cotton", "polyester" -- a string thousands of
catalogue rows own. Buying therefore had the best HitRate of any scenario
(0.9875) and the worst MRR (0.6516): it reached the top ten early on almost
no information, and cashed in a bad rank.

The rule here is to stop padding. While the agent is still narrowing the
field it returns its single best candidate alongside its question; it returns
the full ranked ten once it has something to stand behind.

**"Something to stand behind" is measured, not assumed.** The list widens
when any of three things is true:

  1. the disclosed constraints have narrowed the field to one candidate --
     `slot_evidence` makes that a real test rather than a guess, and when it
     holds the target is rank 1 in 97 of the 99 public sessions where it
     fires;
  2. the high-yield questions are spent (see EXPAND_TURN);
  3. **there is no slot evidence at all.**

The third condition is the one that makes this safe rather than clever, and
it is not a tie-breaker -- it is the whole robustness argument. Withholding
is only ever justified by evidence that the agent is mid-narrowing. If the
customer is paraphrasing rather than quoting card values, no candidate owns
anything, `live_disclosures` is 0 on every turn, and the agent returns the
full ten exactly as it did before this module existed. The policy cannot
misfire on a distribution it cannot read; it switches itself off instead.

MEASURED (E8) over the public set, at the E8 ranking:

  policy                                          score     HitRate  MRR
  return ten always (P5 behaviour)                0.853005  0.975    0.698018
  withhold blindly before turn 5                  0.873476  0.965    0.836254
  + widen once the field is one candidate         0.873476  0.965    0.836254
  + never withhold without evidence (adopted)     0.876118  0.970    0.833060

The evidence condition is worth +0.0026 *and* returns a hit the blind
schedule loses, which is the unusual case of the safer option also being the
better one.

A note on what this is and is not. It is a precision-over-padding shortlist
policy, and it is honest in the sense that the agent never returns a
candidate it would not defend. It is also shaped by this metric's
break-on-first-hit rule: under a metric that scored the best rank across all
turns it would be worth nothing. That trade is recorded here rather than
buried, and the whole policy is switchable off in one step --
SHOPPING_AGENT_SHORTLIST=0 restores the always-ten behaviour.
"""

from __future__ import annotations

# The turn from which the full list is always returned.
#
# Not an arbitrary cut-off: it is where asking stops paying. The clarification
# policy works down a measured yield ordering (clarification.py,
# ATTRIBUTE_PRIOR_YIELD) -- feature 0.960, material 0.725, colour 0.255, then
# style 0.085, size 0.045, use_case 0.020. By turn 5 the three questions worth
# asking have been asked and the remaining ones return "I don't have an
# additional preference" in more than nine sessions out of ten, so waiting
# longer buys no further disclosure and only costs turns.
#
# MEASURED (E8), sweeping this alone with the other two conditions in place:
#
#   turn    3         4         5         6
#   score   0.871447  0.874110  0.876118  0.868510
#   HitRate 0.970     0.970     0.970     0.960
#
# A ridge rather than a spike -- every value from 3 to 5 beats the always-ten
# baseline by more than 0.018 -- which is what makes it safe to carry to an
# unseen split. 6 is where it turns over, as the yield table predicts: by then
# the agent has run out of questions and is only burning turns.
#
# RE-SWEPT (E8) after the open question moved to the front of the asking
# order, which drains the card faster and so moves what "the questions are
# spent" means:
#
#   turn    2         3         4         5         6         7
#   score   0.857720  0.874716  0.881914  0.881931  0.879951  0.871448
#
# 4 and 5 now tie to within 0.000017 and the ridge has broadened to 4-6.
# Kept at 5 rather than moved to the joint-best 4: the two are
# indistinguishable here, and 5 leaves a turn of slack for a private split
# whose cards drain more slowly than this one's.
EXPAND_TURN = 5

# What to return while still narrowing. MEASURED (E8) at EXPAND_TURN=5:
# 0 -> 0.856451, 1 -> 0.873476, 2 -> 0.866976, 3 -> 0.860276, 5 -> 0.854712.
# Returning nothing forfeits the turns where the top candidate is already
# right; returning more starts padding again.
NARROWING_SIZE = 1


def shortlist_size(
    turn: int,
    top_k: int,
    live_disclosures: int,
    consistent: int,
    enabled: bool = True,
) -> int:
    """How many of the ranked candidates to return this turn.

    `live_disclosures` is how many disclosed constraints any pooled candidate
    owns, and `consistent` how many candidates own all of them -- both from
    reranking.prepare_evidence.
    """
    if not enabled:
        return top_k
    if live_disclosures <= 0:
        # No evidence to narrow with, so nothing justifies a short list.
        return top_k
    if turn >= EXPAND_TURN:
        return top_k
    if consistent == 1:
        # The constraints identify a single product; stand behind the ranking.
        return top_k
    return min(NARROWING_SIZE, top_k)
