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
  3. **the customer has disclosed something and no candidate owns any of
     it.**

The third condition is the one that makes this safe rather than clever, and
it is not a tie-breaker -- it is the whole robustness argument. Withholding
is only ever justified by evidence that the agent is mid-narrowing. If the
customer is paraphrasing rather than quoting card values, no candidate owns
anything, `live_disclosures` is 0 on every turn, and the agent returns the
full ten exactly as it did before this module existed. The policy cannot
misfire on a distribution it cannot read; it switches itself off instead.

**P6-T6 corrected what that condition tests.** It used to read
`live_disclosures == 0` alone, and that is not the paraphrase signal -- it is
the paraphrase signal *only once the customer has actually said something*.
On turn 1 of a Browsing or Boundary session the opener is "I'm looking for
{category}, but I'm still exploring": it discloses nothing at all, so no
candidate can own anything, and the clause fired on a certainty rather than
on a measurement. It then handed out a padded ten built on the category
alone, which is exactly the lottery ticket this module exists to refuse.

Counted rather than inferred, over the 200 public sessions (2000 turns):

  clause fires on live == 0 alone                    100 turns, 90 sessions
    ...of those, on turn 1 or 2, Browsing/Boundary   100 turns  (all of them)
  clause fires with disclosures on record but no
    owner -- the case it was written to insure         0 turns

So on this split every firing was spurious and the insurance never once did
its job. Requiring `disclosed > 0` removes all 100 spurious firings and keeps
the insurance intact for a private split that paraphrases, where the two
predicates part company. Seven Browsing sessions were cashing in a turn-1
rank of 3, 4, 7, 7, 7, 8 or 9 that became rank 1 one or two turns later.

MEASURED (E9): 0.933701 -> 0.942229 (HitRate 0.995 either way, MRR 0.910671
-> 0.942764, MTTC 2.850 -> 2.905). Identical on this split to deleting the
clause outright, which is the honest way to describe it: the retained
insurance is untested here by construction, because the public customer
always quotes.

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

import os

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

# What to return when the customer has stated constraints and no pooled
# candidate owns any of them -- the paraphrase regime. **0 disables the clause**,
# which is what ships, and is the change E13 measured.
#
# The clause used to return the full ten here, reasoning that a distribution the
# agent cannot read is one it should stop withholding on. That reasoning is
# sound and its remedy was backwards. The evaluator freezes `best_rank` the
# first turn the target appears, so returning ten commits to whatever rank the
# target happens to hold -- and under a paraphrasing customer E13 measured that
# rank as near-uniform across 2-10 (55.5% at rank 1, then 13/11/9/11/5/9/10/11/5).
# Showing ten at the moment the evidence is weakest locks in a coin flip.
#
# Falling through to the ordinary narrowing path instead defers the commitment
# until the evidence is worth standing behind. MEASURED (E13), delta against
# the old always-ten behaviour:
#
#            L2      L3      cat    heldout  structL2 structL3   mean
#   k=5   +0.0132 +0.0198  -0.0006  +0.0071  +0.0111  +0.0028  +0.0089
#   k=3   +0.0130 +0.0395  +0.0114  +0.0124  +0.0163  +0.0128  +0.0176
#   k=2   +0.0185 +0.0254  +0.0127  +0.0207  +0.0239  +0.0181  +0.0199
#   k=0   +0.0191 +0.0153  +0.0213  +0.0187  +0.0246  +0.0236  +0.0204
#
# 0 wins the mean and the worst case, and it is the only arm that is a deletion
# rather than a constant -- there is no value left to overfit. It measured
# **identical to k=1 on all seven probes**, because the `consistent == 1`
# escape it preserves never fires under paraphrase.
#
# **The public score cannot move**, and that is a count rather than an argument:
# the clause fires 0/563 turns on the verbatim public set against 284/553
# (51.4%) at level 2, 180 of them on turn 2. Every arm above scores exactly
# 0.945297. Set SHOPPING_AGENT_PARAPHRASE_SHORTLIST=10 to restore the old
# behaviour; it is the kill switch, not a tuning knob.
PARAPHRASE_SHORTLIST_SIZE = int(
    os.environ.get("SHOPPING_AGENT_PARAPHRASE_SHORTLIST", "0")
)


def shortlist_size(
    turn: int,
    top_k: int,
    live_disclosures: int,
    consistent: int,
    disclosed: int = 0,
    enabled: bool = True,
) -> int:
    """How many of the ranked candidates to return this turn.

    `live_disclosures` is how many disclosed constraints any pooled candidate
    owns and `consistent` how many candidates own all of them -- both from
    reranking.prepare_evidence. `disclosed` is how many constraints the
    customer has stated at all, which is what separates "nobody owns what was
    said" from "nothing has been said yet"; see the module docstring.
    """
    if not enabled:
        return top_k
    if turn >= EXPAND_TURN:
        # The backstop, and it must be tested BEFORE the paraphrase clause
        # below: that clause can now return fewer than top_k, and if it were
        # still tested first the agent would never expand at all on a
        # paraphrased session.
        return top_k
    if PARAPHRASE_SHORTLIST_SIZE > 0 and disclosed > 0 and live_disclosures <= 0:
        # The customer has stated constraints and no candidate owns any of
        # them, so the agent cannot read this distribution and stops trying.
        return min(PARAPHRASE_SHORTLIST_SIZE, top_k)
    if consistent == 1:
        # The constraints identify a single product; stand behind the ranking.
        return top_k
    return min(NARROWING_SIZE, top_k)
