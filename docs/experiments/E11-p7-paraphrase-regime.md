# E11 (P7): switching the ranking when the customer stops quoting

**Branch**: `phase/7-paraphrase-robustness`
**Base**: `bc6b373` (E10)
**Question**: E10 left the whole remaining paraphrase gap in ranking rather
than retrieval. Can that gap be closed without spending any of the public
score?

---

## What E10 handed over

E10 established, by instrumenting the replay rather than arguing about it,
that under paraphrase the target reaches the rerank pool in **200 of 200
sessions** and is then ranked at a median position of **28** rather than 2.
Pool recall is 1.000. Nothing is missing from the room; the ordering is wrong.

It also priced a perfect reranker against today's retrieval at **0.990300**,
which is the ceiling this experiment is working towards, and left the gap at
roughly 0.294.

## The diagnosis

Reading the weight table against that fact says why the ordering is wrong.
Four of the five heaviest features ask an exact-match question, and the
public simulator answers yes because it builds every customer utterance out
of the target's own field values. Reword the same utterances and the four do
not fail the same way:

| feature | weight | under paraphrase |
|---|---|---|
| `slot_evidence` | 16.0 | near-silent; whole-value identity |
| `constraint_evidence` | 12.0 | scores partial token overlap, so rewording dilutes it (E10 called this noise; see the correction below) |
| `category_exact` | 8.0 | inert once the E9 category filter engages |
| `phrase_evidence` | 6.0 | silent to six decimals (E10) |
| `lexical_rank` | 1.0 | still carries real information |

So the paraphrased ranking is decided by the feature holding the *smallest*
weight in the table, while 34 points of weight are split between saying
nothing and saying something actively wrong. That is the mechanism behind
rank 28, and it is not a tuning error — the table is correctly tuned for a
customer who quotes.

The reranker's real defect is that it has no idea which kind of customer it
is talking to.

## The second failure, which is worse than silence

`slot_evidence` does not merely blunt under paraphrase. It switches off.

`slots.ownership_weights` assigns weight **0.0** to any disclosure that no
pooled candidate owns, on the sound reasoning that such a disclosure is
silence rather than evidence. Under a rewording that describes *every*
disclosure, so every weight is zero, the denominator in `_slot_value` is
zero, and the function returns 0.0 for the entire pool.

This matters for the fix: relaxing the *match* alone would have achieved
nothing, because there would still have been no weight to score it with.
Both had to move together.

## Adopted 1: a conditional weight profile

The regime predicate is the one `shortlist.py` already computes and E9
already validated: the customer has stated constraints and not one pooled
candidate owns any of them. It is deliberately the *same* predicate, so the
shortlist policy and the reranker cannot disagree about which regime a turn
is in.

The cost argument is not a measurement, it is a construction. E9 counted this
predicate over the whole public set and it fired on **0 of 2000 turns**. A
weight profile applied only behind that gate cannot move a score it is never
applied to. E10's equivalent unconditional change cost 0.00075; this costs
nothing, and "nothing" here means identical to six decimals.

## Adopted 2: graded ownership, once the regime has fired

Exact ownership is all-or-nothing: change one word of a nine-word value and
identity scores it the same as a wholly unrelated product. Under the regime
only, the match falls back to the best Jaccard similarity between the
disclosed value's tokens and any single owned value's tokens, and the
selectivity weights are recomputed against that graded notion so the feature
has something to score with.

**This shipped as Jaccard first, and that was wrong.** The reasoning was that
containment — the share of the disclosure's own tokens some owned value
carries — scores a short disclosure buried in a long value at 1.0, so
"cotton" would perfectly own a candidate reading "100% cotton blend twill
lining", reintroducing the E7/E8 impostor at the weight of the table's
dominant feature. Jaccard divides by the union instead, charging for the
extra tokens as well as the missing ones.

That is even-handed in form and not in effect. Dividing by the union charges
the **true owner** for the length of its own card value, so under a
*compressive* paraphrase — a customer saying less than the card holds — it
rewards whichever product owns the shortest phrase built from those words:

    customer says   "wash cold"
    true owner owns "machine wash cold with like colors"   Jaccard 0.333
    impostor   owns "hand wash cold"                       Jaccard 0.667

At a threshold of 0.5 that scored the true owner **0.0** and the impostor
**0.667**, at weight 16.0. On that turn class the feature was not merely
unhelpful, it was **strictly worse than the silence it replaced** — and since
the regime firing also drives `shortlist.py` to return all ten, a misfire can
push the target out of the ten and lose the hit outright rather than only the
rank.

Shipped instead is **containment with two guards**: at least two shared
tokens, and a cap on how much longer the owned value may be than the
disclosure. The true owner now scores 1.0, the impostor also scores 1.0, and
the "cotton" case is refused outright at 0.0 rather than the 0.2 Jaccard gave
it. A tie is the honest outcome — both products genuinely account for what
was said, so the order should fall to other features. Failing to discriminate
is a far cheaper error than discriminating backwards. Selectivity does the
remaining work: `ownership_weights` re-prices down any value much of the pool
accounts for.

**Why the first round of measurement did not catch it, which is the more
useful lesson.** `paraphrase_eval` level 2 is *expansive* — "cotton" becomes
"natural plant fibre" — so a disclosure is never shorter than the value it
came from and the length asymmetry cannot bite. Level 3 is the compressive
one. The threshold had been swept **only at level 2**, which is precisely the
regime in which this defect is invisible. A sweep is only as honest as the
worst case in the fixture it is swept over.

`live_disclosures` and `consistent` are kept **exact** rather than recomputed
from the relaxed pass. Deriving them from the relaxed match would let a
successful relaxation report that the customer is quoting after all, which
would switch off the weight profile that enabled the relaxation one turn
after it started working.

## Results

All numbers below were run. The three guardrail commands were executed on
every configuration reported as adopted.

| configuration | public (submitted) | L0 control | L2 paraphrased | L2 HitRate |
|---|---|---|---|---|
| E10 baseline `bc6b373` | 0.945497 | 0.945497 | 0.696015 | 0.805 |
| + conditional weight profile | **0.945497** | 0.945497 | 0.723034 | 0.830 |
| + graded ownership | **0.945497** | 0.945497 | **0.733138** | **0.845** |

Public set, both changes in: HitRate 1.000, MRR 0.938657, MTTC 2.805 —
identical to E10 in all four figures. 196 tests pass (178 inherited, 18 new).

Runtime is the independent check on the "never fires publicly" claim: the L0
run takes 26.45 s against the baseline's 26.49 s, while L2 rises to 33.50 s.
The relaxed pass is expensive and it is not running on the public path.

## The firing rate, counted rather than inferred

An unchanged public score is consistent with the profile never firing *and*
with it firing and happening to cancel out, which are very different things to
carry to an unseen split. `scripts/paraphrase_regime_audit.py` wraps
`prepare_evidence` and counts, leaving the agent byte-for-byte the submitted
one.

| | rerank calls | with disclosures | regime fired | graded match engaged |
|---|---|---|---|---|
| L0 (public, verbatim) | 561 | 461 | **0** | **0** |
| L2 (paraphrased) | 796 | 586 | 488 (83.3%) | 54 |

The L0 row is the cost argument stated directly. E9 independently reported 0
firings for the same predicate over the public set, so this is corroboration
rather than a fresh claim. (The call count is lower than the turn budget
because a session stops once the target is returned.)

The L2 row carries a caution that the headline number does not. The regime
fires on 83% of disclosing turns, but the *graded ownership* clears its
threshold on only 54 of those 488 — about one turn in nine. Nearly all of the
+0.0371 is therefore the weight profile; graded ownership is a narrow,
targeted fix worth +0.0101 on a thin slice of turns rather than a broad
repair. That thinness is also why its threshold curve below is bumpy: a
handful of turns changing hands moves the composite.

## Sweeps

Both swept with `scripts/paraphrase_profile_sweep.py`, which pays the
catalogue parse once. The instrument was validated three times over against
independently-run points: `lexical_rank` 1.0 reproduces the E10 baseline
0.696015 exactly (the profile is then a no-op), 4.0 reproduces 0.723034, and
threshold 0.7 degenerates to 0.723034 because almost nothing clears the bar.

### Graded-ownership threshold, at L2

| threshold | 0.3 | 0.4 | **0.5** | 0.6 | 0.7 |
|---|---|---|---|---|---|
| score | 0.723355 | 0.714379 | **0.733138** | 0.718234 | 0.723034 |
| HitRate | 0.830 | 0.815 | **0.845** | 0.825 | 0.830 |

0.5 is a genuine local maximum with both neighbours below it, and the high end
degenerates correctly to the no-relaxation score. But the surface is bumpy
rather than a plateau — 0.3 beats 0.4 — and on 200 sessions these gaps are
four to six sessions each. **0.5 is the best measured point, not a
well-resolved optimum.** A permissive threshold engages more turns and scores
*worse*, which is the same lesson `constraint_evidence` teaches: under
paraphrase, admitting noise costs more than staying silent.

## Known limitation: the predicate is session-cumulative and sticky

`state.disclosed_text` accumulates across the session (`orchestrator.py`
extends it each turn), so `prepare_evidence` is always asked about *every*
constraint disclosed so far, not this turn's. `live_disclosures` therefore
counts owners across the whole accumulated set, and the regime requires that
count to be exactly zero.

The consequence is asymmetric and worth stating plainly. One exactly-owned
disclosure anywhere in a session suppresses the regime for every later turn of
that session, however thoroughly the customer rewords afterwards. The
predicate is biased towards concluding "quoting".

On this harness that bias costs little, because L2 rewords everything and the
regime still fires on 83% of disclosing turns. On a *partially* paraphrasing
split — which is the realistic case, and the one the organiser would actually
produce — it would fire far less often, and the profile would stand down
exactly where it is half-needed.

The obvious repair is to make the predicate fractional rather than absolute:
fire when the *share* of disclosures any candidate owns falls below some
level, instead of requiring it to be zero. That is a change to the same
predicate `shortlist.py` reads, so it cannot be made in isolation, and it
needs its own sweep and its own review. It is left for the next phase rather
than folded in here, and it is the highest-value lever this experiment
identifies.

### `lexical_rank` inside the profile, at L2

Swept twice: once with graded ownership off (isolating the weight profile) and
once with it on (the shipped configuration).

| weight | 1.0 | 2.0 | 4.0 | 6.0 | 8.0 | 12.0 |
|---|---|---|---|---|---|---|
| graded off | 0.696015 | 0.711371 | 0.723034 | 0.719192 | 0.726015 | — |
| graded on | — | 0.709058 | **0.733138** | 0.729019 | 0.735979 | 0.735259 |
| HitRate (on) | — | 0.815 | 0.845 | 0.840 | 0.850 | 0.850 |

Both sweeps show the same odd shape: a rise to 4.0, a dip at 6.0, then a
plateau at 8-12 sitting about 0.003 above 4.0.

**That agreement is not the corroboration it looks like.** Both sweeps run the
same deterministic customer over the same 200 sessions with the same seed, so
the second sweep re-asks the identical sessions and necessarily reproduces the
identical bumps. Two runs of one fixture are one measurement, not two, and a
0.003 gap is about one session of HitRate.

**Kept at 4.0 rather than moved to the measured plateau at 8.0.** The
justification is deliberately not "4.0 scored best here", because it did not.
It is that 4.0 is the value E10 arrived at independently, by an unconditional
sweep at a different configuration, and that this fixture cannot resolve
0.003. Moving to 8.0 on this evidence would be tuning to a hand-built synonym
lexicon. The question is left to the held-out paraphrasers, which are the only
instrument here that can answer it.

## Held-out check

`--paraphrase-category` substitutes the category's *words*, which defeats
E10's canonical-form category fix by construction. It is the adversarial probe
E10 established, and nothing in this experiment was tuned against it.

| | E10 | E11 | change |
|---|---|---|---|
| L2 `--paraphrase-category` | 0.627500 | **0.665945** | **+0.038445** |
| HitRate | 0.715 | 0.755 | +0.040 |

The gain on the held-out probe (+0.0384) is slightly *larger* than on the
fixture the constants were chosen against (+0.0371). An overfitted change
shows the opposite pattern, so this is evidence the mechanism is real rather
than lexicon-shaped — which stands to reason, since neither the regime
predicate nor Jaccard overlap knows anything about which words were swapped.

## Every level, before and after

| condition | E10 | E11 | change |
|---|---|---|---|
| `local_evaluator` (the submission) | 0.945497 | **0.945497** | **0** |
| L0 verbatim control | 0.945497 | **0.945497** | **0** |
| L1 synonyms | 0.694494 | 0.733117 | +0.038623 |
| L2 synonyms + destructured | 0.696015 | 0.733138 | +0.037123 |
| L3 + 40% of words dropped | 0.673673 | 0.686577 | +0.012904 |
| L2 `--paraphrase-category` (held out) | 0.627500 | 0.665945 | +0.038445 |

The two rows that matter most are again the ones that do not move.

L3's much smaller gain is the expected shape rather than a disappointment.
E10 records that L3 conflates paraphrasing with information loss and is a
floor rather than a clean measurement: it deletes 40% of the words outright.
A ranking repair can recover a constraint that was reworded; it cannot
recover one that was never said. That the gain shrinks by roughly two thirds
exactly where the harness stops being a paraphrase probe is a small piece of
evidence that the mechanism does what it claims and not something else.

## Correction to E10: `constraint_evidence` is not noise here

E10 measured that zeroing `constraint_evidence` **improved** the paraphrased
score by +0.0157, and concluded that under rewording the feature fills with
noise rather than falling silent. That conclusion was correct at E10's
configuration. It is wrong at this one, and by a wide margin.

Swept inside the profile, at L2:

| weight | 0.0 | 3.0 | 6.0 | 12.0 (shipped) |
|---|---|---|---|---|
| score | 0.722575 | 0.726306 | 0.727946 | **0.733138** |
| HitRate | 0.830 | 0.840 | 0.840 | **0.845** |

Monotone increasing. Deleting the feature now **costs 0.010563** where E10
measured it as gaining 0.0157 — the sign has flipped.

The mechanism is the same one that made E10's reading correct. Partial token
overlap is weak, dilute evidence. When it was the only feature still saying
anything, its weight of 12.0 let that dilute signal dominate the order, and
the ordering it produced was worse than the retrieval ordering underneath it.
Now `lexical_rank` at 4.0 and a working `slot_evidence` carry the order, and
the same dilute signal is demoted to what it should always have been: a
tie-breaker that is weakly right rather than a decider that is strongly
wrong.

This is the project's standing rule — *a rejected idea is only rejected at
the configuration you tested it on* — firing in the unusual direction. Here
an idea that was **accepted** and priced at +0.0157 turns out to be a loss of
0.0106 two changes later. The rule needs stating symmetrically: an accepted
measurement expires exactly as fast as a rejected one, and the levers a
previous phase leaves "priced and ready to ship" must be re-priced before
they are shipped, not treated as banked.

Both of E10's two priced-but-unshipped levers were re-examined here. One
(`lexical_rank`) was adopted, conditionally, and is worth +0.027. The other
(`constraint_evidence` -> 0) was rejected, having reversed sign.
