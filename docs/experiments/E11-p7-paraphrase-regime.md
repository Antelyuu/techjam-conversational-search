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

Shipped instead is **containment with an absolute floor of two shared
tokens**. The true owner now scores 1.0, the impostor also scores 1.0, and
the "cotton" case is refused outright at 0.0 rather than the 0.2 Jaccard gave
it. A tie is the honest outcome — both products genuinely account for what
was said, so the order should fall to other features. Failing to discriminate
is a far cheaper error than discriminating backwards. Selectivity does the
remaining work: `ownership_weights` re-prices down any value much of the pool
accounts for.

**A length cap was tried as that guard first, and was itself a defect.**
Refusing any owned value more than 3x the disclosure's length looks like the
same protection and is not — it refuses honest fragments of a long value.
Against "Machine wash cold with like colors and tumble dry low":

| disclosure | shared / wanted | ratio | under the cap |
|---|---|---|---|
| `machine wash cold` | 3 / 3 | 3.3 | **refused** |
| `tumble dry low` | 3 / 3 | 3.3 | **refused** |
| `wash cold with like colors` | 5 / 5 | 2.0 | 1.0 |

The first two are the true owner's own words, scored 0.0 for quoting a long
value briefly — which is precisely what a summarising customer does, and so
the same compressive case Jaccard had already got wrong. The token floor
refuses the impostor without refusing them, and needs one constant instead of
two.

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
targeted fix worth +0.010104 on a thin slice of turns rather than a broad
repair. (Both figures are Jaccard-era: the split between the two halves was
measured before the review replaced the comparison function, and was not
re-measured after. The end-to-end totals in Results were.) That thinness is also why its threshold curve below is bumpy: a
handful of turns changing hands moves the composite.

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


## Results

All numbers were run. Where a figure was measured under an implementation
that did not ship, it is labelled as such rather than quietly dropped — the
review round below changed the comparison function, which voided a set of
sweeps, and pretending otherwise would leave constants justified by numbers
the shipped code cannot reproduce.

| condition | E10 | E11 | change |
|---|---|---|---|
| **`local_evaluator` (the submission)** | 0.945497 | **0.945497** | **0** |
| L0 verbatim control | 0.945497 | **0.945497** | **0** |
| L1 synonyms | 0.694494 | 0.733847 | +0.039353 |
| L2 synonyms + destructured | 0.696015 | 0.733097 | +0.037082 |
| L3 + 40% of words dropped | 0.673673 | 0.703036 | +0.029363 |
| L2 `--paraphrase-category` (held out) | 0.627500 | 0.652787 | +0.025287 |
| structural L2 (held out) | 0.811102 | 0.819191 | +0.008089 |
| structural L3 (held out) | 0.817477 | **0.806953** | **−0.010524** |

Public HitRate/MRR/MTTC are 1.000 / 0.938657 / 2.805 before and after,
identical in all four figures. 223 tests pass.

**The last row is a regression and is not buried.** Structural L3 is the one
probe on which this phase makes the agent worse. It is also the probe
furthest from what was tuned: a second paraphraser, at its most aggressive
level, whose transformations preserve 99.7% of content tokens. Reported
because a held-out set that only gets consulted when it agrees is not a
held-out set.

## The two held-out paraphrasers, and why the gap between them matters

The synonym paraphraser substitutes vocabulary from a hand-built ~130-entry
lexicon. A second, structurally independent one was built before any tuning
began, precisely so no constant here could be chosen against a single
fixture. It inverts the mechanism: it reframes and reorders while preserving
vocabulary, measured at 99.7% content-token retention against the synonym
mode's 35.8%, and 0% whole-value survival against 8.9%.

Both improve, which is the bar. But the sizes are very different — +0.0371 on
the synonym fixture against +0.0081 on the structural one — and that gap is
the honest calibration:

* The synonym mode destroys tokens, so it damages retrieval *and* ranking
  (0.249 lost). Restoring `lexical_rank` recovers a lot.
* The structural mode leaves the token bag intact, so BM25 still retrieves
  well and only ranking degrades (0.134 lost). There is simply less for this
  change to recover.

Read the synonym number alone and the phase looks worth +0.037. Read both and
the fair statement is that **most of the measured gain reflects how harshly
the synonym fixture damages retrieval**, and the portion that survives a
vocabulary-preserving paraphrase is real but roughly a fifth the size.

One caveat carried from the paraphraser's own report: the two are *not*
independent on the detector question. The regime fires on 0.846 of disclosing
turns under synonym and 0.860 under structural, because both destroy
exactness. The second fixture gives an independent read on the **score**, not
on the **detector**.

## Sweeps, and which of them the review voided

Swept with `scripts/paraphrase_profile_sweep.py`, which pays the catalogue
index once. The instrument validated against independently-run points five
times: `lexical_rank` 1.0 reproduces 0.696015, 4.0 reproduces 0.723034,
threshold 0.7 degenerates to 0.723034 because almost nothing clears the bar,
`constraint_evidence` 12.0 reproduces 0.733138, and threshold 0.5 reproduces
it too.

**Every sweep below was run against the Jaccard implementation of graded
ownership, which the review replaced.** They are kept because they still
answer the question they were asked — `lexical_rank`'s curve is dominated by
the weight profile, not by the ownership function — but the graded-ownership
*threshold* is now a constant inherited from a comparison function that no
longer exists, and it has not been re-swept. That is the largest piece of
unfinished work this experiment leaves, and it is stated here rather than
left for someone to discover.

### `lexical_rank` inside the profile, at L2 (Jaccard-era)

| weight | 1.0 | 2.0 | 4.0 | 6.0 | 8.0 | 12.0 |
|---|---|---|---|---|---|---|
| graded off | 0.696015 | 0.711371 | 0.723034 | 0.719192 | 0.726015 | — |
| graded on | — | 0.709058 | **0.733138** | 0.729019 | 0.735979 | 0.735259 |

Both curves show the same shape: a rise to 4.0, a dip at 6.0, a plateau at
8-12 about 0.003 above 4.0. That agreement is **not** corroboration — both
sweeps run the same deterministic customer over the same 200 sessions with
the same seed, so the second re-asks identical sessions and necessarily
reproduces identical bumps. Two runs of one fixture are one measurement.

**Kept at 4.0 rather than moved to the measured plateau.** Not because 4.0
scored best — it did not — but because 0.003 is about one session of HitRate
here, and 4.0 is the value E10 reached independently by an unconditional
sweep at a different configuration. Moving to 8.0 on this evidence would be
tuning to a hand-built lexicon.

### Graded-ownership threshold, at L2 (Jaccard-era, **stale**)

| threshold | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|
| score | 0.723355 | 0.714379 | **0.733138** | 0.718234 | 0.723034 |

0.5 was a local maximum with both neighbours below it. It is retained under
the containment function on the strength of the end-to-end numbers in the
Results table, not on this curve, which no longer describes the code.

## Correction to E10: `constraint_evidence` is not noise here

E10 measured that zeroing `constraint_evidence` **improved** the paraphrased
score by +0.0157 and concluded that under rewording the feature fills with
noise. That was correct at E10's configuration and is wrong at this one.
Swept inside the profile at L2 (Jaccard-era, but the effect is far larger
than the implementation change):

| weight | 0.0 | 3.0 | 6.0 | 12.0 (shipped) |
|---|---|---|---|---|
| score | 0.722575 | 0.726306 | 0.727946 | **0.733138** |

Monotone increasing: deleting the feature now **costs 0.010563** where E10
measured it as gaining 0.0157. The sign has flipped.

The mechanism is the one that made E10 right at the time. Partial token
overlap is weak, dilute evidence. When it was the only feature still saying
anything, weight 12.0 let that dilute signal decide the order, and the
ordering it produced was worse than the retrieval ordering underneath.
Now `lexical_rank` at 4.0 and a working `slot_evidence` carry the order, and
the same signal is demoted to what it should always have been — a tie-breaker
that is weakly right rather than a decider that is strongly wrong.

This is the standing rule *a rejected idea is only rejected at the
configuration you tested it on* firing in the unusual direction: an
**accepted** measurement, priced and banked as ready to ship, turned into a
loss two changes later. State the rule symmetrically. An accepted measurement
expires exactly as fast as a rejected one, and a lever a previous phase hands
over "priced" is a hypothesis, not a balance.

Both of E10's banked levers were re-priced here. `lexical_rank` was adopted,
conditionally, and is worth +0.027. `constraint_evidence` -> 0 was rejected.

## The review round: one defect shipped, one "fix" refuted

### The defect: Jaccard promoted impostors over the true owner

Recorded in full in the "Adopted 2" section above. In short: dividing by the
union charges the true owner for the length of its own card value, so under a
compressive paraphrase it preferred whichever product owned the shortest
phrase built from those words — the true owner scored 0.0 and a near-
duplicate 0.667, at weight 16.0.

**It hid because the fixture could not express the failure.** Level 2 is
*expansive*: "cotton" becomes "natural plant fibre", so a disclosure is never
shorter than its source and the length asymmetry cannot bite. Level 3 is the
compressive one, and the threshold had been swept only at level 2. The sweep
was honest, reproducible, and blind, because the worst case was not in the
fixture it swept over.

Generalise it: **before sweeping a parameter, ask which failure modes the
fixture is capable of producing.** A plateau found over a fixture that cannot
express a failure is not evidence of safety against it.

Fixing it moved exactly the levels the mechanism predicts, which is the check
that the diagnosis was right rather than merely plausible:

| | Jaccard | containment | change |
|---|---|---|---|
| L2 (expansive — defect invisible) | 0.733138 | 0.733097 | −0.000041 |
| L3 (compressive — defect lives here) | 0.686577 | 0.703036 | **+0.016459** |

### The refuted fix: removing the length cap

Containment needs a guard against a short disclosure being "contained" in a
long value, and the first shipped one was a cap on the owned value's length.
That cap visibly refuses honest fragments — against "Machine wash cold with
like colors and tumble dry low", both "machine wash cold" and "tumble dry
low" share every token they have and are refused at a ratio of 3.3. Replacing
it with an absolute two-shared-token floor fixes that with one constant
instead of two.

Measured, it was worse on four probes of five, and **below the pre-change
baseline on both held-out structural ones**:

| | syn L2 | syn L3 | syn L2+cat | struct L2 | struct L3 |
|---|---|---|---|---|---|
| baseline | 0.696015 | 0.673673 | 0.627500 | 0.811102 | 0.817477 |
| **cap (shipped)** | **0.733097** | **0.703036** | 0.652787 | **0.819191** | **0.806953** |
| floor, no cap | 0.720809 | 0.679837 | 0.655900* | 0.773188 | 0.797113 |

\* printed by the harness as `0.6559`; the only probe on which the reverted
variant beat the shipped one.

Reverted. The cap is not primarily an impostor guard — it is a noise filter,
and admitting short fragments admits several near-duplicates for every true
owner recovered, each credited at the table's dominant weight.

The two failures in this round are mirror images and both are worth carrying.
The defect hid because **the fixture could not express the failure**. The
refuted fix survived reasoning because **the example could not express the
distribution**. Neither was settled by argument; both were settled by running
it.
