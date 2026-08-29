# E7 - phrase containment, and the pool depth it keeps re-pricing

Decision record for the last stretch of P5. Two adopted changes that feed
each other, plus the measurements that declined two findings from the second
pre-merge review.

```yaml
experiment_id: "E7"
phase: "P5"
hypothesis: "After E6, the misses are no longer explained by disclosure quality (target coverage 0.97 in missed sessions): 16 of 26 targets never enter the 50-candidate pool, and the 10 that do are buried under candidates whose token coverage ties at ~1.0. Depth fixes the first only if a rescued deep candidate can be told apart; contiguous phrase containment is what tells it apart."
base_commit: "f771b93"
dataset: "full public set (200 labeled sessions)"
overall_metrics:
  e6_shipped:      {hit_rate_at_10: 0.870, mrr: 0.605760, mttc: 4.170, technical_score: 0.753328}
  plus_phrase_100: {hit_rate_at_10: 0.915, mrr: 0.655387, mttc: 3.760, technical_score: 0.798916}
  plus_depth_250:  {hit_rate_at_10: 0.940, mrr: 0.645534, mttc: 3.650, technical_score: 0.810660}
scenario_metrics:
  e6_shipped:     {buying: 0.8750, browsing: 0.8625, intent_override: 0.833, boundary: 1.000}
  plus_depth_250: {buying: 0.9375, browsing: 0.9250, intent_override: 0.967, boundary: 1.000}
model_api: {model: "none", network_required: false, prompt_tokens: 0, completion_tokens: 0}
known_regressions: []
decision: "ship phrase_evidence at weight 6.0 + RERANK_POOL 250 (+0.057332 over E6, +0.104176 over P5 as handed)"
```

## Where E6 left the misses

Replaying the evaluator loop with a recorder around `rerank()`, at the E6
configuration: 26 misses, of which **16 never entered the 50-candidate pool**
and 10 were pooled but buried. And the E5 discriminator is gone -- mean
evidence coverage of the true target was 0.9853 in hits and **0.9669 in
misses**. Disclosure quality stopped being the constraint the moment short
labels counted; what remained was retrieval depth on one side and coverage
saturation on the other: any near-duplicate of the target ties at ~1.0
token coverage.

## E7-T1: the depth rejection expired a second time

E5 measured pool depth 100-800 flat and rejected it -- correctly, then: at
`MIN_EVIDENCE_TOKENS=3` a rescued deep candidate could not be told apart from
the pool around it. Re-swept at the E6 configuration:

| depth | 50 | 75 | 100 | 150 | 200 | 400 | 800 |
|---|---|---|---|---|---|---|---|
| score | 0.753328 | 0.760814 | **0.766930** | 0.766610 | 0.763124 | 0.764761 | 0.764661 |

A smooth rise to a peak at 100 (+0.0136), then a mild decline as deeper
pools admit more high-coverage impostors. The lesson generalizes past this
repo: **a depth ablation is only as durable as the ranking features it was
measured under.**

## E7-T2: contiguity is what token coverage cannot say

The card quotes each constraint verbatim, so the true target contains every
disclosure as a *contiguous* token stream; a near-duplicate that merely
shares the vocabulary usually does not. `phrase_evidence` scores the
length-weighted share of disclosures contained contiguously, with both sides
normalized to bare tokens (so a details disclosure "Material: 100% Cotton"
matches the colon-free `searchable_text` flattening) and one retry per phrase
without its final token (the card truncates at 180 chars and can clip a
word). Sweeping the weight at depth 100:

| weight | 0.0 | 2.0 | 6.0 | 12.0 |
|---|---|---|---|---|
| score | 0.766930 | 0.798306 | **0.798916** | 0.798291 |

A plateau spanning 2-12 with a 0.0006 spread; 6.0 is its centre and best
point. At the old depth 50 the feature is worth +0.0335 on its own
(0.786811), so the two changes are independently real, not one effect
counted twice. Like token coverage, it fails quiet: no contained phrase
scores 0.0 for every candidate and the ordering falls back to the features
beneath.

## E7-T3: the depth optimum moved again, to 250

With phrase containment discriminating in deep pools, depth kept paying past
100. Swept 100-800 at phrase weight 6:

| depth | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 600 | 800 |
|---|---|---|---|---|---|---|---|---|---|
| score | 0.798916 | 0.807179 | 0.807791 | **0.810660** | 0.810445 | 0.807321 | 0.804144 | 0.800169 | 0.799931 |

250 and 300 tie at the HitRate top (0.940); adopted 250, the earlier and
cheaper point. The tie-breaking weights were re-checked at the new
configuration -- evidence 6/12/24, lexical_rank 0.5/1/2 -- and the incumbents
sit on their plateaus (lexical_rank 0.5 is identical to 1.0 to six decimals;
the ratio saturated again).

## Measured and declined (second pre-merge review)

**Pruning `disclosed_text` on an intent override.** The review argued the
evidence feature keeps ranking by the abandoned intent's quoted sentences.
Measured, clearing disclosures whenever the customer leads with a reversal:
0.810660 falls to **0.754924 (-0.044, HitRate -5 points)**. The finding's
scenario assumes the override abandons target A for target B; this
generator's override swaps between constraints *of the same target's card*
("ignore my earlier preference. What I need is: {hard[0]}"), so the
pre-override disclosures still quote the true target and pruning deletes
real evidence. Declined on this generator; a private set with true
target-switching overrides would reopen it.

**Filler stored as evidence via colon-free replies.** Real in general -- a
free-form "Actually, I need something else" with a question pending would be
stored whole and its filler tokens scored. Unreachable here: every colon-free
reply the simulator can produce ("not quite right yet", "don't have a
preference", the boundary decline) matches `_EMPTY_REPLY_RE`, and override
messages always carry a colon lead-in. Declined as unmeasurable on this
generator; noted for anyone pointing the agent at free-form customers.

## Where the remaining headroom is (and is not)

At the adopted configuration, 12 sessions miss: **1** target never enters the
250-candidate pool, **11** are pooled but buried, and the buried targets'
mean evidence coverage is 0.9541 against 0.9848 for hits. Retrieval is
effectively solved; the residual is targets tied at near-perfect coverage
with catalogue near-duplicates -- listings whose text contains the same
quoted constraints, sometimes verbatim. Text similarity has no more signal
to give there. Anyone attacking the last twelve should look at non-text
discriminators (price, store, category leaf) or at asking questions chosen
to split the surviving candidate set, not at retrieval or the evidence
features, which are measured to be at their ceiling.

## Hardening, re-measured at the adopted configuration

501 replayed turns, no other load: startup 1.34 s, per-turn latency
**37.9 ms median / 72.2 ms p95 / 126.8 ms max**, peak RSS 0.75 GB. The
deeper pool costs ~16 ms of median latency and 0.2 GB against the depth-50
figures E5 reported; still stdlib-only, no network, zero tokens.

## Confirmed end to end

`python3 -m evaluator.local_evaluator`, no environment variables:

```
hit_rate_at_10 0.940   mrr 0.645534   mttc 3.650   technical_score 0.810660
boundary 1.0000 / 4.400    browsing 0.9250 / 3.663
buying   0.9375 / 3.100    intent_override 0.9667 / 4.833
```

118 tests pass.

| milestone | TechnicalScore |
|---|---|
| P4 clarification + reranker | 0.636663 |
| P5 dense retired + evidence | 0.706484 |
| P5 short-label evidence + retune (E6) | 0.753328 |
| **P5 phrase containment + depth 250 (E7)** | **0.810660** |
