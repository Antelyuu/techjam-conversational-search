# E5 - P5 retiring the dense route, and scoring disclosed evidence

Decision record for P5. Two changes are adopted and four ideas are measured
and rejected; the rejections are recorded at equal length because each is
something a later reader would otherwise try again.

```yaml
experiment_id: "E5"
phase: "P5"
hypothesis: "P4 changed what a query is, so P3's retrieval decision no longer holds; and a disclosure is stronger evidence than BM25 can express, because the evaluator quotes it out of the target product itself."
base_commit: "e12f69d"
dataset: "full public set (200 labeled sessions)"
overall_metrics:
  p4_merged:        {hit_rate_at_10: 0.755, mrr: 0.471877, mttc: 5.120, efficiency: 0.5880, technical_score: 0.636663}
  dense_off:        {hit_rate_at_10: 0.820, mrr: 0.512661, mttc: 4.810, efficiency: 0.6190, technical_score: 0.687598}
  plus_evidence:    {hit_rate_at_10: 0.830, mrr: 0.550613, mttc: 4.685, efficiency: 0.6315, technical_score: 0.706484}
scenario_metrics:
  p4_merged:     {buying: 0.7125, browsing: 0.7625, intent_override: 0.800, boundary: 0.900}
  dense_off:     {buying: 0.8250, browsing: 0.8125, intent_override: 0.800, boundary: 0.900}
  plus_evidence: {buying: 0.8375, browsing: 0.8250, intent_override: 0.800, boundary: 0.900}
model_api: {model: "none", network_required: false, prompt_tokens: 0, completion_tokens: 0}
known_regressions: []
decision: "ship dense_off + constraint_evidence at weight 12.0 (+0.069821 over P4)"
```

## P5-T1: the dense route is retired, reversing P3

The team reported that lexical-only search was beating the hybrid across the
board. It contradicts E1 and E2, so it was re-measured before being acted on.
It is correct, and by a wide margin.

`python3 -m scripts.fusion_ablation`:

| config | HitRate@10 | MRR | MTTC | TechnicalScore |
|---|---|---|---|---|
| **lexical_only** | **0.8200** | **0.512661** | **4.810** | **0.687598** |
| rrf | 0.7550 | 0.471562 | 5.115 | 0.636669 |
| weighted | 0.7550 | 0.471877 | 5.120 | 0.636663 |

Per scenario, lexical-only wins or ties everywhere -- Buying 0.8250 against
0.7125, Browsing 0.8125 against 0.7625, Boundary and Intent Override level.
**+0.050935** for deleting a route.

Cross-checked the way this project cross-checks everything since E1: the same
figure was reproduced through an entirely separate path,
`SHOPPING_AGENT_DENSE=0 python3 -m evaluator.local_evaluator`, to six decimal
places, and it differs from the dense rows, so both configurations genuinely
ran.

### Why, measured rather than guessed

Replaying the evaluator's own loop and asking each route separately whether it
held the ground-truth target in a 50-candidate pool:

| turn | median query words | lexical recall | dense recall |
|---|---|---|---|
| 1 | 12 | 0.3800 | 0.3200 |
| 2 | 15 | 0.4200 | 0.2200 |
| 3 | 24 | 0.7100 | 0.2550 |
| 5 | 29 | 0.7450 | 0.3000 |
| 10 | 36 | 0.7400 | 0.3600 |

At turn 1 the routes are comparable, which is the world E1 and E2 measured in
and their conclusion was right for it. From turn 2 the clarification policy
starts feeding the query constraint sentences quoted out of the target's own
`features` and `details`, and the two routes come apart: **lexical recall
nearly doubles while dense stays flat.** BM25 sharpens as terms accumulate;
one fixed-width sentence embedding averages a growing paragraph toward the
corpus mean. P3's decision was not wrong, it expired.

### Down-weighting does not work, and the reason matters

`python3 -m scripts.route_weight_sweep`:

| config | HitRate@10 | MRR | TechnicalScore |
|---|---|---|---|
| **dense_off** | **0.8200** | **0.512661** | **0.687598** |
| lex_0.95 | 0.7600 | 0.474173 | 0.640852 |
| lex_0.9 | 0.7600 | 0.474173 | 0.640852 |
| lex_0.8 | 0.7600 | 0.472506 | 0.640252 |
| lex_0.7 | 0.7600 | 0.472220 | 0.640166 |
| lex_0.6 | 0.7550 | 0.471383 | 0.637015 |
| even_0.5 | 0.7550 | 0.471877 | 0.636663 |

There is a **cliff between off and on, not a slope**: giving dense 5% of the
weight recovers 8% of the gap. That is because `retrieve()` takes the *union*
of both routes' candidates before weighting them, and min-max normalization
floors the worst lexical candidate at 0.0 -- so a dense-only candidate with
any positive weight outscores the tail of the lexical pool and displaces it.
The route was contaminating the pool, not merely voting badly in it.

The code, the flag and the embedding artifact all stay. The finding is about
*this* query distribution; a private set whose customers answer less
verbatim would move the balance back, and `SHOPPING_AGENT_DENSE=1` restores
the P4 behaviour exactly.

## P5-T3: scoring what the customer actually disclosed

The evaluator builds its hidden intent card from the target product's own
`features` and `details` (`local_evaluator.py:52`), whitespace-normalized and
truncated but otherwise **verbatim**. So an answer to a question is a sentence
quoted out of the one product we are looking for. P4 spent that as loose BM25
terms and nothing else.

BM25 cannot express coverage of a whole phrase: it scores a bag of terms, so a
candidate matching *all* of a long disclosure scores much like one matching
most of it. `shopping_agent/evidence.py` asks the question directly -- what
share of the customer's quoted constraints does this candidate's own text
account for, weighting each disclosure by its length so a specific sentence
outvotes a three-word one.

Sweeping the weight, dense off:

| weight | 0.0 | 2.0 | 4.0 | 8.0 | 12.0 | 16.0 | 24.0+ |
|---|---|---|---|---|---|---|---|
| HitRate | 0.8200 | 0.8250 | 0.8250 | 0.8300 | 0.8300 | 0.8300 | 0.8300 |
| MRR | 0.5127 | 0.5263 | 0.5438 | 0.5479 | 0.5506 | 0.5506 | 0.5506 |
| score | 0.6876 | 0.6960 | 0.7017 | 0.7057 | **0.7065** | 0.7065 | 0.7065 |

Identical to six decimals from 12 upward: by then evidence coverage decides
the order outright and the rest of the table only breaks its ties. **Adopted
12.0**, the first point of the plateau. Worth **+0.018886**.

A weight this dominant is safe rather than reckless, because the feature fails
quiet: if a disclosure matches nothing -- a different hidden evaluator, a less
verbatim customer -- coverage is ~0 for every candidate and the ordering falls
back to the features underneath it. The failure mode is silence, not noise.

## Why HitRate stops at 0.830

Of the 34 sessions that never hit:

| | count |
|---|---|
| target never entered the 50-candidate pool | 16 |
| target pooled but never shown (median best rank 23) | 18 |

and the discriminating variable is not retrieval depth but how well the
customer's disclosures describe the target at all:

| | mean evidence coverage of the true target |
|---|---|
| sessions that hit | **0.7324** |
| sessions that missed | 0.3235 |

The misses are sessions where the quoted constraints simply do not pick the
target out. That is a property of the sample, not a ranking defect, which is
why the four ideas below all failed.

## Measured and rejected

Recorded so none of these is tried again blind.

**Re-asking a productive attribute** -- E3 left this open. Each hidden card
holds 3-4 undisclosed constraints and a successful ask returns 2, so `feature`
(96% yield) is not drained by one question, while P4's no-repeat rule retires
it after one use and then spends turns on `size` (4.5%) and `use_case` (2.0%).
The stopping condition already exists and self-corrects, so a repeat costs at
most one wasted turn.

| | HitRate | MRR | score |
|---|---|---|---|
| no repeats | **0.8300** | 0.550613 | **0.706484** |
| repeats | 0.8250 | 0.540038 | 0.698711 |

**Rejected, -0.007773**, and all of it lands on Intent Override (0.800 ->
0.767). Hits cannot register there until the override fires, and a repeated
question crowds out the diversification that finds the new intent. Flag kept:
`SHOPPING_AGENT_REPEAT=1`.

**A second BM25 query over the disclosures alone**, unioned into the pool, to
rescue the 16 targets the cumulative query never retrieves. As first written
it carried the focused query's ranks across and cost -0.016730 (Buying 0.8375
-> 0.7500) because a focused rank 1 is not a main-pool rank 1. Clearing the
extras' ranks so they compete on evidence alone brought it to **0.706584
against 0.706484 -- +0.0001, one session's MTTC**. Rejected: not worth a
second query per turn. Code removed.

**Widening the reranked pool** from 50 to 100, 200, 400 and 800. HitRate is
**0.8300 at every depth**, and the composite moves by less than 0.0012 with no
trend. Retrieval depth is not the constraint; kept at 50, which is also the
cheapest.

**Raising the 40-term query cap.** 8.2% of turns do exceed it, dropping a
median of 11 terms and up to 42, and this looked like a third instance of the
silent-truncation defect P4 found twice. It is not: 60, 80, 120 and 400 all
reproduce 0.706484 exactly. `build_query_text()` ends with the latest raw
message, which for an answered question repeats the disclosure already carried
earlier in the query, so what a cap discards are duplicates. Kept at 40, now
with a measurement behind it instead of an assumption.

**Retuning BM25 field weights** toward `features`/`details`, where disclosures
come from:

| weights | HitRate | MRR | score |
|---|---|---|---|
| **base (0, 6, 4, 2.5, 2.5, 1.5, 1)** | **0.8300** | 0.550613 | **0.706484** |
| features/details up | 0.8150 | 0.561835 | 0.702151 |
| features/details leading | 0.8100 | 0.556347 | 0.697904 |
| flat | 0.8050 | 0.533962 | 0.687089 |
| description up | 0.7900 | 0.537365 | 0.675609 |

All worse. "features/details up" does buy the best MRR of any configuration
measured (0.5618) but pays 1.5 points of HitRate for it, and HitRate carries
50% of the composite against MRR's 30%. P0's weights stand.

## Hardening (P5 checklist)

Measured on this machine, Python 3.14, at the adopted configuration:

| | dense off (default) | dense on |
|---|---|---|
| startup / index build | **1.34 s** | 10.30 s |
| peak RSS | **0.55 GB** | 0.90 GB |
| per-turn latency, median | 22.2 ms | -- |
| per-turn latency, p95 | 47.6 ms | -- |
| per-turn latency, max | 58.4 ms | -- |

500 turns measured for latency. Reported token usage is 0/0: no model, no API,
no network on any path.

Retiring the dense route also retires the only reason the default path needed
`sentence-transformers`, `torch` and a 76 MB artifact. **The submission now
runs on the standard library alone**, which turns "no network required by
default" from a fallback guarantee into a property of the code.

Failure paths from P4-T4 are unchanged and still covered: dense, reranker and
clarification failures each degrade to a valid response rather than costing
the turn, and `tests/test_phase4_fallback.py` exercises them.

## Confirmed end to end

`python3 -m evaluator.local_evaluator`, no environment variables, which is how
the official harness constructs the agent:

```
hit_rate_at_10 0.830   mrr 0.550613   mttc 4.685   technical_score 0.706484
boundary 0.9000 / 5.100    browsing 0.8250 / 4.550
buying   0.8375 / 4.325    intent_override 0.8000 / 5.867
```

110 tests pass.

| milestone | TechnicalScore |
|---|---|
| original BM25 starter | 0.106710 |
| P2 lexical only | 0.115573 |
| P3 MiniLM + weighted fusion | 0.151089 |
| P4 clarification + reranker | 0.636663 |
| **P5 dense retired + evidence** | **0.706484** |
