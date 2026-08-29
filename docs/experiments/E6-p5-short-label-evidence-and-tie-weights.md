# E6 - short labels are evidence, and the weights that only break ties

Decision record for the continuation of P5. Two changes are adopted; the
handoff's open experiment (retuning the reranker weights for the dense-off
world) is run and closed; and two ideas are measured structurally void, with
the verification that proves it.

```yaml
experiment_id: "E6"
phase: "P5"
hypothesis: "The 3-token evidence floor discards the Buying opener's material/color label, silencing the dominant feature exactly where other text is scarcest; and E4's tie-breaking weights are stale twice over -- tuned dense-on and tuned against an evidence feature that said nothing before the first quoted sentence."
base_commit: "9c85b68"
dataset: "full public set (200 labeled sessions)"
overall_metrics:
  p5_as_handed:  {hit_rate_at_10: 0.830, mrr: 0.550613, mttc: 4.685, technical_score: 0.706484}
  min_tokens_1:  {hit_rate_at_10: 0.870, mrr: 0.594407, mttc: 4.175, technical_score: 0.749822}
  plus_retune:   {hit_rate_at_10: 0.870, mrr: 0.605760, mttc: 4.170, technical_score: 0.753328}
scenario_metrics:
  p5_as_handed: {buying: 0.8375, browsing: 0.8250, intent_override: 0.800, boundary: 0.900}
  plus_retune:  {buying: 0.8750, browsing: 0.8625, intent_override: 0.833, boundary: 1.000}
model_api: {model: "none", network_required: false, prompt_tokens: 0, completion_tokens: 0}
known_regressions: []
decision: "ship MIN_EVIDENCE_TOKENS=1 + lexical_rank 1.0 + soft_preferences 2.0 (+0.046844 over P5 as handed)"
```

## E6-T1: the 3-token evidence floor was discarding the strongest early signal

E5 adopted `MIN_EVIDENCE_TOKENS = 3` on the reasoning that a bare label --
"cotton", "color: black" -- matches too much of the catalogue to discriminate
and that slots and BM25 already carry it. The reasoning was never measured,
and it is wrong on both counts at once:

- The evaluator's card puts the corpus-matched material and color **first**
  (`intent_card()` inserts them at positions 0 and 1), so the Buying opener's
  "A key requirement is: {constraint}" discloses exactly such a label.
- At weight 12 the evidence feature decides the order outright -- so ignoring
  the label did not neutrally defer to BM25, it silenced the dominant feature
  for the whole session in the sessions with the least other text. E5's own
  miss analysis (mean target coverage 0.32 in missed sessions) was in part
  this floor zeroing usable evidence.

Sweeping the floor, everything else as shipped:

| min tokens | HitRate@10 | MRR | MTTC | TechnicalScore |
|---|---|---|---|---|
| 3 (E5) | 0.8300 | 0.550613 | 4.685 | 0.706484 |
| 2 | 0.8650 | 0.587327 | 4.365 | 0.741398 |
| **1** | **0.8700** | **0.594407** | **4.175** | **0.749822** |

**+0.043338 for deleting a guard.** Confirmed the way this project confirms
everything: first as an in-process patch, then re-measured through the plain
evaluator after the code change landed -- 0.749822 both times, six decimals.
Length-weighting already keeps a one-token label from outvoting a fourteen
token sentence, so the floor was buying robustness the arithmetic already
provided.

Dropping the floor also closes the sharpest pre-merge review finding: 22% of
catalogue rows produce a card constraint containing ";", `split_disclosures()`
cannot tell an internal semicolon from the simulator's joiner, and the 3-token
floor was discarding the resulting short fragments ("7.8 Ounces"). With the
floor at 1 a fragmented constraint scores within dedup noise of the whole,
because coverage weights each piece by its own token count.

## E6-T2: retuning the tie-breakers (closes the handoff's open experiment)

The handoff's interrupted experiment -- re-sweeping `FEATURE_WEIGHTS` and
`RANK_DECAY` in the dense-off world -- was run first as specified, at
`MIN_EVIDENCE_TOKENS = 3`. As predicted there, nothing decisive: decay
2-30 within ±0.0037 and non-monotonic, hard 0-4 within 0.0001, metadata
0-1 **exactly** flat. The one live signal was soft_preferences (0.1 -> 1.0
worth +0.0078). "No change" was the likely outcome and, for decay, hard and
metadata, the actual one.

Re-swept under min-tokens 1, because T1 changes what the evidence feature is:

| config | HitRate@10 | MRR | TechnicalScore |
|---|---|---|---|
| min1 baseline (lex=2, soft=0.1) | 0.8700 | 0.594407 | 0.749822 |
| evidence=24 | 0.8700 | 0.602829 | 0.752349 |
| lexical_rank=1 | 0.8700 | 0.603246 | 0.752474 |
| evidence=24 + lexical_rank=1 | 0.8700 | 0.603246 | 0.752474 |
| evidence=48 + lexical_rank=1 | 0.8700 | 0.603246 | 0.752474 |
| lexical_rank=1 + soft=2..8 | 0.8700 | 0.605760 | **0.753328** |

Halving lexical_rank and doubling evidence are **identical to six decimals**
-- only the evidence:rank ratio matters, and the plateau extends through
evidence=48, so this is one robust change, not two fragile ones. Adopted
`lexical_rank = 1.0` (the smaller diff) and `soft_preferences = 2.0` (first
point of a plateau flat through 8). HitRate never moves in any of it:
these weights now only decide MRR ties beneath the evidence ordering.
`dense_rank` keeps its E4 value -- it is dead by default and live only under
`SHOPPING_AGENT_DENSE=1`, where none of this was measured.

## Measured and structurally void

**The soft-budget price fingerprint.** "budget around $X" quotes the target's
exact price, so an exact-price feature looked like an identity signal for
sparse-text sessions. Implemented as a patch, it reproduced the baseline to
six decimals at every weight -- the no-op signature -- and the reason is
structural: `intent_card()` appends the budget line after material, color and
every feature/detail, and the card only discloses slots [:4]. **Verified
directly: 0 of the 200 public cards can ever disclose their budget line.**
The same fact makes the reranker's metadata feature exactly flat (E6-T2) and
voids the companion idea of excluding budget text from coverage. Both dead on
this generator; a private set that discloses budgets revives them.

**Re-tuning RANK_DECAY.** 2-30 swept at min-tokens 3 and 2/15/30 re-checked at
min-tokens 1: best +0.0013 over incumbent, non-monotonic, and the min1 retune
subsumes what it was reaching for (rank mattering less). Kept at 5.

## Also in this continuation

The pre-merge review (re-run per convention after the handoff lost the first
one) returned four findings, all acted on alongside T1: disclosures are now
tokenized once per rerank pool instead of once per candidate
(`disclosure_token_sets` / `coverage_from_sets`), `_TOKEN_CACHE` is bounded at
120k entries, and the token definition is shared (`shopping_agent/text.py`)
by the BM25 query builder and evidence scoring, with the two deliberately
different stopword lists documented as such. P4's cosmetic leftovers are
closed: `soft_budget_closeness()` is defined once, dead "xs" removed from
`SIZE_ABBREVIATIONS`. All structural changes reproduced 0.749822 exactly
before the weight retune moved anything.

## Confirmed end to end

`python3 -m evaluator.local_evaluator`, no environment variables, which is how
the official harness constructs the agent:

```
hit_rate_at_10 0.870   mrr 0.605760   mttc 4.170   technical_score 0.753328
boundary 1.0000 / 4.400    browsing 0.8625 / 4.125
buying   0.8750 / 3.675    intent_override 0.8333 / 5.533
```

111 tests pass.

| milestone | TechnicalScore |
|---|---|
| original BM25 starter | 0.106710 |
| P3 MiniLM + weighted fusion | 0.151089 |
| P4 clarification + reranker | 0.636663 |
| P5 dense retired + evidence | 0.706484 |
| **P5 + short-label evidence + retune** | **0.753328** |
