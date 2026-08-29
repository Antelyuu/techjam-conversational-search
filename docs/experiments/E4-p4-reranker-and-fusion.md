# E4 - P4 deterministic reranker, and the P3 fusion re-check

Decision record for P4-T1 (deterministic final scorer), and the re-test E2
asked for: *"If P4 changes retrieval inputs materially, re-run this ablation
rather than assuming weighted still wins."* P4 changed them materially, so it
was re-run.

Produced by `python3 -m scripts.rerank_ablation`, which crosses both variables
against one agent instance. The clarification policy is held at its adopted
default (E3) throughout.

```yaml
experiment_id: "E4"
phase: "P4"
hypothesis: "An explicit per-candidate feature checklist ranks better than the fused score plus boosts it replaces; and weighted fusion still beats RRF once clarification changes the query distribution."
base_commit: "d2bb995"
candidate_commit: "adcde1a"
dataset: "full public set (200 labeled sessions)"
clarification: "soft_plus_wildcard (the E3 default)"
overall_metrics:
  weighted_fused:  {hit_rate_at_10: 0.740, mrr: 0.474613, mttc: 5.250, efficiency: 0.5750, technical_score: 0.627384}
  weighted_rerank: {hit_rate_at_10: 0.735, mrr: 0.328794, mttc: 5.280, efficiency: 0.5720, technical_score: 0.580538}
  rrf_fused:       {hit_rate_at_10: 0.725, mrr: 0.320383, mttc: 5.400, efficiency: 0.5600, technical_score: 0.570615}
  rrf_rerank:      {hit_rate_at_10: 0.740, mrr: 0.323315, mttc: 5.250, efficiency: 0.5750, technical_score: 0.581994}
scenario_metrics:
  weighted_fused:  {buying: 0.7000, browsing: 0.7375, intent_override: 0.800, boundary: 0.900}
  weighted_rerank: {buying: 0.7000, browsing: 0.7375, intent_override: 0.800, boundary: 0.800}
  rrf_fused:       {buying: 0.6750, browsing: 0.7250, intent_override: 0.800, boundary: 0.900}
  rrf_rerank:      {buying: 0.7125, browsing: 0.7375, intent_override: 0.800, boundary: 0.800}
weight_sweep:
  fused_control: {hit_rate_at_10: 0.740, mrr: 0.474613, mttc: 5.250, technical_score: 0.627384}
  spec_order:    {hit_rate_at_10: 0.735, mrr: 0.328794, mttc: 5.280, technical_score: 0.580538}
  sharp_ranks:   {hit_rate_at_10: 0.745, mrr: 0.440911, mttc: 5.245, technical_score: 0.619873}
  retrieval_led: {hit_rate_at_10: 0.730, mrr: 0.336718, mttc: 5.310, technical_score: 0.579815}
  both:          {hit_rate_at_10: 0.755, mrr: 0.459484, mttc: 5.105, technical_score: 0.633245}
model_api: {model: "none", network_required: false, prompt_tokens: 0, completion_tokens: 0}
known_regressions:
  - "as first weighted, the reranker cost 0.146 MRR; corrected by RANK_DECAY 60 -> 5 and retrieval-led weights"
  - "at the adopted weights the reranker still trades -0.015 MRR for +0.015 HitRate against the fused order"
decision: "keep weighted fusion; keep the reranker with retrieval-led weights and RANK_DECAY=5 (+0.005861)"
```

## Weighted fusion survives the re-check

E2 adopted weighted over RRF by **+0.005919**, about one session on HitRate, and
flagged that RRF was the better of the two on Browsing -- the scenario P4
targets. The concern was reasonable and it did not materialise:

| fusion | E2 (no questions) | E4 (with clarification) |
|---|---|---|
| rrf | 0.145170 | 0.570615 |
| weighted | **0.151089** | **0.627384** |
| margin | +0.005919 | **+0.056769** |

The margin grew roughly tenfold, from about one session to about eleven, and
weighted now wins Browsing too (0.7375 vs 0.7250) rather than losing it.
The reason is the same one E2 identified: weighted preserves score margins where
RRF collapses them to ranks, and richer queries give those margins more to say.

`weighted_fused` also reproduces E3's `soft_plus_wildcard` to six decimals
(0.627384), which is this project's standing cross-check against a
configuration silently not being the one under test.

## The reranker loses, and the breakdown says why

As first written -- weights in P4's stated priority order, `hard_constraints`
4.0 down to `soft_preferences` 0.25, and `RANK_DECAY = 60` mirroring the RRF
constant -- the scorer **lost 0.046846 composite**, and lost it entirely in
MRR:

| | HitRate@10 | MRR | TechnicalScore |
|---|---|---|---|
| weighted_fused | 0.740 | **0.474613** | **0.627384** |
| weighted_rerank | 0.735 | 0.328794 | 0.580538 |

HitRate barely moved. The reranker found the same products and **ranked them
worse**, which is the signature of a scorer whose discriminating signal has
been drowned out. The per-candidate breakdown P4-T1 requires is what made this
diagnosable rather than merely visible: `hard_constraints` spans the full 0-1
at weight 4.0, while at `RANK_DECAY = 60` the rank features span only 1.00 to
0.55 over a fifty-candidate pool. So a coarse word-containment check --
`matched_constraints()` asks only whether a constraint's text appears anywhere
in the product's searchable text -- was overriding the score margin that
weighted fusion exists to preserve.

## The sweep confirms the diagnosis

`python3 -m scripts.rerank_weight_sweep`, crossing sharper rank discrimination
(`RANK_DECAY = 5`) with retrieval-led weights:

| config | HitRate@10 | MRR | TechnicalScore |
|---|---|---|---|
| fused_control | 0.740 | 0.474613 | 0.627384 |
| spec_order | 0.735 | 0.328794 | 0.580538 |
| sharp_ranks | 0.745 | 0.440911 | 0.619873 |
| retrieval_led | 0.730 | 0.336718 | 0.579815 |
| **both** | **0.755** | 0.459484 | **0.633245** |

`RANK_DECAY` was the larger half of the mistake: sharpening it alone recovers
most of the lost MRR (0.329 -> 0.441) even under the original weights.
Retrieval-led weights alone recover almost nothing (0.337), because with a rank
feature that spans 0.45 it does not much matter what you multiply it by. Only
both together beat the control.

## Decision

**Keep the reranker, with retrieval-led weights and `RANK_DECAY = 5`.**
0.633245 against 0.627384, **+0.005861**.

Two honest caveats:

1. **The margin is thin** -- comparable to the +0.005919 by which E2 adopted
   weighted fusion. It is +0.015 HitRate (three sessions) and -0.015 MRR: the
   reranker surfaces slightly more targets and ranks them slightly worse than
   the fused order does. It wins on the composite because HitRate carries 50%
   and MRR 30%. If P5 needs to cut moving parts, this is a fair candidate.
2. **The adopted weights contradict P4's stated feature order.** The spec lists
   hard-constraint satisfaction first and retrieval rank third and fourth.
   Weighted that way it measured 0.047 *worse* than not reranking at all. The
   checklist order is retained among the adjustment features -- hard
   constraints over category over metadata over soft preferences -- but
   retrieval rank now outranks all of them. This is recorded as a deviation
   from the phase document, decided on measurement.

### Confirmed end to end

`python3 -m evaluator.local_evaluator` with no environment variables -- the way
the official harness constructs the agent -- reproduces the adopted
configuration exactly:

```
hit_rate_at_10 0.755   mrr 0.459484   mttc 5.105   technical_score 0.633245
boundary 0.900 / 5.000    browsing 0.750 / 5.075
buying   0.725 / 4.775    intent_override 0.800 / 6.100
```

## A rejected refinement: giving the reranker the fused score

Code review observed that `_rank_value` reads only `route_ranks` -- raw
per-route pool positions -- so the reranker discards both the fusion blend and
the P2 category/budget boosts that produced the pool, re-deriving its order
from coarser inputs than the ones it was handed. That is a fair criticism, and
it was tested by adding the fused `combined` score as a seventh feature.

| config | HitRate@10 | MRR | TechnicalScore |
|---|---|---|---|
| adopted (no fused feature) | **0.755** | 0.464476 | 0.634543 |
| fused_light (fused weight 2.0) | 0.750 | 0.476833 | 0.634750 |
| fused_led (fused 4.0, ranks 1.0) | 0.745 | 0.476790 | 0.631237 |

**Rejected.** `fused_light` leads by 0.000207 -- two ten-thousandths, and it
buys that by trading HitRate down (-0.005) for MRR up (+0.012), which is a
wash rather than a gain. Weighting the fused score above the rank features is
clearly worse. A permanent seventh feature is not worth a difference this
size, so the code stays as it was and this is recorded so the idea is not
re-tried blind.

Note the rows above for `spec_order`, `sharp_ranks`, `retrieval_led` and
`fused_control` differ slightly from the first sweep (0.585788 vs 0.580538,
and so on). That is expected: the lead-in cap fix landed between the two runs
and changed the query text every configuration sees. `adopted` and `both` are
the same configuration and reproduce each other exactly, which is the internal
check that the sweep harness is switching what it claims to switch.

## Correction: the category feature was double-counted, and is worth zero

Found in the pre-merge code review of this branch, after the figures above
were recorded. It changes the adopted weights, so it is written here rather
than in a new record.

`category` is the only `DEFAULT_HARD_ATTRIBUTE`, and
`filtering.matched_constraints()` excluded only `budget` from its containment
check. So in any session whose one hard constraint is the category -- the
common case -- a candidate whose text contained the category word scored a
**full `hard_constraints` share (weight 1.0) *and* a full `category` feature
(weight 0.5)**. Category's effective weight was ~1.5 against a stated 0.5, and
every other weight in the table was tuned against that inflated value.

`SCORED_SEPARATELY` now holds both `budget` and `category`, and `_share()`
excludes the same set from its denominator that `matched_constraints()`
excludes from its numerator.

Re-sweeping the category weight alone, everything else at the settings above:

| category weight | HitRate@10 | MRR | TechnicalScore |
|---|---|---|---|
| **0.0** | **0.7550** | **0.471877** | **0.636663** |
| 0.5 | 0.7500 | 0.466940 | 0.631982 |
| 1.0 | 0.7550 | 0.463663 | 0.633899 |
| 1.5 | 0.7550 | 0.457171 | 0.631651 |
| 2.0 | 0.7400 | 0.455060 | 0.621418 |
| 3.0 | 0.7400 | 0.439607 | 0.616982 |

**MRR falls monotonically as category gains weight**, across six points, and
the composite is highest at zero. That is a structural result rather than a
tuning accident: retrieval already applies `score_category()` as a boost when
it builds the pool (`FUSED_BOOST_SCALE`), so the reranker's category feature
re-applies a signal the ordering it is reordering has already accounted for.
The *original* double-count was therefore the same mistake twice over.

**Adopted: `category` weight 0.0**, kept in `FEATURE_WEIGHTS` so the
contribution still appears in `explain()`. Confirmed end to end with no
environment variables:

```
hit_rate_at_10 0.755   mrr 0.471877   mttc 5.120   technical_score 0.636663
boundary 0.900 / 5.000    browsing 0.7625 / 5.000
buying   0.7125 / 4.8875  intent_override 0.800 / 6.100
```

Against the branch as reviewed (0.634293) this is **+0.002370**, and against
the fix alone at the old 0.5 weight (0.631982) it is +0.004681. Note the
figures earlier in this record predate the branch's last two commits, which is
why the "confirmed end to end" block above reads 0.633245 and the branch
measured 0.634293 before this change.

## P4-T5, the optional model reranker

**Not attempted, deliberately.** T5 is marked optional and asks for a local
cross-encoder or external LLM with cost and latency reported. Given that the
*deterministic* scorer only beat plain fused ordering by 0.0059, and only after
its weights were corrected, there is no evidence that reranking is where this
system's remaining headroom lives. A cross-encoder would add a model load, a
per-turn latency cost across fifty candidates, and a second offline-fallback
path, against a lever measured to be worth single-digit thousandths.

The headroom is in dialogue: P4-T2 alone moved the composite by +0.34. Spending
P5 on a cross-encoder rather than on questions would be optimising the wrong
half of the system.
