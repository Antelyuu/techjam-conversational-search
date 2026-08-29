# E2 - P3 fusion ablation

Satisfies P3-T5 ("both are configuration-selectable and independently evaluated")
and the P3 exit criterion that fusion has an ablation. Produced by
`python3 -m scripts.fusion_ablation`, which evaluates all three configurations in a
single run against the same agent instance.

```yaml
experiment_id: "E2"
phase: "P3"
hypothesis: "Fusing a dense route with BM25 improves the composite over lexical-only, and normalized weighted fusion ranks better than RRF because it preserves score margins rather than collapsing them to ranks."
base_commit: "d72e2c9"
candidate_commit: "330e5fd"
dataset: "full public set (200 labeled sessions)"
model: "sentence-transformers/all-MiniLM-L6-v2"
overall_metrics:
  lexical_only: {hit_rate_at_10: 0.135, mrr: 0.074575, mttc: 9.715, efficiency: 0.1285, technical_score: 0.115573}
  rrf:          {hit_rate_at_10: 0.175, mrr: 0.079234, mttc: 9.305, efficiency: 0.1695, technical_score: 0.145170}
  weighted:     {hit_rate_at_10: 0.180, mrr: 0.088964, mttc: 9.280, efficiency: 0.1720, technical_score: 0.151089}
scenario_metrics:
  lexical_only:
    buying:          {n: 80, hit_rate_at_10: 0.2375,   mrr: 0.132758, mttc: 8.625}
    browsing:        {n: 80, hit_rate_at_10: 0.0375,   mrr: 0.008681, mttc: 10.6375}
    intent_override: {n: 30, hit_rate_at_10: 0.133333, mrr: 0.116667, mttc: 10.066667}
    boundary:        {n: 10, hit_rate_at_10: 0.100,    mrr: 0.010000, mttc: 10.000}
  rrf:
    buying:          {n: 80, hit_rate_at_10: 0.250,    mrr: 0.130675, mttc: 8.500}
    browsing:        {n: 80, hit_rate_at_10: 0.100,    mrr: 0.034340, mttc: 10.000}
    intent_override: {n: 30, hit_rate_at_10: 0.133333, mrr: 0.074167, mttc: 10.033333}
    boundary:        {n: 10, hit_rate_at_10: 0.300,    mrr: 0.042063, mttc: 8.000}
  weighted:
    buying:          {n: 80, hit_rate_at_10: 0.300,    mrr: 0.140397, mttc: 8.025}
    browsing:        {n: 80, hit_rate_at_10: 0.0625,   mrr: 0.033056, mttc: 10.375}
    intent_override: {n: 30, hit_rate_at_10: 0.166667, mrr: 0.116667, mttc: 9.800}
    boundary:        {n: 10, hit_rate_at_10: 0.200,    mrr: 0.041667, mttc: 9.000}
performance: {startup_seconds: null, per_turn_latency_ms: null, peak_memory_mb: null}
model_api: {model: "sentence-transformers/all-MiniLM-L6-v2", network_required: false, prompt_tokens: 0, completion_tokens: 0}
known_regressions: ["weighted loses Browsing (-0.0375) and Boundary (-0.100) against RRF"]
decision: "undecided - see recommendation"
```

## Result

Both fusion methods clearly beat lexical-only. That is the robust finding:

| config | technical_score | vs lexical-only |
|---|---|---|
| lexical_only | 0.115573 | -- |
| rrf | 0.145170 | **+0.029597** |
| weighted | **0.151089** | **+0.035516** |

The dense route is worth its dependency. The RRF figure reproduces the earlier MiniLM
run exactly (0.145170), which also confirms that the current HEAD -- with dense
defaulted on and the artifact guard added -- still scores what it did before those
changes.

## weighted vs RRF: nominally better, but thin

Weighted wins the composite by **+0.005919**, and that margin is not solidly outside
noise:

- HitRate differs by 0.005, which is **one session out of 200**.
- The genuine gain is MRR: 0.088964 vs 0.079234 (+0.0097). Weighted ranks the target
  higher when it finds it, which is what normalizing scores rather than collapsing
  them to ranks should do.
- Per scenario the two disagree, and not by much in absolute session counts: weighted
  wins Buying (+0.050, 4 sessions) and Intent Override (+0.033, 1 session); RRF wins
  Browsing (+0.0375, 3 sessions) and Boundary (+0.100, 1 session).

Boundary (n=10) carries no statistical weight either way.

## Recommendation

Adopt **weighted** as the default: it wins the composite, and its advantage sits in
MRR, which is the component least likely to be a fluke of which sessions happened to
land in the top 10.

Two caveats worth carrying forward:

1. The margin over RRF is around one session on HitRate. If P4 changes retrieval
   inputs materially, re-run this ablation rather than assuming weighted still wins.
2. RRF is better on Browsing, the scenario P4's clarification policy targets. Once
   clarification lands and Browsing queries actually gain content, the fusion choice
   should be re-tested -- the ranking is not necessarily stable under a different
   query distribution.

Weights were not tuned: `DEFAULT_ROUTE_WEIGHTS` is an even 0.5/0.5 split. Tuning them
is an untried lever, and given Browsing's weakness a lexical-heavier split is the
obvious first thing to try.
