# E1 - P3 embedding model selection

Decision record for the P3 decision gate ("Compare no dense route, at least two
models... Do not choose by reputation alone").

> **Correction history.** The bge-small figures first recorded here were from a run
> whose dense route never engaged -- they were in fact the lexical-only control
> (see "A discarded bge run" below). Corrected bge-small results were supplied
> afterwards and are the ones recorded below. **MiniLM remains the selected model,
> now on a genuine measured comparison rather than by default.**

```yaml
experiment_id: "E1"
phase: "P3"
hypothesis: "A dense semantic route fused with BM25 improves recall for paraphrases and scenario-based Browsing, and one of two lightweight embedding models does so measurably better."
base_commit: "d72e2c9"
candidate_commit: "330e5fd"
dataset: "full public set (200 labeled sessions)"
fusion_method: "rrf"
overall_metrics:
  original_bm25_starter: {hit_rate_at_10: 0.125, mrr: 0.068034, mttc: 9.81, efficiency: 0.119, technical_score: 0.106710}
  lexical_only_P1_P2:    {hit_rate_at_10: 0.135, mrr: 0.074575, mttc: 9.715, efficiency: 0.1285, technical_score: 0.115573}
  all-MiniLM-L6-v2_rrf:  {hit_rate_at_10: 0.175, mrr: 0.079234, mttc: 9.305, efficiency: 0.1695, technical_score: 0.145170}
  bge-small-en-v1.5_rrf: {hit_rate_at_10: 0.160, mrr: 0.069109, mttc: 9.450, efficiency: 0.1550, technical_score: 0.131733}
scenario_metrics:
  lexical_only_P1_P2:
    buying:          {n: 80, hit_rate_at_10: 0.2375,   mrr: 0.132758, mttc: 8.625}
    browsing:        {n: 80, hit_rate_at_10: 0.0375,   mrr: 0.008681, mttc: 10.6375}
    intent_override: {n: 30, hit_rate_at_10: 0.133333, mrr: 0.116667, mttc: 10.066667}
    boundary:        {n: 10, hit_rate_at_10: 0.100,    mrr: 0.010000, mttc: 10.000}
  all-MiniLM-L6-v2_rrf:
    buying:          {n: 80, hit_rate_at_10: 0.250,    mrr: 0.130675, mttc: 8.500}
    browsing:        {n: 80, hit_rate_at_10: 0.100,    mrr: 0.034340, mttc: 10.000}
    intent_override: {n: 30, hit_rate_at_10: 0.133333, mrr: 0.074167, mttc: 10.033333}
    boundary:        {n: 10, hit_rate_at_10: 0.300,    mrr: 0.042063, mttc: 8.000}
  bge-small-en-v1.5_rrf:
    buying:          {n: 80, hit_rate_at_10: 0.225,    mrr: 0.099717, mttc: 8.750}
    browsing:        {n: 80, hit_rate_at_10: 0.1125,   mrr: 0.039306, mttc: 9.8875}
    intent_override: {n: 30, hit_rate_at_10: 0.100,    mrr: 0.075000, mttc: 10.300}
    boundary:        {n: 10, hit_rate_at_10: 0.200,    mrr: 0.045000, mttc: 9.000}
performance:
  benchmark_sample_texts: 200
  benchmark_environment: "local macOS, Python 3.14, CPU"
  selected_model:
    startup_seconds: 5.733
    per_turn_query_latency_ms: 8.079
    peak_memory_mb: 456.6
    batch_encode_seconds: 1.298
    texts_per_second: 154.07
  bge_comparison:
    startup_seconds: 5.648
    per_turn_query_latency_ms: 8.097
    peak_memory_mb: 602.0
    batch_encode_seconds: 2.670
    texts_per_second: 74.92
  note: "Measured by scripts/embedding_model_benchmark.py on 200 catalogue texts; peak RSS includes the Python/model process and values are hardware/environment dependent."
model_api: {model: "sentence-transformers/all-MiniLM-L6-v2", network_required: false, prompt_tokens: 0, completion_tokens: 0}
newly_won_sessions: []
newly_lost_sessions: []
known_regressions: []
decision: "keep MiniLM - beats bge-small by +0.013437 composite on matched RRF settings"
```

## Decision

**Selected `sentence-transformers/all-MiniLM-L6-v2`.** On matched settings (RRF
fusion), it beats bge-small on every aggregate metric:

| config | HitRate@10 | MRR | MTTC | technical_score |
|---|---|---|---|---|
| lexical only | 0.135 | 0.074575 | 9.715 | 0.115573 |
| bge-small + RRF | 0.160 | 0.069109 | 9.450 | 0.131733 |
| **MiniLM + RRF** | **0.175** | **0.079234** | **9.305** | **0.145170** |
| **MiniLM + weighted** | **0.180** | **0.088964** | **9.280** | **0.151089** |

MiniLM leads bge-small by **+0.013437** composite under RRF, winning HitRate, MRR and
MTTC simultaneously rather than trading between them. Both models are 384-dimensional
and permissively licensed (Apache-2.0, MIT), so neither dimension nor license
separated them.

The retrieval-tuned model losing to the general-purpose one is why the gate's "do not
choose by reputation alone" instruction mattered.

### Two caveats on bge-small

**It is the better model for Browsing.** bge-small beats MiniLM on the paraphrase-heavy
scenario (0.1125 vs 0.100) and improves it most against the lexical control (+0.075).
That is the behaviour its retrieval tuning predicts. It loses overall because it gives
back more elsewhere -- Buying -0.0125 and Intent Override -0.0333 against lexical-only.
This matters for P4: once clarification lets Browsing queries gain content, the query
distribution changes and the model comparison is worth re-running rather than assumed
settled.

**Its MRR is worse than no dense route at all** (0.069109 vs the lexical control's
0.074575, -0.005466). bge-small surfaces more targets but ranks them lower than BM25
alone did. MiniLM does not have this problem (0.079234, above the control).

**Not measured:** bge-small under weighted fusion. To overtake MiniLM+weighted it
would need +0.019 over its own RRF score, roughly three times the gain weighted gave
MiniLM (+0.006), so this is unlikely to change the decision -- but it is untested.

## A discarded bge run

The bge figures first recorded here (0.135 / 0.074575 / 9.715 / 0.115573) were later
found to match the lexical-only control **exactly** -- all five aggregate and all
sixteen per-scenario metrics, MRR to six decimal places across 200 sessions. Two
different retrieval systems cannot do that. That run's dense route never engaged,
most likely because its embedding artifact had not been built yet (committed later in
`a59543a`), so `load_dense_retriever()` returned `None` and the agent silently served
BM25 results.

Recorded here because it is the exact failure mode `dc99b0f` now guards against: the
fallback reason is printed to stderr, so a run like that announces itself instead of
being mistaken for a model comparison. Corrected bge results were supplied afterwards
and are the ones used above.

## Why Browsing is still weak (not a model problem)

Browsing scores far below Buying, but this is **structural at P3**. The agent has no
clarification ability yet: `starter/agent.py` returns `ask_attribute: None` on every
turn, and the simulator only discloses a constraint when asked --
`evaluator/local_evaluator.py:170` replies "Those options are not quite right yet.
Ask me about one specific attribute." whenever `ask_attribute` is absent. Buying
discloses a hard constraint in the opening message; Browsing "begins vague" and so
never gains information.

Recovering hit turns from MTTC makes the closed loop explicit (MiniLM/RRF):

| scenario | hits | mean hit turn |
|---|---|---|
| buying | 20/80 | **1.00** |
| browsing | 8/80 | **1.00** |
| boundary | 3/10 | **1.00** |
| intent_override | 4/30 | 3.75 |

Every hit in Buying, Browsing and Boundary lands on **turn 1**; turns 2-10 contribute
nothing. The single exception is Intent Override, the one scenario where the customer
volunteers new information unprompted on turn 3 or 4. The agent only ever gains
information it was handed for free.

Two consequences: no embedding model can lift Browsing while the query never grows,
and Efficiency (20% of the composite) is currently near-fully determined by HitRate,
since almost every hit is at turn 1. Both unlock in P4-T2 (clarification policy), for
which P1 already left `asked_attributes`, `rejected_attributes` and
`clarification_turns` on `SessionState` -- all still unused.
