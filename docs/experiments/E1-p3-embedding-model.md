# E1 - P3 embedding model selection

Decision record for the P3 decision gate ("Compare no dense route, at least two
models... Do not choose by reputation alone").

```yaml
experiment_id: "E1"
phase: "P3"
hypothesis: "A dense semantic route fused with BM25 improves recall for paraphrases and scenario-based Browsing, and one of two lightweight embedding models does so measurably better."
base_commit: "d72e2c9"          # main, lexical-only baseline
candidate_commit: "675117e"     # shared P3 implementation; per-model config differs
dataset: "full public set (200 labeled sessions)"
fusion_method: "rrf"            # default; weighted fusion NOT yet evaluated
overall_metrics:
  no_dense_baseline: {hit_rate_at_10: 0.125, mrr: 0.068034, mttc: 9.81, efficiency: 0.119, technical_score: 0.106710}
  all-MiniLM-L6-v2: {hit_rate_at_10: 0.175, mrr: 0.079234, mttc: 9.305, efficiency: 0.1695, technical_score: 0.145170}
  bge-small-en-v1.5: {hit_rate_at_10: 0.135, mrr: 0.074575, mttc: 9.715, efficiency: 0.1285, technical_score: 0.115573}
scenario_metrics:
  all-MiniLM-L6-v2:
    buying:          {n: 80, hit_rate_at_10: 0.250,    mrr: 0.130675, mttc: 8.500}
    browsing:        {n: 80, hit_rate_at_10: 0.100,    mrr: 0.034340, mttc: 10.000}
    intent_override: {n: 30, hit_rate_at_10: 0.133333, mrr: 0.074167, mttc: 10.033333}
    boundary:        {n: 10, hit_rate_at_10: 0.300,    mrr: 0.042063, mttc: 8.000}
  bge-small-en-v1.5:
    buying:          {n: 80, hit_rate_at_10: 0.2375,   mrr: 0.132758, mttc: 8.625}
    browsing:        {n: 80, hit_rate_at_10: 0.0375,   mrr: 0.008681, mttc: 10.6375}
    intent_override: {n: 30, hit_rate_at_10: 0.133333, mrr: 0.116667, mttc: 10.066667}
    boundary:        {n: 10, hit_rate_at_10: 0.100,    mrr: 0.010000, mttc: 10.000}
  no_dense_baseline: {}   # NOT MEASURED per scenario -- see "Open gaps" below
performance:
  startup_seconds: null       # T1 measured ~29s for both, but that run included a
                              # one-time model download; not re-measured after the fix
  per_turn_latency_ms: null   # T1's single-sample query timing was unreliable
  peak_memory_mb: null        # T1 reported ~681MB (MiniLM) / ~720MB (bge), same caveat
model_api: {model: "sentence-transformers/all-MiniLM-L6-v2", network_required: false, prompt_tokens: 0, completion_tokens: 0}
newly_won_sessions: []      # not extracted; evaluator reports aggregates only
newly_lost_sessions: []
known_regressions: []       # cannot be determined without baseline scenario metrics
decision: "keep"
```

## Decision

**Selected `sentence-transformers/all-MiniLM-L6-v2`.** It beats bge-small on every
aggregate metric (HitRate +0.040, MRR +0.005, MTTC -0.410) for a composite of 0.145
against 0.116, and beats the no-dense baseline by +0.038 composite (+36%).

Both models are 384-dimensional and permissively licensed (Apache-2.0 and MIT
respectively), so neither dimension nor license separated them.

Notably the retrieval-tuned model lost to the general-purpose one, which is why the
gate's "do not choose by reputation alone" instruction mattered here.

## Confidence

The MiniLM-over-bge margin is **suggestive, not decisive**. The HitRate gap of 0.040 is
8 sessions out of 200, near the noise floor for a sample this size (standard error
~0.03). What raises confidence is that MiniLM wins consistently across three
independent metrics rather than on one. The gain over the **no-dense baseline** is the
better-supported claim.

The boundary scenario (n=10) is statistically meaningless per-scenario -- MiniLM's
0.300 versus bge's 0.100 is 3 sessions against 1 -- and should not be read as signal.

## Why Browsing is still weak (not a model problem)

Browsing scores far below Buying (MiniLM 0.100 vs 0.250), but this is **structural at
P3 and not evidence against the dense route**. The agent has no clarification ability
yet: `starter/agent.py` returns `ask_attribute: None` on every turn, and the simulator
only discloses a constraint when asked --
`evaluator/local_evaluator.py:170` replies "Those options are not quite right yet. Ask
me about one specific attribute." whenever `ask_attribute` is absent. Buying discloses
a hard constraint in the opening message, so it needs no question; Browsing "begins
vague" and therefore never gains information.

Recovering hit turns from MTTC makes the closed loop explicit:

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

Two consequences: no embedding model can lift Browsing while the query never grows, and
Efficiency (20% of the composite) is currently a pure function of HitRate, since every
hit is at turn 1. Both unlock in P4-T2 (clarification policy), for which P1 already
left `asked_attributes`, `rejected_attributes` and `clarification_turns` on
`SessionState` -- all still unused.

## Open gaps

1. **Baseline per-scenario metrics were never measured.** Worth capturing for the
   record so the P3 delta is documented per scenario, but it will not explain
   Browsing -- the cause is established above.
2. **Weighted fusion is unevaluated.** Only the default RRF has been run, so P3-T5's
   "both are configuration-selectable and independently evaluated" is not yet
   satisfied. Given how weak Browsing is, a lexical-weighted blend is worth testing.
3. **Performance figures need re-measuring.** The T1 benchmark that produced the
   startup/RAM/latency numbers had three measurement bugs (model size summed redundant
   export formats, startup included a one-time download, query latency was a single
   untimed-warmup sample). These were fixed in `ec26455` but the benchmark has not been
   re-run, so those fields are recorded as null rather than with known-bad values.
