# E1 - P3 embedding model selection

Decision record for the P3 decision gate ("Compare no dense route, at least two
models... Do not choose by reputation alone").

> **Correction (superseded in part by E2).** The bge-small run recorded here was
> invalid: its dense route never engaged, so the figures originally attributed to
> bge-small are in fact the lexical-only control. See "The bge run was invalid"
> below. MiniLM remains the selected model, but on narrower evidence than first
> claimed -- bge-small has still never actually been measured.

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
  bge-small-en-v1.5:     "NOT MEASURED - see correction"
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
performance:
  startup_seconds: null       # T1's numbers were measured by the pre-fix benchmark
  per_turn_latency_ms: null   # and the corrected benchmark was never re-run
  peak_memory_mb: null
model_api: {model: "sentence-transformers/all-MiniLM-L6-v2", network_required: false, prompt_tokens: 0, completion_tokens: 0}
newly_won_sessions: []
newly_lost_sessions: []
known_regressions: []
decision: "keep MiniLM; bge-small remains untested"
```

## The bge run was invalid

The fusion ablation (E2) measured the lexical-only control at exactly the values
previously recorded for bge-small -- all five aggregate metrics and all sixteen
per-scenario metrics identical, MRR matching to six decimal places across 200
sessions:

| metric | reported as "bge-small" | measured lexical-only |
|---|---|---|
| hit_rate_at_10 | 0.135 | 0.135 |
| mrr | 0.074575 | 0.074575 |
| mttc | 9.715 | 9.715 |
| technical_score | 0.115573 | 0.115573 |
| browsing mrr | 0.008681 | 0.008681 |

Two different retrieval systems cannot produce identical MRR to six decimals over
200 sessions. The bge branch's dense route did not engage during that run -- almost
certainly because its embedding artifact had not been built yet (it was committed
later, in `a59543a`), so `load_dense_retriever()` returned `None` and the agent
silently served BM25 results.

This is precisely the silent-failure mode that `dc99b0f` now guards against by
printing the fallback reason to stderr. Had that warning existed, the invalid run
would have announced itself instead of being recorded as a model comparison.

## Decision

**Selected `sentence-transformers/all-MiniLM-L6-v2`**, but on narrower grounds than
originally recorded. What is actually established:

- MiniLM's dense route plus fusion beats the lexical-only control by a wide margin
  (+0.030 composite with RRF, +0.036 with weighted fusion -- see E2), reproduced
  independently twice.
- **bge-small has never been measured.** The comparison the decision gate asked for
  has not been performed.

MiniLM is retained because it is the configuration that demonstrably works, not
because it was shown superior to bge-small. Re-running bge-small is cheap now that
its artifact exists (`a59543a`) and `scripts/fusion_ablation.py` automates the
comparison; it is worth doing before the model choice is described as evidence-based
in the final report.

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
