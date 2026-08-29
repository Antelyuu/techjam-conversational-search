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
candidate_commit: "5d32397"
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
model_api: {model: "none", network_required: false, prompt_tokens: 0, completion_tokens: 0}
known_regressions: ["the reranker costs 0.146 MRR under weighted fusion"]
decision: "keep weighted fusion; reject the reranker as default, retain behind a flag"
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

## The reranker loses, and loses in a specific way

<!-- pending: weight sweep -->

## Decision

<!-- pending -->
