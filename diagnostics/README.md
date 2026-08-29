# TechJam Diagnostics Harness

Three files:

- **`tracer.py`** — `PipelineTracer`, a per-session logger you call into at
  each pipeline stage. Doesn't change what your Agent returns; only records
  what happened internally.
- **`taxonomy.py`** — classifies every missed session into one failure
  bucket (`filter_excluded`, `never_retrieved`, `retrieved_but_buried`,
  `clarification_stalled`, `override_corrupted`).
- **`metrics.py`** — the full diagnostic metric set (Recall@50 ceiling,
  route-exclusive recall, filter false-negative rate, rank shift from
  fusion to output, BM25-margin AUROC, query-target cosine drift, pool
  narrowing, wasted-turn rate).
- **`run_diagnostics.py`** — runnable demo on synthetic data. Confirmed
  working above. This is also the template for your real integration.

## Wiring this into your real pipeline

Per `techjam-customer-input-pipeline.md`, add tracer calls at these exact
points. Nothing about your actual retrieval/filter/rerank logic changes —
you're just also handing the intermediate lists to the tracer.

```python
from tracer import PipelineTracer

# once per session, when the evaluator calls reset()
tracer = PipelineTracer(session_id=session_id, scenario=scenario_label, target_asin=target_asin)
# scenario_label and target_asin: for your own dev-set runs you know these from
# the labeled public sessions. You will not have target_asin at inference time
# in the real evaluator -- that's fine, this harness is for LOCAL DEV-SET
# DIAGNOSIS ONLY, using your 200 labeled public sessions.

# inside Agent.respond(), per turn:
tracer.start_turn(turn, is_override_turn=<True if this is a detected override turn>)

tracer.log_query(cumulative_query_text, dense_query_embedding_or_None)

tracer.log_retrieval("lexical", bm25_ranked_ids, bm25_scores)
tracer.log_retrieval("dense", dense_ranked_ids, dense_scores, ran=dense_actually_ran)

tracer.log_filter(pre_filter_candidate_ids, post_filter_candidate_ids, reason="hard_budget")

tracer.log_fusion(fused_ranked_ids)          # after step 5 fusion, before Phase 4/5 rerank

tracer.log_rerank(reranked_ids, reranked_scores)   # after Phase 4/5

tracer.log_clarification(ask_attribute_or_None, pool_size_before_ask=len(post_filter_candidate_ids))

tracer.end_turn(output_recommendations)      # the actual top_k ids you return this turn

# when the session ends (hit or turn 10):
session_trace = tracer.finalize(hit_turn=<int or None>, hit_rank=<int or None>)
# collect session_trace into a list across all 200 dev sessions
```

Then run the same analysis as `run_diagnostics.py`'s `print_report()`
against your real `sessions` list instead of the synthetic one.

## Reading the output

1. **Check Recall@50 first.** If it's near 1.0 but HitRate@10 is far lower,
   your problem is ranking (Phase 4/5, fusion), not retrieval — go straight
   to `rank_shift_retrieval_to_output` and the miss taxonomy's
   `retrieved_but_buried` count.
2. **If Recall@50 is closer to your current HitRate@10**, retrieval itself
   is the ceiling — check `route_exclusive_recall` (is dense pulling its
   weight?) and `candidate_pool_jaccard_overlap` (are the two routes
   actually diverse, or redundant?).
3. **`filter_excluded` count > 0** is a correctness bug, not a tuning
   question — every one of those sessions is a target that should have
   been findable and wasn't, due to filter logic/timing. Fix these first,
   they're free points.
4. **`override_corrupted` count** tells you if state purge on override
   turns is actually leaking stale candidates into ranking, independent of
   whether the BM25-margin gate helps or hurts on those turns.
5. **`wasted_turn_rate` near or above 0.5** means your clarification policy
   is roughly coin-flip useful — a strong signal to move to the
   information-gain attribute selector discussed earlier, since it means
   about half your `ask_attribute` turns aren't buying you anything.

## Known limitations of this harness

- `filter_false_negative_rate` and `classify_miss`'s `filter_excluded` label
  use a **mechanical proxy** (present pre-filter, absent post-filter, never
  recovered) rather than true constraint-satisfaction logic, because this
  harness doesn't have access to your actual slot/constraint schema. If you
  want the *true* false-negative rate (target satisfied the constraint AND
  was still excluded), pass a `satisfies_constraint_fn(session, turn)`
  callable into `filter_false_negative_rate` that checks your real slot
  state against the target's actual attributes from the catalog.
- `margin_hit_auroc` assumes the margin gate fires on turn 1 with at least
  2 lexical scores logged; if your gate can fire on a different turn,
  adjust the turn selection in that function.
- `query_target_cosine_drift` needs you to supply `target_embeddings` (the
  dense embedding of each session's true target product) — this is cheap
  to precompute once over your 200 dev-session targets using whatever
  embedding model your dense route already uses.
