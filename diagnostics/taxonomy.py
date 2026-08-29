"""
Classifies every missed session into exactly one failure bucket:

    filter_excluded        target removed by a hard filter despite satisfying the constraint
    never_retrieved         target never appeared in either route's candidate pool, any turn
    retrieved_but_buried    target was retrieved but never made the top-10 output
    clarification_stalled   candidate pool barely shrank turn over turn; ran out the clock
    override_corrupted      miss happens on/after an override turn with pre-override leakage
    unclassified            didn't match any rule above (inspect manually)

Priority order matters: filter_excluded is checked first because a filter bug
is the most actionable/dangerous failure mode and should not be masked by a
"never_retrieved" label just because the filter also happened to remove it
from the pool.
"""

try:
    from .tracer import SessionTrace
except ImportError:  # Support `python3 run_diagnostics.py` from this directory.
    from tracer import SessionTrace

POOL_STALL_RATIO = 0.9   # pool shrank by less than 10% -> considered "stalled"
BURIED_RANK_CUTOFF = 10  # matches evaluator's top-10 scoring window


def classify_miss(session: SessionTrace) -> str:
    if session.hit_turn is not None:
        raise ValueError("classify_miss called on a session that was not a miss")

    target = session.target_asin

    # 1. filter_excluded: target was in the pre-filter pool but not the
    #    post-filter pool on some turn, and no explicit hard-constraint
    #    reason was ever satisfied by the target being genuinely out of bounds.
    #    We can't know true constraint satisfaction here without your slot
    #    state, so this checks the mechanical signal: target present pre-filter,
    #    absent post-filter, on every subsequent turn too (i.e. never recovered).
    ever_excluded_after_presence = False
    for t in session.turns:
        if target in t.pre_filter_pool and target not in t.post_filter_pool:
            ever_excluded_after_presence = True
    if ever_excluded_after_presence:
        # confirm it never came back into a later post-filter pool
        recovered = any(target in t.post_filter_pool for t in session.turns)
        if not recovered:
            return "filter_excluded"

    # 2. never_retrieved: target never appears in lexical or dense ranked ids,
    #    on any turn, at all.
    ever_in_any_route = any(
        target in t.lexical_ranked_ids or target in t.dense_ranked_ids
        for t in session.turns
    )
    if not ever_in_any_route:
        return "never_retrieved"

    # 3. override_corrupted: an override turn happened, and the miss persists
    #    with the target still visible pre-filter/pre-fusion after that turn
    #    but never surfacing in the final output -- suggests stale state
    #    leaking into ranking rather than a retrieval problem.
    override_turns = [t.turn for t in session.turns if t.is_override_turn]
    if override_turns:
        first_override = min(override_turns)
        post_override_turns = [t for t in session.turns if t.turn >= first_override]
        target_present_post_override = any(
            target in t.fused_ranked_ids or target in t.reranked_ids
            for t in post_override_turns
        )
        target_in_output_post_override = any(
            target in t.output_recommendations for t in post_override_turns
        )
        if target_present_post_override and not target_in_output_post_override:
            return "override_corrupted"

    # 4. retrieved_but_buried: target appears in fused/reranked candidates on
    #    some turn but never lands in the final top-10 output.
    ever_in_fused_or_reranked = any(
        target in t.fused_ranked_ids or target in t.reranked_ids
        for t in session.turns
    )
    ever_in_output = any(target in t.output_recommendations for t in session.turns)
    if ever_in_fused_or_reranked and not ever_in_output:
        return "retrieved_but_buried"

    # 5. clarification_stalled: the pool barely shrank across ask_attribute turns
    stalled_count = 0
    ask_count = 0
    for t in session.turns:
        if t.ask_attribute is not None and t.pool_size_before_ask and t.pool_size_after_ask:
            ask_count += 1
            ratio = t.pool_size_after_ask / t.pool_size_before_ask
            if ratio >= POOL_STALL_RATIO:
                stalled_count += 1
    if ask_count > 0 and stalled_count == ask_count:
        return "clarification_stalled"

    return "unclassified"


def summarize_taxonomy(missed_sessions: list) -> dict:
    """missed_sessions: list[SessionTrace] where hit_turn is None"""
    counts = {
        "filter_excluded": 0,
        "never_retrieved": 0,
        "retrieved_but_buried": 0,
        "clarification_stalled": 0,
        "override_corrupted": 0,
        "unclassified": 0,
    }
    session_labels = {}
    for s in missed_sessions:
        label = classify_miss(s)
        counts[label] += 1
        session_labels[s.session_id] = label
    return {"counts": counts, "session_labels": session_labels, "total_misses": len(missed_sessions)}
