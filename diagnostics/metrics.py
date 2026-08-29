"""
Diagnostic metrics computed over a list of SessionTrace objects.

Each function is independent and documented with what it diagnoses, matching
the table discussed in chat. Run `run_diagnostics.py` to get all of them at
once as a single report.
"""

import math
from typing import Optional
try:
    from .tracer import SessionTrace
except ImportError:  # Support `python3 run_diagnostics.py` from this directory.
    from tracer import SessionTrace


def _cosine(a: list, b: list) -> Optional[float]:
    if a is None or b is None:
        return None
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return None
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Retrieval coverage
# ---------------------------------------------------------------------------

def recall_at_k_pre_rerank(sessions: list, k: int = 50) -> float:
    """
    Of all sessions, what fraction have the target in the fused candidate
    pool (top-k, pre-rerank) on at least one turn?

    This is the CEILING on Hit Rate@10 -- nothing downstream of fusion can
    recover a target that never clears this bar.
    """
    hits = 0
    for s in sessions:
        found = any(s.target_asin in t.fused_ranked_ids[:k] for t in s.turns)
        if found:
            hits += 1
    return hits / len(sessions) if sessions else 0.0


def route_exclusive_recall(sessions: list, k: int = 50) -> dict:
    """
    For sessions where the target IS found somewhere, was it found only by
    lexical, only by dense, or both? Low dense-only counts mean the dense
    route isn't earning its complexity cost.
    """
    lexical_only = dense_only = both = neither = 0
    for s in sessions:
        in_lexical = any(s.target_asin in t.lexical_ranked_ids[:k] for t in s.turns)
        in_dense = any(s.target_asin in t.dense_ranked_ids[:k] for t in s.turns if t.dense_ran)
        if in_lexical and in_dense:
            both += 1
        elif in_lexical:
            lexical_only += 1
        elif in_dense:
            dense_only += 1
        else:
            neither += 1
    total = len(sessions) or 1
    return {
        "lexical_only": lexical_only / total,
        "dense_only": dense_only / total,
        "both": both / total,
        "neither": neither / total,
    }


def candidate_pool_jaccard_overlap(sessions: list, k: int = 50) -> float:
    """
    Average Jaccard overlap between lexical top-k and dense top-k candidate
    sets. High overlap (>0.8) means the two routes are largely redundant --
    fusion isn't buying diversity, just cost.
    """
    overlaps = []
    for s in sessions:
        for t in s.turns:
            if not t.dense_ran or not t.lexical_ranked_ids or not t.dense_ranked_ids:
                continue
            lex_set = set(t.lexical_ranked_ids[:k])
            dense_set = set(t.dense_ranked_ids[:k])
            union = lex_set | dense_set
            if union:
                overlaps.append(len(lex_set & dense_set) / len(union))
    return sum(overlaps) / len(overlaps) if overlaps else 0.0


# ---------------------------------------------------------------------------
# Filter correctness
# ---------------------------------------------------------------------------

def filter_false_negative_rate(sessions: list, satisfies_constraint_fn=None) -> float:
    """
    Fraction of sessions where the target was present pre-filter but absent
    post-filter, AND (if satisfies_constraint_fn is provided) genuinely
    satisfied the disclosed constraint -- i.e. it should NOT have been
    filtered out.

    satisfies_constraint_fn: optional callable(session, turn) -> bool.
    If omitted, this reports the purely mechanical rate (present pre-filter,
    absent post-filter, never recovered later) as an upper-bound proxy.
    """
    flagged = 0
    for s in sessions:
        excluded_ever = False
        for t in s.turns:
            if s.target_asin in t.pre_filter_pool and s.target_asin not in t.post_filter_pool:
                if satisfies_constraint_fn is None or satisfies_constraint_fn(s, t):
                    excluded_ever = True
        if excluded_ever:
            recovered = any(s.target_asin in t.post_filter_pool for t in s.turns)
            if not recovered:
                flagged += 1
    return flagged / len(sessions) if sessions else 0.0


def filter_timing_lag(sessions: list, disclosure_turn_fn) -> float:
    """
    Average number of turns between when a hard constraint is disclosed
    (per disclosure_turn_fn(session) -> Optional[int]) and the first turn
    the post-filter pool actually reflects that constraint (i.e. shrinks
    relative to pre-filter pool). Returns average lag in turns across
    sessions where a disclosure turn is known.

    disclosure_turn_fn: callable(session) -> Optional[int]
    """
    lags = []
    for s in sessions:
        disclosure_turn = disclosure_turn_fn(s)
        if disclosure_turn is None:
            continue
        for t in s.turns:
            if t.turn < disclosure_turn:
                continue
            if len(t.post_filter_pool) < len(t.pre_filter_pool):
                lags.append(t.turn - disclosure_turn)
                break
    return sum(lags) / len(lags) if lags else 0.0


# ---------------------------------------------------------------------------
# Ranking quality
# ---------------------------------------------------------------------------

def rank_shift_retrieval_to_output(sessions: list) -> dict:
    """
    Compares the target's rank in the raw fused list vs its rank in the
    final output, on turns where the target is present in both. Positive
    shift = reranking helped (moved target up); negative = reranking hurt.
    """
    shifts = []
    for s in sessions:
        for t in s.turns:
            if s.target_asin in t.fused_ranked_ids and s.target_asin in t.output_recommendations:
                fused_rank = t.fused_ranked_ids.index(s.target_asin) + 1
                output_rank = t.output_recommendations.index(s.target_asin) + 1
                shifts.append(fused_rank - output_rank)  # positive = improved
    if not shifts:
        return {"mean_shift": None, "n": 0, "worsened_fraction": None}
    worsened = sum(1 for x in shifts if x < 0)
    return {
        "mean_shift": sum(shifts) / len(shifts),
        "n": len(shifts),
        "worsened_fraction": worsened / len(shifts),
    }


def margin_hit_auroc(sessions: list) -> Optional[float]:
    """
    AUROC of BM25 top1/top2 relative margin as a binary predictor of session
    success (hit vs miss). Requires t.lexical_scores populated with at least
    2 scores on the turn the margin gate fires (turn 1 is a reasonable
    default; pass a turn selector if you gate on a different turn).

    A value near 0.5 means the margin carries no real signal -- i.e. your
    confidence gate is likely tuned to noise, directly answering "is this
    gate real or coincidence."
    """
    pairs = []  # (margin, label) label=1 for hit, 0 for miss
    for s in sessions:
        first_turn = s.turns[0] if s.turns else None
        if first_turn is None or len(first_turn.lexical_scores) < 2:
            continue
        top, second = first_turn.lexical_scores[0], first_turn.lexical_scores[1]
        if top == 0:
            continue
        margin = (top - second) / abs(top)
        label = 1 if s.hit_turn is not None else 0
        pairs.append((margin, label))

    if not pairs:
        return None

    pos = [m for m, l in pairs if l == 1]
    neg = [m for m, l in pairs if l == 0]
    if not pos or not neg:
        return None

    # AUROC via pairwise comparison (Mann-Whitney U), lower margin should
    # correlate with LESS confident retrieval; adjust sign if your gate logic
    # treats lower margin as "more confident" as in the Phase 5 report.
    count = 0
    for p in pos:
        for n in neg:
            if p > n:
                count += 1
            elif p == n:
                count += 0.5
    return count / (len(pos) * len(neg))


# ---------------------------------------------------------------------------
# Query construction / dilution
# ---------------------------------------------------------------------------

def query_target_cosine_drift(sessions: list, target_embeddings: dict) -> dict:
    """
    Tracks cosine(query_embedding, target_embedding) turn by turn within a
    session. Returns average per-turn-position similarity across sessions,
    so you can see if it declines as turns accumulate (dilution).

    target_embeddings: dict[asin -> embedding vector], precomputed by you.
    """
    by_turn_position = {}
    for s in sessions:
        target_emb = target_embeddings.get(s.target_asin)
        if target_emb is None:
            continue
        for t in s.turns:
            if t.query_embedding is None:
                continue
            sim = _cosine(t.query_embedding, target_emb)
            if sim is None:
                continue
            by_turn_position.setdefault(t.turn, []).append(sim)
    return {turn: sum(vals) / len(vals) for turn, vals in sorted(by_turn_position.items())}


def query_length_by_turn(sessions: list) -> dict:
    by_turn_position = {}
    for s in sessions:
        for t in s.turns:
            length = len(t.query_text.split())
            by_turn_position.setdefault(t.turn, []).append(length)
    return {turn: sum(vals) / len(vals) for turn, vals in sorted(by_turn_position.items())}


# ---------------------------------------------------------------------------
# Clarification effectiveness
# ---------------------------------------------------------------------------

def pool_narrowing_ratio(sessions: list) -> float:
    """
    Average (pool_size_after_ask / pool_size_before_ask) across all turns
    where an ask_attribute was issued. Close to 1.0 means clarification
    questions aren't shrinking the candidate pool at all.
    """
    ratios = []
    for s in sessions:
        for t in s.turns:
            if t.ask_attribute is not None and t.pool_size_before_ask and t.pool_size_after_ask:
                ratios.append(t.pool_size_after_ask / t.pool_size_before_ask)
    return sum(ratios) / len(ratios) if ratios else None


def wasted_turn_rate(sessions: list) -> float:
    """
    Fraction of ask_attribute turns after which the target's rank in the
    NEXT turn's fused/reranked list did not improve (or target still absent
    both before and after). Cleanest single number for "is asking helping."
    """
    wasted = 0
    total = 0
    for s in sessions:
        for i, t in enumerate(s.turns[:-1]):
            if t.ask_attribute is None:
                continue
            total += 1
            next_t = s.turns[i + 1]
            rank_before = (
                t.reranked_ids.index(s.target_asin) + 1 if s.target_asin in t.reranked_ids else None
            )
            rank_after = (
                next_t.reranked_ids.index(s.target_asin) + 1
                if s.target_asin in next_t.reranked_ids
                else None
            )
            improved = (
                rank_after is not None and (rank_before is None or rank_after < rank_before)
            )
            if not improved:
                wasted += 1
    return wasted / total if total else None
