"""
Pipeline instrumentation for the TechJam shopping-copilot agent.

Wire calls to `PipelineTracer` into your existing pipeline at the stage
boundaries described in `techjam-customer-input-pipeline.md`:

    step 3  build cumulative query        -> tracer.log_query(...)
    step 4  lexical / dense retrieval     -> tracer.log_retrieval(...)
    step 5  filter + fuse                 -> tracer.log_filter(...), tracer.log_fusion(...)
    step 6  rerank (Phase 4/5)            -> tracer.log_rerank(...)
    step 7  clarification choice          -> tracer.log_clarification(...)
    (end of turn)                         -> tracer.end_turn(...)

None of this changes what your Agent returns. It only records what happened
internally so `metrics.py` and `taxonomy.py` can diagnose it after the run.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TurnTrace:
    turn: int
    query_text: str = ""
    query_embedding: Optional[list] = None  # dense query vector, if used this turn

    lexical_ranked_ids: list = field(default_factory=list)   # BM25 top-N, in rank order
    lexical_scores: list = field(default_factory=list)       # matching raw scores

    dense_ranked_ids: list = field(default_factory=list)     # dense top-N, in rank order
    dense_scores: list = field(default_factory=list)
    dense_ran: bool = False                                  # False if gated off / unavailable

    pre_filter_pool: list = field(default_factory=list)      # candidate ids before hard filter
    post_filter_pool: list = field(default_factory=list)     # candidate ids after hard filter
    filter_reason: Optional[str] = None                      # e.g. "hard_budget", "category"

    fused_ranked_ids: list = field(default_factory=list)     # after BM25+dense fusion, pre-rerank

    reranked_ids: list = field(default_factory=list)         # after Phase 4/5 scoring
    reranked_scores: list = field(default_factory=list)

    ask_attribute: Optional[str] = None
    pool_size_before_ask: Optional[int] = None
    pool_size_after_ask: Optional[int] = None                # i.e. pool size on the NEXT turn,
                                                               # filled in retroactively by end_turn

    output_recommendations: list = field(default_factory=list)  # final top_k returned this turn
    is_override_turn: bool = False


@dataclass
class SessionTrace:
    session_id: str
    scenario: str  # "buying" | "browsing" | "override" | "boundary"
    target_asin: str
    turns: list = field(default_factory=list)  # list[TurnTrace]
    hit_turn: Optional[int] = None    # turn number of first valid hit, else None
    hit_rank: Optional[int] = None    # 1-indexed rank of target within output_recommendations


class PipelineTracer:
    """
    One tracer instance per session. Call session-scoped methods in order as
    your pipeline runs each turn, then call `end_turn()` before returning
    control to the evaluator, and `finalize(hit_turn, hit_rank)` once the
    session ends (hit or turn 10).
    """

    def __init__(self, session_id: str, scenario: str, target_asin: str):
        self.session = SessionTrace(session_id=session_id, scenario=scenario, target_asin=target_asin)
        self._current: Optional[TurnTrace] = None

    def start_turn(self, turn: int, is_override_turn: bool = False) -> None:
        self._current = TurnTrace(turn=turn, is_override_turn=is_override_turn)

    def log_query(self, query_text: str, query_embedding: Optional[list] = None) -> None:
        self._current.query_text = query_text
        self._current.query_embedding = query_embedding

    def log_retrieval(self, route: str, ranked_ids: list, scores: list, ran: bool = True) -> None:
        """route: 'lexical' or 'dense'"""
        if route == "lexical":
            self._current.lexical_ranked_ids = list(ranked_ids)
            self._current.lexical_scores = list(scores)
        elif route == "dense":
            self._current.dense_ranked_ids = list(ranked_ids)
            self._current.dense_scores = list(scores)
            self._current.dense_ran = ran
        else:
            raise ValueError(f"unknown route: {route}")

    def log_filter(self, pre_filter_pool: list, post_filter_pool: list, reason: Optional[str] = None) -> None:
        self._current.pre_filter_pool = list(pre_filter_pool)
        self._current.post_filter_pool = list(post_filter_pool)
        self._current.filter_reason = reason

    def log_fusion(self, fused_ranked_ids: list) -> None:
        self._current.fused_ranked_ids = list(fused_ranked_ids)

    def log_rerank(self, reranked_ids: list, reranked_scores: list) -> None:
        self._current.reranked_ids = list(reranked_ids)
        self._current.reranked_scores = list(reranked_scores)

    def log_clarification(self, ask_attribute: Optional[str], pool_size_before_ask: int) -> None:
        self._current.ask_attribute = ask_attribute
        self._current.pool_size_before_ask = pool_size_before_ask

    def current_post_filter_pool_size(self) -> int:
        return len(self._current.post_filter_pool)

    def end_turn(self, output_recommendations: list) -> None:
        self._current.output_recommendations = list(output_recommendations)
        # retroactively fill in pool-narrowing denominator on the previous turn
        if len(self.session.turns) > 0:
            prev = self.session.turns[-1]
            if prev.ask_attribute is not None and prev.pool_size_after_ask is None:
                prev.pool_size_after_ask = len(self._current.post_filter_pool) or len(
                    self._current.fused_ranked_ids
                )
        self.session.turns.append(self._current)
        self._current = None

    def finalize(self, hit_turn: Optional[int], hit_rank: Optional[int]) -> SessionTrace:
        self.session.hit_turn = hit_turn
        self.session.hit_rank = hit_rank
        return self.session
