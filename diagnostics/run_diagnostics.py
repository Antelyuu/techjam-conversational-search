"""
End-to-end demo of the diagnostics harness against a SYNTHETIC pipeline, so
you can confirm the harness itself works before wiring it into your real
Agent. Replace `fake_pipeline_turn()` with calls into your actual
lexical/dense/filter/fuse/rerank/clarify code -- the TurnTrace shape and
tracer calls are exactly what your real pipeline should populate.

Run:  python3 run_diagnostics.py
"""

import argparse
import random
import sys
from pathlib import Path
try:
    from .tracer import PipelineTracer
    from .taxonomy import summarize_taxonomy
    from .metrics import (
        recall_at_k_pre_rerank,
        route_exclusive_recall,
        candidate_pool_jaccard_overlap,
        filter_false_negative_rate,
        rank_shift_retrieval_to_output,
        margin_hit_auroc,
        query_target_cosine_drift,
        query_length_by_turn,
        pool_narrowing_ratio,
        wasted_turn_rate,
    )
except ImportError:  # Support `python3 run_diagnostics.py` from this directory.
    from tracer import PipelineTracer
    from taxonomy import summarize_taxonomy
    from metrics import (
        recall_at_k_pre_rerank,
        route_exclusive_recall,
        candidate_pool_jaccard_overlap,
        filter_false_negative_rate,
        rank_shift_retrieval_to_output,
        margin_hit_auroc,
        query_target_cosine_drift,
        query_length_by_turn,
        pool_narrowing_ratio,
        wasted_turn_rate,
    )

random.seed(7)


def fake_catalog(n=200):
    return [f"B{i:05d}" for i in range(n)]


def fake_embedding(seed_str: str, dim: int = 16) -> list:
    rng = random.Random(seed_str)
    return [rng.uniform(-1, 1) for _ in range(dim)]


def fake_pipeline_turn(tracer: PipelineTracer, turn: int, catalog: list, target: str,
                        scenario: str, session_difficulty: float) -> tuple:
    """
    Simulates one turn of retrieval -> filter -> fuse -> rerank -> clarify.
    session_difficulty in [0,1]: higher = target harder to surface (models
    a harder Browsing-style session vs an easy Buying-style session).
    """
    is_override = scenario == "override" and turn == 3
    tracer.start_turn(turn, is_override_turn=is_override)

    query_text = f"turn {turn} query about {target}"
    query_emb = fake_embedding(query_text)
    tracer.log_query(query_text, query_emb)

    # lexical route: target appears with probability depending on difficulty
    lexical_ids = random.sample(catalog, k=30)
    if random.random() > session_difficulty and target not in lexical_ids:
        lexical_ids[0] = target
    lexical_scores = sorted([random.uniform(0, 10) for _ in lexical_ids], reverse=True)
    tracer.log_retrieval("lexical", lexical_ids, lexical_scores)

    # dense route: runs most turns, weaker on hard sessions
    dense_ran = random.random() > 0.1
    dense_ids = random.sample(catalog, k=30)
    if dense_ran and random.random() > (session_difficulty + 0.2) and target not in dense_ids:
        dense_ids[0] = target
    dense_scores = sorted([random.uniform(0, 1) for _ in dense_ids], reverse=True)
    tracer.log_retrieval("dense", dense_ids, dense_scores, ran=dense_ran)

    pre_filter_pool = list(set(lexical_ids) | (set(dense_ids) if dense_ran else set()))
    # simulate an occasional filter bug: 5% chance target gets wrongly excluded
    post_filter_pool = list(pre_filter_pool)
    filter_reason = None
    if scenario == "buying" and target in post_filter_pool and random.random() < 0.05:
        post_filter_pool.remove(target)
        filter_reason = "hard_budget"
    tracer.log_filter(pre_filter_pool, post_filter_pool, filter_reason)

    fused = list(post_filter_pool)
    random.shuffle(fused)
    if target in fused:
        fused.remove(target)
        insert_at = int(len(fused) * session_difficulty * 0.5)
        fused.insert(insert_at, target)
    tracer.log_fusion(fused)

    reranked = list(fused)
    reranked_scores = [random.uniform(0, 1) for _ in reranked]
    tracer.log_rerank(reranked, reranked_scores)

    pool_before = len(post_filter_pool)
    ask_attr = "color" if (turn < 4 and random.random() < 0.4) else None
    tracer.log_clarification(ask_attr, pool_before)

    output_recs = reranked[:10]
    tracer.end_turn(output_recs)

    hit = target in output_recs
    return hit, (output_recs.index(target) + 1 if hit else None)


def run_synthetic_dev_set(n_sessions=200):
    catalog = fake_catalog()
    scenarios = (["buying"] * 80 + ["browsing"] * 80 + ["override"] * 30 + ["boundary"] * 10)
    random.shuffle(scenarios)

    sessions = []
    target_embeddings = {}

    for i in range(n_sessions):
        scenario = scenarios[i]
        target = random.choice(catalog)
        target_embeddings[target] = fake_embedding(target)
        difficulty = {"buying": 0.2, "browsing": 0.6, "override": 0.5, "boundary": 0.4}[scenario]

        tracer = PipelineTracer(session_id=f"sess_{i}", scenario=scenario, target_asin=target)
        hit_turn, hit_rank = None, None
        for turn in range(1, 11):
            hit, rank = fake_pipeline_turn(tracer, turn, catalog, target, scenario, difficulty)
            if hit:
                hit_turn, hit_rank = turn, rank
                break
        session = tracer.finalize(hit_turn, hit_rank)
        sessions.append(session)

    return sessions, target_embeddings


def run_real_dev_set(
    catalog_path="data/catalog.jsonl",
    dataset_path="data/public_set.jsonl",
    *,
    agent=None,
):
    """Run the real Agent/evaluator path while collecting labeled traces."""
    repository_root = Path(__file__).resolve().parent.parent
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

    from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
    from starter.agent import Agent

    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)
    owns_agent = agent is None
    pipeline_agent = agent or Agent(catalog_path)
    sessions = []
    try:
        score = evaluate(
            pipeline_agent,
            samples,
            catalog_ids,
            categories,
            products,
            diagnostic_sessions=sessions,
        )
    finally:
        if owns_agent:
            pipeline_agent.close()
    return sessions, {}, score


def print_report(sessions, target_embeddings, dataset_label="SYNTHETIC"):
    hits = [s for s in sessions if s.hit_turn is not None]
    misses = [s for s in sessions if s.hit_turn is None]

    print("=" * 70)
    print("TECHJAM DIAGNOSTIC REPORT")
    print("=" * 70)
    print(f"\nSessions: {len(sessions)}  |  Hits: {len(hits)}  |  Misses: {len(misses)}")
    print(f"HitRate@10 (from this run): {len(hits) / len(sessions):.4f}")

    print("\n--- Retrieval coverage ---")
    print(f"Recall@50 pre-rerank (ceiling on HitRate@10): {recall_at_k_pre_rerank(sessions):.4f}")
    print(f"Route-exclusive recall: {route_exclusive_recall(sessions)}")
    print(f"Candidate pool Jaccard overlap (lexical vs dense): {candidate_pool_jaccard_overlap(sessions):.4f}")

    print("\n--- Filter correctness ---")
    print(f"Filter false-negative rate (mechanical proxy): {filter_false_negative_rate(sessions):.4f}")

    print("\n--- Ranking quality ---")
    print(f"Rank shift, fused->output: {rank_shift_retrieval_to_output(sessions)}")
    auroc = margin_hit_auroc(sessions)
    print(f"BM25 margin vs hit AUROC: {auroc:.4f}" if auroc is not None else "BM25 margin vs hit AUROC: n/a")

    print("\n--- Query construction ---")
    print(f"Query-target cosine drift by turn: {query_target_cosine_drift(sessions, target_embeddings)}")
    print(f"Query length by turn: {query_length_by_turn(sessions)}")

    print("\n--- Clarification effectiveness ---")
    pnr = pool_narrowing_ratio(sessions)
    print(f"Pool-narrowing ratio (lower=better): {pnr:.4f}" if pnr is not None else "n/a")
    wtr = wasted_turn_rate(sessions)
    print(f"Wasted-turn rate: {wtr:.4f}" if wtr is not None else "n/a")

    print("\n--- Miss taxonomy ---")
    taxonomy = summarize_taxonomy(misses)
    print(f"Total misses: {taxonomy['total_misses']}")
    for label, count in taxonomy["counts"].items():
        print(f"  {label}: {count}")

    print("\n" + "=" * 70)
    if dataset_label == "SYNTHETIC":
        print("This was run on SYNTHETIC data to verify the harness works.")
        print("Use --real to run the instrumented Agent on the public dev set.")
    else:
        print(f"This was run on {dataset_label} through the real Agent pipeline.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TechJam diagnostics harness")
    parser.add_argument("--real", action="store_true", help="run the real public dev pipeline")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    args = parser.parse_args()
    if args.real:
        sessions, target_embeddings, score = run_real_dev_set(args.catalog, args.dataset)
        print_report(sessions, target_embeddings, dataset_label="REAL PUBLIC DEV DATA")
        print(f"Technical score: {score['recommended_technical_score']:.6f}")
    else:
        sessions, target_embeddings = run_synthetic_dev_set(n_sessions=200)
        print_report(sessions, target_embeddings)
