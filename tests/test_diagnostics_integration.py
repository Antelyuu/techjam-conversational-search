from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate
from diagnostics import run_diagnostics as diagnostics_runner
from starter.agent import Agent


CATALOG_ROWS = [
    {
        "parent_asin": "target",
        "title": "Black leather running shoes",
        "categories": ["Shoes"],
        "features": ["lightweight"],
        "price": 40.0,
    },
    {
        "parent_asin": "runner-up",
        "title": "Blue canvas running shoes",
        "categories": ["Shoes"],
        "features": ["breathable"],
        "price": 45.0,
    },
    {
        "parent_asin": "over-budget",
        "title": "Premium semantic running shoes",
        "categories": ["Shoes"],
        "features": ["luxury"],
        "price": 150.0,
    },
]


class FixedRetrievalAgent(Agent):
    def __init__(self, catalog_path: Path, *, strong_margin: bool = False):
        self._strong_margin = strong_margin
        super().__init__(
            catalog_path,
            enable_dense=False,
            enable_clarification=False,
            enable_reranker=True,
        )
        self.dense_search = lambda _query, _limit: [("over-budget", 1.0)]

    def _lexical_search(self, _query_text: str, limit: int):
        second_score = 1.0 if self._strong_margin else 9.99
        return [("target", 10.0), ("runner-up", second_score)][:limit]


class WideRetrievalAgent(Agent):
    def __init__(self, catalog_path: Path):
        super().__init__(
            catalog_path,
            enable_dense=False,
            enable_clarification=False,
            enable_reranker=True,
        )
        self.dense_search = lambda _query, limit: [
            (f"dense-{index}", 1.0 - index / 100.0) for index in range(limit)
        ]

    def _lexical_search(self, _query_text: str, limit: int):
        return [(f"lexical-{index}", 10.0 - index / 100.0) for index in range(limit)]


class DiagnosticsIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.catalog_path = Path(self.directory.name) / "catalog.jsonl"
        self.catalog_path.write_text(
            "".join(json.dumps(row) + "\n" for row in CATALOG_ROWS),
            encoding="utf-8",
        )

    def tearDown(self):
        self.directory.cleanup()

    def test_diagnostic_modules_import_as_a_package(self):
        try:
            metrics = importlib.import_module("diagnostics.metrics")
            taxonomy = importlib.import_module("diagnostics.taxonomy")
        except ModuleNotFoundError as error:
            self.fail(f"diagnostics package imports are broken: {error}")

        self.assertTrue(callable(metrics.recall_at_k_pre_rerank))
        self.assertTrue(callable(taxonomy.classify_miss))

    def test_real_agent_trace_records_each_stage_without_changing_output(self):
        traced = FixedRetrievalAgent(self.catalog_path)
        control = FixedRetrievalAgent(self.catalog_path)
        self.addCleanup(traced.close)
        self.addCleanup(control.close)

        try:
            traced.reset(
                "traced",
                {},
                diagnostic_context={"scenario": "buying", "target_asin": "target"},
            )
        except TypeError as error:
            self.fail(f"agent does not accept diagnostic session context: {error}")
        control.reset("control", {})
        message = "I need running shoes under $50"
        traced_response = traced.respond("traced", message, 1, 2)
        control_response = control.respond("control", message, 1, 2)
        session = traced.finalize_diagnostics("traced", hit_turn=1, hit_rank=1)

        self.assertEqual(traced_response, control_response)
        self.assertEqual(session.session_id, "traced")
        self.assertEqual(session.scenario, "buying")
        self.assertEqual(session.target_asin, "target")
        self.assertEqual(session.hit_turn, 1)
        self.assertEqual(session.hit_rank, 1)
        self.assertEqual(len(session.turns), 1)

        turn = session.turns[0]
        self.assertIn("running shoes", turn.query_text)
        self.assertIsNone(turn.query_embedding)
        self.assertEqual(turn.lexical_ranked_ids, ["target", "runner-up"])
        self.assertEqual(turn.lexical_scores, [10.0, 9.99])
        self.assertTrue(turn.dense_ran)
        self.assertEqual(turn.dense_ranked_ids, ["over-budget"])
        self.assertEqual(turn.dense_scores, [1.0])
        self.assertEqual(
            set(turn.pre_filter_pool), {"target", "runner-up", "over-budget"}
        )
        self.assertEqual(set(turn.post_filter_pool), {"target", "runner-up"})
        self.assertEqual(turn.filter_reason, "over_budget")
        self.assertEqual(set(turn.fused_ranked_ids), {"target", "runner-up"})
        self.assertEqual(turn.reranked_ids, turn.output_recommendations)
        self.assertEqual(turn.ask_attribute, None)
        self.assertEqual(turn.pool_size_before_ask, 2)
        self.assertEqual(
            turn.output_recommendations,
            [item["parent_asin"] for item in traced_response["recommendations"]],
        )

    def test_dense_gate_is_visible_when_semantic_retrieval_is_skipped(self):
        agent = FixedRetrievalAgent(self.catalog_path, strong_margin=True)
        self.addCleanup(agent.close)
        try:
            agent.reset(
                "strong",
                {},
                diagnostic_context={"scenario": "browsing", "target_asin": "target"},
            )
        except TypeError as error:
            self.fail(f"agent does not accept diagnostic session context: {error}")

        agent.respond("strong", "running shoes", 1, 2)
        session = agent.finalize_diagnostics("strong", hit_turn=1, hit_rank=1)

        turn = session.turns[0]
        self.assertFalse(turn.dense_ran)
        self.assertEqual(turn.dense_ranked_ids, [])
        self.assertEqual(turn.dense_scores, [])

    def test_clarification_pool_size_uses_the_full_post_filter_union(self):
        rows = [
            {
                "parent_asin": f"{route}-{index}",
                "title": f"Running shoes {route} {index}",
                "categories": ["Shoes"],
                "price": 40.0,
            }
            for route in ("lexical", "dense")
            for index in range(10)
        ]
        self.catalog_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        agent = WideRetrievalAgent(self.catalog_path)
        self.addCleanup(agent.close)
        agent.reset(
            "wide",
            {},
            diagnostic_context={"scenario": "browsing", "target_asin": "lexical-0"},
        )

        agent.respond("wide", "running shoes", 1, 2)
        session = agent.finalize_diagnostics("wide", hit_turn=1, hit_rank=1)

        turn = session.turns[0]
        self.assertEqual(len(turn.post_filter_pool), 20)
        self.assertEqual(turn.pool_size_before_ask, 20)

    def test_evaluator_collects_completed_real_pipeline_sessions(self):
        agent = FixedRetrievalAgent(self.catalog_path)
        self.addCleanup(agent.close)
        catalog_ids, categories, products = catalog_index(self.catalog_path)
        samples = [
            {
                "sample_id": "diagnostic_1",
                "scenario_type": "buying",
                "user_profile": {},
                "ground_truth": {"parent_asin": "target"},
            }
        ]
        diagnostic_sessions = []

        try:
            result = evaluate(
                agent,
                samples,
                catalog_ids,
                categories,
                products,
                diagnostic_sessions=diagnostic_sessions,
            )
        except TypeError as error:
            self.fail(f"evaluator cannot collect diagnostic sessions: {error}")

        self.assertEqual(result["hit_rate_at_10"], 1.0)
        self.assertEqual(len(diagnostic_sessions), 1)
        self.assertEqual(diagnostic_sessions[0].target_asin, "target")
        self.assertEqual(diagnostic_sessions[0].hit_turn, 1)

    def test_real_diagnostic_runner_uses_the_labeled_pipeline(self):
        dataset_path = Path(self.directory.name) / "public_set.jsonl"
        dataset_path.write_text(
            json.dumps(
                {
                    "sample_id": "diagnostic_1",
                    "scenario_type": "buying",
                    "user_profile": {},
                    "ground_truth": {"parent_asin": "target"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        agent = FixedRetrievalAgent(self.catalog_path)
        self.addCleanup(agent.close)

        try:
            sessions, target_embeddings, score = diagnostics_runner.run_real_dev_set(
                self.catalog_path,
                dataset_path,
                agent=agent,
            )
        except AttributeError as error:
            self.fail(f"real diagnostic runner is unavailable: {error}")

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].target_asin, "target")
        self.assertEqual(target_embeddings, {})
        self.assertEqual(score["recommended_technical_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
