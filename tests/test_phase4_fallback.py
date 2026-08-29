"""P4-T4: a dense or reranker failure still returns a valid response."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluator.local_evaluator import ALLOWED_ATTRIBUTES, TOP_K, normalize_recommendations
from starter.agent import Agent

CATALOG_ROWS = [
    {
        "parent_asin": f"B{index:09d}",
        "title": f"Black leather running shoes {index}",
        "categories": ["Shoes", "Athletic"],
        "features": ["lightweight", "breathable mesh"],
        "details": {"Material": "leather"},
        "store": "Example Store",
        "description": "Cushioned shoes for running and hiking",
        "price": f"{40 + index}.00",
    }
    for index in range(25)
]


class FallbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._directory = tempfile.TemporaryDirectory()
        cls.catalog_path = Path(cls._directory.name) / "catalog.jsonl"
        cls.catalog_path.write_text(
            "".join(json.dumps(row) + "\n" for row in CATALOG_ROWS), encoding="utf-8"
        )
        cls.catalog_ids = {row["parent_asin"] for row in CATALOG_ROWS}

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def agent(self, **kwargs):
        kwargs.setdefault("enable_dense", False)
        return Agent(self.catalog_path, **kwargs)

    def assert_contract_valid(self, response):
        self.assertIsInstance(response, dict)
        self.assertIsInstance(response["message"], str)
        self.assertIsInstance(response["recommendations"], list)
        ask = response["ask_attribute"]
        self.assertTrue(ask is None or ask in ALLOWED_ATTRIBUTES, f"illegal attribute {ask!r}")
        ranked = normalize_recommendations(response["recommendations"], self.catalog_ids)
        self.assertLessEqual(len(ranked), TOP_K)
        self.assertEqual(len(ranked), len(set(ranked)), "duplicate parent_asin emitted")

    def converse(self, agent, messages):
        agent.reset("s", {})
        responses = []
        for turn, message in enumerate(messages, start=1):
            response = agent.respond("s", message, turn, TOP_K)
            self.assert_contract_valid(response)
            responses.append(response)
        return responses

    def test_a_reranker_failure_degrades_to_the_fused_order(self):
        agent = self.agent(enable_reranker=True)
        with patch("starter.agent.rerank", side_effect=RuntimeError("boom")):
            responses = self.converse(agent, ["I'm looking for running shoes"])
        self.assertTrue(responses[0]["recommendations"], "fell back to an empty list")

    def test_a_reranker_failure_is_reported_once(self):
        agent = self.agent(enable_reranker=True)
        with patch("starter.agent.rerank", side_effect=RuntimeError("boom")):
            with patch("sys.stderr") as stderr:
                self.converse(agent, ["running shoes"] * 3)
        written = "".join(call.args[0] for call in stderr.write.call_args_list)
        self.assertEqual(written.count("reranker failed"), 1)

    def test_a_dense_failure_degrades_to_lexical(self):
        agent = self.agent(enable_dense=False)
        agent.dense_search = lambda _query, _limit: (_ for _ in ()).throw(RuntimeError("boom"))
        # The guard lives in dense_retrieval; an unguarded callable must still
        # not take the whole turn down.
        try:
            responses = self.converse(agent, ["running shoes"])
        except RuntimeError:
            self.fail("a dense route failure escaped respond()")
        self.assertTrue(responses[0]["recommendations"])

    def test_a_full_conversation_stays_contract_valid(self):
        agent = self.agent(enable_reranker=True)
        self.converse(
            agent,
            [
                "I'm looking for running shoes, but I'm still exploring.",
                "For that, what matters is: breathable mesh; lightweight.",
                "I don't have an additional preference for color.",
                "Actually, ignore my earlier preference. What I need is: leather.",
                "Those options are not quite right yet. Ask me about one specific attribute.",
            ]
            + ["Those options are not quite right yet."] * 5,
        )

    def test_questions_stop_rather_than_repeat(self):
        agent = self.agent(enable_reranker=True)
        responses = self.converse(agent, ["I'm looking for shoes"] * 10)
        asked = [r["ask_attribute"] for r in responses if r["ask_attribute"] is not None]
        self.assertEqual(len(asked), len(set(asked)), f"repeated a question: {asked}")

    def test_an_empty_query_still_returns_a_valid_response(self):
        agent = self.agent(enable_reranker=True)
        self.converse(agent, ["", "   "])


if __name__ == "__main__":
    unittest.main()
