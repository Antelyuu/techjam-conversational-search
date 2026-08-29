import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent


class FixedLexicalAgent(Agent):
    def __init__(self, catalog_path, lexical_hits):
        self._fixed_lexical_hits = lexical_hits
        super().__init__(
            catalog_path,
            enable_dense=False,
            enable_clarification=False,
            enable_reranker=False,
        )

    def _lexical_search(self, _query_text, limit):
        return self._fixed_lexical_hits[:limit]


class DenseGateTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.catalog_path = Path(self.directory.name) / "catalog.jsonl"
        products = [
            {"parent_asin": "lexical-top", "title": "Leather boots"},
            {"parent_asin": "lexical-runner-up", "title": "Canvas boots"},
            {"parent_asin": "dense-only", "title": "Semantic alternative"},
        ]
        self.catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )

    def tearDown(self):
        self.directory.cleanup()

    def recommendations(self, lexical_hits):
        agent = FixedLexicalAgent(self.catalog_path, lexical_hits)
        agent.dense_search = lambda _query, _limit: [("dense-only", 1.0)]
        try:
            agent.reset("demo", {})
            response = agent.respond("demo", "boots", 1, 2)
            return [item["parent_asin"] for item in response["recommendations"]]
        finally:
            agent.close()

    def test_confident_lexical_margin_skips_the_dense_route(self):
        recommendations = self.recommendations(
            [("lexical-top", 10.0), ("lexical-runner-up", 1.0)]
        )

        self.assertEqual(recommendations, ["lexical-top", "lexical-runner-up"])

    def test_weak_lexical_margin_allows_dense_retrieval(self):
        recommendations = self.recommendations(
            [("lexical-top", 10.0), ("lexical-runner-up", 9.99)]
        )

        self.assertIn("dense-only", recommendations)

    def test_small_but_clear_lexical_margin_skips_dense_retrieval(self):
        recommendations = self.recommendations(
            [("lexical-top", 10.0), ("lexical-runner-up", 9.925)]
        )

        self.assertEqual(recommendations, ["lexical-top", "lexical-runner-up"])


if __name__ == "__main__":
    unittest.main()
