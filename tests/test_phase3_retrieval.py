import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shopping_agent.catalog import normalize_product
from shopping_agent.contracts import SearchRequest, SessionState
from shopping_agent.dense_retrieval import DenseRouteUnavailable, load_dense_retriever
from shopping_agent.embedding_config import ids_path, vectors_path
from shopping_agent.retrieval import FUSION_RRF, FUSION_WEIGHTED, retrieve


class Phase3RetrievalTest(unittest.TestCase):
    def product(self, parent_asin):
        return normalize_product(
            {
                "parent_asin": parent_asin,
                "title": "Black running shoes",
                "categories": ["Shoes"],
                "features": ["lightweight"],
                "details": {},
                "store": "Example Store",
                "description": "Shoes for running",
                "price": "99.99",
            }
        )

    def request(self, top_k=3):
        return SearchRequest("running shoes", SessionState("demo", {}), top_k)

    def test_candidate_union_preserves_route_ranks_and_scores(self):
        products = {asin: self.product(asin) for asin in ("lexical", "dense", "shared")}

        candidates = retrieve(
            self.request(),
            3,
            lambda _query, _limit: [("lexical", 9.0), ("shared", 8.0)],
            products,
            dense_search=lambda _query, _limit: [("dense", 0.9), ("shared", 0.8)],
            fusion_method=FUSION_RRF,
        )

        by_id = {candidate.parent_asin: candidate for candidate in candidates}
        self.assertEqual(by_id["shared"].route_ranks, {"lexical": 2, "dense": 2})
        self.assertEqual(by_id["shared"].route_scores["lexical"], 8.0)
        self.assertEqual(by_id["shared"].route_scores["dense"], 0.8)

    def test_duplicate_route_hits_keep_the_first_rank_and_score(self):
        products = {"shared": self.product("shared")}

        candidates = retrieve(
            self.request(1),
            1,
            lambda _query, _limit: [("shared", 9.0), ("shared", 1.0)],
            products,
        )

        self.assertEqual(candidates[0].route_ranks, {"lexical": 1})
        self.assertEqual(candidates[0].route_scores["lexical"], 9.0)

    def test_weighted_fusion_respects_the_route_weights(self):
        # Each route needs two candidates with distinct scores, otherwise every
        # candidate normalizes to the same value and the assertion would pass
        # on union tie-order no matter which weights were supplied.
        products = {asin: self.product(asin) for asin in ("lex_top", "lex_low", "dense_top", "dense_low")}
        lexical = lambda _query, _limit: [("lex_top", 9.0), ("lex_low", 1.0)]
        dense = lambda _query, _limit: [("dense_top", 0.9), ("dense_low", 0.1)]

        def top_of(weights):
            return retrieve(
                self.request(4), 4, lexical, products,
                dense_search=dense, fusion_method=FUSION_WEIGHTED, route_weights=weights,
            )[0].parent_asin

        self.assertEqual(top_of({"lexical": 1.0, "dense": 0.0}), "lex_top")
        self.assertEqual(top_of({"lexical": 0.0, "dense": 1.0}), "dense_top")

    @unittest.skipUnless(
        importlib.util.find_spec("numpy"), "numpy is only needed for the dense route"
    )
    def test_mismatched_catalogue_artifact_is_rejected(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            embedding_dir = Path(directory)
            np.save(vectors_path(embedding_dir), np.zeros((2, 4), dtype=np.float32))
            ids_path(embedding_dir).write_text(json.dumps(["asin-1", "asin-2"]))

            with self.assertRaises(DenseRouteUnavailable):
                load_dense_retriever(
                    embedding_dir=embedding_dir,
                    expected_ids=["asin-1", "asin-other"],
                    strict=True,
                )

    def test_model_initialization_failure_falls_back_to_lexical(self):
        with patch(
            "shopping_agent.dense_retrieval.DenseRetriever",
            side_effect=RuntimeError("model unavailable"),
        ):
            dense_search = load_dense_retriever()

        self.assertIsNone(dense_search)


if __name__ == "__main__":
    unittest.main()
