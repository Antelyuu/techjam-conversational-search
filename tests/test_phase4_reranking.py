import unittest

from shopping_agent.catalog import normalize_product
from shopping_agent.contracts import Candidate, Constraint
from shopping_agent.reranking import FEATURE_WEIGHTS, rerank, score_candidate


def product(parent_asin, **overrides):
    raw = {
        "parent_asin": parent_asin,
        "title": "Running shoes",
        "categories": ["Shoes"],
        "features": ["lightweight"],
        "details": {},
        "store": "Example",
        "description": "Shoes for running",
        "price": "50.00",
    }
    raw.update(overrides)
    return normalize_product(raw)


def candidate(parent_asin, lexical=None, dense=None, hard=(), soft=()):
    ranks, scores = {}, {}
    if lexical is not None:
        ranks["lexical"], scores["lexical"] = lexical, 1.0
    if dense is not None:
        ranks["dense"], scores["dense"] = dense, 1.0
    scores["combined"] = 0.0
    return Candidate(
        parent_asin=parent_asin,
        route_ranks=ranks,
        route_scores=scores,
        matched_hard_constraints=hard,
        matched_soft_preferences=soft,
    )


class ScoreCandidateTest(unittest.TestCase):
    def test_every_feature_is_reported(self):
        """P4-T1's acceptance: score contributions inspectable per candidate."""
        scored = score_candidate(candidate("a", lexical=1), product("a"), {})
        self.assertEqual(
            [c.feature for c in scored.contributions], list(FEATURE_WEIGHTS)
        )

    def test_the_score_is_the_sum_of_its_contributions(self):
        scored = score_candidate(candidate("a", lexical=1, dense=2), product("a"), {})
        self.assertAlmostEqual(
            scored.score, sum(c.contribution for c in scored.contributions)
        )

    def test_explain_names_every_feature(self):
        text = score_candidate(candidate("a", lexical=1), product("a"), {}).explain()
        for feature in FEATURE_WEIGHTS:
            self.assertIn(feature, text)

    def test_a_route_that_missed_the_candidate_contributes_nothing(self):
        scored = score_candidate(candidate("a", lexical=1), product("a"), {})
        by_feature = {c.feature: c for c in scored.contributions}
        self.assertEqual(by_feature["dense_rank"].value, 0.0)
        self.assertGreater(by_feature["lexical_rank"].value, 0.0)

    def test_a_better_rank_scores_higher(self):
        first = score_candidate(candidate("a", lexical=1), product("a"), {})
        tenth = score_candidate(candidate("b", lexical=10), product("b"), {})
        self.assertGreater(first.score, tenth.score)

    def test_a_hard_constraint_outweighs_retrieval_rank(self):
        """The feature order is meaningful only if a higher feature cannot be
        overturned by everything below it."""
        constraints = {"color": Constraint("color", "black", "hard", 1)}
        matching = score_candidate(
            candidate("a", lexical=50, dense=50, hard=("color",)), product("a"), constraints
        )
        top_ranked = score_candidate(
            candidate("b", lexical=1, dense=1), product("b"), constraints
        )
        self.assertGreater(matching.score, top_ranked.score)

    def test_an_unpriced_item_scores_below_one_known_in_budget(self):
        constraints = {"budget": Constraint("budget", 100.0, "hard", 1)}
        priced = score_candidate(candidate("a", lexical=1), product("a", price="50.00"), constraints)
        unpriced = score_candidate(candidate("b", lexical=1), product("b", price=None), constraints)
        self.assertGreater(priced.score, unpriced.score)


class RerankTest(unittest.TestCase):
    def test_returns_at_most_the_limit(self):
        products = {f"p{i}": product(f"p{i}") for i in range(20)}
        candidates = [candidate(f"p{i}", lexical=i + 1) for i in range(20)]
        self.assertEqual(len(rerank(candidates, products, {}, 10)), 10)

    def test_orders_by_descending_score(self):
        products = {f"p{i}": product(f"p{i}") for i in range(5)}
        candidates = [candidate(f"p{i}", lexical=i + 1) for i in range(5)]
        scores = [item.score for item in rerank(candidates, products, {}, 5)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_a_candidate_missing_from_the_catalogue_is_dropped(self):
        result = rerank([candidate("ghost", lexical=1)], {}, {}, 10)
        self.assertEqual(result, [])

    def test_ties_keep_retrieval_order(self):
        products = {"a": product("a"), "b": product("b")}
        candidates = [candidate("a", lexical=1), candidate("b", lexical=1)]
        self.assertEqual(
            [item.parent_asin for item in rerank(candidates, products, {}, 2)], ["a", "b"]
        )


if __name__ == "__main__":
    unittest.main()
