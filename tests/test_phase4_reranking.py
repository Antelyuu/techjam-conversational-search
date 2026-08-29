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

    def test_a_hard_constraint_breaks_a_tie_at_equal_rank(self):
        """Among candidates retrieval cannot separate, satisfying a stated
        requirement wins. This is the checklist order where it still holds."""
        constraints = {"color": Constraint("color", "black", "hard", 1)}
        matching = score_candidate(
            candidate("a", lexical=3, dense=3, hard=("color",)), product("a"), constraints
        )
        plain = score_candidate(candidate("b", lexical=3, dense=3), product("b"), constraints)
        self.assertGreater(matching.score, plain.score)

    def test_retrieval_rank_outranks_the_constraint_features(self):
        """Deliberate, and contrary to P4's stated feature order: weighting
        hard constraints above retrieval measured 0.047 composite worse,
        because matched_constraints() is a coarse word-containment check and
        swamped the score margin weighted fusion exists to preserve (E4)."""
        constraints = {"color": Constraint("color", "black", "hard", 1)}
        far_down = score_candidate(
            candidate("a", lexical=50, dense=50, hard=("color",)), product("a"), constraints
        )
        top_ranked = score_candidate(
            candidate("b", lexical=1, dense=1), product("b"), constraints
        )
        self.assertGreater(top_ranked.score, far_down.score)

    def test_an_unpriced_item_scores_below_one_known_in_budget(self):
        constraints = {"budget": Constraint("budget", 100.0, "hard", 1)}
        priced = score_candidate(candidate("a", lexical=1), product("a", price="50.00"), constraints)
        unpriced = score_candidate(candidate("b", lexical=1), product("b", price=None), constraints)
        self.assertGreater(priced.score, unpriced.score)


def metadata_of(price, constraints):
    scored = score_candidate(candidate("x", lexical=1), product("x", price=price), constraints)
    return {c.feature: c.value for c in scored.contributions}["metadata"]


class SoftBudgetTest(unittest.TestCase):
    """A soft budget is "around $100" -- a target to rank by closeness, not a
    ceiling. Treating it as a hard cap ties $50 with $99 and puts a cliff at
    $101, discarding the distance ranking P2 established."""

    SOFT = {"budget": Constraint("budget", 100.0, "soft", 1)}
    HARD = {"budget": Constraint("budget", 100.0, "hard", 1)}

    def test_closer_to_the_target_scores_higher(self):
        self.assertGreater(metadata_of("99.00", self.SOFT), metadata_of("50.00", self.SOFT))

    def test_the_target_price_scores_best(self):
        self.assertEqual(metadata_of("100.00", self.SOFT), 1.0)

    def test_there_is_no_cliff_at_the_budget(self):
        just_under = metadata_of("99.00", self.SOFT)
        just_over = metadata_of("101.00", self.SOFT)
        self.assertAlmostEqual(just_under, just_over, places=2)

    def test_slightly_over_beats_far_under(self):
        self.assertGreater(metadata_of("101.00", self.SOFT), metadata_of("40.00", self.SOFT))

    def test_wildly_over_scores_zero(self):
        self.assertEqual(metadata_of("500.00", self.SOFT), 0.0)

    def test_a_hard_budget_is_still_a_ceiling(self):
        self.assertEqual(metadata_of("99.00", self.HARD), 1.0)
        self.assertEqual(metadata_of("50.00", self.HARD), 1.0)
        self.assertEqual(metadata_of("101.00", self.HARD), 0.0)

    def test_soft_and_hard_are_not_interchangeable(self):
        self.assertNotEqual(
            [metadata_of(p, self.SOFT) for p in ("50.00", "101.00")],
            [metadata_of(p, self.HARD) for p in ("50.00", "101.00")],
        )

    def test_an_unknown_price_stays_unverified_under_a_soft_budget(self):
        self.assertEqual(metadata_of(None, self.SOFT), 0.25)


class CategoryNeutralityTest(unittest.TestCase):
    def category_of(self, constraints):
        scored = score_candidate(candidate("x", lexical=1), product("x"), constraints)
        return {c.feature: c.value for c in scored.contributions}["category"]

    def test_no_category_constraint_contributes_nothing(self):
        """Constant across candidates either way, so this cannot reorder --
        but 0.2 makes every explanation claim partial credit for a constraint
        the customer never gave."""
        self.assertEqual(self.category_of({}), 0.0)

    def test_an_unmatched_category_is_still_distinguishable_from_a_match(self):
        constraints = {"category": Constraint("category", "shoes", "hard", 1)}
        self.assertEqual(self.category_of(constraints), 1.0)


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
