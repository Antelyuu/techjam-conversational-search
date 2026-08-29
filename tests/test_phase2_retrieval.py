import unittest

from shopping_agent.catalog import normalize_product
from shopping_agent.contracts import Constraint, SearchRequest, SessionState
from shopping_agent.filtering import evaluate_candidate, evaluate_price, score_category
from shopping_agent.retrieval import retrieve


class Phase2RetrievalTest(unittest.TestCase):
    def product(self, **overrides):
        raw = {
            "parent_asin": "asin-1",
            "title": "Black running shoes",
            "categories": ["Shoes", "Clothing, Shoes & Jewelry"],
            "features": ["lightweight", "rubber sole"],
            "details": {"material": "mesh"},
            "store": "Example Store",
            "description": "Shoes for running",
            "price": "99.99",
        }
        raw.update(overrides)
        return normalize_product(raw)

    def budget_constraint(self, value=120.0):
        return Constraint("budget", value, "hard", 1)

    def soft_budget_constraint(self, value=120.0):
        return Constraint("budget", value, "soft", 1)

    def category_constraint(self, value="shoes"):
        return Constraint("category", value, "hard", 1)

    def test_normalization_handles_missing_and_mixed_fields(self):
        product = normalize_product({
            "parent_asin": 123,
            "title": None,
            "categories": "Shoes, Running",
            "features": {"waterproof": True},
            "details": 7,
            "price": "not listed",
        })

        self.assertEqual(product.parent_asin, "123")
        self.assertEqual(product.title, "")
        self.assertEqual(product.categories, ("shoes", "running"))
        self.assertIsNone(product.price)
        self.assertIn("waterproof True", product.searchable_text)
        self.assertIn("7", product.searchable_text)

    def test_known_over_budget_product_is_excluded(self):
        retained, reason = evaluate_price(self.product(price=150), 120)

        self.assertFalse(retained)
        self.assertEqual(reason, "over_budget")

    def test_known_within_budget_product_is_retained(self):
        retained, reason = evaluate_price(self.product(price=120), 120)

        self.assertTrue(retained)
        self.assertEqual(reason, "within_budget")

    def test_missing_price_is_retained_as_unverified(self):
        retained, reason = evaluate_price(self.product(price=None), 120)

        self.assertTrue(retained)
        self.assertEqual(reason, "budget_unverified")

    def test_soft_budget_does_not_exclude_over_preference_product(self):
        outcome = evaluate_candidate(
            self.product(price=150),
            {"budget": self.soft_budget_constraint()},
        )

        self.assertTrue(outcome.retained)

    def test_soft_budget_prefers_price_closer_to_requested_amount(self):
        state = SessionState(
            session_id="demo",
            user_profile={},
            constraints={"budget": self.soft_budget_constraint()},
        )
        request = SearchRequest("running shoes around 120", state, 2)
        products = {
            "near": self.product(parent_asin="near", price=125),
            "far": self.product(parent_asin="far", price=300),
        }

        candidates = retrieve(
            request,
            2,
            lambda _query, _limit: [("near", 1.0), ("far", 1.0)],
            products,
        )

        self.assertEqual([candidate.parent_asin for candidate in candidates], ["near", "far"])

    def test_category_match_gets_boost(self):
        boost, reason = score_category(self.product(), "shoes")

        self.assertGreater(boost, 0)
        self.assertEqual(reason, "category_match")

    def test_uncertain_category_does_not_exclude_product(self):
        product = self.product(
            title="Generic product",
            categories=["Clothing, Shoes & Jewelry"],
        )

        boost, reason = score_category(product, "shoes")

        self.assertEqual(boost, 0.0)
        self.assertEqual(reason, "category_unverified")

    def test_specific_category_mismatch_is_only_soft_penalty(self):
        product = self.product(title="Gold necklace", categories=["Jewelry"])

        boost, reason = score_category(product, "shoes")

        self.assertLess(boost, 0)
        self.assertEqual(reason, "category_soft_mismatch")

    def test_candidate_audit_reason_combines_budget_and_category(self):
        outcome = evaluate_candidate(
            self.product(price=None),
            {"budget": self.budget_constraint(), "category": self.category_constraint()},
        )

        self.assertTrue(outcome.retained)
        self.assertEqual(outcome.reason, "budget_unverified+category_match")

    def test_retrieve_uses_cumulative_request_and_returns_candidate_contract(self):
        state = SessionState(session_id="demo", user_profile={})
        state.constraints = {
            "budget": self.budget_constraint(),
            "category": self.category_constraint(),
        }
        request = SearchRequest("black running shoes under 120", state, 2)
        calls = []

        def lexical_search(query, limit):
            calls.append((query, limit))
            return [("asin-1", 1.0)]

        candidates = retrieve(request, 2, lexical_search, {"asin-1": self.product()})

        self.assertEqual(calls, [("black running shoes under 120", 10)])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].parent_asin, "asin-1")
        self.assertEqual(candidates[0].route_ranks, {"lexical": 1})
        # Both of this session's constraints are scored by their own feature
        # downstream (score_category, evaluate_price), so neither is reported
        # here as well -- reporting them once and scoring them twice is what
        # inflated category to 1.5 of the reranker's scale.
        self.assertEqual(candidates[0].matched_hard_constraints, ())


if __name__ == "__main__":
    unittest.main()
