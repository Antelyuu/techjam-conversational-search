"""P6-T3: the exact coarse category, reconstructed and stated.

Checked against the evaluator's own coarse_category/initial_message, for the
same reason the slot tests are: the generator is the specification, and a
reconstruction that drifts from it fails silently -- every candidate scores
0.0 and the feature simply stops working, with nothing to notice.
"""

from __future__ import annotations

import unittest

from evaluator.local_evaluator import coarse_category as evaluator_coarse_category
from evaluator.local_evaluator import initial_message
from shopping_agent import slots
from shopping_agent.catalog import normalize_product
from shopping_agent.contracts import Candidate
from shopping_agent.orchestrator import ConversationOrchestrator
from shopping_agent.reranking import FEATURE_WEIGHTS, score_candidate


def candidate(asin: str, rank: int = 1) -> Candidate:
    return Candidate(
        parent_asin=asin,
        route_ranks={"lexical": rank},
        route_scores={"lexical": 1.0, "combined": 1.0},
        matched_hard_constraints=(),
        matched_soft_preferences=(),
    )


CATEGORY_CASES = [
    ["Clothing, Shoes & Jewelry", "Women", "Shoes", "Slippers"],
    ["Clothing", "Men", "Shirts"],
    ["Clothing, Shoes & Jewelry"],
    [],
    ["Girls, Dresses"],
    ["Clothing, Shoes & Jewelry", "Novelty & More", "Clothing", "Hats & Caps"],
]


class CoarseCategoryTest(unittest.TestCase):
    def test_it_reproduces_the_generators_category_exactly(self):
        for values in CATEGORY_CASES:
            self.assertEqual(
                slots.coarse_category(values),
                evaluator_coarse_category(values),
                f"drifted on {values!r}",
            )

    def test_a_product_with_no_categories_gets_the_generators_fallback(self):
        self.assertEqual(slots.coarse_category([]), "clothing item")
        self.assertEqual(slots.coarse_category(None), "clothing item")

    def test_odd_shapes_never_raise(self):
        self.assertEqual(slots.coarse_category(""), "clothing item")
        self.assertIsInstance(slots.coarse_category("Shoes, Boots"), str)
        self.assertIsInstance(slots.coarse_category(7), str)

    def test_the_record_carries_it(self):
        record = normalize_product(
            {"parent_asin": "a", "categories": ["Clothing", "Women", "Shoes", "Flats"]}
        )
        self.assertEqual(record.coarse_category, "Shoes Flats")


class StatedCategoryTest(unittest.TestCase):
    def test_it_recovers_the_category_from_every_opener_shape(self):
        """The three shapes the simulator emits, built by its own code."""
        for scenario, constraints in (
            ("browsing", []),
            ("buying", ["100% Cotton"]),
            ("intent_override", []),
        ):
            sample = {
                "scenario_type": scenario,
                "intent_card": {
                    "hard_constraints": constraints or ["x"],
                    "soft_preferences": ["Machine wash cold"],
                },
                "behavior": {
                    "override": {
                        "turn": 3,
                        "old_value": "Department: womens",
                        "new_value": "x",
                        "message": "",
                    }
                },
            }
            message = initial_message(sample, "Shoes Slippers", set())
            self.assertEqual(
                slots.stated_category(message), "Shoes Slippers", f"{scenario}: {message!r}"
            )

    def test_a_category_containing_the_word_but_survives(self):
        self.assertEqual(
            slots.stated_category("I'm looking for Butter Dishes, but I'm still exploring."),
            "Butter Dishes",
        )

    def test_a_message_that_is_not_an_opener_states_nothing(self):
        for message in (
            "For that, what matters is: 100% Cotton.",
            "Actually, ignore my earlier preference. What I need is: wool.",
            "",
        ):
            self.assertIsNone(slots.stated_category(message))

    def test_the_orchestrator_records_it_on_turn_one_only(self):
        orchestrator = ConversationOrchestrator()
        state = orchestrator.reset("s", {})
        orchestrator.process_turn("s", "I'm looking for Shoes Loafers. A key requirement is: wool.", 1, 10)
        self.assertEqual(state.stated_category, "Shoes Loafers")
        orchestrator.process_turn("s", "For that, what matters is: Machine wash cold.", 2, 10)
        self.assertEqual(state.stated_category, "Shoes Loafers")


class CategoryFeatureTest(unittest.TestCase):
    def product(self, asin, categories):
        return normalize_product({"parent_asin": asin, "categories": categories})

    def test_the_exact_category_outranks_a_merely_similar_one(self):
        exact = self.product("a", ["Clothing", "Women", "Shoes", "Slippers"])
        near = self.product("b", ["Clothing", "Women", "Shoes", "Sandals"])
        a = score_candidate(candidate("a"), exact, {}, stated_category="Shoes Slippers")
        b = score_candidate(candidate("b"), near, {}, stated_category="Shoes Slippers")
        self.assertGreater(a.score, b.score)

    def test_no_stated_category_cannot_reorder(self):
        a = score_candidate(candidate("a"), self.product("a", ["Shoes", "Flats"]), {})
        b = score_candidate(candidate("b"), self.product("b", ["Bags", "Totes"]), {})
        by_feature = {c.feature: c.value for c in a.contributions}
        self.assertEqual(by_feature["category_exact"], 0.0)
        self.assertEqual(a.score, b.score)

    def test_a_category_no_product_reproduces_scores_zero_for_everyone(self):
        for asin, cats in (("a", ["Shoes", "Flats"]), ("b", ["Bags", "Totes"])):
            scored = score_candidate(
                candidate(asin), self.product(asin, cats), {},
                stated_category="Something The Catalogue Never Says",
            )
            value = {c.feature: c.value for c in scored.contributions}["category_exact"]
            self.assertEqual(value, 0.0)

    def test_the_feature_is_registered_and_weighted(self):
        self.assertGreater(FEATURE_WEIGHTS["category_exact"], 0.0)
        # The word-overlap category feature it supersedes stays at 0.0, where
        # E4 measured it: retrieval already applies that boost.
        self.assertEqual(FEATURE_WEIGHTS["category"], 0.0)


class CanonicalCategoryTest(unittest.TestCase):
    """E10: the form two spellings of the same shelf agree on.

    This exists to be the *second* test the agent tries, never the first --
    see starter.Agent._resolve_categories. What is pinned here is only that
    it ignores word order and punctuation and nothing else.
    """

    def test_it_ignores_word_order_and_punctuation(self):
        for a, b in (
            ("Shoes Boots", "Boots Shoes"),
            ("Swim One Pieces", "Swim One-Pieces"),
            ("Jackets & Coats Vests", "Coats Jackets & Vests"),
            ("Shoes Athletic Running", "  athletic   running shoes "),
        ):
            self.assertEqual(
                slots.canonical_category(a), slots.canonical_category(b), (a, b)
            )

    def test_it_does_not_collapse_different_shelves(self):
        for a, b in (
            ("Shoes Boots", "Shoes Sandals"),
            ("Socks Athletic Socks", "Shoes Athletic Running"),
            ("Shoes Boots", "Shoes Boot"),
        ):
            self.assertNotEqual(
                slots.canonical_category(a), slots.canonical_category(b), (a, b)
            )

    def test_nothing_stated_is_the_empty_form(self):
        self.assertEqual(slots.canonical_category(""), "")
        self.assertEqual(slots.canonical_category(None), "")


if __name__ == "__main__":
    unittest.main()
