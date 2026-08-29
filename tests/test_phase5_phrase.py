import unittest

from shopping_agent.catalog import normalize_product
from shopping_agent.contracts import Candidate
from shopping_agent.evidence import (
    disclosure_phrases,
    phrase_coverage_from,
    phrase_text,
)
from shopping_agent.reranking import FEATURE_WEIGHTS, score_candidate


def product(parent_asin, **overrides):
    raw = {
        "parent_asin": parent_asin,
        "title": "Running shoes",
        "categories": ["Shoes"],
        "features": ["Rubber sole", "Machine wash cold with like colors"],
        "details": {"Material": "100% Cotton canvas upper"},
        "store": "Example",
        "description": "Shoes for running",
        "price": "50.00",
    }
    raw.update(overrides)
    return normalize_product(raw)


def phrase_value(target, disclosures):
    return phrase_coverage_from(phrase_text(target), disclosure_phrases(disclosures))


class PhraseCoverageTest(unittest.TestCase):
    def test_a_verbatim_quote_is_contained(self):
        self.assertEqual(
            phrase_value(product("a"), ["Machine wash cold with like colors"]), 1.0
        )

    def test_scattered_tokens_are_not_a_phrase(self):
        """The discriminating case: token coverage ties at 1.0 for a product
        that merely shares the vocabulary; contiguity separates the product
        the customer is actually quoting."""
        scattered = product(
            "b",
            features=["Machine dry only", "wash in cold water", "colors like these"],
            details={},
        )
        disclosures = ["Machine wash cold with like colors"]
        self.assertEqual(phrase_value(scattered, disclosures), 0.0)

    def test_punctuation_differences_do_not_break_containment(self):
        """The card renders details as "key: value" while searchable_text
        flattens them as "key value"; both sides normalize to the same token
        stream."""
        self.assertEqual(phrase_value(product("a"), ["Material: 100% Cotton canvas upper"]), 1.0)

    def test_a_truncated_last_token_still_matches(self):
        """intent_card() truncates constraints at 180 characters, which can
        clip the final word."""
        self.assertEqual(
            phrase_value(product("a"), ["Machine wash cold with like colo"]), 1.0
        )

    def test_longer_phrases_carry_more_weight(self):
        long_hit = phrase_value(
            product("a"), ["Machine wash cold with like colors", "zebra unicorn"]
        )
        short_hit = phrase_value(
            product("a"), ["mahogany walnut dovetail drawer joints", "Rubber sole"]
        )
        self.assertGreater(long_hit, short_hit)

    def test_no_disclosures_is_neutral(self):
        self.assertEqual(phrase_value(product("a"), []), 0.0)


class PhraseFeatureTest(unittest.TestCase):
    def test_the_feature_is_reported(self):
        candidate = Candidate(
            parent_asin="a",
            route_ranks={"lexical": 1},
            route_scores={"lexical": 1.0, "combined": 1.0},
            matched_hard_constraints=(),
            matched_soft_preferences=(),
        )
        scored = score_candidate(candidate, product("a"), {})
        self.assertIn("phrase_evidence", [c.feature for c in scored.contributions])
        self.assertIn("phrase_evidence", FEATURE_WEIGHTS)


if __name__ == "__main__":
    unittest.main()
