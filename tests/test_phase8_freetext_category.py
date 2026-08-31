"""P8-T3: recognise a category in a conversationally-phrased opener.

`stated_category` matches nine lead-in phrasings ("I'm looking for X",
"I want X", ...). The benchmark's simulator always uses one of them, so the
public set never exercises the alternative -- but a real customer opens with
"Hey, do you have any loafers?" and gets no category at all, which stands the
E9 category filter down and widens retrieval from a median 184 rows to all
50,000.

This is the fallback. It asks a different question -- *does this message name a
category I know?* -- against the catalogue's own 1,115 coarse categories, and
runs only when the lead-in patterns fail, so every opener that parses today is
untouched by construction.
"""

from __future__ import annotations

import unittest

from shopping_agent import slots


CATALOGUE = [
    "Shoes Loafers & Slip-Ons",
    "Shoes Athletic",
    "Women Shoes",
    "Men Shoes",
    "Accessories Belts",
    "Clothing Dresses",
    "Novelty & More Novelty",
]


class FreeTextCategoryTest(unittest.TestCase):
    def setUp(self):
        self.index = slots.category_token_index(CATALOGUE)

    def match(self, message):
        return slots.match_category_in_text(message, self.index)

    def test_finds_a_category_named_conversationally(self):
        for message in (
            "Hey, do you have any Shoes Loafers & Slip-Ons?",
            "Can you help me find shoes loafers and slip-ons",
            "show me some loafers & slip-ons shoes please",
        ):
            with self.subTest(message=message):
                self.assertEqual(self.match(message), "Shoes Loafers & Slip-Ons")

    def test_word_order_does_not_matter(self):
        # The tokens are a set, which is the same insensitivity
        # canonical_category already relies on for reworded categories.
        self.assertEqual(self.match("slip-ons loafers shoes"), "Shoes Loafers & Slip-Ons")

    def test_prefers_the_most_specific_category(self):
        # "Women Shoes" is fully named here too, but the longer match is the
        # more informative one and must win.
        self.assertEqual(
            self.match("I'd love some women's shoes loafers & slip-ons"),
            "Shoes Loafers & Slip-Ons",
        )

    def test_returns_none_when_no_category_is_named(self):
        for message in (
            "Hi there!",
            "I need a gift for my wife",
            "something under fifty dollars",
            "",
        ):
            with self.subTest(message=message):
                self.assertIsNone(self.match(message))

    def test_a_partly_named_category_does_not_match(self):
        # "loafers" alone is not enough: every token of the category has to be
        # present, which is what keeps the false-positive rate low.
        self.assertIsNone(self.match("do you have loafers"))

    def test_generic_two_token_categories_still_work(self):
        self.assertEqual(self.match("looking for men shoes"), "Men Shoes")

    def test_is_deterministic_across_equally_specific_matches(self):
        first = self.match("men shoes women shoes")
        second = self.match("men shoes women shoes")
        self.assertEqual(first, second)
        self.assertIn(first, ("Men Shoes", "Women Shoes"))

    def test_empty_catalogue_is_handled(self):
        self.assertIsNone(slots.match_category_in_text("shoes", slots.category_token_index([])))


class LeadInStillWinsTest(unittest.TestCase):
    """The fallback must never displace the pattern that already works."""

    def test_the_nine_lead_ins_are_untouched(self):
        for opener in (
            "I'm looking for Shoes Loafers & Slip-Ons.",
            "I want Shoes Loafers & Slip-Ons.",
            "I need Shoes Loafers & Slip-Ons.",
            "I'd like Shoes Loafers & Slip-Ons.",
            "I'm after Shoes Loafers & Slip-Ons.",
        ):
            with self.subTest(opener=opener):
                self.assertEqual(
                    slots.stated_category(opener), "Shoes Loafers & Slip-Ons"
                )


if __name__ == "__main__":
    unittest.main()


# --- Integration: the contract the Agent implements over the matcher --------

import json
import tempfile
from pathlib import Path

from starter.agent import Agent

_ROWS = [
    {
        "parent_asin": f"B{index:09d}",
        "title": f"Loafer model {index}",
        "categories": ["Clothing, Shoes & Jewelry", "Shoes", "Loafers & Slip-Ons"],
        "features": ["premium microfiber leather upper"],
        "details": {"Material": "leather"},
        "description": "slip on shoe",
    }
    for index in range(6)
]


class AgentFallbackContractTest(unittest.TestCase):
    """`_category_from_free_text` must never displace a working lead-in."""

    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.TemporaryDirectory()
        cls.path = Path(cls._dir.name) / "catalog.jsonl"
        cls.path.write_text(
            "".join(json.dumps(r) + "\n" for r in _ROWS), encoding="utf-8"
        )
        cls.agent = Agent(str(cls.path), enable_dense=False, enable_semantic=False)
        (cls.category,) = {r for r in cls.agent._known_categories}

    @classmethod
    def tearDownClass(cls):
        # Close before the temp directory goes: the agent holds an open SQLite
        # connection to an index built from that catalogue.
        cls.agent.close()
        cls._dir.cleanup()

    def stated(self, message):
        session = f"s-{abs(hash(message))}"
        self.agent.reset(session, {})
        self.agent.respond(session, message, 1, 10)
        return self.agent.orchestrator.store.get(session).stated_category

    def test_a_resolving_lead_in_is_returned_unchanged(self):
        # The public-set case. This is the guarantee that keeps the scored
        # number fixed: whatever the free-text matcher would have found, a
        # stated category the catalogue actually has wins.
        self.assertEqual(self.stated(f"I'm looking for {self.category}."), self.category)

    def test_a_conversational_opener_now_resolves(self):
        found = self.stated(f"Hey, do you have any {self.category}?")
        self.assertEqual(found, self.category)
        self.assertTrue(self.agent._resolve_categories(found))

    def test_a_lead_in_that_returns_junk_is_replaced(self):
        # "I need ..." matches the pattern and hands back the rest of the
        # sentence, which names no category. Before P8-T3 that junk was kept
        # and the filter stood down; now the real category is recovered.
        found = self.stated(f"I need some {self.category.lower()} today")
        self.assertEqual(found, self.category)

    def test_junk_is_preserved_when_no_category_is_named(self):
        # Nothing found means nothing changes -- the old behaviour exactly,
        # rather than a guess.
        self.assertEqual(self.stated("I need a gift for my wife"), "a gift for my wife")

    def test_an_opener_naming_nothing_still_yields_nothing(self):
        self.assertIsNone(self.stated("Hi there!"))
