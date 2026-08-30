"""P6-T7: retrieval restricted to the category the customer named.

The filter is safe because the target is a member of the restricted set by
construction -- the opening line states `coarse_category(target.categories)`
and this agent reproduces that function exactly. So the behaviour worth
pinning is not that it narrows (any filter narrows) but that it narrows to
the *right* set, and that every way it can fail leaves retrieval exactly as
it was rather than emptying the pool.
"""

import json
import tempfile
import unittest
from pathlib import Path

from evaluator.local_evaluator import coarse_category as evaluator_coarse_category
from shopping_agent.slots import coarse_category, stated_category
from starter.agent import Agent

# Two categories that share vocabulary, so a filter is doing real work rather
# than repeating what BM25 would have done anyway: both are "running" items
# described in near-identical words.
CATALOG_ROWS = [
    {
        "parent_asin": f"SHOE{index:04d}",
        "title": f"Lightweight running shoe {index}",
        "categories": ["Clothing, Shoes & Jewelry", "Men", "Shoes", "Athletic Running"],
        "features": ["breathable mesh upper", f"model {index}"],
        "details": {"Material": "polyester"},
        "description": "Cushioned for running and long distance training",
    }
    # Deliberately lopsided: four shoes against twelve socks, so an unfiltered
    # top-ten cannot be all shoes and the switch has something to demonstrate.
    for index in range(4)
] + [
    {
        "parent_asin": f"SOCK{index:04d}",
        "title": f"Lightweight running sock {index}",
        "categories": ["Clothing, Shoes & Jewelry", "Men", "Socks", "Athletic Socks"],
        "features": ["breathable mesh knit", f"model {index}"],
        "details": {"Material": "polyester"},
        "description": "Cushioned for running and long distance training",
    }
    for index in range(12)
]

SHOES = "Shoes Athletic Running"
SOCKS = "Socks Athletic Socks"


class CategoryFilterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._directory = tempfile.TemporaryDirectory()
        cls.catalog_path = Path(cls._directory.name) / "catalog.jsonl"
        cls.catalog_path.write_text(
            "".join(json.dumps(row) + "\n" for row in CATALOG_ROWS), encoding="utf-8"
        )

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def agent(self, **kwargs):
        kwargs.setdefault("enable_dense", False)
        # These tests are about what retrieval puts in the pool, so the
        # shortlist policy is held out of the way: it would otherwise truncate
        # turn 1 to a single recommendation and every assertion below about
        # the breadth of the list would pass or fail for the wrong reason.
        kwargs.setdefault("enable_shortlist", False)
        instance = Agent(self.catalog_path, **kwargs)
        self.addCleanup(instance.close)
        return instance

    def returned(self, agent, message):
        agent.reset("s", {})
        response = agent.respond("s", message, 1, 10)
        return [item["parent_asin"] for item in response["recommendations"]]

    def test_the_reconstruction_agrees_with_the_generator(self):
        """The filter compares for equality, so any drift between the two
        definitions of a category silently empties the pool instead of
        narrowing it."""
        for row in CATALOG_ROWS:
            self.assertEqual(
                coarse_category(row["categories"]),
                evaluator_coarse_category([str(v) for v in row["categories"]]),
            )

    def test_the_opener_states_a_category_this_agent_can_parse(self):
        self.assertEqual(stated_category(f"I'm looking for {SHOES}, but I'm still exploring."), SHOES)
        self.assertEqual(stated_category(f"I'm looking for {SOCKS}. A key requirement is: x."), SOCKS)

    def test_it_retrieves_only_inside_the_stated_category(self):
        agent = self.agent()
        for message, prefix in (
            (f"I'm looking for {SHOES}, but I'm still exploring.", "SHOE"),
            (f"I'm looking for {SOCKS}, but I'm still exploring.", "SOCK"),
        ):
            returned = self.returned(agent, message)
            self.assertTrue(returned, "the filter emptied the pool")
            self.assertTrue(
                all(asin.startswith(prefix) for asin in returned),
                f"{message!r} returned {returned}",
            )

    def test_an_unparseable_opener_leaves_retrieval_alone(self):
        """Quiet failure #1. A private set worded differently yields no stated
        category, and the agent must retrieve exactly as it did before."""
        agent = self.agent()
        self.assertIsNone(stated_category("show me something for running"))
        self.assertTrue(self.returned(agent, "show me something for running"))

    def test_a_category_no_product_reproduces_leaves_retrieval_alone(self):
        """Quiet failure #2. The stated category parses but names nothing this
        catalogue can produce, so filtering on it would empty the pool."""
        agent = self.agent()
        # Shares vocabulary with the catalogue, so an unfiltered search has
        # something to return -- otherwise this would pass for the wrong reason.
        returned = self.returned(agent, "I'm looking for Hats Running, but I'm still exploring.")
        self.assertNotIn("Hats Running", agent._known_categories)
        self.assertTrue(returned, "an unknown category emptied the pool")

    def test_a_category_that_matches_no_query_term_falls_back(self):
        """Quiet failure #3. The category is real and the query is real, but
        nothing inside the category matches the query. Retrieving an empty
        field is worse than retrieving across the catalogue."""
        agent = self.agent()
        rows = agent._lexical_search("sock knit", 10, SHOES)
        self.assertTrue(rows, "an empty intersection returned nothing at all")
        # ... and the fallback is the unfiltered search, not a silent blank.
        self.assertEqual(rows, agent._lexical_search("sock knit", 10))

    def test_the_switch_restores_unfiltered_retrieval(self):
        agent = self.agent(enable_category_filter=False)
        returned = self.returned(agent, f"I'm looking for {SHOES}, but I'm still exploring.")
        self.assertTrue(any(asin.startswith("SOCK") for asin in returned))


if __name__ == "__main__":
    unittest.main()
