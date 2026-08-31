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


class _CategoryFixture:
    """The catalogue and the two helpers both filter test classes below need.

    Deliberately not a TestCase: mixed in rather than inherited from, so the
    reworded-opener class does not re-run the exact-opener class's tests.
    """

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


class CategoryFilterTest(_CategoryFixture, unittest.TestCase):
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


class RewordedCategoryTest(_CategoryFixture, unittest.TestCase):
    """E10: the same filter, reached by a customer who does not say it the
    agent's way.

    Both routes into the filter were exact-string tests against one phrasing:
    the opener had to begin "I'm looking for", and the category had to be
    spelled character for character. Measured on the paraphrase harness, 0 of
    200 openers parsed once the opener was reworded, which stood the filter
    down for every session and cost 0.248 composite -- about half the whole
    paraphrase penalty.

    What is pinned here is that the widened parse and the canonical fallback
    reach the *same* restricted set, and that neither loosens the exact test
    that still runs first.
    """

    def test_the_reworded_lead_ins_state_the_same_category(self):
        for message in (
            f"I want {SHOES}. It has to be breathable.",
            f"I'm after {SHOES} \u2014 the thing that matters is breathable.",
            f"Looking for {SHOES}, and it should be breathable.",
            f"I want {SHOES}, though I am still browsing.",
            f"I need {SHOES}. A key requirement is: breathable.",
            f"I'd like {SHOES}, but I'm still exploring.",
            f"I am looking for {SHOES}, but I'm still exploring.",
        ):
            self.assertEqual(stated_category(message), SHOES, message)

    def test_a_reworded_lead_in_still_filters(self):
        agent = self.agent()
        returned = self.returned(agent, f"I want {SHOES}. It has to be breathable.")
        self.assertTrue(returned, "the filter emptied the pool")
        self.assertTrue(
            all(asin.startswith("SHOE") for asin in returned), returned
        )

    def test_a_reordered_category_resolves_to_the_catalogue_spelling(self):
        """"Athletic Running Shoes" is the same shelf as "Shoes Athletic
        Running", and the exact test cannot see that."""
        agent = self.agent()
        reordered = "Athletic Running Shoes"
        self.assertNotIn(reordered, agent._known_categories)
        self.assertEqual(agent._resolve_categories(reordered), (SHOES,))
        returned = self.returned(
            agent, f"I want {reordered}, though I am still browsing."
        )
        self.assertTrue(returned, "the filter emptied the pool")
        self.assertTrue(all(asin.startswith("SHOE") for asin in returned), returned)

    def test_the_exact_spelling_is_still_preferred(self):
        """The fallback is a second look, not a replacement. A category the
        catalogue spells exactly resolves to itself without the canonical
        index being consulted at all."""
        agent = self.agent()
        self.assertEqual(agent._resolve_categories(SHOES), (SHOES,))
        self.assertEqual(agent._resolve_categories(SOCKS), (SOCKS,))

    def test_a_category_no_spelling_reproduces_still_stands_the_filter_down(self):
        """Quiet failure #2, retested through the new route: "Hats Running"
        has no exact match *and* no canonical one, so the filter does not
        apply and retrieval is unrestricted."""
        agent = self.agent()
        self.assertEqual(agent._resolve_categories("Hats Running"), ())
        self.assertEqual(agent._resolve_categories(None), ())
        self.assertTrue(
            self.returned(agent, "I want Hats Running, though I am still browsing.")
        )

    def test_show_me_is_still_not_an_opener(self):
        """The one lead-in deliberately left out. It routinely introduces
        something that is not a category, and admitting it would filter on a
        phrase the customer never meant as one."""
        self.assertIsNone(stated_category("show me something for running"))


if __name__ == "__main__":
    unittest.main()
