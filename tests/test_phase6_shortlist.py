"""P6-T2: the shortlist policy, and the evidence signal it reads.

The behaviour worth pinning is not the tuned constants -- those are recorded
with their sweeps in shortlist.py and expected to move -- but the three
conditions that decide whether withholding is justified at all. The third of
them is the robustness argument for the whole feature, so it gets the most
direct test here.
"""

from __future__ import annotations

import unittest

from shopping_agent import shortlist
from shopping_agent.catalog import normalize_product
from shopping_agent.contracts import Candidate
from shopping_agent.reranking import prepare_evidence


def raw(asin: str, **fields) -> dict:
    base = {"parent_asin": asin, "title": "Test", "categories": ["Shoes"]}
    base.update(fields)
    return base


def candidate(asin: str, rank: int = 1) -> Candidate:
    return Candidate(
        parent_asin=asin,
        route_ranks={"lexical": rank},
        route_scores={"lexical": 1.0, "combined": 1.0},
        matched_hard_constraints=(),
        matched_soft_preferences=(),
    )


class ShortlistPolicyTest(unittest.TestCase):
    def test_it_withholds_while_still_narrowing(self):
        size = shortlist.shortlist_size(
            turn=1, top_k=10, live_disclosures=1, consistent=40
        )
        self.assertEqual(size, shortlist.NARROWING_SIZE)

    def test_it_returns_everything_once_the_field_is_one_candidate(self):
        size = shortlist.shortlist_size(
            turn=1, top_k=10, live_disclosures=2, consistent=1
        )
        self.assertEqual(size, 10)

    def test_it_returns_everything_once_the_questions_are_spent(self):
        for turn in range(shortlist.EXPAND_TURN, 11):
            self.assertEqual(
                shortlist.shortlist_size(turn, 10, live_disclosures=3, consistent=99),
                10,
                f"turn {turn} should return the full list",
            )

    def test_without_evidence_it_never_withholds(self):
        """The robustness condition. A customer who paraphrases instead of
        quoting leaves live_disclosures at 0 on every turn, and the policy has
        to switch itself off rather than withhold on a ranking it cannot
        measure."""
        for turn in range(1, 11):
            self.assertEqual(
                shortlist.shortlist_size(turn, 10, live_disclosures=0, consistent=0),
                10,
                f"turn {turn} withheld without evidence",
            )

    def test_the_switch_restores_the_previous_behaviour(self):
        for turn in range(1, 11):
            self.assertEqual(
                shortlist.shortlist_size(
                    turn, 10, live_disclosures=5, consistent=99, enabled=False
                ),
                10,
            )

    def test_it_never_exceeds_the_requested_top_k(self):
        for top_k in (0, 1, 2, 10):
            for live, consistent, turn in ((0, 0, 1), (2, 99, 1), (2, 1, 1), (2, 9, 9)):
                self.assertLessEqual(
                    shortlist.shortlist_size(turn, top_k, live, consistent), top_k
                )


class ConsistencySignalTest(unittest.TestCase):
    def products(self, *specs):
        return {asin: normalize_product(raw(asin, features=list(feats)))
                for asin, feats in specs}

    def test_a_uniquely_owned_disclosure_narrows_the_field_to_one(self):
        products = self.products(
            ("a", ["Imported", "Hand-forged titanium clasp"]),
            ("b", ["Imported"]),
            ("c", ["Imported"]),
        )
        candidates = [candidate(x, i + 1) for i, x in enumerate("abc")]
        found = prepare_evidence(
            ["Imported", "Hand-forged titanium clasp."], candidates, products
        )
        self.assertEqual(found.live_disclosures, 2)
        self.assertEqual(found.consistent, 1)

    def test_a_shared_disclosure_leaves_the_field_wide(self):
        products = self.products(("a", ["Imported"]), ("b", ["Imported"]))
        candidates = [candidate("a", 1), candidate("b", 2)]
        found = prepare_evidence(["Imported."], candidates, products)
        self.assertEqual(found.live_disclosures, 1)
        self.assertEqual(found.consistent, 2)

    def test_a_paraphrase_is_not_live_evidence(self):
        products = self.products(("a", ["Machine wash cold"]), ("b", ["Rubber sole"]))
        candidates = [candidate("a", 1), candidate("b", 2)]
        found = prepare_evidence(["please wash it in cold water"], candidates, products)
        self.assertEqual(found.live_disclosures, 0)
        self.assertEqual(found.consistent, 0)
        # ... and that is exactly what makes the policy stand down.
        self.assertEqual(
            shortlist.shortlist_size(1, 10, found.live_disclosures, found.consistent), 10
        )

    def test_nothing_disclosed_yet_is_not_evidence_either(self):
        products = self.products(("a", ["Imported"]))
        found = prepare_evidence([], [candidate("a")], products)
        self.assertEqual((found.live_disclosures, found.consistent), (0, 0))

    def test_an_empty_pool_is_handled(self):
        found = prepare_evidence(["Imported."], [], {})
        self.assertEqual((found.live_disclosures, found.consistent), (0, 0))
        self.assertEqual(found.slot_terms, [])


if __name__ == "__main__":
    unittest.main()
