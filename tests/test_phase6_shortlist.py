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

    def test_unowned_disclosures_keep_narrowing_until_the_backstop(self):
        """P8-T2 (E13), and this assertion is the reverse of what it was.

        A customer who paraphrases instead of quoting leaves live_disclosures
        at 0 on every turn. The policy used to read that as "stop withholding"
        and return the full ten. That reasoning was sound and its remedy was
        backwards: the evaluator freezes `best_rank` the first turn the target
        appears, so showing ten at the moment the evidence is weakest commits
        to a rank E13 measured as near-uniform across 2-10.

        Narrowing instead defers the commitment to a turn worth standing
        behind, and is worth +0.0204 mean across six paraphrase probes at a
        measured cost of exactly zero on the public set -- the clause fires
        0/563 verbatim turns. EXPAND_TURN remains the backstop, so the agent
        cannot withhold forever.
        """
        for turn in range(1, shortlist.EXPAND_TURN):
            self.assertEqual(
                shortlist.shortlist_size(
                    turn, 10, live_disclosures=0, consistent=0, disclosed=3
                ),
                shortlist.NARROWING_SIZE,
                f"turn {turn} committed to a ranking it could not measure",
            )
        for turn in range(shortlist.EXPAND_TURN, 11):
            self.assertEqual(
                shortlist.shortlist_size(
                    turn, 10, live_disclosures=0, consistent=0, disclosed=3
                ),
                10,
                f"turn {turn} withheld past the backstop",
            )

    def test_the_old_always_ten_behaviour_is_one_env_var_away(self):
        """The kill switch, so the change is reversible without a deploy."""
        import importlib
        import os

        os.environ["SHOPPING_AGENT_PARAPHRASE_SHORTLIST"] = "10"
        try:
            restored = importlib.reload(shortlist)
            for turn in range(1, 11):
                self.assertEqual(
                    restored.shortlist_size(
                        turn, 10, live_disclosures=0, consistent=0, disclosed=3
                    ),
                    10,
                )
        finally:
            os.environ.pop("SHOPPING_AGENT_PARAPHRASE_SHORTLIST", None)
            importlib.reload(shortlist)

    def test_having_said_nothing_yet_is_not_the_paraphrase_signal(self):
        """P6-T6, and the distinction the whole clause turns on.

        Zero owned disclosures means "nobody owns what the customer said" only
        once the customer has said something. Before that it is a certainty --
        a Browsing opener discloses nothing at all -- and reading it as the
        paraphrase signal handed out a padded ten built on the category alone.
        """
        for turn in range(1, shortlist.EXPAND_TURN):
            self.assertEqual(
                shortlist.shortlist_size(
                    turn, 10, live_disclosures=0, consistent=0, disclosed=0
                ),
                shortlist.NARROWING_SIZE,
                f"turn {turn} padded a list it had no evidence for",
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
        # ... and the policy now *narrows* on that signal rather than padding
        # a ten it cannot measure (P8-T2, E13). The disclosure count still
        # carries the other half of the distinction -- "nobody owns what was
        # said" is only meaningful once something was said -- it just no
        # longer selects the full list.
        self.assertEqual(
            shortlist.shortlist_size(
                1, 10, found.live_disclosures, found.consistent, disclosed=1
            ),
            shortlist.NARROWING_SIZE,
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
