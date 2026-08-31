"""P7-T2/T3: the paraphrase regime, and what changes once it fires.

Two behaviours are under test and they are deliberately separated. The regime
*predicate* must fire on a rewording and never on a quoting turn, because the
whole cost argument for the conditional profile is that it cannot touch a
split it never fires on. The *response* to the regime -- a different weight
table, and a graded notion of slot ownership -- must then reorder in the
direction claimed, and must leave the exact path byte-identical.
"""

from __future__ import annotations

import unittest

from shopping_agent import slots
from shopping_agent.catalog import normalize_product
from shopping_agent.contracts import Candidate
from shopping_agent.reranking import (
    FEATURE_WEIGHTS,
    PARAPHRASE_WEIGHTS,
    RELAXED_OWNERSHIP_THRESHOLD,
    DisclosureEvidence,
    prepare_evidence,
    score_candidate,
    weights_for,
)


def raw(asin: str, **fields) -> dict:
    base = {"parent_asin": asin, "title": "Test Product", "categories": ["Shoes"]}
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


def evidence(disclosed: int, live: int) -> DisclosureEvidence:
    return DisclosureEvidence([], [], [], live, 0, disclosed)


class RegimePredicateTest(unittest.TestCase):
    def test_it_fires_when_something_was_said_and_nobody_owns_it(self):
        self.assertTrue(evidence(disclosed=2, live=0).paraphrase_regime)

    def test_it_stays_silent_before_the_customer_has_said_anything(self):
        """The distinction P6-T6 was built on: no owner is only evidence of
        rewording once there is something to own. A Browsing opener discloses
        nothing, and that is a certainty rather than a measurement."""
        self.assertFalse(evidence(disclosed=0, live=0).paraphrase_regime)

    def test_it_stays_silent_when_the_customer_is_quoting(self):
        self.assertFalse(evidence(disclosed=2, live=2).paraphrase_regime)

    def test_one_owned_disclosure_is_enough_to_call_it_quoting(self):
        self.assertFalse(evidence(disclosed=3, live=1).paraphrase_regime)


class WeightProfileTest(unittest.TestCase):
    def test_the_quoting_path_gets_the_table_itself_not_a_copy(self):
        """Identity, not equality: the public path must allocate nothing."""
        self.assertIs(weights_for(False), FEATURE_WEIGHTS)

    def test_the_regime_overrides_only_what_it_names(self):
        applied = weights_for(True)
        for feature, weight in FEATURE_WEIGHTS.items():
            if feature not in PARAPHRASE_WEIGHTS:
                self.assertEqual(applied[feature], weight)

    def test_the_override_cannot_invent_a_feature(self):
        self.assertLessEqual(set(PARAPHRASE_WEIGHTS), set(FEATURE_WEIGHTS))

    def test_the_checklist_is_the_same_length_in_either_regime(self):
        """explain() from a paraphrased turn must line up against one from a
        quoted turn, feature for feature."""
        product = normalize_product(raw("a", features=["Rubber sole"]))
        quoted = score_candidate(candidate("a"), product, {}, [])
        reworded = score_candidate(
            candidate("a"), product, {}, [], weights=weights_for(True)
        )
        self.assertEqual(
            [c.feature for c in quoted.contributions],
            [c.feature for c in reworded.contributions],
        )


class OwnershipOverlapTest(unittest.TestCase):
    def test_an_owned_value_scores_one(self):
        owned = frozenset({"machine wash cold"})
        self.assertEqual(slots.ownership_overlap("machine wash cold", owned), 1.0)

    def test_a_rewording_scores_partially(self):
        owned = frozenset({"material 100 cotton"})
        score = slots.ownership_overlap("made of 100 cotton", owned)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_an_unrelated_value_scores_nothing(self):
        owned = frozenset({"material 100 cotton"})
        self.assertEqual(slots.ownership_overlap("rubber sole", owned), 0.0)

    def test_a_compressive_paraphrase_does_not_favour_the_shorter_owner(self):
        """The defect the first version of this feature shipped with.

        Jaccard divides by the union, so it charged the true owner for the
        length of its own card value. A customer who says *less* than the card
        holds therefore scored the true owner 0.333 and a shorter near-
        duplicate 0.667 -- backwards, at the weight of the table's dominant
        feature, and strictly worse than the silence it replaced. It survived
        the first round of measurement because the L2 harness is expansive and
        never produces a disclosure shorter than its source.

        The requirement is not that the true owner wins. Both products
        genuinely account for what was said, so the honest outcome is a tie
        that leaves the ordering to other features. The requirement is that
        the impostor must not win."""
        target = frozenset({"machine wash cold with like colors"})
        shorter = frozenset({"hand wash cold"})
        said = "wash cold"
        self.assertGreaterEqual(
            slots.ownership_overlap(said, target),
            slots.ownership_overlap(said, shorter),
        )
        self.assertGreaterEqual(
            slots.ownership_overlap(said, target), RELAXED_OWNERSHIP_THRESHOLD
        )

    def test_one_shared_word_is_coincidence_not_ownership(self):
        owned = frozenset({"machine wash cold with like colors"})
        self.assertEqual(slots.ownership_overlap("cold brew tumbler", owned), 0.0)

    def test_a_short_value_inside_a_long_one_is_not_full_ownership(self):
        """The impostor path E7/E8 closed, which a containment measure would
        reopen at the weight of the table's dominant feature."""
        owned = frozenset({"100 cotton blend twill lining"})
        self.assertLess(slots.ownership_overlap("cotton", owned), 0.5)

    def test_the_empty_value_scores_nothing(self):
        self.assertEqual(slots.ownership_overlap("", frozenset({"anything"})), 0.0)


class RelaxedOwnershipTest(unittest.TestCase):
    """A quoting pool must be untouched; a reworded one must be re-priced."""

    def products(self, *raws) -> dict:
        return {r["parent_asin"]: normalize_product(r) for r in raws}

    def test_quoting_leaves_ownership_exact(self):
        products = self.products(
            raw("a", features=["Machine wash cold"]),
            raw("b", features=["Rubber sole"]),
        )
        found = prepare_evidence(
            ["Machine wash cold"], [candidate("a"), candidate("b")], products
        )
        self.assertFalse(found.relaxed)
        self.assertFalse(found.paraphrase_regime)

    def test_a_rewording_re_prices_the_slot_terms(self):
        """Exact ownership finds nothing, so ownership_weights would zero
        every term and switch the feature off entirely. The relaxed pass has
        to restore a non-zero weight or there is nothing to score with."""
        products = self.products(
            raw("a", features=["Machine wash cold with like colors"]),
            raw("b", features=["Rubber outsole for traction"]),
        )
        found = prepare_evidence(
            ["machine wash cold with colors"],
            [candidate("a"), candidate("b")],
            products,
        )
        self.assertTrue(found.paraphrase_regime)
        self.assertTrue(found.relaxed)
        self.assertTrue(any(weight > 0.0 for _value, weight in found.slot_terms))

    def test_the_reworded_owner_outscores_an_unrelated_candidate(self):
        products = self.products(
            raw("a", features=["Machine wash cold with like colors"]),
            raw("b", features=["Rubber outsole for traction"]),
        )
        candidates = [candidate("a"), candidate("b")]
        found = prepare_evidence(
            ["machine wash cold with colors"], candidates, products
        )
        scored = {
            asin: score_candidate(
                candidate(asin),
                products[asin],
                {},
                slot_terms=found.slot_terms,
                relaxed_ownership=found.relaxed,
            )
            for asin in ("a", "b")
        }
        value = {
            asin: next(
                c.value for c in scored[asin].contributions
                if c.feature == "slot_evidence"
            )
            for asin in ("a", "b")
        }
        self.assertGreater(value["a"], value["b"])

    def test_relaxation_that_finds_nothing_behaves_as_before(self):
        """A rewording nobody comes close to must not set the flag, so the
        turn is scored exactly as it was before this feature existed."""
        products = self.products(
            raw("a", features=["Rubber outsole for traction"]),
            raw("b", features=["Imported"]),
        )
        found = prepare_evidence(
            ["completely unrelated wording here"],
            [candidate("a"), candidate("b")],
            products,
        )
        self.assertTrue(found.paraphrase_regime)
        self.assertFalse(found.relaxed)

    def test_a_match_below_the_threshold_earns_nothing(self):
        """A single shared word out of seven is coincidence, not ownership.

        Note the value chosen. A bare material or colour word would NOT work
        here, because slots.card_values extracts those as owned values in
        their own right -- a product whose copy reads "100% cotton blend
        twill lining" genuinely owns the value "cotton", and scoring it 1.0
        is correct rather than a false positive."""
        product = normalize_product(
            raw("a", features=["Wash the garment inside out before wearing"])
        )
        self.assertLess(
            slots.ownership_overlap("wearing", product.card_values),
            RELAXED_OWNERSHIP_THRESHOLD,
        )
        scored = score_candidate(
            candidate("a"), product, {},
            slot_terms=[("wearing", 1.0)],
            relaxed_ownership=True,
        )
        value = next(
            c.value for c in scored.contributions if c.feature == "slot_evidence"
        )
        self.assertEqual(value, 0.0)


if __name__ == "__main__":
    unittest.main()
