"""P6-T1: slot ownership -- reconstruction fidelity and scoring behaviour.

The reconstruction tests deliberately check against the evaluator's own
`intent_card`/`_flatten_values`/`_clean_constraint` rather than against
hand-written expectations. Hand-written expectations would encode my reading
of the generator, which is exactly the thing that can be wrong; the generator
is the specification here, and any drift between it and slots.py must fail
loudly rather than show up later as a quietly unowned constraint.
"""

from __future__ import annotations

import unittest

from evaluator.local_evaluator import (
    _clean_constraint,
    _flatten_values,
    intent_card,
)
from shopping_agent import slots
from shopping_agent.catalog import normalize_product
from shopping_agent.contracts import Candidate
from shopping_agent.reranking import (
    FEATURE_WEIGHTS,
    prepare_evidence,
    rerank,
    score_candidate,
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


class ReconstructionTest(unittest.TestCase):
    def test_a_whole_feature_value_is_owned(self):
        owned = slots.card_values(raw("a", features=["Machine wash cold", "Rubber sole"]))
        self.assertIn("Machine wash cold", owned)
        self.assertIn("Rubber sole", owned)

    def test_a_details_dict_is_owned_as_key_colon_value(self):
        owned = slots.card_values(raw("a", details={"Material": "100% Cotton"}))
        self.assertIn("Material: 100% Cotton", owned)

    def test_a_substring_of_a_value_is_not_owned(self):
        """The whole point: containment is not ownership."""
        owned = slots.card_values(raw("a", features=["Machine wash cold with like colors"]))
        self.assertIn("Machine wash cold with like colors", owned)
        self.assertNotIn("Machine wash cold", owned)

    def test_semicolon_pieces_are_owned_alongside_the_whole(self):
        """The reply is split on ";" before it reaches us, and a semicolon
        inside one constraint is indistinguishable from the joiner."""
        owned = slots.card_values(raw("a", features=["10 x 8 inches; 7.8 Ounces"]))
        self.assertIn("10 x 8 inches; 7.8 Ounces", owned)
        self.assertIn("10 x 8 inches", owned)
        self.assertIn("7.8 Ounces", owned)

    def test_the_material_label_uses_the_generators_field_order(self):
        """The generator scans title, features, details, description,
        categories, store -- in that order -- and keeps the first hit.
        catalog.py flattens the same fields in a different order for search,
        so a product whose categories and features name different materials
        pins the ordering down."""
        product = raw("a", categories=["Leather Goods"], features=["100% cotton blend"])
        self.assertIn("cotton", slots.card_values(product))
        self.assertNotIn("leather", slots.card_values(product))

    def test_a_clipped_value_keeping_its_comma_still_matches_the_reply(self):
        """The generator trims edge punctuation *before* clipping at 180, so a
        clipped value can end in a comma while the customer's quote of it ends
        in ",." -- 4 of the 200 public targets hit this. Both sides have to
        canonicalize or the target stops owning its own constraint."""
        long_value = "x" * 178 + ", and more text beyond the clip"
        card_value = _clean_constraint(long_value, 180)
        self.assertTrue(card_value.endswith(","), "fixture no longer exercises the clip")
        owned = slots.card_values(raw("a", features=[long_value]))
        # The customer quotes the card value and appends the sentence's stop.
        self.assertIn(slots.normalize_disclosure(card_value + "."), owned)

    def test_reconstruction_agrees_with_the_generator_on_a_built_product(self):
        product = raw(
            "a",
            features=["Imported", "Machine wash cold; tumble dry low"],
            details={"Material": "100% Cotton", "Department": "Womens"},
            price=19.99,
        )
        owned = slots.card_values(product)
        expected = [
            *_flatten_values(product.get("features")),
            *_flatten_values(product.get("details")),
        ]
        for value in expected:
            cleaned = slots.canonical(_clean_constraint(value, 180))
            self.assertIn(cleaned, owned, f"generator value not owned: {value!r}")

    def test_every_constraint_the_card_can_disclose_is_owned(self):
        product = raw(
            "a",
            features=["Pull On closure", "Machine Wash"],
            details={"Fabric type": "95% Cotton, 5% Spandex"},
        )
        card = intent_card(product)
        owned = slots.card_values(product)
        disclosable = [*card["hard_constraints"], *card["soft_preferences"]]
        self.assertTrue(disclosable)
        for value in disclosable:
            self.assertIn(slots.normalize_disclosure(value), owned)

    def test_an_empty_or_odd_product_never_raises(self):
        self.assertEqual(slots.card_values({"parent_asin": "a"}), frozenset())
        self.assertIn("7", slots.card_values(raw("a", features=7)))
        slots.card_values(raw("a", features=[None, ""], details={"k": []}))


class OwnershipWeightTest(unittest.TestCase):
    def test_a_disclosure_everyone_owns_is_worth_almost_nothing(self):
        common, rare = slots.ownership_weights([100, 1], pool_size=100)
        self.assertLess(common, rare)
        self.assertAlmostEqual(common, 1.0, places=6)

    def test_a_disclosure_nobody_owns_is_silence_not_evidence(self):
        self.assertEqual(slots.ownership_weights([0], pool_size=100), [0.0])

    def test_weights_are_positive_and_finite(self):
        for weight in slots.ownership_weights([1, 2, 5, 50], pool_size=50):
            self.assertGreater(weight, 0.0)
            self.assertLess(weight, 100.0)


class SlotFeatureTest(unittest.TestCase):
    def product(self, asin, **fields):
        return normalize_product(raw(asin, **fields))

    def test_owning_the_disclosure_beats_merely_containing_it(self):
        """The near-duplicate case E7 could not separate: both candidates
        contain the sentence, only one holds it as a value of its own."""
        owner = self.product("a", features=["Machine wash cold"])
        impostor = self.product(
            "b", features=["Machine wash cold with like colors and tumble dry"]
        )
        disclosures = ["Machine wash cold."]
        a = score_candidate(candidate("a"), owner, {}, disclosures)
        b = score_candidate(candidate("b"), impostor, {}, disclosures)
        self.assertGreater(a.score, b.score)

    def test_with_nothing_disclosed_the_feature_cannot_reorder(self):
        a = score_candidate(candidate("a"), self.product("a", features=["X"]), {}, [])
        b = score_candidate(candidate("b"), self.product("b", features=["Y"]), {}, [])
        by_feature = {c.feature: c.value for c in a.contributions}
        self.assertEqual(by_feature["slot_evidence"], 0.0)
        self.assertEqual(a.score, b.score)

    def test_a_paraphrased_disclosure_scores_zero_for_everyone(self):
        """The designed failure mode: silence, not noise. If a private set
        paraphrases instead of quoting, the ordering falls back to the
        features beneath this one rather than being corrupted by it."""
        products = {
            "a": self.product("a", features=["Machine wash cold"]),
            "b": self.product("b", features=["Rubber sole"]),
        }
        candidates = [candidate("a", 1), candidate("b", 2)]
        disclosures = ["please wash it in cold water"]
        terms = prepare_evidence(disclosures, candidates, products).slot_terms
        for asin in ("a", "b"):
            scored = score_candidate(
                candidate(asin), products[asin], {}, slot_terms=terms
            )
            value = {c.feature: c.value for c in scored.contributions}["slot_evidence"]
            self.assertEqual(value, 0.0)

    def test_a_rare_disclosure_outweighs_a_ubiquitous_one(self):
        """Selectivity is the whole reason this beats phrase containment: the
        candidate owning the rare constraint must win over the one owning only
        the constraint everybody shares."""
        common = "Imported"
        rare = "Hand-forged titanium clasp"
        products = {
            "a": self.product("a", features=[common, rare]),
            "b": self.product("b", features=[common]),
            "c": self.product("c", features=[common]),
            "d": self.product("d", features=[common]),
        }
        candidates = [candidate(a, i + 1) for i, a in enumerate("abcd")]
        terms = prepare_evidence([common, rare], candidates, products).slot_terms
        by_value = dict(terms)
        self.assertGreater(by_value[rare], by_value[common])

    def test_the_uniquely_owned_disclosure_takes_rank_one(self):
        products = {
            "a": self.product("a", features=["Imported"]),
            "b": self.product("b", features=["Imported", "Hand-forged titanium clasp"]),
        }
        # "b" starts behind on every other feature in the table.
        candidates = [candidate("a", 1), candidate("b", 50)]
        ordered = rerank(
            candidates, products, {}, 10, ["Imported", "Hand-forged titanium clasp."]
        )
        self.assertEqual(ordered[0].parent_asin, "b")

    def test_the_feature_is_registered_and_weighted(self):
        self.assertIn("slot_evidence", FEATURE_WEIGHTS)
        self.assertGreater(FEATURE_WEIGHTS["slot_evidence"], 0.0)
        scored = score_candidate(candidate("a"), self.product("a"), {}, [])
        self.assertIn("slot_evidence", {c.feature for c in scored.contributions})


if __name__ == "__main__":
    unittest.main()
