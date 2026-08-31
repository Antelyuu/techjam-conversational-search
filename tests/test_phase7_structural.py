"""P7-T2: the structural paraphraser must move form and nothing else.

scripts/paraphrase_eval.py's second paraphraser only earns its place as a
held-out check if three things are true, and all three are cheap to assert:

  * level 0 is an exact no-op, because it is the control that gives every
    other number a meaning (`--paraphraser structural --level 0` has to
    reproduce the harness's 0.945497 to six decimals, and it cannot do that
    if the transform touches anything at level 0);
  * one seed gives one answer, because the project's whole method is
    reproducible numbers;
  * it PARAPHRASES rather than SUMMARISES -- every content token the card
    value carried is still there afterwards. Without this the mode would be
    an information-loss probe wearing a paraphrase probe's name, and the two
    answer different questions. The synonym mode's level 3 is the
    information-loss probe and is deliberately kept separate.

The corpus below is not decoration either: it is one instance of each shape
measured across the public set's 800 disclosable constraints, so a rule that
only works on the shape it was written for fails here. When the catalogue is
present the same invariants are re-checked over all 800 for real.
"""

from __future__ import annotations

import random
import unittest
from pathlib import Path

from scripts.paraphrase_eval import (
    HEAD_NOUNS,
    PARAPHRASERS,
    SURFACE_VARIANTS,
    _content,
    audit,
    constraint_values,
    restructure,
    restructure_phrase,
    surface_variant,
)

SEED = 20260831

# One of each shape the census found, plus the two that broke an earlier draft
# (a percentage glued to the next word, and a glued number+unit).
CORPUS = [
    "cotton",
    "Imported",
    "Rubber sole",
    "Zipper closure",
    "Hand Wash Only",
    "color: black",
    "Material: 100% Cotton",
    "100% Polyester",
    "54% Cotton, 36% Polyester, 10% Spandex",
    "Fabric: 70%Cotton, 28.5%Polyester, 1.5%Spandex",
    "Textile Upper. Textile Covered EVA Footbed. Rubber Outsole",
    'Platform measures approximately 0.5"',
    'Shaft measures approximately 8" from arch',
    "Gold-tone 18mm stainless steel expansion band fits up to 8-inch wrist",
    "Sold by piece! 14g nose hoop ring size: 18mm",
    "Hand Wash or Machine Wash with laundry bag (30°C Max)",
    "Made in the USA or Imported",
    "【COMFORTABLE TO WEAR】—Acrylic Leopard dangle earring; lightweight",
    "Received APMA (American Podiatric Medical Association) Seal of Acceptance",
]

CATALOG = Path("data/catalog.jsonl")
DATASET = Path("data/public_set.jsonl")
_HAS_PUBLIC_SET = CATALOG.exists() and DATASET.exists()


def _run(level: int, seed: int = SEED, values: list[str] | None = None) -> list[str]:
    rng = random.Random(seed)
    return [restructure(value, level, rng) for value in values or CORPUS]


class StructuralLevelZeroTest(unittest.TestCase):
    """Level 0 is the control and must be indistinguishable from no probe."""

    def test_level_zero_returns_the_value_unchanged(self):
        for value in CORPUS:
            self.assertEqual(restructure(value, 0, random.Random(SEED)), value)
            self.assertEqual(restructure_phrase(value, 0, random.Random(SEED)), value)

    def test_level_zero_consumes_no_randomness(self):
        """The harness draws the carrier and opener from the SAME rng.

        If level 0 consumed a draw, the stream would advance and the control
        would stop being the simulator's own wording. Asserted by checking
        that the generator is where it started, which is stronger than
        comparing outputs: it is the property the harness actually relies on.
        """
        rng = random.Random(SEED)
        before = rng.getstate()
        for value in CORPUS:
            restructure(value, 0, rng)
            restructure_phrase(value, 0, rng)
        self.assertEqual(rng.getstate(), before)

    def test_negative_levels_are_also_the_identity(self):
        self.assertEqual(restructure("cotton", -1, random.Random(SEED)), "cotton")


class StructuralDeterminismTest(unittest.TestCase):
    def test_same_seed_gives_byte_identical_output(self):
        for level in (1, 2, 3):
            with self.subTest(level=level):
                self.assertEqual(_run(level), _run(level))

    def test_a_different_seed_moves_something(self):
        """Not a style point: if the seed did nothing, the levels that draw
        would be silently deterministic in a way that hides a dead knob."""
        self.assertNotEqual(_run(2), _run(2, seed=SEED + 1))

    def test_paraphraser_registry_defaults_to_synonym(self):
        """The default has to stay `synonym` or every command that predates
        --paraphraser silently changes what it measures."""
        self.assertEqual(sorted(PARAPHRASERS), ["structural", "synonym"])
        self.assertIs(PARAPHRASERS["structural"].value, restructure)
        self.assertIs(PARAPHRASERS["structural"].phrase, restructure_phrase)


class StructuralContentPreservationTest(unittest.TestCase):
    """The mode paraphrases; it never summarises."""

    def assert_lossless(self, values: list[str], level: int) -> None:
        rng = random.Random(SEED)
        for value in values:
            out = restructure(value, level, rng)
            src, dst = _content(value), _content(out)
            for token in src - dst:
                replacement = surface_variant(token)
                self.assertIsNotNone(
                    replacement,
                    f"level {level} deleted {token!r} from {value!r} -> {out!r}",
                )
                self.assertTrue(
                    set(replacement) <= dst,
                    f"level {level} replaced {token!r} in {value!r} with nothing "
                    f"usable -> {out!r}",
                )

    def test_every_content_token_survives_at_every_level(self):
        for level in (1, 2, 3):
            with self.subTest(level=level):
                self.assert_lossless(CORPUS, level)

    def test_level_one_changes_no_token_at_all(self):
        """Level 1 only reframes, so it may add function words and must not
        touch the content ones -- not even through a declared variant."""
        rng = random.Random(SEED)
        for value in CORPUS:
            self.assertLessEqual(_content(value), _content(restructure(value, 1, rng)))

    def test_percentages_and_units_do_not_swallow_the_next_word(self):
        """The bug this test was written for: substituting "%" in place turned
        "70%Cotton" into "70 percentCotton", whose only token is
        "percentcotton" -- the word "cotton" deleted by a missing space."""
        rng = random.Random(SEED)
        out = restructure("Fabric: 70%Cotton, 28.5%Polyester, 1.5%Spandex", 2, rng)
        self.assertIn("cotton", _content(out))
        self.assertIn("polyester", _content(out))
        self.assertIn("spandex", _content(out))

    def test_surface_variant_covers_the_glued_number_unit_form(self):
        self.assertEqual(surface_variant("mm"), ("millimeters",))
        self.assertEqual(surface_variant("18mm"), ("18", "millimeters"))
        self.assertEqual(surface_variant("6mm"), ("millimeters",))  # "6" is dropped
        self.assertIsNone(surface_variant("cotton"))
        for abbreviation in SURFACE_VARIANTS:
            self.assertIsNotNone(surface_variant(abbreviation))


class StructuralMechanismTest(unittest.TestCase):
    """It has to actually defeat the thing it claims to defeat."""

    def test_whole_value_equality_is_always_broken(self):
        """slot_evidence (weight 16.0) asks whether the disclosure IS a card
        value. If any output still equalled its input the mode would be
        scoring the agent on verbatim quoting for that value."""
        rng = random.Random(SEED)
        for value in CORPUS:
            out = restructure(value, 2, rng)
            self.assertNotEqual(out.strip(), value.strip())

    def test_output_carries_no_colon_or_semicolon(self):
        """Both are hazards rather than tidiness. A colon lets the
        orchestrator's _LEAD_IN_RE strip the reframe back off and hand slot
        ownership its exact value again; a semicolon is what
        evidence.split_disclosures splits a reply on, so one inside a value
        fragments the disclosure and measures a different effect."""
        for level in (1, 2, 3):
            rng = random.Random(SEED)
            for value in CORPUS:
                out = restructure(value, level, rng)
                self.assertNotIn(":", out)
                self.assertNotIn(";", out)

    def test_a_head_noun_phrase_is_inverted(self):
        """"Rubber sole" is <modifier> <head>, and every frame has to put the
        head first and the modifier last -- that inversion is the only
        structural move a two-word value offers."""
        self.assertIn("sole", HEAD_NOUNS)
        rng = random.Random(SEED)
        outs = {restructure("Rubber sole", 1, rng) for _ in range(20)}
        self.assertGreater(len(outs), 1, "every frame collapsed to one form")
        for out in outs:
            self.assertLess(out.index("sole"), out.index("Rubber"), out)

    def test_a_labelled_value_is_spoken_as_its_label(self):
        rng = random.Random(SEED)
        outs = {restructure("color: black", 1, rng) for _ in range(20)}
        self.assertGreater(len(outs), 1, "every frame collapsed to one form")
        for out in outs:
            self.assertIn("the color ", out)
            self.assertTrue(out.endswith(" black"), out)

    def test_clause_order_moves_at_level_two_and_not_at_level_one(self):
        value = "54% Cotton, 36% Polyester, 10% Spandex"
        rng = random.Random(SEED)
        self.assertTrue(restructure(value, 1, rng).endswith("10% Spandex"))
        rotated = {restructure(value, 2, random.Random(s)) for s in range(12)}
        self.assertGreater(len(rotated), 1, rotated)

    def test_a_noun_phrase_stays_a_noun_phrase(self):
        """--paraphrase-category feeds this into "I'm looking for {}", so the
        sentence frame must not appear."""
        rng = random.Random(SEED)
        for value in CORPUS:
            out = restructure_phrase(value, 2, rng)
            self.assertFalse(out.startswith("it is "), out)
            self.assertFalse(out.startswith("the "), out)


@unittest.skipUnless(_HAS_PUBLIC_SET, "needs data/catalog.jsonl and data/public_set.jsonl")
class StructuralPublicSetTest(StructuralContentPreservationTest):
    """The same invariants over all 800 constraints the harness really uses.

    Subclasses the corpus test so the losslessness assertion is written once.
    The catalogue is gitignored, so this skips rather than fails where it is
    absent -- the corpus above still covers every shape it contains.
    """

    @classmethod
    def setUpClass(cls):
        cls.values = constraint_values(str(DATASET), str(CATALOG))

    def test_the_public_set_is_the_size_the_docstrings_claim(self):
        self.assertEqual(len(self.values), 800)

    def test_every_content_token_survives_over_the_whole_public_set(self):
        for level in (1, 2, 3):
            with self.subTest(level=level):
                self.assert_lossless(self.values, level)

    def test_audit_reports_the_two_modes_attacking_different_axes(self):
        """The claim the second paraphraser exists to support, as a test.

        Structural keeps the vocabulary and destroys the exactness; synonym
        destroys the vocabulary and leaves some exactness standing. If these
        ever converge, the two modes have stopped bracketing anything and the
        held-out check is no longer held out.
        """
        report = audit(str(DATASET), str(CATALOG), 2, SEED)
        structural = report["modes"]["structural"]
        synonym = report["modes"]["synonym"]
        self.assertTrue(structural["token_preserving"])
        self.assertGreater(structural["content_token_retention"], 0.99)
        self.assertLess(synonym["content_token_retention"], 0.50)
        self.assertEqual(structural["whole_value_survives"], 0.0)
        self.assertGreater(synonym["whole_value_survives"], 0.0)
