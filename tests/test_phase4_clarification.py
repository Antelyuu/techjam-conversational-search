import unittest

from shopping_agent.catalog import normalize_product
from shopping_agent.clarification import (
    ATTRIBUTE_PRIOR_YIELD,
    DEAD_ATTRIBUTES,
    MAX_CLARIFICATIONS,
    WILDCARD_ATTRIBUTE,
    attribute_coverage,
    choose_attribute,
    interpret_reply,
)
from shopping_agent.contracts import ALLOWED_ATTRIBUTES, Constraint, SessionState
from shopping_agent.orchestrator import ConversationOrchestrator


def product(parent_asin, **overrides):
    raw = {
        "parent_asin": parent_asin,
        "title": "Running shoes",
        "categories": ["Shoes"],
        "features": ["lightweight"],
        "details": {},
        "store": "Example",
        "description": "Shoes for running",
        "price": "50.00",
    }
    raw.update(overrides)
    return normalize_product(raw)


class ChooseAttributeTest(unittest.TestCase):
    def state(self, **overrides):
        state = SessionState("demo", {})
        for key, value in overrides.items():
            setattr(state, key, value)
        return state

    def test_returns_an_allowed_attribute(self):
        chosen = choose_attribute(self.state(), [])
        self.assertIn(chosen, ALLOWED_ATTRIBUTES)

    def test_never_asks_a_dead_attribute(self):
        """brand, category and budget measured 0/200 yield on the public set;
        asking one can only waste a turn."""
        state = self.state()
        for _ in range(MAX_CLARIFICATIONS):
            chosen = choose_attribute(state, [])
            if chosen is None:
                break
            self.assertNotIn(chosen, DEAD_ATTRIBUTES)
            state.asked_attributes.add(chosen)

    def test_never_repeats_an_attribute(self):
        state = self.state()
        seen = []
        for _ in range(MAX_CLARIFICATIONS):
            chosen = choose_attribute(state, [])
            if chosen is None:
                break
            self.assertNotIn(chosen, seen)
            seen.append(chosen)
            state.asked_attributes.add(chosen)
        self.assertGreater(len(seen), 1)

    def test_never_asks_a_fixed_slot(self):
        state = self.state(
            constraints={
                "color": Constraint("color", "black", "soft", 1),
                "material": Constraint("material", "leather", "soft", 1),
            }
        )
        for _ in range(MAX_CLARIFICATIONS):
            chosen = choose_attribute(state, [])
            if chosen is None:
                break
            self.assertNotIn(chosen, ("color", "material"))
            state.asked_attributes.add(chosen)

    def test_never_asks_a_rejected_attribute(self):
        state = self.state(rejected_attributes={"feature"})
        for _ in range(MAX_CLARIFICATIONS):
            chosen = choose_attribute(state, [])
            if chosen is None:
                break
            self.assertNotEqual(chosen, "feature")
            state.asked_attributes.add(chosen)

    def test_returns_none_when_the_budget_is_exhausted(self):
        state = self.state(clarification_turns=MAX_CLARIFICATIONS)
        self.assertIsNone(choose_attribute(state, []))

    def test_returns_none_when_every_attribute_is_used_up(self):
        state = self.state()
        while choose_attribute(state, []) is not None:
            state.asked_attributes.add(choose_attribute(state, []))
        self.assertIsNone(choose_attribute(state, []))

    def test_wildcard_is_withheld_unless_enabled(self):
        state = self.state()
        chosen = []
        while True:
            attribute = choose_attribute(state, [], allow_wildcard=False)
            if attribute is None:
                break
            chosen.append(attribute)
            state.asked_attributes.add(attribute)
        self.assertNotIn(WILDCARD_ATTRIBUTE, chosen)

    def test_the_open_question_is_asked_first_when_enabled(self):
        """P4 asked it last, on E3's finding. P6 asks it first, on E8's:
        the simulator answers it with *any* undisclosed constraint, so its
        yield dominates every specific attribute at every point, and queueing
        it behind six narrower questions left 44 of the 200 sessions not
        finishing their card until turn 8 (MTTC 3.900 -> 3.655)."""
        state = self.state()
        chosen = []
        while True:
            attribute = choose_attribute(state, [], allow_wildcard=True)
            if attribute is None:
                break
            chosen.append(attribute)
            state.asked_attributes.add(attribute)
        self.assertEqual(chosen[0], WILDCARD_ATTRIBUTE)
        self.assertNotIn(WILDCARD_ATTRIBUTE, chosen[1:])
        # Every specific attribute is still asked behind it. E3's warning was
        # about running *out* of questions, and that failure mode is what
        # keeping them all available prevents.
        self.assertEqual(len(chosen), len(ATTRIBUTE_PRIOR_YIELD))

    def test_prefers_the_highest_yielding_attribute_with_no_pool_signal(self):
        self.assertEqual(choose_attribute(self.state(), []), "feature")


class CoverageTest(unittest.TestCase):
    def test_absent_values_are_not_counted_as_disagreement(self):
        """P4-T3: candidates that simply do not state a colour must not look
        like candidates that state different colours."""
        silent = [product(f"p{index}", description="a product") for index in range(10)]
        coverage, disagreement = attribute_coverage(silent, "color")
        self.assertEqual(coverage, 0.0)
        self.assertEqual(disagreement, 0.0)

    def test_agreeing_candidates_score_no_disagreement(self):
        same = [product(f"p{index}", description="a black shoe") for index in range(10)]
        coverage, disagreement = attribute_coverage(same, "color")
        self.assertEqual(coverage, 1.0)
        self.assertEqual(disagreement, 0.0)

    def test_split_candidates_score_disagreement(self):
        colors = ["black", "white", "blue", "red"] * 3
        split = [product(f"p{i}", description=f"a {c} shoe") for i, c in enumerate(colors)]
        coverage, disagreement = attribute_coverage(split, "color")
        self.assertEqual(coverage, 1.0)
        self.assertGreater(disagreement, 0.5)

    def test_low_coverage_suppresses_disagreement(self):
        pool = [product("a", description="a black shoe"), product("b", description="a white shoe")]
        pool += [product(f"p{index}", description="a product") for index in range(18)]
        coverage, disagreement = attribute_coverage(pool, "color")
        self.assertLess(coverage, 0.2)
        self.assertEqual(disagreement, 0.0)

    def test_empty_pool_is_neutral(self):
        self.assertEqual(attribute_coverage([], "color"), (0.0, 0.0))


class LeadInStrippingTest(unittest.TestCase):
    """The lead-in cap was 60 characters, and a Buying opener's colon sits
    after the category name -- past 60 for 27 of the 87 such openers on the
    public set. Those sessions dropped their hard constraint instead of
    keeping it."""

    def absorb(self, message, asked="feature"):
        state = SessionState("demo", {})
        state.pending_attribute = asked
        ConversationOrchestrator._absorb_answer(state, message, False)
        return state.disclosed_text

    def test_a_long_category_opener_still_yields_its_constraint(self):
        message = (
            "I'm looking for Novelty Costumes Accessories Hats Caps. "
            "A key requirement is: 100% Leather."
        )
        self.assertEqual(self.absorb(message, asked=None), ["100% Leather."])

    def test_a_short_category_opener_behaves_the_same(self):
        message = "I'm looking for Shoes. A key requirement is: 100% Leather."
        self.assertEqual(self.absorb(message, asked=None), ["100% Leather."])

    # P5 stores each quoted constraint separately rather than as one blob, so
    # the evidence feature can score coverage of each. The framing is still
    # what gets stripped; only the granularity of what is kept changed.
    def test_a_disclosure_keeps_only_the_constraints(self):
        message = "For that, what matters is: 100% Cotton; Imported."
        self.assertEqual(self.absorb(message), ["100% Cotton", "Imported."])

    def test_a_colon_inside_a_value_is_not_treated_as_framing(self):
        message = "For that, what matters is: color: black; Imported."
        self.assertEqual(self.absorb(message), ["color: black", "Imported."])

    def test_an_empty_reply_contributes_nothing(self):
        self.assertEqual(
            self.absorb("Those options are not quite right yet. Ask me about one specific attribute."),
            [],
        )

    def test_a_vague_opener_is_not_persisted(self):
        self.assertEqual(self.absorb("I'm looking for boots, but I'm still exploring.", asked=None), [])


class AnswerReplacedTest(unittest.TestCase):
    """Only a reversal stated up front displaces the answer to our question.
    detect_override_cue() is deliberately broad -- it fires on "instead" or
    "no longer" anywhere in the text -- and disclosures are raw product copy,
    so the anchor is what stops an answered question being re-asked."""

    def asked_after(self, message):
        state = SessionState("demo", {})
        state.pending_attribute = "feature"
        state.asked_attributes.add("feature")
        ConversationOrchestrator._absorb_answer(state, message, True)
        return state.asked_attributes

    def test_a_leading_reversal_reopens_the_question(self):
        self.assertNotIn("feature", self.asked_after("Actually, I want boots instead."))

    def test_a_leading_ignore_reopens_the_question(self):
        self.assertNotIn(
            "feature", self.asked_after("Ignore my previous request; I want boots instead.")
        )

    def test_the_phrase_buried_in_a_disclosure_does_not(self):
        """Regression: the "ignore ..." alternative was unanchored, so this
        product copy discarded an answer we had actually received."""
        message = (
            "For that, what matters is: wipe clean with a damp cloth; "
            "ignore the previous packaging note, it no longer applies."
        )
        self.assertIn("feature", self.asked_after(message))


class InterpretReplyTest(unittest.TestCase):
    def test_an_empty_answer_rejects_the_attribute(self):
        rejected, boundary = interpret_reply("I don't have an additional preference for color.")
        self.assertEqual(rejected, "color")
        self.assertFalse(boundary)

    def test_a_boundary_no_preference_does_not_reject(self):
        """The Boundary scenario burns the first question regardless of which
        attribute was asked, so that attribute may still hold an answer."""
        rejected, boundary = interpret_reply(
            "I don't have a preference for color; please use your judgment."
        )
        self.assertIsNone(rejected)
        self.assertTrue(boundary)

    def test_a_disclosure_rejects_nothing(self):
        rejected, boundary = interpret_reply("For that, what matters is: 100% Cotton; Machine wash.")
        self.assertIsNone(rejected)
        self.assertFalse(boundary)

    def test_the_no_question_nudge_rejects_nothing(self):
        rejected, boundary = interpret_reply(
            "Those options are not quite right yet. Ask me about one specific attribute."
        )
        self.assertIsNone(rejected)
        self.assertFalse(boundary)

    def test_unknown_attribute_names_are_ignored(self):
        rejected, _ = interpret_reply("I don't have an additional preference for vibes.")
        self.assertIsNone(rejected)


if __name__ == "__main__":
    unittest.main()
