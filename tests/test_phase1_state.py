import unittest

from shopping_agent.contracts import SessionState
from shopping_agent.intent import (
    classify_intent,
    detect_override_cue,
    extract_candidate_slots,
)
from shopping_agent.state import SessionStore, apply_candidates, build_query_text


class Phase1StateTest(unittest.TestCase):
    def test_sessions_are_isolated(self):
        store = SessionStore()
        first = store.create("session-1", {"rating_style": "high"})
        second = store.create("session-2", {"rating_style": "high"})

        first.history.append("I want shoes")

        self.assertEqual(first.history, ["I want shoes"])
        self.assertEqual(second.history, [])
        self.assertIsNot(first, second)

    def test_missing_session_requires_reset(self):
        with self.assertRaisesRegex(RuntimeError, "reset must be called"):
            SessionStore().get("missing-session")

    def test_slots_accumulate_across_turns(self):
        state = SessionState(session_id="demo", user_profile={})

        apply_candidates(state, extract_candidate_slots("I need shoes"), 1)
        apply_candidates(state, extract_candidate_slots("Black running shoes"), 2)
        apply_candidates(state, extract_candidate_slots("Under $120"), 3)

        self.assertEqual(state.constraints["category"].value, "shoes")
        self.assertEqual(state.constraints["color"].value, "black")
        self.assertEqual(state.constraints["use_case"].value, "running")
        self.assertEqual(state.constraints["budget"].value, 120.0)
        self.assertIn("black", build_query_text(state, "Under $120"))

    def test_same_slot_value_is_replaced(self):
        state = SessionState(session_id="demo", user_profile={})

        apply_candidates(state, extract_candidate_slots("I need black shoes"), 1)
        apply_candidates(state, extract_candidate_slots("Actually, brown shoes"), 2)

        self.assertEqual(state.constraints["color"].value, "brown")
        self.assertEqual(state.constraints["color"].source_turn, 2)

    def test_category_change_clears_dependent_slots(self):
        state = SessionState(session_id="demo", user_profile={})

        apply_candidates(state, extract_candidate_slots("I need running shoes"), 1)
        apply_candidates(state, extract_candidate_slots("Actually, I want a necklace"), 2)

        self.assertEqual(state.constraints["category"].value, "necklace")
        self.assertNotIn("use_case", state.constraints)

    def test_common_slots_are_extracted(self):
        candidates = dict(
            (attribute, (value, strength))
            for attribute, value, strength in extract_candidate_slots(
                "I need black Nike running shoes under $120"
            )
        )

        self.assertEqual(candidates["category"][0], "shoes")
        self.assertEqual(candidates["color"][0], "black")
        self.assertEqual(candidates["brand"][0], "Nike")
        self.assertEqual(candidates["use_case"][0], "running")
        self.assertEqual(candidates["budget"][0], 120.0)

    def test_intent_routing_and_override_detection(self):
        buying = extract_candidate_slots("I need black shoes")
        browsing = extract_candidate_slots("I am just browsing for something casual")

        self.assertEqual(classify_intent("I need black shoes", buying, False), "buying")
        self.assertEqual(classify_intent("I am just browsing for something casual", browsing, False), "browsing")
        self.assertTrue(detect_override_cue("Actually, I want boots instead"))


class SizeExtractionTest(unittest.TestCase):
    """Regression: \\b treats an apostrophe as a word boundary, so bare
    single-letter sizes matched inside contractions. Every evaluator session
    opens with "I'm looking for ...", which set size="m" on all 200."""

    def sizes(self, text):
        return [value for attribute, value, _ in extract_candidate_slots(text) if attribute == "size"]

    def test_contractions_are_not_sizes(self):
        self.assertEqual(self.sizes("I'm looking for boots, but I'm still exploring."), [])
        self.assertEqual(self.sizes("It's a gift and I'll need it soon"), [])

    def test_single_letter_sizes_need_a_size_cue(self):
        self.assertEqual(self.sizes("I need size M"), ["m"])
        self.assertEqual(self.sizes("size: L please"), ["l"])

    def test_word_sizes_still_match_bare(self):
        self.assertEqual(self.sizes("a medium sweater"), ["medium"])
        self.assertEqual(self.sizes("I'll take the wide fit"), ["wide"])

    def test_numeric_sizes_still_match(self):
        self.assertEqual(self.sizes("I need size 10"), ["10"])


if __name__ == "__main__":
    unittest.main()
