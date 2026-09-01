"""P4-T4: a dense or reranker failure still returns a valid response."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluator.local_evaluator import ALLOWED_ATTRIBUTES, TOP_K, normalize_recommendations
from shopping_agent import clarification
from starter.agent import Agent

CATALOG_ROWS = [
    {
        "parent_asin": f"B{index:09d}",
        "title": f"Black leather running shoes {index}",
        "categories": ["Shoes", "Athletic"],
        "features": ["lightweight", "breathable mesh"],
        "details": {"Material": "leather"},
        "store": "Example Store",
        "description": "Cushioned shoes for running and hiking",
        "price": f"{40 + index}.00",
    }
    for index in range(25)
]


class FallbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._directory = tempfile.TemporaryDirectory()
        cls.catalog_path = Path(cls._directory.name) / "catalog.jsonl"
        cls.catalog_path.write_text(
            "".join(json.dumps(row) + "\n" for row in CATALOG_ROWS), encoding="utf-8"
        )
        cls.catalog_ids = {row["parent_asin"] for row in CATALOG_ROWS}

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def agent(self, **kwargs):
        kwargs.setdefault("enable_dense", False)
        # Off for the same reason dense is: these tests are not about
        # the semantic feature, and leaving it on makes every agent
        # here load a 66 MB artifact and a 340M-parameter model.
        # tests/test_phase7_semantic_evidence.py covers it directly.
        kwargs.setdefault("enable_semantic", False)
        instance = Agent(self.catalog_path, **kwargs)
        # Each test builds its own agent; without this every one leaks a
        # SQLite connection until collection, which shows up as ResourceWarning.
        self.addCleanup(instance.close)
        return instance

    def assert_contract_valid(self, response):
        self.assertIsInstance(response, dict)
        self.assertIsInstance(response["message"], str)
        self.assertIsInstance(response["recommendations"], list)
        ask = response["ask_attribute"]
        self.assertTrue(ask is None or ask in ALLOWED_ATTRIBUTES, f"illegal attribute {ask!r}")
        ranked = normalize_recommendations(response["recommendations"], self.catalog_ids)
        self.assertLessEqual(len(ranked), TOP_K)
        self.assertEqual(len(ranked), len(set(ranked)), "duplicate parent_asin emitted")

    def converse(self, agent, messages):
        agent.reset("s", {})
        responses = []
        for turn, message in enumerate(messages, start=1):
            response = agent.respond("s", message, turn, TOP_K)
            self.assert_contract_valid(response)
            responses.append(response)
        return responses

    def test_a_reranker_failure_degrades_to_the_fused_order(self):
        agent = self.agent(enable_reranker=True)
        with patch("starter.agent.rerank", side_effect=RuntimeError("boom")):
            responses = self.converse(agent, ["I'm looking for running shoes"])
        self.assertTrue(responses[0]["recommendations"], "fell back to an empty list")

    def test_a_reranker_failure_is_reported_once(self):
        agent = self.agent(enable_reranker=True)
        with patch("starter.agent.rerank", side_effect=RuntimeError("boom")):
            with patch("sys.stderr") as stderr:
                self.converse(agent, ["running shoes"] * 3)
        written = "".join(call.args[0] for call in stderr.write.call_args_list)
        self.assertEqual(written.count("reranker failed"), 1)

    def test_a_dense_failure_degrades_to_lexical(self):
        agent = self.agent(enable_dense=False)
        agent.dense_search = lambda _query, _limit: (_ for _ in ()).throw(RuntimeError("boom"))
        # The guard lives in dense_retrieval; an unguarded callable must still
        # not take the whole turn down.
        try:
            responses = self.converse(agent, ["running shoes"])
        except RuntimeError:
            self.fail("a dense route failure escaped respond()")
        self.assertTrue(responses[0]["recommendations"])

    def test_a_full_conversation_stays_contract_valid(self):
        agent = self.agent(enable_reranker=True)
        self.converse(
            agent,
            [
                "I'm looking for running shoes, but I'm still exploring.",
                "For that, what matters is: breathable mesh; lightweight.",
                "I don't have an additional preference for color.",
                "Actually, ignore my earlier preference. What I need is: leather.",
                "Those options are not quite right yet. Ask me about one specific attribute.",
            ]
            + ["Those options are not quite right yet."] * 5,
        )

    def assert_no_question_loop(self, asked):
        """The no-repeat guarantee, as it stands after E14.

        A *specific* attribute still never repeats: asking "color" twice can
        only return the same nothing, because the customer's answer to it does
        not change. The open question is exempt, because it matches whatever
        is still undisclosed and so its yield does not decay -- but only up to
        `WILDCARD_REPEAT_CAP` times in a row, so a session cannot degenerate
        into asking "what else matters?" over and over.
        """
        specific = [a for a in asked if a != clarification.WILDCARD_ATTRIBUTE]
        self.assertEqual(
            len(specific), len(set(specific)), f"repeated a specific question: {asked}"
        )
        run = 0
        for attribute in asked:
            run = run + 1 if attribute == clarification.WILDCARD_ATTRIBUTE else 0
            self.assertLessEqual(
                run,
                clarification.WILDCARD_REPEAT_CAP,
                f"asked the open question {run} times running: {asked}",
            )

    def test_questions_stop_rather_than_repeat(self):
        agent = self.agent(enable_reranker=True)
        responses = self.converse(agent, ["I'm looking for shoes"] * 10)
        asked = [r["ask_attribute"] for r in responses if r["ask_attribute"] is not None]
        self.assert_no_question_loop(asked)

    def test_an_override_word_inside_a_disclosure_does_not_reopen_a_question(self):
        """Disclosures are raw product copy, so "instead" or "no longer" turn
        up inside them. Only a reversal stated up front displaces an answer."""
        agent = self.agent(enable_reranker=True)
        responses = self.converse(
            agent,
            [
                "I'm looking for shoes",
                "For that, what matters is: wear these instead of boots; no longer stiff.",
                "For that, what matters is: breathable mesh.",
                "For that, what matters is: lightweight.",
            ],
        )
        asked = [r["ask_attribute"] for r in responses if r["ask_attribute"] is not None]
        self.assert_no_question_loop(asked)

    def test_a_real_override_reopens_the_displaced_question(self):
        """The override message arrives instead of the answer, so the question
        it displaced is unanswered and gets asked again straight away."""
        agent = self.agent(enable_reranker=True)
        agent.reset("s", {})
        first = agent.respond("s", "I'm looking for shoes", 1, TOP_K)
        second = agent.respond(
            "s", "Actually, ignore my earlier preference. What I need is: leather.", 2, TOP_K
        )
        self.assertEqual(second["ask_attribute"], first["ask_attribute"])

    def test_an_empty_query_still_returns_a_valid_response(self):
        agent = self.agent(enable_reranker=True)
        self.converse(agent, ["", "   "])


if __name__ == "__main__":
    unittest.main()


class WildcardCapBindsUnconditionallyTest(unittest.TestCase):
    """E14 review finding: the cap has to bind however the wildcard arrived.

    The first implementation tested `exempt and consecutive_wildcard >= CAP`,
    which short-circuits whenever `exempt` is False -- and the wildcard is
    available without the exemption in two real situations: the first ask of a
    session, and after an override calls `asked_attributes.discard()` so the
    displaced question can be re-asked. Measured on a paraphrased replay, 17 of
    200 sessions ran the open question four times consecutively against a cap
    of three before this was fixed.
    """

    def state(self, consecutive, asked):
        from shopping_agent.contracts import SessionState

        s = SessionState(session_id="s", user_profile={})
        s.consecutive_wildcard = consecutive
        s.asked_attributes = set(asked)
        return s

    def choose(self, state, paraphrasing):
        return clarification.choose_attribute(
            state, [], allow_wildcard=True, paraphrasing=paraphrasing
        )

    def test_at_the_cap_the_wildcard_is_refused_even_when_unasked(self):
        # `asked` is empty, so nothing but the cap can withhold the wildcard.
        for paraphrasing in (True, False):
            with self.subTest(paraphrasing=paraphrasing):
                chosen = self.choose(
                    self.state(clarification.WILDCARD_REPEAT_CAP, set()), paraphrasing
                )
                self.assertNotEqual(chosen, clarification.WILDCARD_ATTRIBUTE)

    def test_below_the_cap_the_wildcard_is_still_available(self):
        chosen = self.choose(
            self.state(clarification.WILDCARD_REPEAT_CAP - 1, set()), True
        )
        self.assertEqual(chosen, clarification.WILDCARD_ATTRIBUTE)

    def test_a_specific_question_resets_the_run(self):
        """The documented behaviour, asserted so the limitation is explicit:
        the cap bounds a *run*, not a session total."""
        from shopping_agent.orchestrator import ConversationOrchestrator

        s = self.state(clarification.WILDCARD_REPEAT_CAP, set())
        ConversationOrchestrator.record_question(s, "material")
        self.assertEqual(s.consecutive_wildcard, 0)
        self.assertEqual(self.choose(s, True), clarification.WILDCARD_ATTRIBUTE)
