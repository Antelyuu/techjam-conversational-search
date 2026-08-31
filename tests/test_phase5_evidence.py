import unittest

from shopping_agent.catalog import normalize_product
from shopping_agent.contracts import Candidate, SessionState
from shopping_agent.evidence import (
    MIN_EVIDENCE_TOKENS,
    coverage,
    split_disclosures,
)
from shopping_agent.orchestrator import ConversationOrchestrator
from shopping_agent.reranking import FEATURE_WEIGHTS, score_candidate


def product(parent_asin, **overrides):
    raw = {
        "parent_asin": parent_asin,
        "title": "Running shoes",
        "categories": ["Shoes"],
        "features": ["Rubber sole", "Machine wash cold with like colors"],
        "details": {"Material": "100% Cotton canvas upper"},
        "store": "Example",
        "description": "Shoes for running",
        "price": "50.00",
    }
    raw.update(overrides)
    return normalize_product(raw)


class SplitDisclosuresTest(unittest.TestCase):
    def test_the_simulator_joins_constraints_with_semicolons(self):
        self.assertEqual(
            split_disclosures("100% Cotton canvas upper; Machine wash cold"),
            ["100% Cotton canvas upper", "Machine wash cold"],
        )

    def test_a_single_constraint_is_left_whole(self):
        self.assertEqual(split_disclosures("Rubber sole"), ["Rubber sole"])

    def test_empty_parts_are_dropped(self):
        self.assertEqual(split_disclosures(" ; ; "), [])


class CoverageTest(unittest.TestCase):
    def test_no_disclosures_is_neutral(self):
        self.assertEqual(coverage(product("a"), []), 0.0)

    def test_a_quoted_feature_scores_full_coverage(self):
        """The evaluator builds its intent card verbatim from the target's own
        features, so an exact quote is the signal this feature exists for."""
        self.assertEqual(coverage(product("a"), ["Machine wash cold with like colors"]), 1.0)

    def test_an_unrelated_disclosure_scores_nothing(self):
        self.assertEqual(coverage(product("a"), ["stainless steel bezel wristwatch"]), 0.0)

    def test_a_partial_quote_scores_between(self):
        value = coverage(product("a"), ["Machine wash cold with zebra unicorn"])
        self.assertGreater(value, 0.0)
        self.assertLess(value, 1.0)

    def test_a_short_label_counts_as_evidence(self):
        """The Buying opener discloses the card's first constraint, which is
        usually a bare material or color label. Ignoring those cost 4 points
        of HitRate (E6); they count now, down-weighted by their own length."""
        self.assertEqual(MIN_EVIDENCE_TOKENS, 1)
        self.assertEqual(coverage(product("a"), ["cotton canvas"]), 1.0)

    def test_an_all_stopword_disclosure_is_ignored(self):
        """No content tokens means nothing to cover -- and must not divide
        by zero."""
        self.assertEqual(coverage(product("a"), ["with your, and the"]), 0.0)

    def test_a_longer_disclosure_outweighs_a_shorter_one(self):
        """Flat averaging would let a trivial match outvote a specific one."""
        long_hit_short_miss = coverage(
            product("a"),
            ["Machine wash cold with like colors", "brass zipper pull tab"],
        )
        long_miss_short_hit = coverage(
            product("a"),
            ["mahogany walnut dovetail drawer joint", "Rubber sole here"],
        )
        self.assertGreater(long_hit_short_miss, long_miss_short_hit)


class EvidenceFeatureTest(unittest.TestCase):
    def candidate(self, parent_asin):
        return Candidate(
            parent_asin=parent_asin,
            route_ranks={"lexical": 1},
            route_scores={"lexical": 1.0, "combined": 1.0},
            matched_hard_constraints=(),
            matched_soft_preferences=(),
        )

    def test_the_feature_is_reported_even_with_no_disclosures(self):
        scored = score_candidate(self.candidate("a"), product("a"), {})
        self.assertIn("constraint_evidence", [c.feature for c in scored.contributions])

    def test_evidence_outranks_a_better_retrieval_rank(self):
        """The whole point of the feature: a candidate the customer has
        effectively described beats one BM25 merely liked."""
        described = Candidate(
            parent_asin="a", route_ranks={"lexical": 40},
            route_scores={"lexical": 0.1, "combined": 0.1},
            matched_hard_constraints=(), matched_soft_preferences=(),
        )
        top_ranked = self.candidate("b")
        disclosures = ["Machine wash cold with like colors"]
        a = score_candidate(described, product("a"), {}, disclosures)
        b = score_candidate(top_ranked, product("b", features=["Rubber sole"]), {}, disclosures)
        self.assertGreater(a.score, b.score)

    def test_with_no_disclosures_the_feature_cannot_reorder(self):
        a = score_candidate(self.candidate("a"), product("a"), {}, [])
        b = score_candidate(self.candidate("b"), product("b", features=["Nothing alike"]), {}, [])
        by_feature = {c.feature: c.value for c in a.contributions}
        self.assertEqual(by_feature["constraint_evidence"], 0.0)
        self.assertEqual(a.score, b.score)

    # What the customer actually told us, however it reached us: the
    # constraints they quoted (the three evidence features) and the category
    # they named (category_exact). Everything else is generic retrieval and
    # constraint machinery.
    # P7 retired constraint_evidence to 0.0 (E11) and put semantic_evidence in
    # its place, so the set of *live* disclosure features has changed
    # membership. constraint_evidence stays in the table for inspectability
    # but is deliberately excluded here: at 0.0 it carries no disclosure
    # signal, and including it would make the invariant below unsatisfiable
    # rather than false.
    DISCLOSURE_FEATURES = frozenset({
        "semantic_evidence", "slot_evidence", "phrase_evidence", "category_exact",
    })
    RETIRED_FEATURES = frozenset({"constraint_evidence", "category"})

    def test_what_the_customer_said_outranks_the_structural_features(self):
        """P5 pinned constraint_evidence as the single largest weight. P6
        displaced it twice -- slot_evidence (+0.0316) then category_exact
        (+0.0475) -- and P7 displaced it again with semantic_evidence, which
        asks constraint_evidence's own question with a matcher that survives
        paraphrase. The invariant worth holding is the general one the numbers
        support across all four phases: signals carrying what the customer
        actually told us outrank every structural feature, and the sharpest of
        them leads."""
        ordered = sorted(FEATURE_WEIGHTS, key=FEATURE_WEIGHTS.get, reverse=True)
        self.assertEqual(ordered[0], "semantic_evidence")
        structural = set(FEATURE_WEIGHTS) - self.DISCLOSURE_FEATURES - self.RETIRED_FEATURES
        self.assertTrue(self.DISCLOSURE_FEATURES <= set(FEATURE_WEIGHTS))
        # A retired feature stays in the table so explain() can still show it;
        # if one is ever revived, this fails and the sets above get revisited.
        for name in self.RETIRED_FEATURES:
            self.assertEqual(FEATURE_WEIGHTS[name], 0.0, name)
        self.assertGreater(
            min(FEATURE_WEIGHTS[name] for name in self.DISCLOSURE_FEATURES),
            max(FEATURE_WEIGHTS[name] for name in structural),
        )


class DisclosuresReachTheStateSplitTest(unittest.TestCase):
    def test_a_multi_constraint_answer_is_stored_as_separate_disclosures(self):
        state = SessionState("demo", {})
        state.pending_attribute = "feature"
        ConversationOrchestrator._absorb_answer(
            state, "For that, what matters is: 100% Cotton; Imported.", False
        )
        self.assertEqual(state.disclosed_text, ["100% Cotton", "Imported."])


if __name__ == "__main__":
    unittest.main()
