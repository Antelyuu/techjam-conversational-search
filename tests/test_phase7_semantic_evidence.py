"""P7-T1: the semantic evidence feature and its failure paths.

These tests build their own tiny artifact and inject a deterministic encoder,
so nothing here downloads a model or reads the 66 MB shipped artifact. What is
being checked is the wiring and the degradation behaviour, both of which are
what make the feature safe to have on by default -- the *quality* claim is
E11's and lives in docs/experiments/, because it needs the full public set.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from shopping_agent import reranking
from shopping_agent.catalog import ProductRecord
from shopping_agent.contracts import Candidate
from shopping_agent.semantic_evidence import (
    SemanticEvidenceUnavailable,
    SemanticScorer,
    load_semantic_scorer,
    values_digest,
)

# A three-dimensional toy space: one axis per "meaning", so a query aimed at an
# axis matches the value on that axis exactly and the others not at all.
VALUES = ["cotton fabric", "leather upper", "steel buckle"]
AXES = {
    "cotton fabric": [1.0, 0.0, 0.0],
    "leather upper": [0.0, 1.0, 0.0],
    "steel buckle": [0.0, 0.0, 1.0],
}
# The paraphrase a lexical feature cannot see, pointed at the cotton axis.
QUERIES = {
    "natural plant fibre": [1.0, 0.0, 0.0],
    "tanned animal hide": [0.0, 1.0, 0.0],
    "nothing in this catalogue": [0.0, 0.0, 0.0],
}


def encoder(texts):
    return np.array([QUERIES.get(text, [0.0, 0.0, 0.0]) for text in texts], dtype="float32")


def product(asin, values):
    """Built directly rather than through normalize_product, because these
    tests need exact control over card_values -- that set is what the feature
    scores against."""
    return ProductRecord(
        parent_asin=asin,
        title=asin,
        categories=("Clothing",),
        price=None,
        price_is_lower_bound=False,
        searchable_text=" | ".join(values),
        card_values=frozenset(values),
        coarse_category="Clothing",
    )


def candidate(asin, rank=1):
    return Candidate(
        parent_asin=asin,
        route_ranks={"lexical": rank},
        route_scores={"lexical": 1.0},
        matched_hard_constraints=(),
        matched_soft_preferences=(),
    )


class ArtifactTestCase(unittest.TestCase):
    """Writes a float32 or int8 artifact into a temporary embedding dir."""

    def temp_dir(self):
        handle = tempfile.TemporaryDirectory()
        self.addCleanup(handle.cleanup)
        return Path(handle.name)

    def artifact(self, precision="float32"):
        directory = self.temp_dir()
        matrix = np.array([AXES[value] for value in VALUES], dtype="float32")
        metadata = {"values": len(VALUES), "dimensions": 3}
        if precision == "int8":
            scale = 127.0 / float(np.abs(matrix).max())
            stored = np.clip(np.round(matrix * scale), -127, 127).astype("int8")
            metadata["quantization_scale"] = scale
            metadata["precision"] = "int8"
        else:
            stored = matrix
        slug = "voyage-4-nano-values256"
        np.save(directory / f"{slug}.npy", stored)
        (directory / f"{slug}.values.json").write_text(json.dumps(VALUES))
        (directory / f"{slug}.meta.json").write_text(json.dumps(metadata))
        return directory


class SemanticScorerTest(ArtifactTestCase):
    def scorer(self, precision="float32", **kwargs):
        return SemanticScorer(self.artifact(precision), encoder=encoder, **kwargs)

    def pool(self):
        products = {
            "cotton": product("cotton", ["cotton fabric"]),
            "leather": product("leather", ["leather upper"]),
            "both": product("both", ["cotton fabric", "steel buckle"]),
        }
        candidates = [candidate(asin, i + 1) for i, asin in enumerate(products)]
        return candidates, products

    def test_a_paraphrase_scores_the_product_that_owns_the_meaning(self):
        """The whole point: 'natural plant fibre' shares no token with 'cotton
        fabric', so constraint_evidence scores it 0, and this scores it 1."""
        candidates, products = self.pool()
        scores = self.scorer().score_pool(
            candidates, products, [("natural plant fibre", 2.0)]
        )
        self.assertAlmostEqual(scores["cotton"], 1.0, places=5)
        self.assertAlmostEqual(scores["leather"], 0.0, places=5)

    def test_the_best_matching_value_wins_not_the_average(self):
        """`both` owns an irrelevant second value; taking the max over values
        rather than the mean is what stops that diluting the match."""
        candidates, products = self.pool()
        scores = self.scorer().score_pool(
            candidates, products, [("natural plant fibre", 2.0)]
        )
        self.assertAlmostEqual(scores["both"], scores["cotton"], places=5)

    def test_disclosures_are_weighted_by_their_token_count(self):
        """A two-token disclosure the candidate matches and a six-token one it
        does not must not average to a half."""
        candidates, products = self.pool()
        scores = self.scorer().score_pool(
            candidates,
            products,
            [("natural plant fibre", 2.0), ("tanned animal hide", 6.0)],
        )
        self.assertAlmostEqual(scores["cotton"], 2.0 / 8.0, places=5)
        self.assertAlmostEqual(scores["leather"], 6.0 / 8.0, places=5)

    def test_a_disclosure_matching_nothing_scores_zero_for_everyone(self):
        """The quiet-failure property the weight of 192 rests on."""
        candidates, products = self.pool()
        scores = self.scorer().score_pool(
            candidates, products,
            [("nothing in this catalogue", 4.0)],
        )
        self.assertEqual(set(scores.values()), {0.0})

    def test_no_usable_disclosure_scores_nothing_at_all(self):
        candidates, products = self.pool()
        self.assertEqual(self.scorer().score_pool(candidates, products, []), {})

    def test_int8_storage_agrees_with_float32(self):
        """The artifact ships quantized to fit the repository; E11 measured the
        difference on the full set at six millionths, and it should be inside
        rounding here."""
        candidates, products = self.pool()
        weights = [("natural plant fibre", 2.0)]
        exact = self.scorer("float32").score_pool(
            candidates, products, weights
        )
        quantized = self.scorer("int8").score_pool(
            candidates, products, weights
        )
        for asin in exact:
            self.assertAlmostEqual(exact[asin], quantized[asin], places=4, msg=asin)

    def test_a_candidate_missing_from_the_catalogue_is_skipped(self):
        candidates, products = self.pool()
        candidates.append(candidate("ghost", 9))
        scores = self.scorer().score_pool(
            candidates, products, [("natural plant fibre", 2.0)]
        )
        self.assertNotIn("ghost", scores)


class DegradationTest(ArtifactTestCase):
    """Every path that must return None or {} rather than raise."""

    def test_a_missing_artifact_disables_the_feature_rather_than_failing(self):
        self.assertIsNone(load_semantic_scorer(self.temp_dir()))

    def test_strict_mode_raises_so_a_build_script_can_tell(self):
        with self.assertRaises(SemanticEvidenceUnavailable):
            load_semantic_scorer(self.temp_dir(), strict=True)

    def test_an_artifact_that_disagrees_with_its_value_list_is_refused(self):
        directory = self.artifact()
        slug = "voyage-4-nano-values256"
        np.save(directory / f"{slug}.npy", np.zeros((2, 3), dtype="float32"))
        with self.assertRaises(SemanticEvidenceUnavailable):
            SemanticScorer(directory, encoder=encoder)

    def test_an_int8_artifact_without_its_scale_is_refused(self):
        """Loading it anyway would score every candidate off vectors 388x too
        long -- wrong rather than absent, which is the worse failure."""
        directory = self.artifact("int8")
        slug = "voyage-4-nano-values256"
        (directory / f"{slug}.meta.json").write_text(json.dumps({"precision": "int8"}))
        with self.assertRaises(SemanticEvidenceUnavailable):
            SemanticScorer(directory, encoder=encoder)

    def test_an_artifact_from_another_catalogue_is_refused(self):
        """A count match is not an identity match. The row labelling is derived
        from the catalogue rather than shipped, so without this a catalogue with
        the same number of distinct values would map every row to the wrong
        string -- silently, at the heaviest weight in the table."""
        directory = self.artifact()
        slug = "voyage-4-nano-values256"
        metadata = json.loads((directory / f"{slug}.meta.json").read_text())
        metadata["values_sha256"] = values_digest(VALUES)
        (directory / f"{slug}.meta.json").write_text(json.dumps(metadata))

        # Same count, different strings.
        with self.assertRaises(SemanticEvidenceUnavailable):
            SemanticScorer(directory, encoder=encoder,
                           expected_values=["a", "b", "c"])
        # The real labelling still loads.
        SemanticScorer(directory, encoder=encoder, expected_values=list(VALUES))

    def test_a_missing_dependency_disables_the_feature_at_load(self):
        """Requirement: no sentence-transformers must mean None from
        load_semantic_scorer, so the operator hears about it at startup rather
        than on the first turn that happens to carry a disclosure."""
        real_find_spec = importlib.util.find_spec

        def missing(name, *args, **kwargs):
            if name == "sentence_transformers":
                return None
            return real_find_spec(name, *args, **kwargs)

        directory = self.artifact()
        with mock.patch.object(importlib.util, "find_spec", missing):
            self.assertIsNone(load_semantic_scorer(directory))
            # An injected encoder needs no such dependency, so it still loads.
            self.assertIsNotNone(load_semantic_scorer(directory, encoder=encoder))

    def test_the_encoder_s_own_array_is_not_normalized_in_place(self):
        """`encoder` is a public seam, so an injected one may hand back a cached
        or shared array. Normalizing it in place would corrupt the caller's copy
        and make the second call read a doubly-normalized value."""
        handed_out = np.array([[3.0, 4.0, 0.0]], dtype="float32")
        scorer = SemanticScorer(self.artifact(), encoder=lambda texts: handed_out)
        scorer._embed(["anything"])
        np.testing.assert_array_equal(handed_out, np.array([[3.0, 4.0, 0.0]], "float32"))

    def test_an_encoder_that_raises_costs_the_feature_and_not_the_turn(self):
        def broken(texts):
            raise RuntimeError("boom")

        scorer = load_semantic_scorer(self.artifact(), encoder=broken)
        self.assertIsNotNone(scorer)
        products = {"cotton": product("cotton", ["cotton fabric"])}
        candidates = [candidate("cotton")]
        self.assertEqual(scorer(candidates, products, [("x", 1.0)]), {})


class RerankingIntegrationTest(ArtifactTestCase):
    def test_prepare_evidence_carries_the_scores_into_the_ranking(self):
        products = {
            "cotton": product("cotton", ["cotton fabric"]),
            "leather": product("leather", ["leather upper"]),
        }
        candidates = [candidate(asin, i + 1) for i, asin in enumerate(products)]
        scorer = load_semantic_scorer(self.artifact(), encoder=encoder)

        prepared = reranking.prepare_evidence(
            ["natural plant fibre"], candidates, products, semantic_scorer=scorer
        )
        self.assertAlmostEqual(prepared.semantic["cotton"], 1.0, places=5)

        ranked = reranking.rerank(candidates, products, {}, 10, prepared=prepared)
        self.assertEqual(ranked[0].parent_asin, "cotton")
        contribution = {c.feature: c.value for c in ranked[0].contributions}
        self.assertAlmostEqual(contribution["semantic_evidence"], 1.0, places=5)

    def test_without_a_scorer_the_feature_is_inert(self):
        """No scorer must mean the ranking this agent had before P7, not a
        different one -- so the feature is present, scored 0.0, and orders
        nothing."""
        products = {
            "cotton": product("cotton", ["cotton fabric"]),
            "leather": product("leather", ["leather upper"]),
        }
        candidates = [candidate(asin, i + 1) for i, asin in enumerate(products)]
        prepared = reranking.prepare_evidence(["natural plant fibre"], candidates, products)
        self.assertEqual(prepared.semantic, {})

        ranked = reranking.rerank(candidates, products, {}, 10, prepared=prepared)
        values = {
            item.parent_asin: {c.feature: c.value for c in item.contributions}[
                "semantic_evidence"
            ]
            for item in ranked
        }
        self.assertEqual(set(values.values()), {0.0})

    def gate_pool(self):
        products = {
            "cotton": product("cotton", ["cotton fabric"]),
            "leather": product("leather", ["leather upper"]),
        }
        candidates = [candidate(asin, i + 1) for i, asin in enumerate(products)]
        return candidates, products, load_semantic_scorer(self.artifact(), encoder=encoder)

    def test_a_disclosure_someone_owns_silences_the_semantic_feature(self):
        """The gate that makes this free on a quoting customer: when slot
        ownership has the answer, the semantic feature must not get a vote --
        at weight 192 it would outvote slot_evidence's 16."""
        candidates, products, scorer = self.gate_pool()
        prepared = reranking.prepare_evidence(
            ["cotton fabric"], candidates, products, semantic_scorer=scorer
        )
        self.assertEqual(prepared.live_disclosures, 1)
        self.assertEqual(set(prepared.semantic.values()), {0.0})

    def test_a_disclosure_nobody_owns_leaves_the_feature_at_full_voice(self):
        candidates, products, scorer = self.gate_pool()
        prepared = reranking.prepare_evidence(
            ["natural plant fibre"], candidates, products, semantic_scorer=scorer
        )
        self.assertEqual(prepared.live_disclosures, 0)
        self.assertAlmostEqual(prepared.semantic["cotton"], 1.0, places=5)

    def test_a_partly_reworded_turn_scales_the_feature_rather_than_switching_it(self):
        """One of two disclosures owned means the feature speaks at half
        voice. This is the whole reason the soft gate beats the hard one: a
        hard switch would silence it exactly when it is half needed."""
        candidates, products, scorer = self.gate_pool()
        prepared = reranking.prepare_evidence(
            ["cotton fabric", "natural plant fibre"],
            candidates,
            products,
            semantic_scorer=scorer,
        )
        self.assertEqual(prepared.live_disclosures, 1)
        # "cotton fabric" weighs 2 tokens and "natural plant fibre" 3, of 5
        # total. Only the second is a query the encoder places on the cotton
        # axis, so ungated the product scores 3/5. One of the two disclosures
        # is owned, so the gate halves it.
        self.assertAlmostEqual(prepared.semantic["cotton"], (3.0 / 5.0) * 0.5, places=5)

    def test_the_feature_is_in_the_table_and_leads_it(self):
        self.assertIn("semantic_evidence", reranking.FEATURE_WEIGHTS)
        self.assertEqual(
            max(reranking.FEATURE_WEIGHTS, key=reranking.FEATURE_WEIGHTS.get),
            "semantic_evidence",
        )


if __name__ == "__main__":
    unittest.main()
