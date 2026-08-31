"""P7-T1: score a disclosure against a candidate's field values by meaning.

`constraint_evidence` asks how many of the disclosure's content tokens appear
anywhere in the candidate's text. E10 measured what that is worth once the
customer stops quoting and found it *negative*: the feature scores partial
overlap, so rewording does not silence it, it fills it with whichever words
happened to survive. Zeroing the feature made the paraphrased score go up,
0.696015 -> 0.711681. At weight 12 that was noise with a loud voice.

This asks the same question with a matcher that survives rewording:

    value(p) = sum_d w_d * max_v cos(emb(d), emb(v)) / sum_d w_d

over the disclosures `d` and the candidate's own **card values** `v`.

Two choices, both inherited from features that already work rather than
invented here:

  * **Card values, not product text.** `slot_evidence` (weight 16, the
    strongest feature in the table) asks whether a candidate *owns* the
    disclosed string as a whole field value, and E8 showed that granularity is
    what makes it sharp -- "machine wash cold" inside an impostor's longer
    bullet is not evidence, the same string standing alone as one of its
    values is. This asks that structural question semantically, so where slot
    ownership goes silent this degrades instead of falling back to
    bag-of-words.
  * **`w_d` is the disclosure's distinct content-token count**, exactly the
    weight `evidence.coverage_from_sets` uses, so the two features price each
    disclosure identically and only the matching rule differs.

MEASURED (E11), full public set replayed through the paraphrasing customer:

    condition                        public     paraphrased  HitRate
    shipped (token containment)      0.945497   0.696015     0.805
    feature deleted                  0.945297   0.711681     0.825
    this, voyage-4-nano @192         0.944430   0.862054     0.975

and the target's median rank under paraphrase moves 28 -> 7 of 400 while pool
recall stays 1.000, so what moved is ranking rather than retrieval -- which is
what E10 said the whole remaining gap was.

**Optional by construction, exactly like dense_retrieval.** If the artifact is
missing, or numpy/sentence-transformers/torch are not installed, or the model
cannot be loaded, `load_semantic_scorer()` returns None, the feature scores
0.0 for every candidate alike, and the ordering falls through to the rest of
the table -- which is the agent as it shipped before this module existed. The
reason is written to stderr rather than swallowed, because a silently missing
scorer looks exactly like a working agent that scores worse.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

from .catalog import ProductRecord
from .contracts import Candidate
from .embedding_config import (
    EMBEDDING_DIR,
    VALUE_MODEL_ID,
    value_metadata_path,
    value_strings_path,
    value_vectors_path,
)

# (disclosures, candidates, products) -> {parent_asin: 0-1 score}.
SemanticScoreFn = Callable[[list[str], list[Candidate], dict], dict]


class SemanticEvidenceUnavailable(RuntimeError):
    """The scorer cannot run: missing artifact, missing dependency, or an
    artifact that does not line up with its value list."""


class SemanticScorer:
    """Holds the value artifact and the query encoder, and scores a pool."""

    def __init__(
        self,
        embedding_dir: str | Path = EMBEDDING_DIR,
        model_id: str = VALUE_MODEL_ID,
        encoder: Callable[[list[str]], object] | None = None,
        expected_values: list[str] | None = None,
    ) -> None:
        """`encoder` overrides how queries are embedded.

        Left None the real model is loaded on first use. Supplying one is how
        the tests exercise this module without a 340M-parameter download, and
        is the seam that keeps them hermetic.

        `expected_values` is the row labelling, derived from the catalogue by
        the caller. The build writes the same list in sorted order, so passing
        it here means the 19 MB `.values.json` never has to ship: the agent
        already holds the catalogue those strings come from. It doubles as the
        artifact-matches-catalogue guard, the same role `expected_ids` plays
        for the dense route -- a mismatch means the artifact was built from a
        different catalogue and the feature stands down rather than scoring
        against the wrong rows. Falls back to the file when None, which is
        what the offline comparison scripts use.
        """
        try:
            import numpy as np
        except ImportError as error:  # pragma: no cover - environment guard
            raise SemanticEvidenceUnavailable(f"numpy is required: {error}") from error

        self._np = np
        vectors = value_vectors_path(embedding_dir)
        if not vectors.exists():
            raise SemanticEvidenceUnavailable(
                f"value embedding artifact not found at {vectors}; build it with "
                f"scripts/build_value_embeddings.py"
            )

        values = expected_values
        if values is None:
            strings = value_strings_path(embedding_dir)
            if not strings.exists():
                raise SemanticEvidenceUnavailable(
                    f"no expected_values given and no {strings.name} beside the "
                    f"artifact; one of the two has to label the rows"
                )
            values = json.loads(strings.read_text())

        matrix = np.load(vectors)
        if matrix.shape[0] != len(values):
            raise SemanticEvidenceUnavailable(
                f"artifact mismatch: {matrix.shape[0]} vectors vs {len(values)} "
                f"catalogue values; rebuild the artifact against this catalogue"
            )

        metadata = {}
        metadata_file = value_metadata_path(embedding_dir)
        if metadata_file.exists():
            metadata = json.loads(metadata_file.read_text())
        # The artifact ships int8 to fit the repository; the scale that
        # dequantizes it lives beside it rather than being inferred, so a
        # float32 artifact and a quantized one load through the same path.
        self._scale = float(metadata.get("quantization_scale") or 0.0) or None
        if matrix.dtype == np.int8 and self._scale is None:
            raise SemanticEvidenceUnavailable(
                "int8 artifact has no quantization_scale in its .meta.json"
            )

        self._matrix = matrix
        self._index = {value: row for row, value in enumerate(values)}
        self._model_id = model_id
        self._model = None
        self._injected_encoder = encoder
        self._dimensions = int(matrix.shape[1])
        # Cache of per-product row indices and of query vectors. Disclosures
        # accumulate across a session, so the same sentence is re-scored on
        # every later turn.
        self._rows: dict[str, object] = {}
        self._vectors: dict[str, object] = {}

    # -- encoding ---------------------------------------------------------

    def _encode(self, texts: list[str]):
        if self._injected_encoder is not None:
            return self._injected_encoder(texts)
        if self._model is None:
            from .voyage_compat import load_model

            self._model = load_model()
        return self._model.encode_query(
            texts, convert_to_numpy=True, truncate_dim=self._dimensions
        )

    def _embed(self, texts: list[str]):
        np = self._np
        missing = [text for text in texts if text not in self._vectors]
        if missing:
            encoded = self._encode(missing)
            encoded = np.asarray(encoded, dtype="float32")
            encoded /= np.linalg.norm(encoded, axis=1, keepdims=True).clip(min=1e-12)
            for text, vector in zip(missing, encoded):
                self._vectors[text] = vector
        return np.stack([self._vectors[text] for text in texts])

    # -- scoring ----------------------------------------------------------

    def _rows_for(self, parent_asin: str, product: ProductRecord):
        rows = self._rows.get(parent_asin)
        if rows is None:
            rows = self._np.array(
                [self._index[value] for value in product.card_values if value in self._index],
                dtype="int64",
            )
            self._rows[parent_asin] = rows
        return rows

    def _vectors_for(self, rows):
        """Gather rows and, if the artifact is quantized, restore them.

        Dequantizing the gathered rows rather than the whole matrix keeps the
        66 MB artifact at 66 MB in memory: a pool touches a few thousand rows
        per turn, against 268,564 in the file.
        """
        np = self._np
        gathered = self._matrix[rows]
        if self._scale is None:
            return gathered
        restored = gathered.astype("float32") / self._scale
        restored /= np.linalg.norm(restored, axis=1, keepdims=True).clip(min=1e-12)
        return restored

    def score_pool(
        self,
        disclosures: list[str],
        candidates: list[Candidate],
        products: dict[str, ProductRecord],
        weights: list[tuple[str, float]],
    ) -> dict:
        """{parent_asin: 0-1} for the pool, or {} if there is nothing to score.

        `weights` is (disclosure text, content-token count), prepared by the
        caller so this module and `evidence` cannot disagree about which
        disclosures count or what each is worth.
        """
        np = self._np
        if not weights or not candidates:
            return {}

        query = self._embed([text for text, _ in weights])
        token_weights = np.array([weight for _, weight in weights], dtype="float32")
        total = float(token_weights.sum())
        if total <= 0.0:
            return {}

        scores: dict[str, float] = {}
        for candidate in candidates:
            product = products.get(candidate.parent_asin)
            if product is None:
                continue
            rows = self._rows_for(candidate.parent_asin, product)
            if rows.size == 0:
                continue
            # Apple's Accelerate BLAS leaks floating-point status flags out of
            # matmul, raising spurious invalid/divide-by-zero warnings here.
            # Checked rather than assumed (E11): over 2993 real pools the
            # output had 0 non-finite entries and matched a float64 einsum
            # reference to 3.9e-07. Silenced so a host that turns warnings
            # into errors cannot push the reranker onto its fallback path.
            with np.errstate(all="ignore"):
                similarity = self._vectors_for(rows) @ query.T
            best = np.clip(similarity.max(axis=0), 0.0, 1.0)
            scores[candidate.parent_asin] = float((best * token_weights).sum() / total)
        return scores


def _guard_scoring_failures(scorer: SemanticScorer) -> SemanticScoreFn:
    """Keep a per-turn scoring failure from costing the turn.

    Same contract as dense_retrieval._guard_query_failures: an exception
    escaping respond() becomes an empty recommendation list for that turn, so
    an encode error would score ~0 across every remaining session with no
    diagnostic. Returning no scores instead degrades that turn to the rest of
    the feature table, and the reason is reported once rather than per turn.
    """
    warned = False

    def guarded(disclosures, candidates, products, weights):
        nonlocal warned
        try:
            return scorer.score_pool(disclosures, candidates, products, weights)
        except Exception as error:
            if not warned:
                print(
                    f"[shopping_agent] semantic evidence failed, scoring it 0 for "
                    f"the rest of this run: {error}",
                    file=sys.stderr,
                )
                warned = True
            return {}

    return guarded


def load_semantic_scorer(
    embedding_dir: str | Path = EMBEDDING_DIR,
    model_id: str = VALUE_MODEL_ID,
    strict: bool = False,
    encoder: Callable[[list[str]], object] | None = None,
    expected_values: list[str] | None = None,
):
    """Build the scorer, or return None if it cannot run.

    Returning None is the supported path, not an error: the reranker then
    scores this feature 0.0 for every candidate and the ordering falls through
    to the rest of the table. Pass strict=True to raise instead.
    """
    try:
        return _guard_scoring_failures(
            SemanticScorer(embedding_dir, model_id, encoder, expected_values)
        )
    except Exception as error:
        if strict:
            raise
        print(
            f"[shopping_agent] semantic evidence unavailable, ranking without it: {error}",
            file=sys.stderr,
        )
        return None


def catalogue_values(products: dict) -> list[str]:
    """The row labelling the artifact was built with.

    Must stay identical to scripts/build_value_embeddings.distinct_values --
    same set, same sort -- because it is what maps a matrix row back to the
    string it encodes. Deriving it here rather than shipping it keeps 19 MB of
    JSON out of the repository.
    """
    return sorted({value for product in products.values() for value in product.card_values})
