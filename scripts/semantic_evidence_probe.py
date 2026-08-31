"""E11: score a paraphrased disclosure against a candidate's own field values.

`constraint_evidence` (rerank weight 12.0) asks how many of the disclosure's
content tokens appear anywhere in the candidate's text. E10 measured what that
is worth once the customer stops quoting, and the answer was *negative*: the
feature scores partial overlap, so rewording fills it with whichever words
happened to survive rather than silencing it, and zeroing it makes the
paraphrased score go up. At weight 12 that is noise with a loud voice.

This replaces the matching rule and keeps everything else:

    value(p) = sum_d w_d * max_v cos(emb(d), emb(v))  /  sum_d w_d

for disclosures d and the candidate's own card values v. Two deliberate
choices, both inherited rather than invented:

  * **Card values, not product text.** `slot_evidence` (weight 16, the table's
    strongest feature) already asks whether a candidate *owns* the disclosed
    string as a whole field value, and that granularity is what makes it
    sharp. This asks the same structural question with a semantic matcher
    instead of string equality, so it degrades where slot ownership goes
    silent instead of falling back to bag-of-words.
  * **w_d is the disclosure's distinct content-token count**, exactly the
    weight `evidence.coverage_from_sets` already uses, so the two features
    agree on what each disclosure is worth and only the matching rule differs.

Nothing in shopping_agent/ changes. FEATURE_WEIGHTS is patched in place and
rerank is wrapped, the same way scripts/paraphrase_weight_probe.py does it,
so this measures a candidate change without being one.

**The control that makes the rest trustworthy** is `--weight 0
--keep-token-feature`: the wrapper still runs but contributes nothing, and it
must reproduce the shipped score to six decimals (0.945497 at level 0,
0.696015 at level 2). If it does not, the harness is adding something of its
own and every delta from it is void.

Usage:
    python3 -m scripts.semantic_evidence_probe --model minilm --level 2 --weight 24
    python3 -m scripts.semantic_evidence_probe --model bge --level 2 --weight 96 --heldout
    python3 -m scripts.semantic_evidence_probe --weight 0 --keep-token-feature   # control

Requires an artifact from scripts/build_value_embeddings.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from evaluator import local_evaluator as ev
from scripts.build_value_embeddings import MODELS, MAX_SEQ_LENGTH, load_voyage_model
from scripts.paraphrase_eval import install_paraphrasing_customer
from shopping_agent import evidence, reranking
from shopping_agent.embedding_config import EMBEDDING_DIR

# The query-side prefix for each model, paired with the document-side prefix
# build_value_embeddings.py used. BGE is asymmetric and takes its instruction
# here and nowhere else; MiniLM is symmetric and must take none; voyage
# carries named prompts and gets them through encode_query.
QUERY_PREFIX = {
    "minilm": "",
    "bge": "Represent this sentence for searching relevant passages: ",
    "voyage": None,
}

# The raw disclosure text for the turn being scored. rerank() is handed only
# the *normalized* evidence by starter.agent, and embedding needs the original
# sentence, so prepare_evidence stashes it on the way past. One turn at a
# time, single threaded -- the same assumption scripts/paraphrase_headroom.py
# makes about session state.
_CURRENT: dict = {"disclosures": []}


class SemanticEvidence:
    """Cosine between a disclosure and a candidate's own card values."""

    def __init__(self, slug: str, model_key: str, truncate_dim: int = 0):
        self.matrix = np.load(Path(EMBEDDING_DIR) / f"{slug}.npy")
        values = json.loads((Path(EMBEDDING_DIR) / f"{slug}.values.json").read_text())
        if len(values) != self.matrix.shape[0]:
            raise ValueError(
                f"artifact mismatch: {self.matrix.shape[0]} vectors vs {len(values)} values"
            )
        if truncate_dim:
            # Matryoshka truncation: keep the leading dimensions and restore
            # unit norm, so a dot product is still a cosine.
            self.matrix = self.matrix[:, :truncate_dim].copy()
            self.matrix /= np.linalg.norm(self.matrix, axis=1, keepdims=True).clip(min=1e-12)
        self.index = {value: row for row, value in enumerate(values)}
        self.truncate_dim = truncate_dim
        self.model_key = model_key
        self._rows: dict[str, np.ndarray] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._encoder = None
        self.encode_seconds = 0.0
        self.encode_batches = 0

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self._encoder is None:
            if self.model_key == "voyage":
                model = load_voyage_model()
                model.max_seq_length = MAX_SEQ_LENGTH
                extra = {"truncate_dim": self.truncate_dim} if self.truncate_dim else {}
                self._encoder = lambda batch: model.encode_query(
                    batch, convert_to_numpy=True, **extra
                )
            else:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(MODELS[self.model_key][0])
                prefix = QUERY_PREFIX[self.model_key]
                self._encoder = lambda batch: model.encode(
                    [prefix + text for text in batch],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
        started = time.time()
        vectors = np.asarray(self._encoder(texts), dtype="float32")
        self.encode_seconds += time.time() - started
        self.encode_batches += 1
        return vectors

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed, reusing anything seen before.

        Disclosures accumulate across a session, so the same sentence is
        re-scored on every later turn; caching turns ~2000 encodes per run
        into a few hundred.
        """
        missing = [text for text in texts if text not in self._vectors]
        if missing:
            vectors = self._encode(missing)
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
            for text, vector in zip(missing, vectors):
                self._vectors[text] = vector
        return np.stack([self._vectors[text] for text in texts])

    def _candidate_rows(self, parent_asin: str, product) -> np.ndarray:
        rows = self._rows.get(parent_asin)
        if rows is None:
            rows = np.array(
                [self.index[value] for value in product.card_values if value in self.index],
                dtype="int64",
            )
            self._rows[parent_asin] = rows
        return rows

    def scores(self, disclosures: list[str], candidates: list, products: dict):
        """0-1 per candidate, in candidate order, or None if nothing usable.

        None is the quiet-failure path: no usable disclosure means the feature
        has nothing to say and must not reorder anything, exactly as
        coverage_from_sets returns a neutral 0.0.
        """
        usable = [
            (text, float(len(frozenset(evidence._content_tokens(text)))))
            for text in disclosures
        ]
        usable = [pair for pair in usable if pair[1] >= evidence.MIN_EVIDENCE_TOKENS]
        if not usable:
            return None

        query = self.embed([text for text, _ in usable])
        weights = np.array([weight for _, weight in usable], dtype="float32")
        best = np.zeros((len(candidates), len(usable)), dtype="float32")
        scored = np.zeros(len(candidates), dtype=bool)

        for position, candidate in enumerate(candidates):
            product = products.get(candidate.parent_asin)
            if product is None:
                continue
            rows = self._candidate_rows(candidate.parent_asin, product)
            if rows.size == 0:
                continue
            scored[position] = True
            # Apple's Accelerate BLAS leaks floating-point status flags out of
            # matmul, so this raises spurious invalid/divide-by-zero warnings.
            # Checked rather than assumed: over 2993 real pools the output had
            # 0 non-finite entries and matched a float64 einsum reference to
            # 3.9e-07. Silenced here so a host that turns warnings into errors
            # cannot push the reranker onto its fallback path.
            with np.errstate(all="ignore"):
                similarity = self.matrix[rows] @ query.T
            best[position] = np.clip(similarity.max(axis=0), 0.0, 1.0)

        out = (best * weights).sum(axis=1) / float(weights.sum())
        out[~scored] = 0.0
        return out


def install(scorer: SemanticEvidence, weight: float, keep_token_feature: bool) -> None:
    """Wrap rerank so the semantic feature is added, without editing the agent."""
    real_prepare = reranking.prepare_evidence
    real_rerank = reranking.rerank

    def prepare_evidence(disclosures, candidates, products):
        _CURRENT["disclosures"] = list(disclosures or [])
        return real_prepare(disclosures, candidates, products)

    def rerank(candidates, products, constraints, limit,
               disclosures=None, prepared=None, stated_category=None):
        # Score the whole pool before truncating, so a candidate the semantic
        # feature would promote out of the tail is not cut off first.
        ordered = real_rerank(candidates, products, constraints, len(candidates),
                              disclosures, prepared, stated_category)
        raw = disclosures if disclosures is not None else _CURRENT["disclosures"]
        semantic = scorer.scores(raw, candidates, products)
        if semantic is None:
            return ordered[:limit]
        by_asin = {c.parent_asin: value for c, value in zip(candidates, semantic)}
        rescored = [
            reranking.RerankedCandidate(
                parent_asin=item.parent_asin,
                score=item.score + weight * by_asin.get(item.parent_asin, 0.0),
                contributions=item.contributions,
            )
            for item in ordered
        ]
        # Stable, so candidates the added feature cannot separate keep the
        # order the shipped scorer gave them.
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored[:limit]

    reranking.prepare_evidence = prepare_evidence
    reranking.rerank = rerank
    # starter.agent imported both names by value, so rebinding the module
    # attribute alone would leave the agent calling the originals.
    import starter.agent as agent_module

    agent_module.prepare_evidence = prepare_evidence
    agent_module.rerank = rerank

    if not keep_token_feature:
        reranking.FEATURE_WEIGHTS["constraint_evidence"] = 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--model", default="minilm", choices=sorted(MODELS))
    parser.add_argument("--slug", default="", help="artifact slug (default: the model's)")
    parser.add_argument("--dim", type=int, default=0, help="Matryoshka truncation")
    parser.add_argument("--level", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--weight", type=float, default=24.0)
    parser.add_argument(
        "--keep-token-feature", action="store_true",
        help="add the semantic feature alongside constraint_evidence rather "
             "than in place of it",
    )
    parser.add_argument("--paraphrase-category", action="store_true")
    parser.add_argument(
        "--heldout", action="store_true",
        help="swap in the disjoint held-out lexicon (scripts/heldout_lexicon.py)",
    )
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    if args.heldout:
        from scripts import heldout_lexicon

        heldout_lexicon.install()

    slug = args.slug or MODELS[args.model][2]
    scorer = SemanticEvidence(slug, args.model, args.dim)
    install(scorer, args.weight, args.keep_token_feature)
    install_paraphrasing_customer(args.level, args.paraphrase_category, args.seed)

    samples = ev.load_jsonl(args.dataset)
    catalog_ids, categories, products = ev.catalog_index(args.catalog)
    started = time.time()
    result = ev.evaluate(ev.Agent(args.catalog), samples, catalog_ids, categories, products)

    print(json.dumps({
        "model": args.model,
        "slug": slug,
        "dim": args.dim or int(scorer.matrix.shape[1]),
        "level": args.level,
        "weight": args.weight,
        "token_feature_kept": args.keep_token_feature,
        "paraphrase_category": args.paraphrase_category,
        "heldout_lexicon": args.heldout,
        "score": result["recommended_technical_score"],
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "wall_seconds": round(time.time() - started, 1),
        "encode_batches": scorer.encode_batches,
        "encode_seconds": round(scorer.encode_seconds, 1),
    }))


if __name__ == "__main__":
    main()
