"""E11: embed every distinct *card value* in the catalogue.

scripts/build_embeddings.py encodes one vector per product, which is what a
retrieval route needs. This encodes one vector per **card value** -- the whole
feature/detail strings `local_evaluator.intent_card` builds its hidden card
from -- because that is the granularity the customer actually quotes at, and
therefore the granularity a semantic evidence feature has to score at.

Why distinct values rather than per product: the 50,000 catalogue rows carry
615,776 card-value slots between them but only **268,564 distinct strings**,
because values like "Imported" or "100% Cotton" recur across thousands of
rows. Deduplicating removes 56% of the encoding work and shrinks the artifact
by the same fraction; `shopping_agent.slots` already guarantees a product's
values are exact members of that set, so the mapping back is a dict lookup.

The build is deterministic in the same sense build_embeddings.py is: values
are encoded in sorted order, so row i of the .npy is always values[i] in the
.values.json, and the same catalogue plus the same model always produce the
same matrix. Vectors are L2-normalized here, so cosine is a plain dot product
at scoring time.

Three models are selectable because E11's question is which of them scores a
paraphrased sentence best -- this is a comparison script, so the model is a
flag rather than embedding_config.py's single-source constant.

**Prefixes are not decoration.** MiniLM is symmetric and takes none. BGE is
trained for asymmetric retrieval: the *query* takes an instruction prefix and
the document side takes none. voyage-4-nano carries named `query`/`document`
prompts in its own config and needs both. Getting this wrong does not error --
it quietly degrades, which is the failure E1's correction history records.
The query side of each pairing lives in scripts/semantic_evidence_probe.py and
must match what is used here.

Usage:
    python3 -m scripts.build_value_embeddings --model minilm
    python3 -m scripts.build_value_embeddings --model voyage --batch 128

Writes data/embeddings/<slug>-values.npy plus a .values.json and a .meta.json.
Those artifacts are gitignored: at 268,564 rows even MiniLM's 384 dimensions
come to 393 MB, and only a 256-dimension int8 build fits GitHub's 100 MB
per-file limit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shopping_agent.catalog import load_catalog
from shopping_agent.embedding_config import EMBEDDING_DIR

# (hugging face id, document-side prefix, artifact slug).
MODELS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "", "all-MiniLM-L6-v2-values"),
    "bge": ("BAAI/bge-small-en-v1.5", "", "bge-small-en-v1.5-values"),
    "voyage": ("voyageai/voyage-4-nano", None, "voyage-4-nano-values"),
}

# Card values are short -- median 47 characters, 180 at the 95th percentile,
# and the card generator clips at 180 -- so this truncates nothing real while
# keeping attention cost down on the larger model.
MAX_SEQ_LENGTH = 256


def load_voyage_model():
    """Load voyageai/voyage-4-nano under transformers 5.x.

    The published remote code targets transformers 4.51 and does not load as
    written on 5.x. Two incompatibilities, both mechanical:

      * `Qwen3BidirectionalModel` declares no `config_class`, and AutoModel's
        registration path dereferences it.
      * `create_causal_mask` renamed `input_embeds` to `inputs_embeds` and
        dropped `cache_position`, which the model still passes.

    The shim is confined to this function so nothing else has to know. It was
    verified rather than assumed: the model card's own retrieval example picks
    the right document (0.650 against 0.541/0.533/0.403), encoding is
    padding-invariant, and the shimmed mask agrees with transformers' native
    `create_bidirectional_mask` to within 0.995-1.000 -- two independent
    constructions of the same mask.
    """
    from transformers import Qwen3Config
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from sentence_transformers import SentenceTransformer

    model_class = get_class_from_dynamic_module(
        "modeling_qwen3_bidirectional.Qwen3BidirectionalModel", "voyageai/voyage-4-nano"
    )
    model_class.config_class = Qwen3Config
    module = sys.modules[model_class.__module__]
    if not hasattr(module, "_original_create_causal_mask"):
        module._original_create_causal_mask = module.create_causal_mask

    original = module._original_create_causal_mask

    def create_causal_mask(*args, **kwargs):
        kwargs.pop("cache_position", None)
        if "input_embeds" in kwargs:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        return original(*args, **kwargs)

    module.create_causal_mask = create_causal_mask
    return SentenceTransformer(
        "voyageai/voyage-4-nano",
        trust_remote_code=True,
        model_kwargs={"attn_implementation": "sdpa"},
    )


def distinct_values(catalog_path: str | Path) -> list[str]:
    """Every card value in the catalogue, deduplicated and sorted."""
    products = load_catalog(catalog_path)
    return sorted({value for product in products.values() for value in product.card_values})


def build(
    model_key: str,
    catalog_path: str | Path = "data/catalog.jsonl",
    embedding_dir: str | Path = EMBEDDING_DIR,
    batch_size: int = 256,
    truncate_dim: int = 0,
    chunk: int = 20_000,
) -> dict:
    import numpy as np

    model_id, doc_prefix, slug = MODELS[model_key]
    values = distinct_values(catalog_path)
    print(f"distinct card values: {len(values)}", flush=True)

    if model_key == "voyage":
        model = load_voyage_model()
        model.max_seq_length = MAX_SEQ_LENGTH
        extra = {"truncate_dim": truncate_dim} if truncate_dim else {}
        # encode_document applies the model's own document prompt.
        encode = lambda batch: model.encode_document(
            batch, batch_size=batch_size, convert_to_numpy=True, **extra
        )
        if truncate_dim:
            slug = f"{slug}{truncate_dim}"
    else:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_id)
        encode = lambda batch: model.encode(
            [doc_prefix + text for text in batch],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    started = time.time()
    blocks = []
    for start in range(0, len(values), chunk):
        blocks.append(encode(values[start : start + chunk]).astype("float32"))
        done = min(start + chunk, len(values))
        rate = done / (time.time() - started)
        print(f"  {done}/{len(values)}  {rate:.0f}/s  eta {(len(values)-done)/rate/60:.1f} min",
              flush=True)

    matrix = np.vstack(blocks)
    # Normalize unconditionally rather than trusting the encoder's flag, so a
    # dot product is a cosine for every model on this path.
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-12)

    directory = Path(embedding_dir)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / f"{slug}.npy", matrix)
    (directory / f"{slug}.values.json").write_text(json.dumps(values))
    metadata = {
        "model_id": model_id,
        "values": len(values),
        "dimensions": int(matrix.shape[1]),
        "build_minutes": round((time.time() - started) / 60, 2),
        "megabytes": round(matrix.nbytes / 1024 / 1024, 1),
    }
    (directory / f"{slug}.meta.json").write_text(json.dumps(metadata, indent=2))
    print(f"wrote {slug}.npy {matrix.shape} -- {json.dumps(metadata)}", flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="minilm", choices=sorted(MODELS))
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument(
        "--dim", type=int, default=0,
        help="Matryoshka output dimension (voyage only; 0 keeps the native 2048)",
    )
    args = parser.parse_args()
    build(args.model, args.catalog, batch_size=args.batch, truncate_dim=args.dim)


if __name__ == "__main__":
    main()
