"""P3-T2 reproducible embedding artifact build.

Encodes every catalog product once and writes the vectors to disk, so the
Agent never has to embed the catalog at request time. Not part of the
Agent runtime -- run this manually before enabling the dense route.

The build is deterministic: products are encoded in sorted parent_asin
order, so the same catalog and the same model always produce the same
matrix, and row i of the .npy always corresponds to ids[i] in the
.ids.json. Vectors are L2-normalized at build time, which lets the search
adapter use a plain dot product for cosine similarity.

Which model gets used comes from shopping_agent/embedding_config.py --
that is the single file the per-model comparison branches differ on.

Requires `sentence-transformers` (see requirements.txt), which is not
installed by default in this repo:

    pip install -r requirements.txt
    python3 -m scripts.build_embeddings

Writes to data/embeddings/ (gitignored -- artifacts are large binaries and
are rebuilt per checkout rather than committed). The first run downloads
model weights from the Hugging Face Hub and needs network access; every
run afterwards, and the dense route itself, work fully offline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make this runnable regardless of invocation style (`python3 file.py` or `-m`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shopping_agent.catalog import load_catalog
from shopping_agent.embedding_config import (
    EMBEDDING_DIR,
    MODEL_ID,
    MODEL_SLUG,
    QUERY_PREFIX,
    ids_path,
    metadata_path,
    vectors_path,
)

ENCODE_BATCH_SIZE = 64


def build(
    catalog_path: str | Path = "data/catalog.jsonl",
    embedding_dir: str | Path = EMBEDDING_DIR,
) -> dict:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    products = load_catalog(catalog_path)
    # Sorted so the row order is a property of the catalog, not of dict
    # insertion order -- this is what makes rebuilds byte-comparable.
    ordered_ids = sorted(products)
    texts = [products[parent_asin].searchable_text for parent_asin in ordered_ids]

    model = SentenceTransformer(MODEL_ID)
    vectors = model.encode(
        texts,
        batch_size=ENCODE_BATCH_SIZE,
        # Unit vectors mean cosine similarity is just a dot product at query time.
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)

    output_dir = Path(embedding_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(vectors_path(output_dir), vectors)
    ids_path(output_dir).write_text(json.dumps(ordered_ids), encoding="utf-8")

    metadata = {
        "model_id": MODEL_ID,
        "model_slug": MODEL_SLUG,
        "query_prefix": QUERY_PREFIX,
        "dimensions": int(vectors.shape[1]),
        "product_count": int(vectors.shape[0]),
        "normalized": True,
        "catalog_path": str(catalog_path),
    }
    metadata_path(output_dir).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
