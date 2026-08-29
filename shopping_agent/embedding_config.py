"""Which embedding model the dense route uses.

This is the ONLY module that differs between the per-model comparison
branches (phase/3-dense-minilm and phase/3-dense-bge). Everything else in
the dense pipeline -- the artifact build, the vector search adapter, the
candidate union, the fusion -- is model-agnostic and shared, so a fix in
the retrieval logic does not have to be made twice.

query_prefix is prepended to QUERIES ONLY, never to product text. Models
trained for asymmetric retrieval (bge) need it; symmetric ones (MiniLM)
leave it empty.
"""

from __future__ import annotations

from pathlib import Path

MODEL_ID = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Filesystem-safe name for this model's artifact files.
MODEL_SLUG = MODEL_ID.rsplit("/", maxsplit=1)[-1]

EMBEDDING_DIR = Path("data/embeddings")


def vectors_path(embedding_dir: Path | str = EMBEDDING_DIR) -> Path:
    return Path(embedding_dir) / f"{MODEL_SLUG}.npy"


def ids_path(embedding_dir: Path | str = EMBEDDING_DIR) -> Path:
    return Path(embedding_dir) / f"{MODEL_SLUG}.ids.json"


def metadata_path(embedding_dir: Path | str = EMBEDDING_DIR) -> Path:
    return Path(embedding_dir) / f"{MODEL_SLUG}.meta.json"
