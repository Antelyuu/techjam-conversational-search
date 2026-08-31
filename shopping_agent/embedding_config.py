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

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
QUERY_PREFIX = ""

# Filesystem-safe name for this model's artifact files.
MODEL_SLUG = MODEL_ID.rsplit("/", maxsplit=1)[-1]

# Anchored to the repository, not the process working directory. The official
# harness constructs Agent(catalog_path) and may run from anywhere; a relative
# path silently fails to find the bundled artifact, and the dense route then
# degrades to BM25 while the run still looks healthy.
EMBEDDING_DIR = Path(__file__).resolve().parent.parent / "data" / "embeddings"


def vectors_path(embedding_dir: Path | str = EMBEDDING_DIR) -> Path:
    return Path(embedding_dir) / f"{MODEL_SLUG}.npy"


def ids_path(embedding_dir: Path | str = EMBEDDING_DIR) -> Path:
    return Path(embedding_dir) / f"{MODEL_SLUG}.ids.json"


def metadata_path(embedding_dir: Path | str = EMBEDDING_DIR) -> Path:
    return Path(embedding_dir) / f"{MODEL_SLUG}.meta.json"


# ---------------------------------------------------------------------------
# The value-level artifact (P7, E11), which is a different model doing a
# different job and so is configured separately from the dense route above.
#
# The dense route embeds one vector per *product* and answers "what is this
# query about". This embeds one vector per distinct *card value* and answers
# "does this candidate own a field value that means what the customer just
# said" -- the semantic form of the question slots.py asks exactly.
#
# voyage-4-nano rather than MiniLM on measurement, not reputation (E11):
# 0.862054 paraphrased against MiniLM's 0.799906 and bge-small's 0.825260, and
# it is the only one of the three whose artifact fits the repository. Its
# Matryoshka training lets the 2048-dimension build be cut to 256, and its
# quantization-aware training lets that be stored int8, which is 66 MB against
# 393 MB for either 384-dimension alternative at float32.
VALUE_MODEL_ID = "voyageai/voyage-4-nano"
VALUE_MODEL_SLUG = "voyage-4-nano-values256"


def value_vectors_path(embedding_dir: Path | str = EMBEDDING_DIR) -> Path:
    return Path(embedding_dir) / f"{VALUE_MODEL_SLUG}.npy"


def value_strings_path(embedding_dir: Path | str = EMBEDDING_DIR) -> Path:
    return Path(embedding_dir) / f"{VALUE_MODEL_SLUG}.values.json"


def value_metadata_path(embedding_dir: Path | str = EMBEDDING_DIR) -> Path:
    return Path(embedding_dir) / f"{VALUE_MODEL_SLUG}.meta.json"
