"""P3-T1 embedding model benchmark.

Read-only comparison tool, not part of the Agent runtime: measures
dimensions, on-disk model size, license, peak RAM, startup time, batch
encoding throughput, and single-query latency for the embedding model
candidates under consideration for Phase 3 dense retrieval, alongside a
no-dense baseline for reference. Per docs/phase-skeleton-ai.md's P3
decision gate, this is meant to produce real numbers to choose from,
not a reputation-based pick.

Requires `sentence-transformers` (see requirements.txt), which is not
installed by default in this repo. Install it yourself before running:

    pip install -r requirements.txt
    python3 -m scripts.embedding_model_benchmark

Each real model is benchmarked in its own subprocess so RAM/startup
measurements for one candidate aren't inflated by a previous
candidate's model still resident in the same process. License and
model-size lookups hit the Hugging Face Hub API, so this script needs
network access to run (the models themselves, once downloaded, run
fully offline).
"""

from __future__ import annotations

import json
import resource
import subprocess
import sys
import time

from shopping_agent.catalog import load_catalog

SAMPLE_SIZE = 200

SAMPLE_QUERIES = (
    "waterproof hiking boots under $100",
    "a gift for my mom who likes gardening",
    "wireless noise cancelling headphones for travel",
    "durable phone case for a toddler",
    "eco-friendly yoga mat, non-slip",
)

# query_prefix is prepended to queries only, never to product text -- bge-small
# is trained for asymmetric retrieval and needs this to perform as documented.
CANDIDATES = (
    {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": "",
    },
    {
        "name": "BAAI/bge-small-en-v1.5",
        "model_id": "BAAI/bge-small-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
)

WORKER_FLAG = "--worker"


def _peak_rss_mb() -> float:
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return raw / divisor


def _model_license_and_size_mb(model_id: str) -> tuple[str, float | None]:
    from huggingface_hub import HfApi

    info = HfApi().model_info(model_id, files_metadata=True)

    license_id = "unknown"
    if info.card_data and info.card_data.get("license"):
        license_id = str(info.card_data["license"])
    else:
        for tag in info.tags or ():
            if tag.startswith("license:"):
                license_id = tag.split(":", 1)[1]
                break

    weight_suffixes = (".safetensors", ".bin", ".onnx", ".pt")
    siblings = info.siblings or ()
    sizes = [getattr(s, "size", None) for s in siblings]
    weight_bytes = sum(
        size for s, size in zip(siblings, sizes)
        if size and s.rfilename.endswith(weight_suffixes)
    )
    total_bytes = weight_bytes or sum(size for size in sizes if size)
    return license_id, (total_bytes / (1024.0 * 1024.0)) if total_bytes else None


def _run_worker(model_id: str, query_prefix: str, product_texts: list[str], query: str) -> dict:
    from sentence_transformers import SentenceTransformer

    start = time.perf_counter()
    model = SentenceTransformer(model_id)
    startup_seconds = time.perf_counter() - start

    encode_start = time.perf_counter()
    vectors = model.encode(product_texts, show_progress_bar=False)
    encode_seconds = time.perf_counter() - encode_start

    query_start = time.perf_counter()
    model.encode([query_prefix + query], show_progress_bar=False)
    query_seconds = time.perf_counter() - query_start

    return {
        "dimensions": int(vectors.shape[1]),
        "startup_seconds": startup_seconds,
        "batch_encode_seconds": encode_seconds,
        "texts_per_second": len(product_texts) / encode_seconds if encode_seconds > 0 else None,
        "single_query_latency_seconds": query_seconds,
        "peak_rss_mb": _peak_rss_mb(),
    }


def _benchmark_in_subprocess(
    model_id: str, query_prefix: str, product_texts: list[str], query: str
) -> dict:
    payload = json.dumps(
        {
            "model_id": model_id,
            "query_prefix": query_prefix,
            "product_texts": product_texts,
            "query": query,
        }
    )
    result = subprocess.run(
        [sys.executable, __file__, WORKER_FLAG],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def benchmark(catalog_path: str = "data/catalog.jsonl", sample_size: int = SAMPLE_SIZE) -> dict:
    products = load_catalog(catalog_path)
    product_texts = [record.searchable_text for record in list(products.values())[:sample_size]]
    query = SAMPLE_QUERIES[0]

    results: dict = {
        "no_dense_baseline": {
            "license": None,
            "model_size_mb": None,
            "dimensions": None,
            "startup_seconds": 0.0,
            "batch_encode_seconds": 0.0,
            "texts_per_second": None,
            "single_query_latency_seconds": 0.0,
            "peak_rss_mb": None,
            "note": "existing BM25 lexical route; no embedding model involved",
        }
    }

    for candidate in CANDIDATES:
        license_id, model_size_mb = _model_license_and_size_mb(candidate["model_id"])
        timing = _benchmark_in_subprocess(
            candidate["model_id"], candidate["query_prefix"], product_texts, query
        )
        results[candidate["name"]] = {
            "license": license_id,
            "model_size_mb": model_size_mb,
            **timing,
        }
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == WORKER_FLAG:
        request = json.loads(sys.stdin.read())
        print(
            json.dumps(
                _run_worker(
                    request["model_id"],
                    request["query_prefix"],
                    request["product_texts"],
                    request["query"],
                )
            )
        )
    else:
        print(json.dumps(benchmark(), indent=2))
