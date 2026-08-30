# The dense semantic route

Off by default since Phase 5, on measurement.

The [README](../README.md#the-one-model-we-built-and-then-deleted) carries the argument
and the two measurements behind the decision: the composite before and after
clarification, and the recall-by-turn table showing lexical retrieval pulling away from
dense as quoted constraints accumulate. This file is the operational reference — how to
turn the route back on, which model it uses, and when to rebuild the artifact.

## Enabling it

```bash
pip install -r requirements.txt      # sentence-transformers (pulls in torch, numpy)
SHOPPING_AGENT_DENSE=1 python3 -m evaluator.local_evaluator
```

`SHOPPING_AGENT_FUSION` selects the blend: `weighted` (the default) or `rrf`. Every other
flag is listed in [SETUP.md](SETUP.md#configuration-flags).

Enabling the route costs 0.0509 on the current ranking, so the numbers it produces will
not match the ones quoted in the README.

## Model

`sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, Apache-2.0. We chose it over
`bge-small-en-v1.5` on a benchmark (+0.013437 composite, recorded in E1).

The model is set in `shopping_agent/embedding_config.py`. The first `pip install` run
downloads weights from the Hugging Face Hub and needs network; both models cache locally
and run fully offline afterwards.

## The prebuilt artifact

`data/embeddings/all-MiniLM-L6-v2.npy` is bundled so the route works in a fresh runtime
without a rebuild. Rebuild it only when the frozen catalogue or the selected model
changes:

```bash
python3 -m scripts.build_embeddings
```

`bge-small-en-v1.5` is deliberately not bundled — that model line was discontinued by team
decision in E1, and its artifact is 73 MB.

## Where the measurements live

| | |
|---|---|
| `experiments/E1-p3-embedding-model.md` | MiniLM against bge-small |
| `experiments/E2-p3-fusion-ablation.md` | lexical-only against RRF and weighted fusion |
| `experiments/E5-p5-retrieval-reversal-and-evidence.md` | the removal, and the recall-by-turn audit |

Reproduce the fusion comparison directly with `python3 -m scripts.fusion_ablation`.
