"""Load voyageai/voyage-4-nano under transformers 5.x.

The weights are Apache-2.0 and published on the Hugging Face Hub, so the model
runs locally with no API key and no network at inference time. What it does
need is a shim: the published remote code targets transformers 4.51 and does
not import as written on 5.x. Two incompatibilities, both mechanical:

  * `Qwen3BidirectionalModel` declares no `config_class`, and AutoModel's
    registration path dereferences it (`AttributeError: 'NoneType' object has
    no attribute '__name__'`).
  * `create_causal_mask` renamed `input_embeds` to `inputs_embeds` and dropped
    `cache_position`, both of which the model still passes.

Keeping this in one module means the artifact builder and the runtime scorer
patch the model exactly once and identically -- a mismatch between how the
catalogue was encoded and how a query is encoded is silent, and would look
like a model that simply scores worse.

**The shim was verified rather than assumed** (E11), because a wrong attention
mask degrades quality without erroring:

  * the model card's own retrieval example still picks the right document,
    scoring Mars at 0.650 against 0.541 / 0.533 / 0.403;
  * encoding is padding-invariant, so no padding leaks through the mask;
  * the shimmed mask agrees with transformers' native
    `create_bidirectional_mask` to 0.995-1.000 -- two independent
    constructions of the same thing.
"""

from __future__ import annotations

import sys

MODEL_ID = "voyageai/voyage-4-nano"

# Card values are short -- median 47 characters, and the card generator clips
# constraints at 180 -- so this truncates nothing real.
MAX_SEQ_LENGTH = 256


def _patch_remote_code() -> None:
    from transformers import Qwen3Config
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    model_class = get_class_from_dynamic_module(
        "modeling_qwen3_bidirectional.Qwen3BidirectionalModel", MODEL_ID
    )
    if getattr(model_class, "config_class", None) is None:
        model_class.config_class = Qwen3Config

    module = sys.modules[model_class.__module__]
    if getattr(module, "_shimmed_for_transformers_5", False):
        return
    original = module.create_causal_mask

    def create_causal_mask(*args, **kwargs):
        kwargs.pop("cache_position", None)
        if "input_embeds" in kwargs:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        return original(*args, **kwargs)

    module.create_causal_mask = create_causal_mask
    module._shimmed_for_transformers_5 = True


def load_model():
    """Return a ready SentenceTransformer, or raise if it cannot be loaded."""
    _patch_remote_code()
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        MODEL_ID,
        trust_remote_code=True,
        model_kwargs={"attn_implementation": "sdpa"},
    )
    model.max_seq_length = MAX_SEQ_LENGTH
    return model
