"""Port pretrained GPT-2 weights into the local module definitions.

The one real trap: HuggingFace's GPT-2 stores the attention and MLP projections
as `Conv1D`, whose weight is [in_features, out_features] -- the transpose of what
`nn.Linear` expects. Getting this wrong produces a model that runs happily and
emits fluent-looking nonsense, so every tensor is shape-checked on the way in
rather than assigned on faith.
"""

from __future__ import annotations

import torch

from .config import PRESETS, GPTConfig
from .model import GPT

# Projections stored transposed by HuggingFace's Conv1D.
_TRANSPOSED = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")


def _assign(dst: torch.Tensor, src: torch.Tensor, name: str) -> None:
    if dst.shape != src.shape:
        raise ValueError(f"shape mismatch for {name}: local {tuple(dst.shape)} vs source {tuple(src.shape)}")
    with torch.no_grad():
        dst.copy_(src)


def load_pretrained(model_name: str = "gpt2", dtype: torch.dtype = torch.float32) -> GPT:
    """Build a local `GPT` and fill it with the released `model_name` weights."""
    from transformers import GPT2LMHeadModel  # imported lazily: not needed at inference

    if model_name not in PRESETS:
        raise KeyError(f"unknown model {model_name!r}; known: {sorted(PRESETS)}")
    cfg: GPTConfig = PRESETS[model_name]

    hf = GPT2LMHeadModel.from_pretrained(model_name)
    hf.eval()
    src = hf.state_dict()

    model = GPT(cfg)
    _assign(model.wte.weight, src["transformer.wte.weight"], "wte")
    _assign(model.wpe.weight, src["transformer.wpe.weight"], "wpe")
    _assign(model.ln_f.weight, src["transformer.ln_f.weight"], "ln_f.weight")
    _assign(model.ln_f.bias, src["transformer.ln_f.bias"], "ln_f.bias")

    for i, block in enumerate(model.h):
        p = f"transformer.h.{i}."
        _assign(block.ln_1.weight, src[p + "ln_1.weight"], p + "ln_1.weight")
        _assign(block.ln_1.bias, src[p + "ln_1.bias"], p + "ln_1.bias")
        _assign(block.ln_2.weight, src[p + "ln_2.weight"], p + "ln_2.weight")
        _assign(block.ln_2.bias, src[p + "ln_2.bias"], p + "ln_2.bias")

        for suffix, dst in (
            ("attn.c_attn", block.attn.c_attn),
            ("attn.c_proj", block.attn.c_proj),
            ("mlp.c_fc", block.mlp.c_fc),
            ("mlp.c_proj", block.mlp.c_proj),
        ):
            w = src[p + suffix + ".weight"]
            # Transpose only if the source is stored Conv1D-style. Newer
            # transformers releases have migrated some of these to nn.Linear,
            # so decide from the shape rather than the version number.
            if suffix + ".weight" in _TRANSPOSED and w.shape == (dst.in_features, dst.out_features):
                w = w.t()
            _assign(dst.weight, w, p + suffix + ".weight")
            _assign(dst.bias, src[p + suffix + ".bias"], p + suffix + ".bias")

    model = model.to(dtype=dtype)
    model.eval()
    return model


def load_tokenizer(model_name: str = "gpt2"):
    from transformers import GPT2TokenizerFast

    return GPT2TokenizerFast.from_pretrained(model_name)
