"""GPT-2, written out rather than imported.

Nothing here calls into `transformers` at runtime -- that library is used only
to fetch the pretrained weights (see `weights.py`) and as a correctness oracle
in the tests. The forward pass, the attention maths and the decode path are all
local, because the point of the exercise is to own the code the KV-cache lives in.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cache import KVCache
from .config import GPTConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(
        self, x: torch.Tensor, cache: KVCache | None = None, static: bool = False
    ) -> torch.Tensor:
        B, T, C = x.shape
        n_head, head_dim = self.cfg.n_head, self.cfg.head_dim

        q, k, v = self.c_attn(x).split(C, dim=2)
        # [B, T, C] -> [B, n_head, T, head_dim]
        q = q.view(B, T, n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, n_head, head_dim).transpose(1, 2)

        if static:
            # Constant-shape decode: attend over the whole reserved window and
            # mask, rather than slicing to the live length. Costs attention over
            # padding, buys a kernel sequence a CUDA graph can capture.
            k, v = cache.append_static(self.layer_idx, k, v)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=cache.valid_mask())
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            return self.c_proj(y)

        if cache is None:
            # No cache: the query block is the whole sequence, so a plain causal
            # mask applies. This is the naive path -- correct, and quadratic.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            past = cache.pos
            k, v = cache.append(self.layer_idx, k, v)
            if T == 1:
                # Decode step: one query attending to every cached position.
                # Every key is at a position <= the query's, so no mask is needed
                # -- and skipping it avoids materialising a [1, S] mask per layer.
                y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            else:
                # Prefill (or a chunked extend) with a non-empty cache: the causal
                # boundary sits at an offset, so `is_causal=True` would mask the
                # wrong cells. Build the offset mask explicitly.
                total = past + T
                qi = torch.arange(past, total, device=x.device).unsqueeze(1)
                ki = torch.arange(total, device=x.device).unsqueeze(0)
                mask = (ki <= qi).view(1, 1, T, total)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        # GPT-2 shipped with the tanh approximation; matching it matters for
        # bit-level parity against the reference implementation.
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg, layer_idx)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(
        self, x: torch.Tensor, cache: KVCache | None = None, static: bool = False
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), cache, static)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.h = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # GPT-2 ties the output projection to the input embedding.
        self.lm_head.weight = self.wte.weight

    def forward(
        self,
        idx: torch.Tensor,
        cache: KVCache | None = None,
        last_only: bool = False,
        static: bool = False,
    ) -> torch.Tensor:
        """Return logits for every position in `idx`, or just the last one.

        With a cache, `idx` holds only the *new* tokens; positions are numbered
        from where the cache left off, which is what makes an incremental decode
        step see the same positional embeddings the naive path would have used.

        `last_only` drops the vocab projection on every position except the final
        one. Generation reads only that position, and the projection is the widest
        tensor in the model -- [batch, T, 50257]. Measured on an A5000, computing
        it across a 10-token prompt cost 2.01 MiB per sequence, which at batch
        1024 is 2 GiB spent on values that are immediately discarded.
        """
        B, T = idx.shape

        if static:
            # Position comes from the device tensor, not a Python int, so the
            # embedding lookup is a kernel the graph can record.
            x = self.wte(idx) + self.wpe(cache.pos_dev).view(1, 1, self.cfg.n_embd)
            for block in self.h:
                x = block(x, cache, static=True)
            cache.advance_static()
            return self.lm_head(self.ln_f(x))

        past = cache.pos if cache is not None else 0
        if past + T > self.cfg.block_size:
            raise ValueError(
                f"sequence length {past + T} exceeds context window {self.cfg.block_size}"
            )

        pos = torch.arange(past, past + T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)

        for block in self.h:
            x = block(x, cache)

        if cache is not None:
            cache.advance(T)

        x = self.ln_f(x)
        if last_only:
            x = x[:, -1:, :]
        return self.lm_head(x)

    # -- convenience ------------------------------------------------------

    @torch.no_grad()
    def new_cache(self, batch_size: int, max_seq: int, kv_dtype: str = "auto"):
        """Allocate a cache. `kv_dtype="int8"` swaps in the quantised store.

        Returns a different class, not a differently-configured one -- but both
        expose the same surface, so nothing in the forward pass branches on it.
        """
        p = next(self.parameters())
        if kv_dtype == "int8":
            from .quant import QuantizedKVCache

            return QuantizedKVCache(
                self.cfg, batch_size, max_seq, device=p.device, dtype=p.dtype
            )
        return KVCache(self.cfg, batch_size, max_seq, device=p.device, dtype=p.dtype)

    def n_params(self) -> int:
        # lm_head is tied to wte, so count distinct tensors only.
        seen: set[int] = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total
