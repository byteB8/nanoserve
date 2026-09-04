"""Pre-allocated key/value cache.

The cache is the whole point of this project, so it gets its own module rather
than living as a tuple passed between functions.

Why pre-allocate instead of concatenating each step? Concatenation reallocates
and copies the entire cache on every generated token, which quietly turns an
O(1)-per-step decode back into O(t). Real serving engines reserve the memory up
front for exactly this reason -- and doing so forces you to state the memory
budget as a number before the run starts.
"""

from __future__ import annotations

import torch

from .config import GPTConfig


class KVCache:
    """Per-layer K/V storage of shape [batch, n_head, max_seq, head_dim]."""

    def __init__(
        self,
        cfg: GPTConfig,
        batch_size: int,
        max_seq: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if max_seq > cfg.block_size:
            raise ValueError(
                f"max_seq={max_seq} exceeds the model's context window "
                f"({cfg.block_size}); GPT-2 has no positions beyond that."
            )
        self.cfg = cfg
        self.batch_size = batch_size
        self.max_seq = max_seq
        self.device = torch.device(device)
        self.dtype = dtype

        shape = (batch_size, cfg.n_head, max_seq, cfg.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]

        # How many positions currently hold real data. Advanced once per forward
        # pass by the model, not once per layer.
        self.pos = 0

    # -- memory -----------------------------------------------------------

    def nbytes(self) -> int:
        """Total bytes reserved by this cache."""
        elems = self.batch_size * self.cfg.n_head * self.max_seq * self.cfg.head_dim
        return 2 * self.cfg.n_layer * elems * self.dtype.itemsize

    def summary(self) -> str:
        mib = self.nbytes() / 2**20
        per_tok = self.cfg.kv_bytes_per_token(self.dtype.itemsize)
        return (
            f"KVCache batch={self.batch_size} max_seq={self.max_seq} "
            f"dtype={self.dtype} -> {mib:.1f} MiB reserved "
            f"({per_tok / 1024:.0f} KiB per token per sequence)"
        )

    # -- read / write -----------------------------------------------------

    def append(
        self, layer: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write this step's K/V for `layer`, and return everything cached so far.

        `k`/`v` arrive as [batch, n_head, q_len, head_dim]. The returned views
        span positions [0, pos + q_len), which is exactly what the query must
        attend over.
        """
        q_len = k.size(2)
        end = self.pos + q_len
        if end > self.max_seq:
            raise RuntimeError(
                f"KV-cache overflow: writing {q_len} token(s) at position "
                f"{self.pos} exceeds max_seq={self.max_seq}."
            )
        self.k[layer][:, :, self.pos : end] = k
        self.v[layer][:, :, self.pos : end] = v
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]

    def advance(self, q_len: int) -> None:
        """Commit `q_len` positions. Called once per forward pass."""
        self.pos += q_len

    def reset(self) -> None:
        self.pos = 0
