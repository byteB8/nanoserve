"""INT8 KV cache.

§4 of RESULTS.md established that the KV-cache, not the weights, sets how many
sequences you can serve: at batch 1024 it was 19 GiB of a 20 GiB peak. Storing it
in int8 instead of fp32 attacks precisely that term.

**Scheme.** Symmetric, per-token and per-head:

    scale = max(|x|) over head_dim / 127
    q     = round(x / scale)  clamped to [-127, 127]

Each token's K (and V) vector across one head gets its own scale, computed when
the token is written and never revisited. That matches how a cache is actually
used -- positions arrive one at a time and are then immutable -- whereas a
per-channel scheme would need to see the whole sequence before it could pick
scales. A per-tensor scale would be cheaper still but has to cover every token's
dynamic range at once, which is what makes per-tensor int8 KV lossy in practice.

**What this does and does not buy.** Attention runs in floating point: values are
dequantised on read, because doing the matmul in int8 needs kernels beyond what
PyTorch exposes here. So the saving is memory, not bandwidth or arithmetic --
which is the term that matters for the concurrency ceiling. Note also that the
dequantised window for the layer being processed is a real fp tensor, so the
saving is less than the naive 4x. `nbytes` reports storage; the benchmark reports
measured peak, and the gap between them is exactly that transient.
"""

from __future__ import annotations

import torch

from .config import GPTConfig

QMAX = 127.0


def quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-token/per-head int8. `x` is [batch, head, seq, head_dim]."""
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / QMAX
    q = (x / scale).round().clamp(-QMAX, QMAX).to(torch.int8)
    return q, scale


def dequantize(q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return q.to(dtype) * scale.to(dtype)


class QuantizedKVCache:
    """Drop-in int8 replacement for `KVCache`.

    Deliberately exposes the same surface as the fp cache -- `append`, `advance`,
    `reset`, `nbytes` -- so the model never learns which one it is holding. The
    quantisation happens on write and the dequantisation on read, both inside
    `append`.
    """

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
                f"max_seq={max_seq} exceeds the model's context window ({cfg.block_size})"
            )
        self.cfg = cfg
        self.batch_size = batch_size
        self.max_seq = max_seq
        self.device = torch.device(device)
        self.dtype = dtype  # dtype attention runs in, after dequantisation

        shape = (batch_size, cfg.n_head, max_seq, cfg.head_dim)
        sshape = (batch_size, cfg.n_head, max_seq, 1)
        self.k_q = [torch.zeros(shape, device=device, dtype=torch.int8) for _ in range(cfg.n_layer)]
        self.v_q = [torch.zeros(shape, device=device, dtype=torch.int8) for _ in range(cfg.n_layer)]
        self.k_s = [torch.zeros(sshape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]
        self.v_s = [torch.zeros(sshape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]

        self.pos = 0
        self.pos_dev = torch.zeros(1, dtype=torch.long, device=device)
        self.window = max_seq

    # -- memory -----------------------------------------------------------

    def nbytes(self) -> int:
        """Bytes of *stored* cache. Excludes the transient dequantised window."""
        n = self.batch_size * self.cfg.n_head * self.max_seq
        payload = 2 * self.cfg.n_layer * n * self.cfg.head_dim * 1  # int8 K and V
        scales = 2 * self.cfg.n_layer * n * self.dtype.itemsize  # one scale per token/head
        return payload + scales

    def summary(self) -> str:
        per_tok = self.nbytes() / (self.batch_size * self.max_seq)
        return (
            f"QuantizedKVCache batch={self.batch_size} max_seq={self.max_seq} int8 "
            f"-> {self.nbytes() / 2**20:.1f} MiB stored "
            f"({per_tok / 1024:.1f} KiB per token per sequence)"
        )

    # -- read / write -----------------------------------------------------

    def append(
        self, layer: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_len = k.size(2)
        end = self.pos + q_len
        if end > self.max_seq:
            raise RuntimeError(
                f"KV-cache overflow: writing {q_len} token(s) at position "
                f"{self.pos} exceeds max_seq={self.max_seq}."
            )

        kq, ks = quantize(k)
        vq, vs = quantize(v)
        self.k_q[layer][:, :, self.pos : end] = kq
        self.k_s[layer][:, :, self.pos : end] = ks
        self.v_q[layer][:, :, self.pos : end] = vq
        self.v_s[layer][:, :, self.pos : end] = vs

        return (
            dequantize(self.k_q[layer][:, :, :end], self.k_s[layer][:, :, :end], self.dtype),
            dequantize(self.v_q[layer][:, :, :end], self.v_s[layer][:, :, :end], self.dtype),
        )

    def advance(self, q_len: int) -> None:
        self.pos += q_len
        self.pos_dev.fill_(self.pos)

    def reset(self) -> None:
        self.pos = 0
        self.pos_dev.zero_()

    # -- static-shape path ------------------------------------------------

    def append_static(
        self, layer: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kq, ks = quantize(k)
        vq, vs = quantize(v)
        self.k_q[layer].index_copy_(2, self.pos_dev, kq)
        self.k_s[layer].index_copy_(2, self.pos_dev, ks)
        self.v_q[layer].index_copy_(2, self.pos_dev, vq)
        self.v_s[layer].index_copy_(2, self.pos_dev, vs)
        w = self.window
        return (
            dequantize(self.k_q[layer][:, :, :w], self.k_s[layer][:, :, :w], self.dtype),
            dequantize(self.v_q[layer][:, :, :w], self.v_s[layer][:, :, :w], self.dtype),
        )

    def valid_mask(self) -> torch.Tensor:
        ar = torch.arange(self.window, device=self.device)
        return (ar <= self.pos_dev).view(1, 1, 1, self.window)

    def advance_static(self) -> None:
        self.pos_dev += 1
