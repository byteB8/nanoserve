"""CUDA graph capture for the decode step.

Motivation is measured, not assumed. Decode of GPT-2 124M at batch 1 runs at
3.2 ms/token on an A5000, while the memory-bandwidth floor (475 MiB of weights
at 768 GB/s) is about 0.6 ms. The missing ~80% is per-kernel launch and Python
dispatch across 12 layers -- roughly 120 launches to produce a single token.

A CUDA graph records that kernel sequence once and replays it with a single
launch, which removes the dispatch cost entirely. The price is rigidity: a graph
replays fixed kernels against fixed addresses, so every shape and every pointer
must be identical on each replay.

Three things had to change to make the decode step capturable, all in the
`static=True` path:

  * the cache position lives in a device tensor (`pos_dev`), not a Python int,
    so the write target can move without re-recording;
  * attention spans the whole reserved window with a mask derived on device,
    rather than slicing to the live length, which would change shapes each step;
  * the input token and output logits are fixed buffers the caller writes into
    and reads out of, rather than fresh allocations.
"""

from __future__ import annotations

import torch

from .cache import KVCache
from .model import GPT


DEFAULT_BUCKETS = (128, 256, 512, 1024)


class GraphedDecoder:
    """Captured single-token decode steps, bucketed by attention window.

    A single full-window graph removes launch overhead but replaces it with a new
    cost: attention over every reserved slot on every step, including the ones a
    short sequence never reaches. Measured on an A5000, that turned a 2.29x win at
    64 tokens into 1.11x at 1000 -- the launch saving is constant, the wasted
    attention grows with the reservation.

    So capture several graphs at increasing windows and replay the smallest one
    that covers the current position. Each graph is still fixed-shape, which is
    all capture requires; the choice between them is an ordinary Python branch
    outside any graph.

    Logits live in a per-bucket buffer that the next `step` overwrites::

        decoder = GraphedDecoder(model, cache, batch_size)
        logits = decoder.step(token)      # valid until the next step()
    """

    def __init__(
        self,
        model: GPT,
        cache: KVCache,
        batch_size: int,
        buckets: tuple[int, ...] | None = None,
        warmup: int = 3,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA graphs require a CUDA device")
        if cache.batch_size != batch_size:
            raise ValueError(f"cache is sized for batch {cache.batch_size}, not {batch_size}")

        self.model = model
        self.cache = cache
        self.batch_size = batch_size

        chosen = sorted({min(b, cache.max_seq) for b in (buckets or DEFAULT_BUCKETS)})
        self.buckets = [b for b in chosen if b <= cache.max_seq] or [cache.max_seq]
        if self.buckets[-1] < cache.max_seq:
            self.buckets.append(cache.max_seq)

        self.token = torch.zeros(batch_size, 1, dtype=torch.long, device=cache.device)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.outputs: dict[int, torch.Tensor] = {}

        saved_pos = int(cache.pos_dev.item())
        saved_window = cache.window

        # Warm up once, on a side stream, before any capture. Capture records
        # whatever kernels run, so one-off initialisation -- cuBLAS handles,
        # autotuning, lazy module setup -- must be flushed out beforehand or it
        # gets baked into the first graph.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(warmup):
                cache.pos_dev.fill_(saved_pos)
                model(self.token, cache, static=True)
        torch.cuda.current_stream().wait_stream(stream)

        # One graph per bucket. They share the same weights and the same cache
        # storage; only the attention window differs.
        pool = None
        for window in self.buckets:
            cache.window = window
            cache.pos_dev.fill_(min(saved_pos, window - 1))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool), torch.no_grad():
                self.outputs[window] = model(self.token, cache, static=True)
            pool = graph.pool()  # share one memory pool across buckets
            self.graphs[window] = graph

        cache.window = saved_window
        cache.pos_dev.fill_(saved_pos)

        # Host-side mirror of the decode position. Reading `pos_dev` would mean a
        # device-to-host sync on every token, stalling the pipeline for exactly
        # the overhead the graph exists to remove. The position advances by one
        # per step and by nothing else, so it can be tracked here and reconciled
        # once per generation via `sync()`.
        self._pos = saved_pos

    def sync(self) -> None:
        """Re-read the device position. Call once after a prefill, not per token."""
        self._pos = int(self.cache.pos_dev.item())

    def _bucket_for(self, pos: int) -> int:
        for b in self.buckets:
            if pos < b:
                return b
        return self.buckets[-1]

    def step(self, token: torch.Tensor) -> torch.Tensor:
        """Decode one token. Returns logits [batch, 1, vocab] valid until next call."""
        window = self._bucket_for(self._pos)
        self.token.copy_(token)
        self.graphs[window].replay()
        self._pos += 1
        return self.outputs[window]
