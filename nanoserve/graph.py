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


class GraphedDecoder:
    """A captured single-token decode step, replayable at a fixed batch size.

    Usage mirrors a forward pass, except the caller must accept that the returned
    logits live in a buffer that the next `step` overwrites::

        decoder = GraphedDecoder(model, cache, batch_size)
        logits = decoder.step(token)      # valid until the next step()
    """

    def __init__(self, model: GPT, cache: KVCache, batch_size: int, warmup: int = 3) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA graphs require a CUDA device")
        if cache.batch_size != batch_size:
            raise ValueError(f"cache is sized for batch {cache.batch_size}, not {batch_size}")

        self.model = model
        self.cache = cache
        self.batch_size = batch_size

        device = cache.device
        self.token = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

        # Warm up on a side stream first. Capture records whatever kernels run,
        # so any one-off initialisation -- cuBLAS handles, autotuning, lazy module
        # setup -- must happen before capture or it gets baked into the graph.
        # Warmup also advances the cache, so the position is restored afterwards.
        saved_pos = int(cache.pos_dev.item())
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(warmup):
                cache.pos_dev.fill_(saved_pos)
                model(self.token, cache, static=True)
        torch.cuda.current_stream().wait_stream(stream)

        cache.pos_dev.fill_(saved_pos)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.logits = model(self.token, cache, static=True)

        # Capture itself ran the step once and advanced the position; undo that
        # so the caller starts from where they left off.
        cache.pos_dev.fill_(saved_pos)

    def step(self, token: torch.Tensor) -> torch.Tensor:
        """Decode one token. Returns logits [batch, 1, vocab] valid until next call."""
        self.token.copy_(token)
        self.graph.replay()
        return self.logits
