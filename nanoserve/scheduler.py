"""Continuous batching: a slot pool, and a scheduler that keeps it full.

Every batch figure in RESULTS.md §3 was measured the easy way -- take N identical
prompts, run them in lockstep, wait for all of them. Real traffic does not arrive
that way. Requests turn up at different times and ask for different lengths, so a
lockstep batch spends most of its life partly idle: once the shortest sequence
finishes, its slot sits unused until the longest one drains.

Continuous batching removes that. Slots are an explicitly managed pool; a request
is admitted the moment a slot frees, mid-flight, without waiting for the batch
around it. The prize is quantified in §3 -- 29x the throughput at batch 32 -- and
the cost is that a single shared decode position no longer works, because every
slot is now at a different point in its own sequence.

Two pieces:

  `SlotCache`     -- KV storage whose position is a vector, one entry per slot,
                     so slots advance independently.
  `Scheduler`     -- admission, eviction, and the step loop over active slots.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import torch

from .config import GPTConfig
from .generate import _next_token
from .model import GPT


@dataclass
class Request:
    """One in-flight generation."""

    prompt: torch.Tensor  # [1, prompt_len]
    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    rid: int = field(default_factory=itertools.count().__next__)

    tokens: list[int] = field(default_factory=list)
    slot: int | None = None
    arrived: float = 0.0
    started: float = 0.0
    finished: float = 0.0

    @property
    def done(self) -> bool:
        return len(self.tokens) >= self.max_tokens


class _SlotView:
    """Presents one slot of a `SlotCache` as if it were a batch-1 cache.

    Prefill is variable-length and therefore cannot ride along with a decode step,
    which is fixed at one token. Rather than build a second code path, a new
    request is prefilled through the ordinary `KVCache` interface against a view
    of its own slot, then joins the decode batch.
    """

    def __init__(self, parent: SlotCache, slot: int) -> None:
        self.parent = parent
        self.slot = slot
        self.cfg = parent.cfg
        self.pos = 0

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        end = self.pos + k.size(2)
        if end > self.parent.max_seq:
            raise RuntimeError(f"slot {self.slot} overflow: {end} > {self.parent.max_seq}")
        s = slice(self.slot, self.slot + 1)
        self.parent.k[layer][s, :, self.pos : end] = k
        self.parent.v[layer][s, :, self.pos : end] = v
        return (
            self.parent.k[layer][s, :, :end],
            self.parent.v[layer][s, :, :end],
        )

    def advance(self, q_len: int) -> None:
        self.pos += q_len


class SlotCache:
    """KV storage for `n_slots` independent sequences.

    Identical layout to `KVCache` -- [slots, head, seq, head_dim] per layer -- but
    the position is a vector rather than a scalar, and writes scatter to each
    slot's own offset. That single change is what lets sequences at different
    points share one decode step.
    """

    def __init__(
        self,
        cfg: GPTConfig,
        n_slots: int,
        max_seq: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if max_seq > cfg.block_size:
            raise ValueError(f"max_seq={max_seq} exceeds context window {cfg.block_size}")
        self.cfg = cfg
        self.batch_size = n_slots
        self.n_slots = n_slots
        self.max_seq = max_seq
        self.device = torch.device(device)
        self.dtype = dtype
        self.window = max_seq

        shape = (n_slots, cfg.n_head, max_seq, cfg.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]

        # One position per slot. This is the whole difference from KVCache.
        self.pos_dev = torch.zeros(n_slots, dtype=torch.long, device=device)
        self._arange = torch.arange(max_seq, device=device)

    def nbytes(self) -> int:
        elems = self.n_slots * self.cfg.n_head * self.max_seq * self.cfg.head_dim
        return 2 * self.cfg.n_layer * elems * self.dtype.itemsize

    def view(self, slot: int) -> _SlotView:
        return _SlotView(self, slot)

    # -- decode step ------------------------------------------------------

    def append_static(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        """Write one token per slot, each at that slot's own position.

        `index_copy_` cannot express this -- it takes a single index shared by the
        whole batch. `scatter_` takes an index per element, so each slot lands at
        its own offset in one kernel.
        """
        b, h, _, d = k.shape
        idx = self.pos_dev.view(b, 1, 1, 1).expand(b, h, 1, d)
        self.k[layer].scatter_(2, idx, k)
        self.v[layer].scatter_(2, idx, v)
        w = self.window
        return self.k[layer][:, :, :w], self.v[layer][:, :, :w]

    def valid_mask(self) -> torch.Tensor:
        """[slots, 1, 1, window] -- each slot sees only its own written positions."""
        return (self._arange[: self.window].unsqueeze(0) <= self.pos_dev.unsqueeze(1)).view(
            self.n_slots, 1, 1, self.window
        )

    def advance_static(self) -> None:
        self.pos_dev += 1

    def release(self, slot: int) -> None:
        """Return a slot to the pool. Stale KV is harmless: the mask hides it."""
        self.pos_dev[slot] = 0


class Scheduler:
    """Keeps `n_slots` busy by admitting waiting requests as slots free."""

    def __init__(self, model: GPT, n_slots: int, max_seq: int) -> None:
        self.model = model
        self.n_slots = n_slots
        p = next(model.parameters())
        self.cache = SlotCache(model.cfg, n_slots, max_seq, device=p.device, dtype=p.dtype)
        self.free: list[int] = list(range(n_slots))
        self.active: dict[int, Request] = {}
        self.waiting: list[Request] = []
        self.finished: list[Request] = []
        # Token most recently produced by each slot; the input to the next step.
        self.last = torch.zeros(n_slots, 1, dtype=torch.long, device=p.device)
        self.steps = 0

    # -- admission --------------------------------------------------------

    def submit(self, req: Request) -> None:
        self.waiting.append(req)

    @torch.no_grad()
    def _admit(self, now: float) -> int:
        """Fill every free slot that a waiting request can use. Returns how many."""
        admitted = 0
        while self.free and self.waiting:
            req = self.waiting.pop(0)
            slot = self.free.pop(0)
            req.slot, req.started = slot, now

            view = self.cache.view(slot)
            logits = self.model(req.prompt, view, last_only=True)
            token = _next_token(logits, req.temperature, req.top_p, None)

            self.cache.pos_dev[slot] = view.pos
            self.last[slot] = token
            req.tokens.append(int(token.item()))
            self.active[slot] = req
            admitted += 1
        return admitted

    def _evict(self, now: float) -> int:
        done = [s for s, r in self.active.items() if r.done]
        for slot in done:
            req = self.active.pop(slot)
            req.finished = now
            self.finished.append(req)
            self.cache.release(slot)
            self.free.append(slot)
        return len(done)

    # -- the loop ---------------------------------------------------------

    @torch.no_grad()
    def step(self, now: float = 0.0) -> int:
        """Advance every active slot by one token. Returns tokens produced.

        Idle slots are stepped too. Their output is discarded, but they still ride
        along in the batch, because a fixed batch shape is what makes the step one
        kernel launch per layer instead of one per sequence. That waste is the
        argument for keeping the pool full -- which is the scheduler's job.
        """
        self._evict(now)
        self._admit(now)
        if not self.active:
            return 0

        logits = self.model(self.last, self.cache, static=True)
        for slot, req in list(self.active.items()):
            token = _next_token(logits[slot : slot + 1], req.temperature, req.top_p, None)
            self.last[slot] = token
            req.tokens.append(int(token.item()))

        self.steps += 1
        return len(self.active)

    @property
    def busy(self) -> bool:
        return bool(self.active or self.waiting)

    def run_to_completion(self, clock=lambda: 0.0) -> list[Request]:
        while self.busy:
            self.step(clock())
        self._evict(clock())
        return self.finished
