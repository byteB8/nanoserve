"""Two decode loops that produce identical tokens at very different costs.

`generate_naive` is the strawman everyone writes first: no state is kept between
steps, so producing token t re-runs attention over all t-1 previous tokens. Total
work is O(T^2) in the sequence length.

`generate_cached` keeps each layer's K/V, so a step only projects the single new
token and attends it against stored keys. Total work is O(T).

The two share a sampler so that any measured difference is attributable to the
cache and nothing else.
"""

from __future__ import annotations

import torch

from .model import GPT


def _next_token(logits: torch.Tensor, temperature: float, top_p: float, generator) -> torch.Tensor:
    """Pick the next token from the final position's logits. Greedy when temperature == 0."""
    logits = logits[:, -1, :]
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    if top_p < 1.0:
        ordered, idx = torch.sort(probs, descending=True, dim=-1)
        cumulative = ordered.cumsum(dim=-1)
        # Keep the smallest prefix whose mass exceeds top_p; the shift keeps the
        # first token above the threshold rather than dropping it.
        drop = cumulative - ordered > top_p
        ordered[drop] = 0.0
        ordered /= ordered.sum(dim=-1, keepdim=True)
        choice = torch.multinomial(ordered, num_samples=1, generator=generator)
        return idx.gather(-1, choice)

    return torch.multinomial(probs, num_samples=1, generator=generator)


@torch.no_grad()
def generate_naive(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    generator=None,
) -> torch.Tensor:
    """Regenerate from the full sequence every step. No cache, quadratic work."""
    for _ in range(max_new_tokens):
        window = idx[:, -model.cfg.block_size :]
        logits = model(window, last_only=True)
        idx = torch.cat([idx, _next_token(logits, temperature, top_p, generator)], dim=1)
    return idx


@torch.no_grad()
def generate_cached(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    generator=None,
    cache=None,
) -> torch.Tensor:
    """Prefill the prompt once, then advance one token at a time through the cache.

    Pass an existing `cache` to reuse its storage. Allocating and zeroing a cache
    is not free -- tens of MiB of memset at long context -- and a server does it
    once per slot, not once per request. Benchmarks should hoist it out of the
    timed region or they attribute setup cost to the decode algorithm.
    """
    batch, prompt_len = idx.shape
    if cache is None:
        cache = model.new_cache(batch_size=batch, max_seq=prompt_len + max_new_tokens)
    else:
        cache.reset()

    # Prefill: the whole prompt in a single pass. This is the compute-bound
    # phase -- one big matmul per layer, high arithmetic intensity.
    logits = model(idx, cache, last_only=True)
    token = _next_token(logits, temperature, top_p, generator)
    out = [idx, token]

    # Decode: one token per pass. Memory-bandwidth-bound -- the model reads all
    # of its weights to produce a single token, which is why batching pays.
    for _ in range(max_new_tokens - 1):
        logits = model(token, cache, last_only=True)
        token = _next_token(logits, temperature, top_p, generator)
        out.append(token)

    return torch.cat(out, dim=1)


@torch.no_grad()
def stream_cached(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    generator=None,
    cache=None,
    stop_ids: set[int] | None = None,
):
    """Yield token ids one at a time, as they are produced.

    Same decode loop as `generate_cached`, surfaced as a generator so a server can
    flush each token to the client instead of waiting for the whole completion.
    Kept separate rather than refactoring `generate_cached` into it: that function
    is what every benchmark in RESULTS.md timed, and wrapping its inner loop in
    generator machinery would quietly change the numbers.
    """
    batch, prompt_len = idx.shape
    if batch != 1:
        raise ValueError("streaming is single-sequence; batch decoding goes through generate_cached")
    if cache is None:
        cache = model.new_cache(batch_size=1, max_seq=prompt_len + max_new_tokens)
    else:
        cache.reset()

    logits = model(idx, cache, last_only=True)
    token = _next_token(logits, temperature, top_p, generator)

    for _ in range(max_new_tokens):
        tid = int(token.item())
        if stop_ids and tid in stop_ids:
            return
        yield tid
        logits = model(token, cache, last_only=True)
        token = _next_token(logits, temperature, top_p, generator)


@torch.no_grad()
def generate_graphed(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    generator=None,
    cache=None,
    decoder=None,
) -> torch.Tensor:
    """Same as `generate_cached`, with the decode step replayed from a CUDA graph.

    Prefill still runs eagerly: it happens once, its shape depends on the prompt,
    and capturing a graph per prompt length would cost more than it saves. Only
    the decode loop -- the part executed hundreds of times at a fixed shape -- is
    captured.

    Pass a `decoder` to reuse a capture across calls. Capturing takes a few
    hundred milliseconds, so a benchmark that captures inside its timed region is
    measuring the wrong thing.
    """
    from .graph import GraphedDecoder

    batch, prompt_len = idx.shape
    if cache is None:
        cache = model.new_cache(batch_size=batch, max_seq=prompt_len + max_new_tokens)
    else:
        cache.reset()

    logits = model(idx, cache, last_only=True)
    token = _next_token(logits, temperature, top_p, generator)
    out = [idx, token]

    if decoder is None:
        decoder = GraphedDecoder(model, cache, batch_size=batch)
    else:
        decoder.sync()  # one device read per generation, not per token

    for _ in range(max_new_tokens - 1):
        logits = decoder.step(token)
        token = _next_token(logits, temperature, top_p, generator)
        out.append(token)

    return torch.cat(out, dim=1)
