"""Continuous batching correctness.

The property that matters: a sequence's output must not depend on what happens to
share the batch with it. Slots are admitted and evicted at arbitrary times, so if
positions or masks leaked between them the tokens would drift -- plausibly, and
silently.
"""

import pytest
import torch

from nanoserve.generate import generate_cached
from nanoserve.scheduler import Request, Scheduler
from nanoserve.weights import load_pretrained, load_tokenizer


@pytest.fixture(scope="module")
def model():
    return load_pretrained("gpt2")


@pytest.fixture(scope="module")
def prompts():
    tok = load_tokenizer("gpt2")
    return [
        torch.tensor([tok.encode(t)])
        for t in (
            "The key insight about autoregressive decoding is that",
            "A KV cache works by",
            "Continuous batching means",
            "GPU memory is",
        )
    ]


def _reference(model, prompt, n):
    out = generate_cached(model, prompt, n, cache=model.new_cache(1, prompt.size(1) + n))
    return out[0, prompt.size(1) :].tolist()


def test_single_slot_matches_sequential(model, prompts):
    """One request through the scheduler == plain cached generation."""
    n = 16
    sched = Scheduler(model, n_slots=1, max_seq=prompts[0].size(1) + n)
    sched.submit(Request(prompt=prompts[0], max_tokens=n))
    got = sched.run_to_completion()[0].tokens
    assert got == _reference(model, prompts[0], n)


def test_batched_slots_are_independent(model, prompts):
    """Four different prompts sharing a batch must each match their solo run."""
    n = 16
    longest = max(p.size(1) for p in prompts) + n
    sched = Scheduler(model, n_slots=4, max_seq=longest)
    for p in prompts:
        sched.submit(Request(prompt=p, max_tokens=n))
    done = {r.rid: r for r in sched.run_to_completion()}

    for req in done.values():
        want = _reference(model, req.prompt, n)
        assert req.tokens == want, f"slot {req.slot} drifted from its solo run"


def test_staggered_admission_is_clean(model, prompts):
    """A slot reused by a later request must not inherit the earlier one's state.

    Two slots, four requests of differing lengths: slots free at different times
    and get refilled mid-flight, which is the whole point of the scheduler and the
    most likely place for stale KV or a stale position to leak through.
    """
    lengths = [8, 20, 12, 16]
    longest = max(p.size(1) for p in prompts) + max(lengths)
    sched = Scheduler(model, n_slots=2, max_seq=longest)
    for p, n in zip(prompts, lengths):
        sched.submit(Request(prompt=p, max_tokens=n))

    done = sched.run_to_completion()
    assert len(done) == 4
    for req in done:
        assert len(req.tokens) == req.max_tokens
        assert req.tokens == _reference(model, req.prompt, req.max_tokens), (
            f"request {req.rid} (slot {req.slot}) diverged after slot reuse"
        )


def test_more_requests_than_slots(model, prompts):
    """Queueing works: 4 requests through a single slot, run one after another."""
    n = 10
    longest = max(p.size(1) for p in prompts) + n
    sched = Scheduler(model, n_slots=1, max_seq=longest)
    for p in prompts:
        sched.submit(Request(prompt=p, max_tokens=n))
    done = sched.run_to_completion()
    assert len(done) == 4
    for req in done:
        assert req.tokens == _reference(model, req.prompt, n)
