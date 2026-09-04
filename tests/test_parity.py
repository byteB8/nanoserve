"""Correctness gates.

Two claims need to hold before any benchmark number means anything:

  1. The from-scratch model reproduces HuggingFace's GPT-2 logits.
  2. The cached decode path reproduces the naive one, token for token.

If (2) fails, the speedup is just the cost of being wrong faster.
"""

import pytest
import torch

from nanoserve.weights import load_pretrained, load_tokenizer

torch.manual_seed(0)


@pytest.fixture(scope="module")
def model():
    return load_pretrained("gpt2")


@pytest.fixture(scope="module")
def tokens():
    tok = load_tokenizer("gpt2")
    text = "The key insight about autoregressive decoding is that"
    return torch.tensor([tok.encode(text)])


def test_logits_match_reference(model, tokens):
    """Our forward pass agrees with the reference implementation."""
    from transformers import GPT2LMHeadModel

    ref = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    with torch.no_grad():
        ours = model(tokens)
        theirs = ref(tokens).logits

    max_abs = (ours - theirs).abs().max().item()
    assert max_abs < 1e-3, f"logits diverge from reference by {max_abs:.2e}"
    assert ours.argmax(-1).equal(theirs.argmax(-1)), "argmax token predictions differ"


def test_cache_matches_naive(model, tokens):
    """Incremental decode with a cache == recomputing the prefix every step.

    This is the load-bearing test of the whole project: it certifies that the
    KV-cache is an optimisation and not a change in behaviour.
    """
    with torch.no_grad():
        full = model(tokens)

        cache = model.new_cache(batch_size=1, max_seq=64)
        # Prefill everything but the last token, then step the final one through
        # the incremental path.
        model(tokens[:, :-1], cache)
        stepped = model(tokens[:, -1:], cache)

    max_abs = (stepped[:, -1] - full[:, -1]).abs().max().item()
    assert max_abs < 1e-4, f"cached decode diverges from naive by {max_abs:.2e}"


def test_cache_token_by_token(model, tokens):
    """Feeding one token at a time must match a single full-sequence pass."""
    with torch.no_grad():
        full = model(tokens)
        cache = model.new_cache(batch_size=1, max_seq=64)
        outs = [model(tokens[:, i : i + 1], cache) for i in range(tokens.size(1))]
    stepwise = torch.cat(outs, dim=1)

    max_abs = (stepwise - full).abs().max().item()
    assert max_abs < 1e-4, f"token-by-token decode diverges by {max_abs:.2e}"


def test_cache_overflow_is_loud(model, tokens):
    """Running past the reserved window should raise, not silently corrupt."""
    cache = model.new_cache(batch_size=1, max_seq=4)
    with torch.no_grad(), pytest.raises(RuntimeError, match="overflow"):
        model(tokens[:, :8], cache)
