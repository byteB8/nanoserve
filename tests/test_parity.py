"""Correctness gates.

Two claims need to hold before any benchmark number means anything:

  1. The from-scratch model reproduces HuggingFace's GPT-2 logits.
  2. The cached decode path reproduces the naive one, token for token.

If (2) fails, the speedup is just the cost of being wrong faster.
"""

import copy

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


def test_last_only_matches_full_projection(model, tokens):
    """Skipping the vocab projection on discarded positions must not change output.

    `last_only` exists to avoid materialising a [batch, T, 50257] tensor whose
    non-final rows generation never reads.

    Note this is *not* bit-identical, and shouldn't be asserted as such. Slicing
    before the projection hands the GEMM a [1, 768] input instead of [10, 768],
    which selects a different kernel with a different reduction order. Floating
    point addition is not associative, so the results differ in the last bits --
    measured at ~3e-5 absolute on logits of magnitude ~130, a relative error of
    3e-7, essentially fp32 epsilon. What must hold exactly is the *decision*:
    the ranking generation samples from.
    """
    with torch.no_grad():
        full = model(tokens)
        last = model(tokens, last_only=True)

    assert last.shape[1] == 1, f"expected a single position, got {last.shape[1]}"

    ref = full[:, -1]
    diff = (last[:, 0] - ref).abs().max().item()
    rel = diff / ref.abs().max().item()
    assert rel < 1e-6, f"last-position logits diverge by {rel:.2e} relative -- beyond rounding"
    assert torch.equal(last[:, 0].topk(5).indices, ref.topk(5).indices), "token ranking changed"


def test_static_path_matches_eager(model, tokens):
    """The constant-shape decode path must agree with the sliced one.

    `static=True` attends over the whole reserved window and masks, instead of
    slicing the cache to the live length. That is a different kernel shape for
    the same maths, so this checks the maths survived -- on CPU, where no graph
    is involved, isolating the shape change from the capture machinery.
    """
    with torch.no_grad():
        eager = model.new_cache(batch_size=1, max_seq=64)
        model(tokens[:, :-1], eager)
        want = model(tokens[:, -1:], eager, last_only=True)

        stat = model.new_cache(batch_size=1, max_seq=64)
        model(tokens[:, :-1], stat)          # prefill uses the ordinary path
        got = model(tokens[:, -1:], stat, static=True)

    diff = (got - want).abs().max().item()
    assert diff < 1e-4, f"static decode diverges from eager by {diff:.2e}"
    assert int(stat.pos_dev.item()) == stat.pos + 1, "static path did not advance the position"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a CUDA device")
def test_graphed_decode_matches_cached(model, tokens):
    """Replaying a captured graph must produce the same tokens as eager decode.

    A graph replays fixed kernels against fixed addresses. If the cache position
    or the mask were captured as constants instead of read from device memory,
    every step after the first would attend over the wrong window -- and would
    still return plausible tokens. Only comparing full sequences catches that.
    """
    from nanoserve.generate import generate_cached, generate_graphed

    # deepcopy, not model.cuda(): nn.Module.cuda() mutates in place, and this
    # fixture is module-scoped. Moving it would strand every later test on a
    # different device from its inputs.
    gpu = copy.deepcopy(model).cuda()
    prompt = tokens.cuda()
    n = 32

    want = generate_cached(gpu, prompt, n)
    got = generate_graphed(gpu, prompt, n)

    assert torch.equal(got, want), (
        "graphed decode diverged at token "
        f"{int((got[0] != want[0]).nonzero()[0].item()) - prompt.size(1)}"
    )


def test_int8_kv_preserves_generation(model, tokens):
    """An int8 KV cache must not change what the model says.

    Per-token symmetric scaling should be well within the margin that separates
    competing tokens: quantisation error lands around 1/127 of each vector's own
    dynamic range, while argmax gaps are typically far wider. If this ever fails,
    the scheme is wrong -- not the tolerance.
    """
    from nanoserve.generate import generate_cached

    n = 32
    prompt_len = tokens.size(1)
    fp = generate_cached(model, tokens, n, cache=model.new_cache(1, prompt_len + n))
    q8 = generate_cached(
        model, tokens, n, cache=model.new_cache(1, prompt_len + n, kv_dtype="int8")
    )

    same = fp[0, prompt_len:] == q8[0, prompt_len:]
    assert bool(same.all()), (
        f"int8 KV diverged at generated token {int((~same).nonzero()[0].item())} "
        f"({int(same.sum())}/{n} agreed)"
    )


def test_int8_kv_is_smaller(model):
    """The point of the exercise: stored bytes must actually drop."""
    fp = model.new_cache(batch_size=4, max_seq=512)
    q8 = model.new_cache(batch_size=4, max_seq=512, kv_dtype="int8")
    ratio = fp.nbytes() / q8.nbytes()
    # 4x from int8, minus one fp scale per token per head (head_dim=64) -> ~3.76x
    assert 3.5 < ratio < 4.0, f"expected ~3.8x storage reduction, got {ratio:.2f}x"
