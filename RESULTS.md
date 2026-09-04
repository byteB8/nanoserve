# Results

All numbers below are reproducible with `python -m nanoserve.bench all`, which writes raw harness output to `results/` (gitignored — regenerate rather than trust a stale file).

**Environment** — Intel i5-1035G1 (4 cores / 8 threads, Ice Lake), 15 GiB RAM, no GPU. PyTorch 2.9.0, Python 3.12.3, 4 threads, GPT-2 124M in fp32.

Every measurement is best-of-N wall clock. Best-of rather than mean: we want the machine's capability, not its worst scheduling luck.

---

## 1. What the KV-cache buys

| New tokens | Naive (s) | +KV-cache (s) | Speedup | ms/token (cached) |
|---|---|---|---|---|
| 32 | 2.05 | 0.64 | **3.2×** | 19.8 |
| 64 | 6.19 | 1.31 | **4.7×** | 20.4 |
| 128 | 18.69 | 2.50 | **7.5×** | 19.5 |
| 256 | 57.24 | 5.11 | **11.2×** | 20.0 |

The single number people usually quote — "the KV-cache makes it ~10× faster" — is the least interesting thing here. **The speedup is not a constant; it grows with sequence length.** That growth is the actual evidence that the asymptotic complexity changed.

Look at the per-doubling scaling factors:

| | 32→64 | 64→128 | 128→256 | Interpretation |
|---|---|---|---|---|
| Naive | 3.02× | 3.02× | 3.06× | superlinear — work grows faster than length |
| Cached | 2.05× | 1.91× | 2.04× | **exactly linear** — 2× the tokens, 2× the time |

The cached path holds a flat ~20 ms/token regardless of how far into the sequence it is. That flatness *is* the O(1)-per-step property, measured directly.

### Why naive scales as T^1.6, not T²

Theory says naive decoding should be O(T²). Measured, it's ~T^1.6 (since 2^1.6 ≈ 3.03, matching the observed per-doubling factor almost exactly).

This is not an error in the theory — it's a second effect partially cancelling the first. Naive step *t* processes a *t*-token block, and larger matmuls run at much higher efficiency. Measured on this CPU with a GPT-2-shaped matmul:

```
T=  1    0.25 ms   →   18.7 GFLOP/s     (one token: memory-bandwidth-bound)
T=256    4.21 ms   →  286.7 GFLOP/s     (a block: compute-bound)
```

A **15× efficiency spread**. So while naive decoding does quadratically more arithmetic, it does that arithmetic at a rising hardware efficiency, and the observed exponent lands between 1 and 2. On hardware with a wider gap between compute and bandwidth — any modern GPU — the exponent moves back toward 2 and the cache wins by more.

Extrapolating the fitted scaling to lengths too slow to run repeatedly here: **≈17× at 512 tokens, ≈26× at 1024.**

---

## 2. Batch scaling, and why decode is the odd phase

| Batch | Wall (s) | Total tokens/s | Per-sequence tokens/s | Throughput vs batch 1 |
|---|---|---|---|---|
| 1 | 1.27 | 50.6 | 50.6 | 1.0× |
| 2 | 1.32 | 96.6 | 48.3 | **1.9×** |
| 4 | 2.30 | 111.2 | 27.8 | 2.2× |
| 8 | 3.23 | 158.4 | 19.8 | 3.1× |

**Batch 2 is nearly free**: 1.9× the throughput for 1.04× the wall time. That is the signature of a bandwidth-bound workload — generating one token requires reading all 475 MiB of weights, and reading them to serve two sequences costs essentially the same as reading them for one.

Then it stops being free. By batch 4 the marginal gain has collapsed (2.2× instead of 4×), and per-sequence latency has dropped from 50.6 to 27.8 tokens/s. With only 4 cores, this machine crosses from bandwidth-bound to compute-bound almost immediately.

**This is exactly why batching is a GPU technique.** A GPU has enough arithmetic units that the free-batching regime extends to batch 32, 64, or beyond, instead of ending at 2. The shape of this curve is the argument for continuous batching in real serving engines — and reproducing it on a GPU is the next experiment (see Open questions).

---

## 3. The memory bill

GPT-2's KV-cache costs `2 × n_layer × n_embd × dtype_bytes` per token = **72 KiB per token per sequence** in fp32.

| Context | Batch | KV-cache | Weights | KV / weights |
|---|---|---|---|---|
| 512 | 1 | 36 MiB | 475 MiB | 0.08× |
| 512 | 8 | 288 MiB | 475 MiB | 0.61× |
| 1024 | 1 | 72 MiB | 475 MiB | 0.15× |
| 1024 | 4 | 288 MiB | 475 MiB | 0.61× |
| 1024 | 8 | **576 MiB** | 475 MiB | **1.21×** |

At full context and batch 8, **the cache is larger than the model itself.**

This is the point where the two halves of the project meet. The cache is what makes decoding fast, and it is also what stops you from serving more sequences. Weights are a fixed cost paid once; KV grows with *batch × context*, so on any real deployment it — not the model — is what sets the concurrency ceiling.

Everything that looks like exotica in production serving stacks falls out of this one table:

- **Paged attention** — allocate KV in fixed blocks so a sequence that stops early returns its memory, instead of reserving for a worst-case length it never reaches.
- **KV quantisation (INT8/INT4)** — the row above shrinks 2–4×, buying proportionally more concurrency.
- **Multi-query / grouped-query attention** — later architectures share K/V across heads specifically to shrink this term.

None of these are optimisations of the model. They are all optimisations of this table.

---

## Open questions

1. **Does the batch curve stay flat longer on a GPU?** The prediction is that free-batching extends well past batch 2. Worth measuring the point where it breaks.
2. **Where is the OOM boundary?** On a fixed-VRAM device, the memory table above predicts a specific batch size that fails. Does the prediction hold?
3. **What does INT8 KV cost in quality?** Halving cache memory should be measurable against perplexity on a held-out set.
4. **Does the T^1.6 exponent move toward 2 on a GPU?** If the compute/bandwidth gap explanation is right, it should.

---

## Reproducing

```bash
pytest                                              # correctness gates first
python -m nanoserve.bench all --device cpu --out results/laptop.md
./scripts/remote.sh bench all --device cuda         # same code, GPU box
```
