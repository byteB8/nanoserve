# Results

Reproducible with `python -m nanoserve.bench all`. Raw harness output goes to `results/` (gitignored — regenerate rather than trust a stale file).

All timings are best-of-3 after 2 warmup passes. Best-of rather than mean: the interest is the machine's capability, not its worst scheduling luck. CUDA timings synchronise before stopping the clock, or they would measure kernel launches rather than kernel work.

| | **laptop** | **bhaskar** |
|---|---|---|
| Device | Intel i5-1035G1, 4 cores / 8 threads | NVIDIA RTX A5000, 24 GiB |
| Config | 15 GiB RAM, 4 torch threads | 32-core host, 251 GiB RAM |
| Stack | PyTorch 2.9.0, Python 3.12.3 | PyTorch 2.6.0+cu124, Python 3.10.12 |

Model is GPT-2 124M throughout. fp32 unless stated.

---

## 1. What the KV-cache buys — and where it loses

**Laptop (fp32)**

| New tokens | Naive (s) | +KV-cache (s) | Speedup | ms/token cached |
|---|---|---|---|---|
| 32 | 1.69 | 0.61 | 2.8× | 19.1 |
| 64 | 5.47 | 1.31 | 4.2× | 20.5 |
| 128 | 14.22 | 2.57 | 5.5× | 20.1 |
| 256 | 43.12 | 5.28 | **8.2×** | 20.6 |

**A5000 (fp32)**

| New tokens | Naive (s) | +KV-cache (s) | Speedup | ms/token cached |
|---|---|---|---|---|
| 64 | 0.17 | 0.21 | **0.8×** | 3.28 |
| 128 | 0.38 | 0.41 | 0.9× | 3.20 |
| 256 | 1.01 | 0.82 | 1.2× | 3.20 |
| 512 | 3.08 | 1.63 | 1.9× | 3.18 |
| 1000 | 10.45 | 3.19 | **3.3×** | 3.19 |

Two things to take from this, and the second is the interesting one.

**The cache makes decode O(1) per step.** Cached ms/token is flat — 3.18 to 3.28 across a 16× range in sequence length on the GPU, ~20 on the laptop. That flatness *is* the property, measured directly. It is more informative than any single speedup figure, because a speedup number depends on which length you happened to pick.

**On fast hardware at short context, the cache is a net loss.** At 64 tokens on the A5000 it runs at 0.8× — slower than recomputing everything. This is not a measurement artifact; hoisting cache allocation out of the timed region did not change it.

The reason is that decode on this model is not compute-bound, it is *launch*-bound. The floor set by memory bandwidth is roughly 475 MiB of weights ÷ 768 GB/s ≈ 0.6 ms/token, but measured decode is 3.2 ms/token. The missing ~80% is Python and kernel-launch overhead across 12 layers. Both paths run the same number of sequential steps and issue the same number of launches; naive simply does more arithmetic per launch, and on an otherwise idle A5000 that arithmetic is free. The cache only pays once the quadratic term grows enough to matter.

This is the argument for CUDA graphs, kernel fusion, and larger batches — everything that attacks per-step overhead rather than per-step FLOPs.

## 2. Naive decoding is not quite quadratic

Theory says recomputing the prefix every step is O(T²). Neither machine measures that.

Per-doubling scaling factor of the naive path (2.0 would mean linear, 4.0 quadratic):

| | 32→64 | 64→128 | 128→256 | 256→512 | 512→1000 |
|---|---|---|---|---|---|
| Laptop | 3.24 | 2.60 | 3.03 | — | — |
| A5000 | — | 2.24 | 2.66 | 3.05 | 3.39 |

Fitted exponents: laptop ≈ **T^1.56** overall; A5000 rises from **T^1.16** to **T^1.82** as length grows.

The cause is a second effect cancelling the first. Naive step *t* processes a *t*-token block, and larger matmuls run at far higher hardware efficiency. Measured on the laptop CPU with a GPT-2-shaped matmul:

```
T=  1    0.25 ms  →   18.7 GFLOP/s   (single token: memory-bound)
T=256    4.21 ms  →  286.7 GFLOP/s   (a block: compute-bound)
```

A 15× efficiency spread. Naive does quadratically more arithmetic but does it at a rising efficiency, so the observed exponent lands between 1 and 2.

The GPU behaviour confirms the mechanism rather than just repeating it: as sequences lengthen and the GPU saturates, efficiency stops improving and the exponent climbs toward 2 (1.16 → 1.82). The CPU, already near its efficiency ceiling, stays flat around 1.56. **This was predicted from the CPU data before the GPU runs, and held.**

## 3. Batch scaling: where "free" ends

**Laptop (fp32, 64 tokens)**

| Batch | Wall (s) | Total tok/s | Per-seq tok/s | vs batch 1 |
|---|---|---|---|---|
| 1 | 1.30 | 49.3 | 49.3 | 1.0× |
| 2 | 1.35 | 95.0 | 47.5 | 1.9× |
| 4 | 2.42 | 105.6 | 26.4 | 2.1× |
| 8 | 3.21 | 159.7 | 20.0 | 3.2× |
| 16 | 5.46 | 187.7 | 11.7 | 3.8× |

**A5000 (fp32, 256 tokens)**

| Batch | Wall (s) | Total tok/s | Per-seq tok/s | vs batch 1 | KV (MiB) | Peak (MiB) |
|---|---|---|---|---|---|---|
| 1 | 0.81 | 314.8 | 314.8 | 1.0× | 19 | 504 |
| 8 | 0.84 | 2,441 | 305.2 | 7.8× | 150 | 638 |
| 32 | 0.89 | 9,173 | 286.7 | **29.1×** | 598 | 1,095 |
| 64 | 1.35 | 12,128 | 189.5 | 38.5× | 1,197 | 1,709 |
| 128 | 2.45 | 13,366 | 104.4 | 42.5× | 2,394 | 2,934 |
| 256 | 4.57 | 14,331 | 56.0 | 45.5× | 4,788 | 5,385 |
| 512 | 8.89 | 14,749 | 28.8 | 46.9× | 9,576 | 10,283 |
| 1024 | 17.66 | 14,840 | 14.5 | 47.1× | 19,152 | 20,034 |
| 1152 | 19.99 | 14,755 | 12.8 | 46.9× | 21,546 | 22,484 |

**Batching is free until it isn't, and where that happens is a property of the hardware.** On the A5000, batch 1 → 32 costs 10% more wall time and returns **29× the throughput**. On a 4-core laptop the same regime ends at batch 2.

That is the entire case for continuous batching, measured on both sides: decode reads the full weight matrix to produce one token, so amortising that read across many sequences is nearly free until the arithmetic units saturate. Past the knee, per-sequence latency degrades linearly while aggregate throughput flattens at ~14,800 tok/s — the classic latency-for-throughput trade a scheduler has to arbitrate.

## 4. A predictive memory model

Fitting measured peak VRAM as `fixed + per_seq × batch + KV(batch)`:

| Precision | Fixed | Per-sequence | Weights (actual) |
|---|---|---|---|
| fp32 | 484 MiB | 0.394 MiB | 475 MiB |
| fp16 | 253 MiB | 0.197 MiB | 237 MiB |

The fixed term recovers the weights to within ~2% (the remainder is CUDA context), and the per-sequence term halves exactly with precision. The model predicts the concurrency ceiling as `(free − fixed) / (per_seq + KV_per_seq)`:

| Precision | Predicted ceiling | Measured |
|---|---|---|
| fp32 | 1,205 | 1,152 OK |
| fp16 | 2,434 | **2,304 OK, 2,560 OOM** |

Predicted 2,434 lands inside the measured bracket. Spot-checking peaks: batch 1024 fp32 predicted 20,085 MiB against 20,034 measured (**0.25%**); batch 2304 fp16 predicted 22,237 against 22,253 (**0.07%**).

### The model found a bug

The first fit gave a per-sequence term of **2.16 MiB**, not 0.394. Backing that out: the prefill was computing logits for all 10 prompt positions — `10 × 50257 vocab × 4 B = 2.01 MiB per sequence`, 93% of the anomaly — when generation reads only the last. At batch 1024 that is **2 GiB spent on values immediately discarded**.

Adding `last_only` to skip the projection on unread positions saved a measured 1.73 MiB/sequence against 1.81 predicted, and dropped the per-sequence term to 0.394.

The fix is not bit-identical, which is worth knowing rather than hiding: slicing before the projection hands the GEMM a `[1, 768]` input instead of `[10, 768]`, selecting a different kernel with a different reduction order. Floating-point addition is not associative, so results differ by ~3e-7 relative — fp32 epsilon. The top-5 ranking is unchanged, and the test asserts exactly that rather than exact equality.

### Why this table is the whole game

At full context and batch 8, the cache already exceeds the weights (576 MiB vs 475). At batch 1024 it is **19 GiB of a 20 GiB peak**. Weights are a fixed cost paid once; KV grows with *batch × context*, so it — not the model — sets the concurrency ceiling.

Everything exotic-looking in a production serving stack falls out of this:

- **Paged attention** — allocate KV in blocks so a sequence that ends early returns memory instead of reserving a worst-case length.
- **KV quantisation** — shrink the dominant term 2–4×, buy proportional concurrency.
- **Multi-query / grouped-query attention** — later architectures share K/V across heads specifically to shrink this term.

None optimise the model. All optimise this table.

## 5. Precision: half the memory, none of the speed

A5000, batch 1, 256 tokens:

| dtype | ms/token | Peak (MiB) | Agrees with fp32 | First divergence |
|---|---|---|---|---|
| float32 | 3.24 | 504 | reference | — |
| float16 | 3.43 | 265 | **256/256** | none |
| bfloat16 | 3.38 | 265 | 208/256 | **token 12** |

**Half precision does not speed up single-stream decode — it is marginally slower.** Consistent with §1: decode at batch 1 is launch-bound, so halving arithmetic width buys nothing, while the extra casts cost a little.

But at batch, where the workload *is* bandwidth-bound, the picture inverts completely — fp16 peaks at **48,392 tok/s against fp32's 14,840, a 3.3× throughput gain**. Whether half precision helps depends entirely on which regime you are in, and quoting one number without the batch size is meaningless.

**fp16 beats bf16 for this workload.** Identical memory, but fp16 reproduced all 256 fp32 tokens exactly while bf16 diverged at token 12. bf16 spends 8 bits on mantissa to fp16's 10, trading precision for exponent range. GPT-2's inference activations do not need that range, so the extra mantissa bits win. The common "prefer bf16" guidance comes from *training*, where gradient range genuinely matters — carrying it into inference unexamined costs accuracy for nothing.

## 6. Two measurement traps

Both were caught before publishing, and both would have produced a confidently wrong claim.

**Allocation inside the timed region.** `generate_cached` originally allocated and zeroed the cache on every call — tens of MiB of memset charged to the cached path and never to naive. At short lengths this was large enough to invert the comparison. Fixed by hoisting allocation out; a server allocates per slot, not per request.

**Allocator reservations faking an OOM.** The sweep reported OOM at batch 1024, and the memory model said it should fit in 20 GiB of 24. Investigating, batch 1024 ran fine *standalone* — the allocator was still holding batch 512's 9.5 GiB reservation, so the sweep was asking for 28.7 GiB on a 24 GiB card. **The OOM was in the harness, not the model.** With `empty_cache()` between steps, batch 1024 completes at 20,034 MiB. Published as-is, this would have claimed a hardware limit ~2× too pessimistic.

The general lesson: when a measurement contradicts a model that has been accurate elsewhere, suspect the measurement.

## Open questions

1. **Do CUDA graphs close the launch-overhead gap?** §1 attributes ~80% of decode to launch overhead. Capturing the decode step should move measured ms/token toward the 0.6 ms bandwidth floor, and would make the cache win at short context too.
2. **Where does INT8 KV quantisation land on the accuracy axis?** §4 predicts another 2× concurrency; §5 gives the method for measuring what it costs — token agreement against an fp32 reference.
3. **Does the exponent reach 2.0 on a larger model?** The A5000 hit 1.82 at 1000 tokens with a 124M model. A model that saturates the GPU sooner should get closer.
4. **Continuous batching.** Every batch number here pads to a fixed length. Admitting sequences mid-flight is the difference between this and a real serving engine.

## Reproducing

```bash
pytest                                                    # correctness gates first
python -m nanoserve.bench all --device cpu --out results/laptop.md
./scripts/remote.sh bench all --device cuda               # same code, GPU box
./scripts/remote.sh bench dtype --device cuda --dtypes float32 float16 bfloat16
```
