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

## 6. CUDA graphs: cashing in the launch overhead

§1 attributed ~80% of per-token decode time to launch and dispatch rather than arithmetic. That is a falsifiable claim, and a CUDA graph tests it: capture the ~120 kernels of a decode step once, replay them with a single launch.

A5000, batch 1, greedy. Short lengths at `--repeat 8`, long at `--repeat 3`:

| New tokens | Eager ms/token | Graphed ms/token | Speedup |
|---|---|---|---|
| 64 | 3.26 | 1.43 | **2.29×** |
| 128 | 3.24 | 1.41 | **2.31×** |
| 256 | 3.22 | 1.51 | 2.14× |
| 512 | 3.22 | 1.77 | 1.82× |
| 1000 | 3.17 | 2.33 | 1.36× |

**The attribution was broadly right.** Per-token cost falls from 3.2 ms to ~1.4 — from 5.3× the 0.6 ms bandwidth floor down to 2.3×. Graphs removed roughly 60% of the overhead; what remains is replay cost, sampling, the Python loop, and mask construction, none of which the graph subsumes.

Note this also **flips the §1 result**: at 64 tokens the KV-cache was a net loss against naive decode. Graphed, cached decode runs at 1.43 ms/token, and the cache wins at every length again. The cache was never the problem — the dispatch around it was.

### Graphs are not free: capture rigidity costs attention

Capture requires constant shapes, so the static path attends over the whole reserved window rather than slicing to the live position. A run reserving 1010 slots pays full-window attention from its first token. The first implementation showed exactly that — the launch saving is constant, the wasted attention grows:

| New tokens | Single full-window graph | Bucketed graphs |
|---|---|---|
| 128 | 2.15× | **2.31×** |
| 256 | 1.88× | **2.14×** |
| 512 | 1.51× | **1.82×** |
| 1000 | 1.11× | **1.36×** |

The fix is to capture several graphs at increasing windows (128 / 256 / 512 / max) and replay the smallest that covers the current position. Each stays fixed-shape — all capture requires — and choosing between them is an ordinary Python branch outside any graph. That recovered most of the loss, and the residual decline at 1000 tokens is simply bucket coarseness: finer buckets would recover more, at the cost of capture time and memory.

**A trap worth naming:** the obvious `step()` reads the cache position from the device to pick a bucket. That forces a device-to-host sync every token — stalling the pipeline for precisely the overhead graphs exist to remove. The position advances by exactly one per step and by nothing else, so it is tracked host-side and reconciled once per generation instead.

## 7. INT8 KV cache: concurrency bought with latency

§4 established the KV-cache as the term that caps concurrency. Storing it in int8 attacks that term directly. Scheme is symmetric, per-token and per-head: each token's K (and V) vector across one head gets a scale computed when it is written and never revisited — which matches how a cache is used, since positions arrive one at a time and are then immutable.

A5000, 256 tokens:

| KV store | Batch | Stored | Peak VRAM | ms/token | Agreement with fp32 |
|---|---|---|---|---|---|
| fp32 | 1 | 19 MiB | 504 MiB | 3.24 | reference |
| int8 | 1 | **5 MiB** | 493 MiB | 6.00 | **256/256** |
| fp32 | 64 | 1,197 MiB | 1,709 MiB | 5.20 | — |
| int8 | 64 | **318 MiB** | **966 MiB** | 9.44 | **256/256** |
| fp32 | 256 | 4,788 MiB | 5,386 MiB | 17.53 | — |
| int8 | 256 | **1,272 MiB** | **2,410 MiB** | 31.70 | **256/256** |

**Accuracy is a non-issue.** Every configuration reproduced fp32's tokens exactly — identical text, no divergence anywhere. Quantisation error lands around 1/127 of each vector's own dynamic range, and the gaps between competing tokens are wider than that. This is the cheap half of the trade.

**Storage falls 3.76×, but peak VRAM only 2.23×.** The difference is the honest cost of dequantise-on-read: attention still runs in floating point, so the layer being processed materialises a full fp copy of its window. The memory model quantifies it — the per-sequence term rose from **0.394 MiB (fp32) to 2.549 MiB (int8)**, and `2 × 12 layers × 266 × 64 × 4 B = 1.63 MiB` accounts for the increase. Roughly a third of the saving is handed back.

**It is 1.8× slower.** Quantise on write, dequantise on read, every layer, every step — arithmetic the fp path never does. Keeping the matmul in int8 needs kernels PyTorch does not expose here; that is what production engines use int8 attention for.

### The ceiling, predicted then measured

Extending §4's model to int8 gives `485 + 2.549 × batch + stored(batch)`, and a predicted ceiling of **3,061 sequences**.

| Batch | Predicted peak | Measured peak | Error |
|---|---|---|---|
| 1024 | 8,182 MiB | 8,186 MiB | 0.05% |
| 2048 | 15,879 MiB | 15,884 MiB | 0.03% |
| 2816 | 21,653 MiB | 21,625 MiB | 0.13% |
| 3008 | 23,096 MiB | 23,068 MiB | 0.12% |
| 3072 | 23,564 MiB | **OOM** | — |

Measured: 3,008 runs, 3,072 does not. **The predicted 3,061 sits inside that 64-wide bracket.** Concurrency rises from ~1,205 (fp32) to ~3,061 — **2.54×**.

So the summary is conditional, which is the point: **int8 KV buys concurrency and costs latency.** Right for a throughput-bound server, wrong for a latency-bound single stream. A benchmark that reported only the 3.76× storage figure would be describing a third of the picture.

*(A methodology note worth keeping: an early run of this used `--tokens 64` rather than 256, which shrank `max_seq` from 266 to 74 and let everything fit. The ceiling is a function of batch × context, not batch alone.)*

## 8. Continuous batching: it depends on there being a queue

Every batch figure above was measured the easy way — N identical prompts, lockstep, wait for all. Real traffic varies in length, so a lockstep batch idles: once the shortest sequence finishes, its slot keeps stepping until the longest drains.

Workload: 128 requests, 12,264 tokens asked, output lengths drawn from 16–256. A5000, fp32.

| Slots | Static tok/s | Cont. tok/s | Throughput | Static work | Cont. work | Static lat mean/p95 | Cont. lat mean/p95 |
|---|---|---|---|---|---|---|---|
| 8 | 975 | **1,623** | **1.66×** | 2.36× | **1.05×** | 7.23 / 12.57 s | **3.92 / 6.78 s** |
| 32 | 3,231 | **4,114** | **1.27×** | 2.67× | **1.37×** | 2.34 / 3.76 s | **1.40 / 2.43 s** |
| 64 | 4,294 | **4,556** | 1.06× | 2.67× | 1.61× | 2.08 / 2.80 s | **1.18 / 2.17 s** |
| 128 | **4,766** | 3,647 | **0.77×** | 2.67× | 2.55× | 2.51 / 2.51 s | **1.43** / 3.30 s |

"Work" is tokens *computed* ÷ tokens *asked for* — the mechanism behind everything else.

**Continuous batching wins when there is a queue, and only then.** At 8 slots for 128 requests, static computes 2.36× the tokens requested while continuous computes 1.05×, and throughput follows: 1.66×. As the pool grows the advantage decays, and at 128 slots for a 128-request burst it **reverses** — 0.77×.

That reversal is not a bug. With a pool as large as the burst, every request is admitted immediately, nothing ever waits, and no slot is ever refilled. The work ratios confirm it (2.55× vs 2.67× — essentially identical), so the scheduler's machinery is pure overhead against a lockstep batch that can use the cheaper sliced-attention path. Production engines run with slots well below peak demand, which is exactly the regime where this pays.

**Latency tells the other half, and it is unambiguous.** Continuous batching improves *mean* latency at every configuration — 1.67× to 1.84× — including the one where it loses on throughput. A short request no longer waits for whatever long sequence it happened to be batched with.

The 128-slot row is the most interesting: static gives every request the same 2.51 s, because they all finish together. Continuous gives a mean of 1.43 s but a p95 of 3.30 s. Static batching does not make requests fast, it makes them *uniformly slow* — which flatters p95 while being worse for almost every individual caller.

### Three traps between the naive scheduler and this one

The first working version measured 1.47× at 8 slots and 0.52× at 128. Getting to 1.66× / 0.77× meant removing three costs, all of which had nothing to do with the model:

1. **A host sync per slot per step.** Reading each slot's token back with `.item()` as it was produced meant 128 device-to-host syncs per step at 128 slots. Batching the sampler into one call and collecting each request's tokens once, at eviction, fixed it. (→ 1.61× / 0.57×)
2. **Full-window attention** — the same trap as §6. The static decode path spans `cache.window` regardless of how far any sequence has got, so a pool reserved for 266 tokens paid full-window attention from step one. Every position is known host-side, so the window widens in buckets as sequences actually grow. (→ 1.78× / 0.75×)
3. **Idle slots still stepping.** A slot with no live request still costs a full column of attention and MLP. Allocating from a heap keeps live requests packed into a contiguous prefix, so the step can slice to just that prefix. (→ 1.74× / 0.77×, and 64 slots crossed from 0.99× to 1.07×)

Trap 3 surfaced a bug worth recording: slicing the *input* to the active prefix while leaving `pos_dev` at full width made `wpe(pos_dev)` silently broadcast the batch back up to every slot. It failed loudly here only because the shapes happened to collide; with a different slot count it would have produced plausible tokens from the wrong positions.

## 9. Two measurement traps

Both were caught before publishing, and both would have produced a confidently wrong claim.

**Allocation inside the timed region.** `generate_cached` originally allocated and zeroed the cache on every call — tens of MiB of memset charged to the cached path and never to naive. At short lengths this was large enough to invert the comparison. Fixed by hoisting allocation out; a server allocates per slot, not per request.

**Allocator reservations faking an OOM.** The sweep reported OOM at batch 1024, and the memory model said it should fit in 20 GiB of 24. Investigating, batch 1024 ran fine *standalone* — the allocator was still holding batch 512's 9.5 GiB reservation, so the sweep was asking for 28.7 GiB on a 24 GiB card. **The OOM was in the harness, not the model.** With `empty_cache()` between steps, batch 1024 completes at 20,034 MiB. Published as-is, this would have claimed a hardware limit ~2× too pessimistic.

The general lesson: when a measurement contradicts a model that has been accurate elsewhere, suspect the measurement.

## Open questions

1. **How much further can bucketing go?** §6 leaves 2.33 ms/token at 1000 tokens against 1.41 at 128, purely from bucket coarseness. Finer buckets trade capture time and memory for it; the curve of that trade is unmeasured.
2. **Would int8 attention kernels remove §7's latency cost?** The 1.8× slowdown is entirely dequantise-on-read. Kernels that keep the matmul in int8 should erase it and shrink the transient that ate a third of the memory saving.
3. **Does the exponent reach 2.0 on a larger model?** The A5000 hit 1.82 at 1000 tokens with a 124M model. A model that saturates the GPU sooner should get closer.
4. **Staggered arrivals.** §8 admits every request at t=0, which isolates the length-variance effect but understates the case: real arrivals are spread over time, so a large pool is rarely full and admission latency matters more than it does here.
5. **Wire the scheduler into the HTTP endpoint.** The server still serialises requests behind a semaphore; §8's scheduler is what it needs to serve concurrently.

## Reproducing

```bash
pytest                                                    # correctness gates first
python -m nanoserve.bench all --device cpu --out results/laptop.md
./scripts/remote.sh bench all --device cuda               # same code, GPU box
./scripts/remote.sh bench graph --device cuda --repeat 8  # graphs vs eager
./scripts/remote.sh bench dtype --device cuda --dtypes float32 float16 bfloat16
./scripts/remote.sh bench kvquant --device cuda            # fp vs int8 KV
./scripts/remote.sh bench serving --device cuda            # static vs continuous
```

Serving:

```bash
uvicorn nanoserve.server:app --port 8000
python scripts/make_demo.py                                # regenerate docs/demo.gif
```
