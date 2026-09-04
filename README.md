# nanoserve

A GPT-2 inference engine written from scratch, built to measure one thing precisely: **what a KV-cache actually buys you, and what it costs.**

No `transformers.generate()`. The model definition, the attention maths, the cache and the decode loop are all local code — because the interesting engineering in LLM serving lives in exactly the parts a library hides.

```bash
pip install -r requirements.txt
pytest                                          # prove it matches real GPT-2
python -m nanoserve.bench kvcache --device cpu  # measure the speedup
```

## The question

Generating a token requires attending over every previous token. The obvious implementation re-runs the whole prefix each step, so producing *T* tokens costs **O(T²)**. Caching each layer's keys and values makes a step depend only on the new token: **O(T)**.

That is the textbook claim. This repo measures it, then asks the follow-up question that matters more in production: *if the cache is free speed, why can't you serve unlimited sequences?* The answer is memory, and the arithmetic is unforgiving.

## What's here

| Module | Role |
|---|---|
| `nanoserve/model.py` | GPT-2 — attention, MLP, blocks — written out in full |
| `nanoserve/cache.py` | Pre-allocated per-layer K/V store, with explicit memory accounting |
| `nanoserve/generate.py` | Two decode loops: `generate_naive` (no cache) and `generate_cached` |
| `nanoserve/weights.py` | Ports pretrained GPT-2 weights, shape-checking every tensor |
| `nanoserve/bench.py` | Benchmark harness; emits markdown straight into `RESULTS.md` |
| `tests/test_parity.py` | Correctness gates (see below) |

## Correctness first

A fast decoder that produces different tokens is not an optimisation, so the benchmarks are gated behind four tests:

1. Local logits match HuggingFace GPT-2 to `< 1e-3`.
2. Cached decode matches naive decode.
3. Feeding tokens one at a time matches a single full-sequence pass.
4. Overflowing the reserved cache raises, rather than silently corrupting.

```
$ pytest -q
4 passed
```

Two details cost real debugging time and are worth flagging for anyone reading the code:

- **HuggingFace stores GPT-2's projections as `Conv1D`**, whose weight is the transpose of what `nn.Linear` expects. Get it wrong and the model still runs, still emits fluent-looking text, and is simply wrong. `weights.py` decides from the tensor shape rather than trusting a convention.
- **A cached prefill needs an offset causal mask.** During decode (`q_len == 1`) no mask is needed at all, since every cached key precedes the query. But prefilling *into a non-empty cache* puts the causal boundary at an offset, where `is_causal=True` masks the wrong cells.

## Results

Measured numbers, the memory arithmetic, and the analysis live in **[RESULTS.md](RESULTS.md)**.

Headline from a 4-core laptop CPU (Intel i5-1035G1, no GPU):

| New tokens | Naive | +KV-cache | Speedup |
|---|---|---|---|
| 32 | 2.05 s | 0.64 s | 3.2× |
| 64 | 6.19 s | 1.31 s | 4.7× |
| 128 | 18.69 s | 2.50 s | 7.5× |
| 256 | 57.24 s | 5.11 s | **11.2×** |

The speedup **grows with sequence length** — which is the real signature of the quadratic-to-linear change. A single fixed number would not tell you that.

And at full context with batch 8, the cache costs **576 MiB against 475 MiB of weights** — it is bigger than the model. That tension is what [RESULTS.md](RESULTS.md) is actually about.

## Running on a GPU

Code stays local and authoritative; a remote box is treated as a disposable executor, so no commit is needed to run an experiment.

```bash
cp scripts/remote.env.example scripts/remote.env   # fill in host details (gitignored)
./scripts/remote.sh setup                          # remote venv + deps
./scripts/remote.sh bench all --device cuda        # sync, run, fetch results/
```

## Roadmap

- [x] GPT-2 from scratch, parity-tested against the reference
- [x] Pre-allocated KV-cache, naive vs cached benchmark
- [x] Batch-scaling and memory-footprint experiments
- [ ] GPU numbers: batch saturation curve, VRAM-limited OOM boundary
- [ ] INT8 KV-cache quantisation — memory saved vs quality lost
- [ ] Continuous batching: admit new sequences mid-flight instead of padding to the longest
- [ ] Streaming HTTP endpoint

## License

MIT
