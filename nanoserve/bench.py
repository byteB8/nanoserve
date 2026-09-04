"""Benchmark harness. Emits markdown so results land straight in RESULTS.md.

Four experiments:

  kvcache  -- naive vs cached decode across generation lengths. The speedup should
              grow roughly linearly with length, because naive is O(T^2) against
              the cache's O(T).
  batch    -- cached decode throughput against batch size. Decode is bandwidth-
              bound, so serving many sequences at once costs barely more wall-clock
              per step than serving one, until compute saturates.
  dtype    -- half precision against fp32: memory and speed won, tokens changed.
  memory   -- reserved KV bytes against batch and context, next to the weights.
              Shows which of the two actually limits how much you can serve.

Usage:
    python -m nanoserve.bench kvcache --device cpu
    python -m nanoserve.bench batch   --device cuda --model gpt2
    python -m nanoserve.bench dtype   --device cuda --dtypes float32 float16
    python -m nanoserve.bench memory  --model gpt2
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass

import torch

from .config import PRESETS
from .generate import generate_cached, generate_graphed, generate_naive
from .weights import load_pretrained, load_tokenizer

PROMPT = "The key insight about autoregressive decoding is that"


# -- plumbing -------------------------------------------------------------


def _sync(device: str) -> None:
    """CUDA kernels are async; without this we would time the launch, not the work."""
    if device == "cuda":
        torch.cuda.synchronize()


def _time(fn, device: str, warmup: int = 2, repeat: int = 3) -> float:
    """Best-of-`repeat` wall-clock seconds. Best-of, not mean: we want the machine's
    capability, not its worst scheduling luck."""
    for _ in range(warmup):
        fn()
    _sync(device)
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        best = min(best, time.perf_counter() - t0)
    return best


def _describe(device: str) -> dict:
    info = {
        "device": device,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "threads": torch.get_num_threads(),
    }
    if device == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    else:
        info["cpu"] = platform.processor() or platform.machine()
    return info


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


# -- experiments ----------------------------------------------------------


def run_kvcache(model, prompt_ids, device: str, lengths: list[int], repeat: int, kv_dtype: str = "auto") -> str:
    rows = []
    prompt_len = prompt_ids.size(1)
    budget = model.cfg.block_size - prompt_len
    usable = [n for n in lengths if n <= budget]
    for n in sorted(set(lengths) - set(usable)):
        print(f"  skipping {n}: prompt({prompt_len}) + {n} exceeds the {model.cfg.block_size}-token window")

    for n in usable:
        # Hoist cache allocation out of the timed region. Otherwise the cached
        # path is charged a multi-MiB memset that the naive path never pays,
        # which at short lengths is large enough to invert the comparison.
        cache = model.new_cache(batch_size=prompt_ids.size(0), max_seq=prompt_len + n, kv_dtype=kv_dtype)
        naive = _time(lambda: generate_naive(model, prompt_ids, n), device, repeat=repeat)
        cached = _time(lambda: generate_cached(model, prompt_ids, n, cache=cache), device, repeat=repeat)
        rows.append(
            [
                str(n),
                f"{naive:.2f}",
                f"{cached:.2f}",
                f"**{naive / cached:.1f}x**",
                f"{n / cached:.1f}",
                f"{cached / n * 1000:.1f}",
            ]
        )
        print(f"  {n:5d} tokens: naive {naive:7.2f}s  cached {cached:6.2f}s  -> {naive/cached:5.1f}x")
    return _table(
        ["New tokens", "Naive (s)", "+KV-cache (s)", "Speedup", "Tokens/s (cached)", "ms/token"],
        rows,
    )


def run_batch(model, prompt_ids, device: str, batches: list[int], n_tokens: int, repeat: int, kv_dtype: str = "auto") -> str:
    rows = []
    single = None
    for b in batches:
        batched = prompt_ids.repeat(b, 1)
        try:
            # Allocated outside the timed region, as in run_kvcache: cache setup
            # scales with batch, so timing it here would confound the throughput
            # curve with allocator behaviour.
            cache = model.new_cache(batch_size=b, max_seq=prompt_ids.size(1) + n_tokens, kv_dtype=kv_dtype)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            secs = _time(lambda: generate_cached(model, batched, n_tokens, cache=cache), device, repeat=repeat)
            peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
            # What the cache alone accounts for -- the gap against measured peak
            # is activations, and it is not small.
            kv_mib = cache.nbytes() / 2**20
        except (torch.cuda.OutOfMemoryError if device == "cuda" else RuntimeError) as exc:
            print(f"  batch {b:4d}: out of memory -- {type(exc).__name__}")
            rows.append([str(b), "**OOM**", "-", "-", "-", "-", "-"])
            break
        total_tps = b * n_tokens / secs
        if single is None:
            single = total_tps
        rows.append(
            [
                str(b),
                f"{secs:.2f}",
                f"{total_tps:.1f}",
                f"{n_tokens / secs:.1f}",
                f"{total_tps / single:.1f}x",
                f"{kv_mib:.0f}",
                f"{peak:.0f}" if peak else "-",
            ]
        )
        print(
            f"  batch {b:4d}: {secs:6.2f}s  total {total_tps:8.1f} tok/s  "
            f"per-seq {n_tokens/secs:6.1f} tok/s  kv {kv_mib:6.0f} MiB  peak {peak:7.0f} MiB"
        )
        # Release this step's cache before sizing the next one. Without it the
        # allocator still holds the previous (smaller) reservation when the next
        # allocation is attempted, and the sweep reports an OOM that the model
        # would not hit on its own -- a measurement artifact, not a real limit.
        del cache
        if device == "cuda":
            torch.cuda.empty_cache()
    return _table(
        [
            "Batch",
            "Wall (s)",
            "Total tokens/s",
            "Per-sequence tokens/s",
            "vs batch 1",
            "KV-cache (MiB)",
            "Peak VRAM (MiB)",
        ],
        rows,
    )


def run_graph(model, prompt_ids, device: str, lengths: list[int], repeat: int) -> str:
    """Eager decode against a CUDA-graph replay of the same step.

    §1 attributed ~80% of per-token decode time to launch and dispatch overhead.
    A graph collapses the whole step into one launch, so this measures how much
    of that attribution was right.
    """
    from .graph import GraphedDecoder

    prompt_len = prompt_ids.size(1)
    rows = []
    for n in [x for x in lengths if x <= model.cfg.block_size - prompt_len]:
        cache = model.new_cache(batch_size=prompt_ids.size(0), max_seq=prompt_len + n)

        eager = _time(lambda: generate_cached(model, prompt_ids, n, cache=cache), device, repeat=repeat)

        # Capture once, outside the timed region: a server captures per slot at
        # startup, not per request.
        cache.reset()
        model(prompt_ids, cache, last_only=True)
        decoder = GraphedDecoder(model, cache, batch_size=prompt_ids.size(0))
        graphed = _time(
            lambda: generate_graphed(model, prompt_ids, n, cache=cache, decoder=decoder),
            device,
            repeat=repeat,
        )

        rows.append(
            [
                str(n),
                f"{eager:.3f}",
                f"{graphed:.3f}",
                f"**{eager / graphed:.2f}x**",
                f"{eager / n * 1000:.2f}",
                f"{graphed / n * 1000:.2f}",
            ]
        )
        print(
            f"  {n:5d} tokens: eager {eager:6.3f}s ({eager/n*1000:5.2f} ms/tok)  "
            f"graphed {graphed:6.3f}s ({graphed/n*1000:5.2f} ms/tok)  -> {eager/graphed:4.2f}x"
        )
        del cache, decoder
        if device == "cuda":
            torch.cuda.empty_cache()

    return _table(
        ["New tokens", "Eager (s)", "Graphed (s)", "Speedup", "Eager ms/token", "Graphed ms/token"],
        rows,
    )


def run_serving(model, prompt_ids, device: str, slots_list: list[int], n_requests: int, seed: int = 0) -> str:
    """Static batching against continuous batching on the same workload.

    Static batching takes a group of requests, runs them in lockstep, and waits
    for the longest before starting the next group. When output lengths vary --
    which they always do -- every slot whose sequence finished early keeps
    stepping until the batch drains. The wasted work is the gap between tokens
    *asked for* and tokens *computed*.

    Continuous batching returns a slot the moment its request finishes and admits
    the next one immediately. Same model, same tokens, same hardware; the only
    difference is who decides when a slot is free.

    Lengths are drawn from a spread deliberately similar to real traffic, where a
    short answer and a long one differ by an order of magnitude.
    """
    import random

    from .scheduler import Request, Scheduler

    rng = random.Random(seed)
    lengths = [rng.choice([16, 24, 32, 48, 64, 96, 128, 192, 256]) for _ in range(n_requests)]
    asked = sum(lengths)
    prompt_len = prompt_ids.size(1)
    max_seq = prompt_len + max(lengths)

    print(f"  workload: {n_requests} requests, {asked} tokens asked, lengths {min(lengths)}-{max(lengths)}")
    rows = []
    for slots in slots_list:
        # -- static: fixed groups, everyone runs to the group's longest ------
        _sync(device)
        t0 = time.perf_counter()
        computed = 0
        for i in range(0, n_requests, slots):
            group = lengths[i : i + slots]
            n = max(group)
            batched = prompt_ids.repeat(len(group), 1)
            cache = model.new_cache(batch_size=len(group), max_seq=prompt_len + n)
            generate_cached(model, batched, n)
            computed += len(group) * n
            del cache
            if device == "cuda":
                torch.cuda.empty_cache()
        _sync(device)
        static_s = time.perf_counter() - t0

        # -- continuous: refill a slot as soon as it frees -------------------
        sched = Scheduler(model, n_slots=slots, max_seq=max_seq)
        for n in lengths:
            sched.submit(Request(prompt=prompt_ids, max_tokens=n))
        _sync(device)
        t0 = time.perf_counter()
        sched.run_to_completion()
        _sync(device)
        cont_s = time.perf_counter() - t0
        cont_computed = sched.steps * slots

        rows.append(
            [
                str(slots),
                f"{static_s:.2f}",
                f"{cont_s:.2f}",
                f"**{static_s / cont_s:.2f}x**",
                f"{asked / static_s:.0f}",
                f"{asked / cont_s:.0f}",
                f"{computed / asked:.2f}x",
                f"{cont_computed / asked:.2f}x",
            ]
        )
        print(
            f"  slots {slots:3d}: static {static_s:6.2f}s ({asked/static_s:7.0f} tok/s, "
            f"{computed/asked:.2f}x work)   continuous {cont_s:6.2f}s "
            f"({asked/cont_s:7.0f} tok/s, {cont_computed/asked:.2f}x work)   -> {static_s/cont_s:.2f}x"
        )
        del sched
        if device == "cuda":
            torch.cuda.empty_cache()

    return _table(
        [
            "Slots",
            "Static (s)",
            "Continuous (s)",
            "Speedup",
            "Static tok/s",
            "Continuous tok/s",
            "Static work",
            "Continuous work",
        ],
        rows,
    )


def run_kvquant(model, prompt_ids, device: str, batches: list[int], n_tokens: int, repeat: int) -> str:
    """fp KV against int8 KV: stored bytes, measured peak, speed, and agreement.

    Storage drops ~3.8x (4x from int8, less one fp scale per token per head).
    Measured peak drops by less, because attention still runs in floating point:
    the layer being processed holds a dequantised copy of its window. That gap
    between stored and peak is the honest cost of dequantise-on-read, and it is
    why the concurrency gain is not the 4x the storage figure suggests.
    """
    prompt_len = prompt_ids.size(1)
    rows = []
    ref = None
    for kv in ("auto", "int8"):
        for b in batches:
            batched = prompt_ids.repeat(b, 1)
            try:
                cache = model.new_cache(batch_size=b, max_seq=prompt_len + n_tokens, kv_dtype=kv)
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                secs = _time(lambda: generate_cached(model, batched, n_tokens, cache=cache), device, repeat=repeat)
                peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
                out = generate_cached(model, batched, n_tokens, cache=cache)[0, prompt_len:]
            except (torch.cuda.OutOfMemoryError if device == "cuda" else RuntimeError):
                print(f"  {kv:5s} batch {b:5d}: OOM")
                rows.append([kv, str(b), "-", "**OOM**", "-", "-"])
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue

            if ref is None:
                ref, agree = out, "reference"
            else:
                same = out == ref
                agree = f"{int(same.sum())}/{n_tokens}"

            rows.append(
                [
                    kv,
                    str(b),
                    f"{cache.nbytes() / 2**20:.0f}",
                    f"{peak:.0f}" if peak else "-",
                    f"{secs / n_tokens * 1000:.2f}",
                    agree,
                ]
            )
            print(
                f"  {kv:5s} batch {b:5d}: stored {cache.nbytes()/2**20:7.0f} MiB  "
                f"peak {peak:8.0f} MiB  {secs/n_tokens*1000:6.2f} ms/tok  agree {agree}"
            )
            del cache
            if device == "cuda":
                torch.cuda.empty_cache()

    return _table(
        ["KV store", "Batch", "Stored (MiB)", "Peak VRAM (MiB)", "ms/token", "Agreement"], rows
    )


def run_dtype(model_name: str, device: str, dtypes: list[str], n_tokens: int, repeat: int) -> str:
    """Precision trade-off: what half precision buys, and what it costs.

    Memory and speed are the easy half. The half that decides whether you can
    ship it is agreement: fp16 rounds every intermediate, errors compound across
    a sequential decode, and eventually the argmax flips and the two runs say
    different things. Reporting MiB without reporting that divergence point
    describes only the upside.
    """
    tok = load_tokenizer(model_name)
    prompt = torch.tensor([tok.encode(PROMPT)], device=device)
    prompt_len = prompt.size(1)

    ref: torch.Tensor | None = None
    rows = []
    for name in dtypes:
        dt = getattr(torch, name)
        model = load_pretrained(model_name, dtype=dt).to(device)
        cache = model.new_cache(batch_size=1, max_seq=prompt_len + n_tokens)

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        secs = _time(lambda: generate_cached(model, prompt, n_tokens, cache=cache), device, repeat=repeat)
        peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0

        out = generate_cached(model, prompt, n_tokens, cache=cache)[0, prompt_len:]
        if ref is None:
            ref, agree, diverge = out, "reference", "-"
        else:
            same = out == ref
            agree = f"{int(same.sum())}/{n_tokens}"
            diverge = "none" if bool(same.all()) else str(int((~same).nonzero()[0].item()))

        rows.append(
            [
                name,
                f"{model.n_params() * dt.itemsize / 2**20:.0f}",
                f"{model.cfg.kv_bytes_per_token(dt.itemsize) / 1024:.0f}",
                f"{secs / n_tokens * 1000:.2f}",
                f"{peak:.0f}" if peak else "-",
                agree,
                diverge,
            ]
        )
        print(f"  {name:9s} {secs/n_tokens*1000:6.2f} ms/token  peak {peak:7.0f} MiB  agree {agree}  first divergence {diverge}")

        del model, cache
        if device == "cuda":
            torch.cuda.empty_cache()

    return _table(
        [
            "dtype",
            "Weights (MiB)",
            "KV (KiB/token)",
            "ms/token",
            "Peak VRAM (MiB)",
            "Tokens agreeing with fp32",
            "First divergence at",
        ],
        rows,
    )


def run_memory(model_name: str, batches: list[int], contexts: list[int], dtype_bytes: int) -> str:
    cfg = PRESETS[model_name]
    weights_mib = cfg.n_params() * dtype_bytes / 2**20
    per_token = cfg.kv_bytes_per_token(dtype_bytes)

    rows = []
    for ctx in contexts:
        for b in batches:
            kv_mib = per_token * ctx * b / 2**20
            rows.append(
                [
                    str(ctx),
                    str(b),
                    f"{kv_mib:.0f}",
                    f"{weights_mib:.0f}",
                    f"{kv_mib / weights_mib:.2f}x",
                ]
            )
    header = (
        f"Model `{model_name}`: {cfg.n_params()/1e6:.0f}M params, "
        f"{weights_mib:.0f} MiB of weights at {dtype_bytes} bytes/value. "
        f"KV-cache costs **{per_token/1024:.0f} KiB per token per sequence**.\n\n"
    )
    return header + _table(
        ["Context", "Batch", "KV-cache (MiB)", "Weights (MiB)", "KV / weights"], rows
    )


# -- entry point ----------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", choices=["kvcache", "batch", "memory", "dtype", "graph", "kvquant", "serving", "all"])
    p.add_argument("--model", default="gpt2", choices=sorted(PRESETS))
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--lengths", type=int, nargs="+", default=[64, 128, 256, 512])
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--tokens", type=int, default=128, help="tokens to generate in the batch sweep")
    p.add_argument("--contexts", type=int, nargs="+", default=[256, 512, 1024])
    p.add_argument("--dtypes", nargs="+", default=["float32", "float16"], help="precisions to compare in the dtype experiment")
    p.add_argument("--kv-dtype", dest="kv_dtype", default="auto", choices=["auto", "int8"], help="KV cache storage precision")
    p.add_argument("--requests", type=int, default=64, help="requests in the serving workload")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    p.add_argument("--out", default=None, help="write markdown here instead of stdout")
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but no CUDA device is visible")

    dtype = getattr(torch, args.dtype)
    dtype_bytes = torch.empty(0, dtype=dtype).element_size()

    sections = [f"_Generated by `nanoserve.bench {args.experiment}`._\n"]
    want = ["kvcache", "batch", "graph", "dtype", "kvquant", "serving", "memory"] if args.experiment == "all" else [args.experiment]

    needs_model = any(w in ("kvcache", "batch", "graph", "kvquant", "serving") for w in want)
    if needs_model:
        print(f"Loading {args.model} on {args.device} ({args.dtype}) ...")
        model = load_pretrained(args.model, dtype=dtype).to(args.device)
        tok = load_tokenizer(args.model)
        prompt_ids = torch.tensor([tok.encode(PROMPT)], device=args.device)
        env = _describe(args.device)
        sections.append(
            "**Environment** — "
            + ", ".join(f"{k}: {v}" for k, v in env.items())
            + f", model: {args.model} ({model.n_params()/1e6:.0f}M params), dtype: {args.dtype}\n"
        )

    for w in want:
        if w == "kvcache":
            print("\nKV-cache: naive vs cached decode")
            sections.append("## KV-cache: naive vs cached decode\n\n" + run_kvcache(model, prompt_ids, args.device, args.lengths, args.repeat, args.kv_dtype))
        elif w == "batch":
            print("\nBatch scaling (cached decode)")
            sections.append("## Batch scaling\n\n" + run_batch(model, prompt_ids, args.device, args.batches, args.tokens, args.repeat, args.kv_dtype))
        elif w == "graph":
            print("\nCUDA graph vs eager decode")
            sections.append("## CUDA graph vs eager decode\n\n" + run_graph(model, prompt_ids, args.device, args.lengths, args.repeat))
        elif w == "serving":
            print("\nStatic vs continuous batching")
            sections.append("## Static vs continuous batching\n\n" + run_serving(model, prompt_ids, args.device, args.batches, args.requests))
        elif w == "kvquant":
            print("\nINT8 KV cache")
            sections.append("## INT8 KV cache\n\n" + run_kvquant(model, prompt_ids, args.device, args.batches, args.tokens, args.repeat))
        elif w == "dtype":
            print("\nPrecision trade-off")
            sections.append("## Precision trade-off\n\n" + run_dtype(args.model, args.device, args.dtypes, args.tokens, args.repeat))
        elif w == "memory":
            print("\nKV-cache memory footprint")
            sections.append("## KV-cache memory footprint\n\n" + run_memory(args.model, args.batches, args.contexts, dtype_bytes))

    md = "\n\n".join(sections) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(md)
        print(f"\nWrote {args.out}")
    else:
        print("\n" + md)


if __name__ == "__main__":
    main()
