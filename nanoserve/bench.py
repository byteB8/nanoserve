"""Benchmark harness. Emits markdown so results land straight in RESULTS.md.

Three experiments:

  kvcache  -- naive vs cached decode across generation lengths. The speedup should
              grow roughly linearly with length, because naive is O(T^2) against
              the cache's O(T).
  batch    -- cached decode throughput against batch size. Decode is bandwidth-
              bound, so serving many sequences at once costs barely more wall-clock
              per step than serving one, until compute saturates.
  memory   -- reserved KV bytes against batch and context, next to the weights.
              Shows which of the two actually limits how much you can serve.

Usage:
    python -m nanoserve.bench kvcache --device cpu
    python -m nanoserve.bench batch   --device cuda --model gpt2
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
from .generate import generate_cached, generate_naive
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


def run_kvcache(model, prompt_ids, device: str, lengths: list[int], repeat: int) -> str:
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
        cache = model.new_cache(batch_size=prompt_ids.size(0), max_seq=prompt_len + n)
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


def run_batch(model, prompt_ids, device: str, batches: list[int], n_tokens: int, repeat: int) -> str:
    rows = []
    single = None
    for b in batches:
        batched = prompt_ids.repeat(b, 1)
        try:
            # Allocated outside the timed region, as in run_kvcache: cache setup
            # scales with batch, so timing it here would confound the throughput
            # curve with allocator behaviour.
            cache = model.new_cache(batch_size=b, max_seq=prompt_ids.size(1) + n_tokens)
            secs = _time(lambda: generate_cached(model, batched, n_tokens, cache=cache), device, repeat=repeat)
        except (torch.cuda.OutOfMemoryError if device == "cuda" else RuntimeError) as exc:
            print(f"  batch {b:4d}: out of memory -- {type(exc).__name__}")
            rows.append([str(b), "OOM", "OOM", "OOM", "-"])
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
            ]
        )
        print(f"  batch {b:4d}: {secs:6.2f}s  total {total_tps:8.1f} tok/s  per-seq {n_tokens/secs:6.1f} tok/s")
    return _table(
        ["Batch", "Wall (s)", "Total tokens/s", "Per-sequence tokens/s", "Throughput vs batch 1"],
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
    p.add_argument("experiment", choices=["kvcache", "batch", "memory", "all"])
    p.add_argument("--model", default="gpt2", choices=sorted(PRESETS))
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--lengths", type=int, nargs="+", default=[64, 128, 256, 512])
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--tokens", type=int, default=128, help="tokens to generate in the batch sweep")
    p.add_argument("--contexts", type=int, nargs="+", default=[256, 512, 1024])
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
    want = ["kvcache", "batch", "memory"] if args.experiment == "all" else [args.experiment]

    needs_model = any(w in ("kvcache", "batch") for w in want)
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
            sections.append("## KV-cache: naive vs cached decode\n\n" + run_kvcache(model, prompt_ids, args.device, args.lengths, args.repeat))
        elif w == "batch":
            print("\nBatch scaling (cached decode)")
            sections.append("## Batch scaling\n\n" + run_batch(model, prompt_ids, args.device, args.batches, args.tokens, args.repeat))
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
