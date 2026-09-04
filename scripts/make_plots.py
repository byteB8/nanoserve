"""Render docs/plots/*.png from the numbers recorded in RESULTS.md.

The measurements live here as literals rather than being re-run: every figure is
transcribed from a specific section of RESULTS.md, cited in the constant's name
and comment, so a plot can never drift from the text without the drift being
visible in a diff. Re-running the benchmarks is the way to change them.

    python scripts/make_plots.py
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "plots"

# -- palette ---------------------------------------------------------------
# One hue per idea, held constant across every figure: the baseline is always
# blue, the improvement always teal, and failure always red.
INK = "#16202e"
MUTED = "#5c6672"
GRID = "#dde2e8"
BASE = "#2f6fb0"       # naive / eager / static -- the thing being improved on
GOOD = "#0d7a6f"       # cached / graphed / continuous -- the improvement
FP16 = "#c77e23"
INT8 = "#7c5cbf"
BAD = "#b23b3b"        # OOM, wasted work
FACE = "#ffffff"


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": FACE,
        "axes.facecolor": FACE,
        "savefig.facecolor": FACE,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "600",
        "axes.labelsize": 10,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRID,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
        "figure.dpi": 160,
    })


def tidy(ax, ygrid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ygrid:
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)
        ax.xaxis.grid(False)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.savefig(p, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  {p.relative_to(OUT.parent.parent)}  ({p.stat().st_size / 1024:.0f} KiB)")


# =========================================================================
# Measured data -- RESULTS.md
# =========================================================================

# §1 -- naive vs cached decode, seconds
CPU_LEN = [32, 64, 128, 256]
CPU_NAIVE = [1.69, 5.47, 14.22, 43.12]
CPU_CACHED = [0.61, 1.31, 2.57, 5.28]

GPU_LEN = [64, 128, 256, 512, 1000]
GPU_NAIVE = [0.17, 0.38, 1.01, 3.08, 10.45]
GPU_CACHED = [0.21, 0.41, 0.82, 1.63, 3.19]

# §3 -- batch scaling on the A5000, fp32, 256 tokens
B_BATCH = [1, 8, 32, 64, 128, 256, 512, 1024, 1152]
B_TOTAL = [314.8, 2441, 9173, 12128, 13366, 14331, 14749, 14840, 14755]
B_PERSEQ = [314.8, 305.2, 286.7, 189.5, 104.4, 56.0, 28.8, 14.5, 12.8]
B_PEAK = [504, 638, 1095, 1709, 2934, 5385, 10283, 20034, 22484]
B_KV = [19, 150, 598, 1197, 2394, 4788, 9576, 19152, 21546]

LAP_BATCH = [1, 2, 4, 8, 16]
LAP_TOTAL = [49.3, 95.0, 105.6, 159.7, 187.7]

# §4/§7 -- memory model: fixed MiB, per-sequence MiB, KV per sequence MiB, ceiling
MODEL = {
    "fp32 KV": dict(fixed=484, per_seq=0.394, kv=18.703, ceiling=1205, colour=BASE,
                    measured=[(1, 504), (8, 638), (32, 1095), (64, 1709), (128, 2934),
                              (256, 5385), (512, 10283), (1024, 20034), (1152, 22484)],
                    oom=None),
    "fp16 KV": dict(fixed=253, per_seq=0.197, kv=9.351, ceiling=2434, colour=FP16,
                    measured=[(1, 262), (32, 557), (256, 2702), (1024, 10052),
                              (2048, 19803), (2304, 22253)],
                    oom=2560),
    "int8 KV": dict(fixed=485, per_seq=2.549, kv=4.968, ceiling=3061, colour=INT8,
                    measured=[(1, 493), (64, 966), (256, 2410), (1024, 8186),
                              (2048, 15884), (2816, 21625), (3008, 23068)],
                    oom=3072),
}
VRAM_FREE = 23494  # MiB reported free on the A5000

# §5 -- precision at batch 1, 256 tokens
PREC = {
    "fp32": dict(ms=3.24, peak=504, agree=256),
    "fp16": dict(ms=3.43, peak=265, agree=256),
    "bf16": dict(ms=3.38, peak=265, agree=208),
}
FP16_BATCH = [1, 32, 256, 1024, 2048, 2304]
FP16_TOTAL = [296.2, 8716.9, 44908.8, 48391.7, 47908.2, 47906.0]

# §6 -- CUDA graphs, ms/token
G_LEN = [64, 128, 256, 512, 1000]
G_EAGER = [3.26, 3.24, 3.22, 3.22, 3.17]
G_BUCKETED = [1.43, 1.41, 1.51, 1.77, 2.33]
G_FULLWIN = [1.42, 1.51, 1.72, 2.13, 2.88]
BANDWIDTH_FLOOR = 0.62  # 475 MiB of weights / 768 GB/s

# §8 -- static vs continuous batching, 128 requests
S_SLOTS = [8, 32, 64, 128]
S_STATIC_TPS = [975, 3231, 4294, 4766]
S_CONT_TPS = [1623, 4114, 4556, 3647]
S_STATIC_WORK = [2.36, 2.67, 2.67, 2.67]
S_CONT_WORK = [1.05, 1.37, 1.61, 2.55]
S_STATIC_LAT = [7.23, 2.34, 2.08, 2.51]
S_CONT_LAT = [3.92, 1.40, 1.18, 1.43]
S_STATIC_P95 = [12.57, 3.76, 2.80, 2.51]
S_CONT_P95 = [6.78, 2.43, 2.17, 3.30]


# =========================================================================
# Figures
# =========================================================================


def fig_kvcache() -> None:
    """The cache makes decode O(1) per step -- and that is visible as flatness."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.0))

    a.plot(GPU_LEN, [s / n * 1000 for s, n in zip(GPU_NAIVE, GPU_LEN)], "o-",
           color=BASE, label="naive (recompute prefix)")
    a.plot(GPU_LEN, [s / n * 1000 for s, n in zip(GPU_CACHED, GPU_LEN)], "o-",
           color=GOOD, label="+ KV-cache")
    a.set_xscale("log", base=2)
    a.set_xticks(GPU_LEN)
    a.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    a.set_ylim(0, 11.5)
    a.set_xlabel("tokens generated")
    a.set_ylabel("ms per token")
    a.set_title("Cost per token — RTX A5000, fp32", loc="left")
    a.legend(loc="upper left")
    a.annotate("flat: cost does not grow\nwith position in the sequence",
               xy=(700, 3.19), xytext=(150, 6.2), color=GOOD, fontsize=8.5,
               arrowprops=dict(arrowstyle="->", color=GOOD, lw=1.1))
    tidy(a)

    cpu_sp = [n / c for n, c in zip(CPU_NAIVE, CPU_CACHED)]
    gpu_sp = [n / c for n, c in zip(GPU_NAIVE, GPU_CACHED)]
    b.plot(CPU_LEN, cpu_sp, "s-", color=BASE, label="laptop CPU (i5-1035G1)")
    b.plot(GPU_LEN, gpu_sp, "o-", color=GOOD, label="RTX A5000")
    b.axhline(1.0, color=BAD, lw=1.2, ls="--")
    b.text(300, 0.47, "below 1× the cache is a net loss", color=BAD, fontsize=8.5)
    b.set_ylim(0.3, 9)
    b.set_xscale("log", base=2)
    b.set_xticks(sorted(set(CPU_LEN + GPU_LEN)))
    b.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    b.set_xlabel("tokens generated")
    b.set_ylabel("speedup vs naive (×)")
    b.set_title("Speedup grows with length — and starts below 1× on GPU", loc="left")
    b.legend(loc="upper left")
    tidy(b)

    fig.suptitle("KV-cache: what it buys, and where it does not",
                 x=0.005, ha="left", fontsize=13, fontweight="600", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "kvcache.png")


def fig_batch() -> None:
    """Batching is free until the arithmetic units saturate."""
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(B_BATCH, B_TOTAL, "o-", color=GOOD, label="A5000 — aggregate throughput")
    ax.plot(LAP_BATCH, LAP_TOTAL, "s-", color=BASE, label="laptop CPU — aggregate throughput")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch size (concurrent sequences)")
    ax.set_ylabel("tokens / second (all sequences)")
    ax.set_xticks([1, 2, 8, 32, 64, 128, 256, 512, 1024])
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    tidy(ax)

    ax.axvline(32, color=GOOD, lw=1.0, ls=":", alpha=0.8)
    ax.axvline(2, color=BASE, lw=1.0, ls=":", alpha=0.8)
    ax.annotate("A5000 knee — batch 1→32 returns\n29× throughput for 10% more wall time",
                xy=(32, 9173), xytext=(60, 1500), fontsize=8.5, color=GOOD,
                arrowprops=dict(arrowstyle="->", color=GOOD, lw=1.1))
    ax.annotate("laptop knee\n(batch 2)", xy=(2, 95), xytext=(1.15, 300),
                fontsize=8.5, color=BASE,
                arrowprops=dict(arrowstyle="->", color=BASE, lw=1.1))

    ax2 = ax.twinx()
    ax2.plot(B_BATCH, B_PERSEQ, "o--", color=MUTED, lw=1.4, ms=4,
             label="A5000 — per-sequence rate")
    ax2.set_ylabel("tokens / second (single sequence)", color=MUTED)
    ax2.set_yscale("log")
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right")
    ax.set_title("Batch scaling: where “free” ends is a property of the hardware", loc="left")
    fig.tight_layout()
    save(fig, "batch_scaling.png")


def fig_memory() -> None:
    """Peak VRAM is predictable, and the prediction locates the OOM boundary."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.4, 4.3),
                               gridspec_kw={"width_ratios": [1.35, 1]})

    for name, d in MODEL.items():
        xs = [x for x, _ in d["measured"]]
        ys = [y for _, y in d["measured"]]
        grid = [1] + list(range(64, int(d["ceiling"]) + 1, 32))
        pred = [d["fixed"] + (d["per_seq"] + d["kv"]) * g for g in grid]
        a.plot(grid, pred, "-", color=d["colour"], lw=1.4, alpha=0.55)
        a.plot(xs, ys, "o", color=d["colour"], label=f"{name} (measured)", ms=5)
        if d["oom"]:
            a.plot([d["oom"]], [d["fixed"] + (d["per_seq"] + d["kv"]) * d["oom"]],
                   "x", color=BAD, ms=9, mew=2.2)

    a.axhline(VRAM_FREE, color=BAD, lw=1.3, ls="--")
    a.text(1.15, VRAM_FREE + 550, "24 GiB card — free VRAM", color=BAD, fontsize=8.5, ha="left")
    a.set_xscale("log", base=2)
    a.set_xticks([1, 8, 64, 256, 1024, 3072])
    a.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    a.set_xlabel("concurrent sequences (batch)")
    a.set_ylabel("peak VRAM (MiB)")
    a.set_title("Lines are predicted, dots measured; × marks a real OOM", loc="left")
    a.legend(loc="center left")
    a.set_ylim(0, 27000)
    tidy(a)

    names = list(MODEL)
    ceilings = [MODEL[n]["ceiling"] for n in names]
    colours = [MODEL[n]["colour"] for n in names]
    bars = b.bar(names, ceilings, color=colours, width=0.6)
    for n, bar in zip(names, bars):
        d = MODEL[n]
        b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 60,
               f"{d['ceiling']:,}", ha="center", fontsize=9.5, color=INK, fontweight="600")
        lo = max((x for x, _ in d["measured"]))
        hi = d["oom"]
        if hi:
            b.plot([bar.get_x() + bar.get_width() / 2] * 2, [lo, hi], color=INK, lw=1.2)
            b.plot([bar.get_x() + bar.get_width() / 2], [lo], "_", color=INK, ms=12, mew=1.6)
            b.plot([bar.get_x() + bar.get_width() / 2], [hi], "_", color=BAD, ms=12, mew=1.6)
    b.set_ylabel("max concurrent sequences")
    b.set_title("Predicted ceiling vs measured OK/OOM bracket", loc="left")
    b.set_ylim(0, 3600)
    tidy(b)

    fig.suptitle("A memory model accurate to 0.25% — and predictive across precisions",
                 x=0.005, ha="left", fontsize=13, fontweight="600", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "memory_model.png")


def fig_graphs() -> None:
    """CUDA graphs cash in the launch overhead -- and bucketing keeps the win."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.0))

    a.plot(G_LEN, G_EAGER, "o-", color=BASE, label="eager decode")
    a.plot(G_LEN, G_FULLWIN, "o--", color=MUTED, lw=1.5, ms=4,
           label="graphed, single full-window capture")
    a.plot(G_LEN, G_BUCKETED, "o-", color=GOOD, label="graphed, bucketed by window")
    a.axhline(BANDWIDTH_FLOOR, color=BAD, lw=1.2, ls="--")
    a.text(1000, BANDWIDTH_FLOOR + 0.13, "memory-bandwidth floor (0.62 ms)",
           color=BAD, fontsize=8.5, ha="right")
    a.set_xscale("log", base=2)
    a.set_xticks(G_LEN)
    a.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    a.set_ylim(0, 3.8)
    a.set_xlabel("tokens generated")
    a.set_ylabel("ms per token")
    a.set_title("Eager decode sits 5× above the bandwidth floor", loc="left")
    a.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86))
    tidy(a)

    x = range(len(G_LEN))
    w = 0.38
    full = [e / g for e, g in zip(G_EAGER, G_FULLWIN)]
    buck = [e / g for e, g in zip(G_EAGER, G_BUCKETED)]
    b.bar([i - w / 2 for i in x], full, w, color=MUTED, label="single full-window graph")
    b.bar([i + w / 2 for i in x], buck, w, color=GOOD, label="bucketed graphs")
    b.axhline(1.0, color=BAD, lw=1.1, ls="--")
    b.set_xticks(list(x))
    b.set_xticklabels(G_LEN)
    b.set_xlabel("tokens generated")
    b.set_ylabel("speedup over eager (×)")
    b.set_title("Capture rigidity costs attention; bucketing recovers it", loc="left")
    b.set_ylim(0, 2.85)
    b.legend(loc="upper right")
    for i, (f, k) in enumerate(zip(full, buck)):
        b.text(i + w / 2, k + 0.04, f"{k:.2f}×", ha="center", fontsize=8.5, color=GOOD)
    tidy(b)

    fig.suptitle("CUDA graphs: ~80% of decode time was launch overhead",
                 x=0.005, ha="left", fontsize=13, fontweight="600", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "cuda_graphs.png")


def fig_precision() -> None:
    """Half precision halves memory, does not help latency, and bf16 loses accuracy."""
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(12, 3.7))
    names = list(PREC)
    cols = [BASE, FP16, INT8]

    a.bar(names, [PREC[n]["peak"] for n in names], color=cols, width=0.55)
    for i, n in enumerate(names):
        a.text(i, PREC[n]["peak"] + 12, f"{PREC[n]['peak']}", ha="center", fontsize=9, color=INK)
    a.set_ylabel("peak VRAM (MiB)")
    a.set_title("Memory — halved", loc="left", fontsize=11)
    a.set_ylim(0, 600)
    tidy(a)

    b.bar(names, [PREC[n]["ms"] for n in names], color=cols, width=0.55)
    for i, n in enumerate(names):
        b.text(i, PREC[n]["ms"] + 0.06, f"{PREC[n]['ms']:.2f}", ha="center", fontsize=9, color=INK)
    b.set_ylabel("ms per token (batch 1)")
    b.set_title("Latency — no better", loc="left", fontsize=11)
    b.set_ylim(0, 4.2)
    tidy(b)

    c.bar(names, [PREC[n]["agree"] for n in names], color=cols, width=0.55)
    c.axhline(256, color=MUTED, lw=1.0, ls=":")
    for i, n in enumerate(names):
        v = PREC[n]["agree"]
        c.text(i, v + 5, f"{v}/256", ha="center", fontsize=9,
               color=BAD if v < 256 else INK, fontweight="600" if v < 256 else "normal")
    c.annotate("diverges at token 12 —\n8 mantissa bits vs fp16's 10",
               xy=(2, 208), xytext=(-0.35, 105), fontsize=8.5, color=BAD,
               arrowprops=dict(arrowstyle="->", color=BAD, lw=1.1))
    c.set_ylabel("tokens matching fp32")
    c.set_title("Accuracy — bf16 drifts", loc="left", fontsize=11)
    c.set_ylim(0, 300)
    tidy(c)

    fig.suptitle("Precision: the answer depends on which regime you are in",
                 x=0.005, ha="left", fontsize=13, fontweight="600", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "precision.png")


def fig_continuous() -> None:
    """Continuous batching helps when there is a queue; latency wins regardless."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.2, 4.2))
    x = range(len(S_SLOTS))
    w = 0.38

    a.bar([i - w / 2 for i in x], S_STATIC_TPS, w, color=BASE, label="static batching")
    a.bar([i + w / 2 for i in x], S_CONT_TPS, w, color=GOOD, label="continuous batching")
    for i, (s, c) in enumerate(zip(S_STATIC_TPS, S_CONT_TPS)):
        r = c / s
        a.text(i, max(s, c) + 180, f"{r:.2f}×", ha="center", fontsize=9,
               color=GOOD if r >= 1 else BAD, fontweight="600")
    a.set_xticks(list(x))
    a.set_xticklabels([f"{s} slots" for s in S_SLOTS])
    a.set_ylabel("tokens / second")
    a.set_title("Throughput: wins only while requests queue", loc="left")
    a.set_ylim(0, 5900)
    a.legend(loc="upper left")
    tidy(a)

    b.bar([i - w / 2 for i in x], S_STATIC_LAT, w, color=BASE, label="static — mean")
    b.bar([i + w / 2 for i in x], S_CONT_LAT, w, color=GOOD, label="continuous — mean")
    b.plot([i - w / 2 for i in x], S_STATIC_P95, "_", color=INK, ms=14, mew=2,
           label="p95")
    b.plot([i + w / 2 for i in x], S_CONT_P95, "_", color=INK, ms=14, mew=2)
    for i in x:
        b.plot([i - w / 2] * 2, [S_STATIC_LAT[i], S_STATIC_P95[i]], color=INK, lw=1.0, alpha=0.5)
        b.plot([i + w / 2] * 2, [S_CONT_LAT[i], S_CONT_P95[i]], color=INK, lw=1.0, alpha=0.5)
    b.set_xticks(list(x))
    b.set_xticklabels([f"{s} slots" for s in S_SLOTS])
    b.set_ylabel("per-request latency (s)")
    b.set_title("Latency: better everywhere, 1.7–1.8× on the mean", loc="left")
    b.legend(loc="upper right")
    tidy(b)

    fig.suptitle("Static vs continuous batching — 128 requests, output lengths 16–256",
                 x=0.005, ha="left", fontsize=13, fontweight="600", color=INK)
    fig.text(0.005, -0.02,
             "At 128 slots for a 128-request burst nothing ever queues, so no slot is "
             "refilled and the scheduler is pure overhead — the work ratios converge "
             "(2.55× vs 2.67×).",
             fontsize=9, color=MUTED, ha="left")
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    save(fig, "continuous_batching.png")


def fig_timeline() -> None:
    """Why continuous batching wins, as a picture of slot occupancy.

    This is a *schedule*, not a timing run: it replays the same length
    distribution the benchmark uses and lays out when each slot is busy under
    both policies. The x axis is decode steps, so the shaded gaps are exactly the
    tokens static batching computes for sequences that have already finished.
    """
    slots = 6
    rng = random.Random(0)
    lengths = [rng.choice([16, 24, 32, 48, 64, 96, 128, 192, 256]) for _ in range(18)]

    # static: fixed groups, everyone runs to the group's longest
    static_bars, t = [], 0
    for i in range(0, len(lengths), slots):
        group = lengths[i : i + slots]
        span = max(group)
        for s, n in enumerate(group):
            static_bars.append((s, t, n, span - n))  # slot, start, useful, wasted
        t += span

    # continuous: a slot is refilled the moment it frees
    free_at = [0] * slots
    cont_bars = []
    for n in lengths:
        s = min(range(slots), key=lambda i: free_at[i])
        cont_bars.append((s, free_at[s], n, 0))
        free_at[s] += n

    fig, (a, b) = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True)
    for ax, bars, title in (
        (a, static_bars, "Static batching — a slot that finishes early keeps stepping until the batch drains"),
        (b, cont_bars, "Continuous batching — the slot is refilled immediately"),
    ):
        for s, start, useful, wasted in bars:
            ax.barh(s, useful, left=start, height=0.62, color=GOOD, edgecolor=FACE, lw=0.8)
            if wasted:
                ax.barh(s, wasted, left=start + useful, height=0.62,
                        color=BAD, alpha=0.30, edgecolor=FACE, lw=0.8, hatch="///")
        ax.set_yticks(range(slots))
        ax.set_yticklabels([f"slot {i}" for i in range(slots)])
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True)
        ax.yaxis.grid(False)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)

    total_static = max(s + u + w for _, s, u, w in static_bars)
    total_cont = max(s + u for _, s, u, _ in cont_bars)
    b.set_xlabel("decode steps")

    from matplotlib.patches import Patch

    fig.legend(handles=[Patch(color=GOOD, label="tokens the caller asked for"),
                        Patch(facecolor=BAD, alpha=0.30, hatch="///",
                              label="tokens computed for a sequence that already finished")],
               loc="upper right", ncol=2, bbox_to_anchor=(0.995, 0.945), fontsize=9)
    b.axvline(total_cont, color=INK, lw=1.0, ls=":")
    b.axvline(total_static, color=MUTED, lw=1.0, ls=":")

    fig.suptitle("The mechanism: 6 slots, 18 requests, output lengths 16–256",
                 x=0.005, y=0.995, ha="left", fontsize=13, fontweight="600", color=INK)
    fig.text(0.005, -0.01,
             f"Same {sum(lengths)} tokens either way: {total_static} decode steps under static "
             f"batching, {total_cont} under continuous — {total_static / total_cont:.2f}× shorter. "
             "A schedule, not a timing run.",
             fontsize=9, color=MUTED, ha="left")
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    save(fig, "batching_timeline.png")


def main() -> None:
    style()
    print("rendering plots ...")
    fig_kvcache()
    fig_batch()
    fig_memory()
    fig_graphs()
    fig_precision()
    fig_continuous()
    fig_timeline()


if __name__ == "__main__":
    main()
