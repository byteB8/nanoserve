"""Render docs/demo.gif from a real streaming request.

Not a mock-up: the script issues an actual request against a running server and
records when each token arrives, then draws those arrival times back at 1x. The
pauses you see in the GIF are the model's real per-token latency.

    uvicorn nanoserve.server:app --port 8000        # in one shell
    python scripts/make_demo.py                     # in another
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Terminal palette: a muted blue-black ground, so the accent colours carry the
# eye rather than the background competing with them.
BG = (26, 27, 38)
CHROME = (36, 40, 59)
PROMPT = (158, 206, 106)
CMD = (192, 202, 245)
DIM = (86, 95, 137)
OUT = (125, 207, 255)
META = (187, 154, 247)

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
SIZE = 15
PAD = 18
COLS = 74
FPS = 20  # 50 ms per frame


def capture(url: str, prompt: str, max_tokens: int) -> list[tuple[float, str]]:
    body = json.dumps(
        {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    events: list[tuple[float, str]] = []
    with urllib.request.urlopen(req) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            piece = json.loads(line[6:])["choices"][0]["delta"].get("content")
            if piece:
                events.append((time.perf_counter() - t0, piece))
    return events


def wrap(text: str, cols: int) -> list[str]:
    """Wrap while preserving the exact characters, so partial words render mid-stream."""
    lines, cur = [], ""
    for ch in text:
        if ch == "\n":
            lines.append(cur)
            cur = ""
            continue
        cur += ch
        if len(cur) >= cols:
            lines.append(cur)
            cur = ""
    lines.append(cur)
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--prompt", default="The key insight about autoregressive decoding is that")
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--out", default="docs/demo.gif")
    args = ap.parse_args()

    print(f"streaming from {args.url} ...")
    events = capture(args.url, args.prompt, args.max_tokens)
    total = events[-1][0]
    print(f"  {len(events)} tokens in {total:.2f}s ({total/len(events)*1000:.0f} ms/token)")

    font = ImageFont.truetype(FONT_PATH, SIZE)
    bold = ImageFont.truetype(FONT_BOLD, SIZE)
    cw = font.getlength("M")
    lh = SIZE + 6
    cmd = "curl -N localhost:8000/v1/chat/completions -d '{…,\"stream\":true}'"
    cmd_lines = wrap(cmd, COLS - 2)

    # Size the canvas to the finished frame so the GIF has no dead space, and so
    # nothing reflows as text arrives.
    full = "".join(p for _, p in events)
    out_lines = len(wrap(full, COLS))
    width = int(PAD * 2 + cw * COLS)
    height = 26 + PAD * 2 + lh * (len(cmd_lines) + out_lines + 2) + lh // 2

    def frame(shown: str, done: bool) -> Image.Image:
        img = Image.new("RGB", (width, height), BG)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, width, 26], fill=CHROME)
        for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
            d.ellipse([13 + i * 18, 9, 21 + i * 18, 17], fill=c)
        d.text((width // 2 - 46, 6), "nanoserve", font=font, fill=DIM)

        y = 26 + PAD
        d.text((PAD, y), "$", font=bold, fill=PROMPT)
        for i, ln in enumerate(cmd_lines):
            d.text((PAD + cw * 2, y + i * lh), ln, font=font, fill=CMD)
        y += lh * len(cmd_lines) + lh // 2

        for ln in wrap(shown, COLS):
            d.text((PAD, y), ln, font=font, fill=OUT)
            y += lh

        if done:
            y += lh // 2
            d.text(
                (PAD, y),
                f"{len(events)} tokens · {total:.2f}s · {total/len(events)*1000:.0f} ms/token · gpt2 on cpu",
                font=font,
                fill=META,
            )
        return img

    frames, durations = [], []
    step = 1.0 / FPS
    t, i, shown = 0.0, 0, ""
    while i < len(events):
        while i < len(events) and events[i][0] <= t:
            shown += events[i][1]
            i += 1
        frames.append(frame(shown, False))
        durations.append(int(step * 1000))
        t += step

    frames.append(frame(shown, True))
    durations.append(2500)  # hold on the final frame

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True
    )
    print(f"wrote {out} ({len(frames)} frames, {out.stat().st_size/1024:.0f} KiB)")


if __name__ == "__main__":
    main()
