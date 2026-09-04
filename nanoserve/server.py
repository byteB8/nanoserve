"""An OpenAI-compatible HTTP front end for the engine.

    uvicorn nanoserve.server:app --port 8000

Why OpenAI-shaped rather than something bespoke: every LLM client already speaks
it, so the engine can be pointed at from `curl`, the `openai` SDK, or any chat UI
without adapters. Being wire-compatible is close to free and makes the thing
usable rather than merely demonstrable.

**Honest limitations**, because a serving front end invites assumptions:

  * GPT-2 is a *base* model with no chat template and no special tokens for turn
    structure. Messages are flattened into a plain prompt, so `/v1/chat/completions`
    accepts the chat schema but the model completes text -- it does not converse.
  * Requests are serialised. Each holds its own KV-cache and runs to completion
    before the next starts, so concurrency is bounded by a semaphore rather than
    shared across a batch. Fixing that is continuous batching: admitting arrivals
    into free cache slots mid-flight instead of running one sequence at a time.
    The §3 batch curve says what it is worth -- 29x the throughput at batch 32.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .generate import stream_cached
from .weights import load_pretrained, load_tokenizer

MODEL_NAME = os.environ.get("NANOSERVE_MODEL", "gpt2")
DEVICE = os.environ.get("NANOSERVE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MAX_CONCURRENCY = int(os.environ.get("NANOSERVE_CONCURRENCY", "1"))


# -- schema ---------------------------------------------------------------


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[Message]
    max_tokens: int = Field(default=64, ge=1, le=1024)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    stream: bool = False
    seed: int | None = None


@dataclass
class Engine:
    model: object = None
    tokenizer: object = None
    lock: threading.Semaphore = field(default_factory=lambda: threading.Semaphore(MAX_CONCURRENCY))


engine = Engine()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load weights once at startup rather than per request."""
    engine.model = load_pretrained(MODEL_NAME).to(DEVICE)
    engine.tokenizer = load_tokenizer(MODEL_NAME)
    print(f"nanoserve: {MODEL_NAME} on {DEVICE}, concurrency {MAX_CONCURRENCY}")
    yield


app = FastAPI(title="nanoserve", version="0.1", lifespan=lifespan)


# -- helpers --------------------------------------------------------------


def _flatten(messages: list[Message]) -> str:
    """Collapse the chat schema into a plain prompt.

    GPT-2 has no turn structure to honour, so anything more elaborate than this
    would be inventing a template the weights were never trained against.
    """
    return "\n".join(m.content for m in messages)


def _encode(prompt: str) -> torch.Tensor:
    ids = engine.tokenizer.encode(prompt)
    budget = engine.model.cfg.block_size
    if not ids:
        raise HTTPException(400, "empty prompt")
    if len(ids) >= budget:
        raise HTTPException(400, f"prompt is {len(ids)} tokens; the context window is {budget}")
    return torch.tensor([ids], device=DEVICE)


def _clamp(prompt_len: int, requested: int) -> int:
    return max(1, min(requested, engine.model.cfg.block_size - prompt_len))


def _chunk(cid: str, created: int, delta: dict, finish: str | None) -> str:
    body = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_NAME,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(body)}\n\n"


# -- endpoints ------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    ready = engine.model is not None
    out = {"status": "ok" if ready else "loading", "model": MODEL_NAME, "device": DEVICE}
    if ready and DEVICE == "cuda":
        out["vram_allocated_mib"] = round(torch.cuda.memory_allocated() / 2**20)
    return out


@app.get("/v1/models")
def models() -> dict:
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "nanoserve"}]}


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    idx = _encode(_flatten(req.messages))
    prompt_len = idx.size(1)
    n = _clamp(prompt_len, req.max_tokens)

    gen = None
    if req.seed is not None:
        gen = torch.Generator(device=DEVICE).manual_seed(req.seed)

    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def tokens():
        """Run the decode loop under the concurrency guard.

        Starlette runs a sync generator in a worker thread, so blocking torch
        calls here do not stall the event loop.
        """
        with engine.lock:
            cache = engine.model.new_cache(batch_size=1, max_seq=prompt_len + n)
            yield from stream_cached(
                engine.model, idx, n, req.temperature, req.top_p, gen, cache=cache
            )

    if not req.stream:
        text = engine.tokenizer.decode(list(tokens()))
        return {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_len,
                "completion_tokens": n,
                "total_tokens": prompt_len + n,
            },
        }

    def sse():
        yield _chunk(cid, created, {"role": "assistant", "content": ""}, None)
        # Decode incrementally rather than per token: GPT-2's BPE splits many
        # words across tokens, and decoding each id alone emits replacement
        # characters mid-word. Holding the ids and re-decoding lets multi-token
        # characters resolve before they are sent.
        ids: list[int] = []
        sent = 0
        for tid in tokens():
            ids.append(tid)
            text = engine.tokenizer.decode(ids)
            if "�" in text[sent:]:
                continue  # incomplete UTF-8; wait for the next token
            piece, sent = text[sent:], len(text)
            if piece:
                yield _chunk(cid, created, {"content": piece}, None)
        yield _chunk(cid, created, {}, "length")
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser(description="Serve nanoserve over HTTP")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
