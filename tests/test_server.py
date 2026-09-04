"""Endpoint smoke tests.

Deliberately checks wire shape, not text quality. The value of being
OpenAI-compatible is that existing clients work unmodified, and that only holds
if the response actually carries the fields they read.
"""

import json
import os

import pytest

os.environ.setdefault("NANOSERVE_DEVICE", "cpu")

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from nanoserve.server import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # context manager runs the startup hook
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "gpt2"


def test_completion_shape(client):
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 8, "temperature": 0},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"]
    assert body["usage"]["completion_tokens"] == 8


def test_stream_emits_chunks_then_done(client):
    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 8,
            "temperature": 0,
            "stream": True,
        },
    )
    assert r.status_code == 200
    lines = [ln for ln in r.text.splitlines() if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"

    chunks = [json.loads(ln[6:]) for ln in lines[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text, "stream carried no content"


def test_streaming_matches_non_streaming(client):
    """Same seed and settings must give the same text either way."""
    payload = {
        "messages": [{"role": "user", "content": "The cache is"}],
        "max_tokens": 12,
        "temperature": 0,
    }
    whole = client.post("/v1/chat/completions", json=payload).json()
    streamed = client.post("/v1/chat/completions", json={**payload, "stream": True})
    pieces = [
        json.loads(ln[6:])["choices"][0]["delta"].get("content", "")
        for ln in streamed.text.splitlines()
        if ln.startswith("data: ") and ln != "data: [DONE]"
    ]
    assert "".join(pieces) == whole["choices"][0]["message"]["content"]


def test_overlong_prompt_is_rejected(client):
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "word " * 1200}], "max_tokens": 4},
    )
    assert r.status_code == 400
    assert "context window" in r.json()["detail"]
