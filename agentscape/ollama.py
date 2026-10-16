from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager

import httpx


class OllamaError(RuntimeError):
    pass


class Cancelled(RuntimeError):
    pass


class Ollama:
    """All model inference/lifecycle operations go through Ollama's native API."""

    def __init__(self, url="http://localhost:11434", parallel=1, timeout=180):
        self.url = url.rstrip("/")
        self.slots = threading.BoundedSemaphore(parallel)
        self.timeout = timeout

    def request(self, method, path, data=None):
        try:
            with httpx.Client(timeout=self.timeout, trust_env=False) as client:
                response = client.request(method, self.url + path, json=data)
                response.raise_for_status()
                result = response.json()
                if result.get("error"):
                    raise OllamaError(result["error"])
                return result
        except (httpx.HTTPError, ValueError) as exc:
            raise OllamaError(f"Ollama request failed: {exc}") from exc

    def tags(self):
        return self.request("GET", "/api/tags").get("models", [])

    def running(self):
        return self.request("GET", "/api/ps").get("models", [])

    def show(self, model):
        return self.request("POST", "/api/show", {"model": model})

    def unload(self, model):
        return self.request("POST", "/api/generate", {"model": model, "keep_alive": 0, "stream": False})

    def pull(self, model, callback=lambda _: None):
        try:
            with httpx.Client(timeout=httpx.Timeout(600, connect=10), trust_env=False) as client:
                with client.stream("POST", self.url + "/api/pull", json={"model": model, "stream": True}) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line:
                            continue
                        item = json.loads(line)
                        if item.get("error"):
                            raise OllamaError(item["error"])
                        callback(item)
        except (httpx.HTTPError, ValueError) as exc:
            raise OllamaError(f"Model download failed: {exc}") from exc

    @contextmanager
    def slot(self, cancel):
        while not self.slots.acquire(timeout=0.1):
            if cancel.is_set():
                raise Cancelled("Cancelled while waiting for inference.")
        try:
            if cancel.is_set():
                raise Cancelled("Cancelled.")
            yield
        finally:
            self.slots.release()

    def chat(self, model, messages, config, *, schema=None, tools=None, cancel=None, on_token=None, on_start=None):
        cancel = cancel or threading.Event()
        payload = {
            "model": model, "messages": messages, "stream": True,
            "keep_alive": "10m", "think": config.thinking,
            "options": {"num_ctx": config.context, "num_predict": config.max_tokens, "temperature": config.temperature},
        }
        if schema is not None:
            payload["format"] = schema
        if tools:
            payload["tools"] = tools
        start = time.monotonic()
        content, thinking, calls = [], [], []
        final = None
        with self.slot(cancel):
            if on_start:
                on_start()
            try:
                with httpx.Client(timeout=httpx.Timeout(self.timeout, connect=10), trust_env=False) as client:
                    with client.stream("POST", self.url + "/api/chat", json=payload) as response:
                        if response.is_error:
                            response.read()
                            raise OllamaError(response.text[:1200])
                        for line in response.iter_lines():
                            if cancel.is_set():
                                raise Cancelled("Cancelled during inference.")
                            if not line:
                                continue
                            item = json.loads(line)
                            if item.get("error"):
                                raise OllamaError(item["error"])
                            msg = item.get("message", {})
                            token = msg.get("content", "")
                            content.append(token)
                            thinking.append(msg.get("thinking", ""))
                            calls.extend(msg.get("tool_calls", []))
                            if on_token and token:
                                on_token(token)
                            if item.get("done"):
                                final = item
                if final is None:
                    raise OllamaError("Ollama stream ended without a completion marker.")
            except (httpx.HTTPError, ValueError) as exc:
                raise OllamaError(f"Ollama inference failed: {exc}") from exc
        final["message"] = {"role": "assistant", "content": "".join(content)}
        if calls:
            final["message"]["tool_calls"] = calls
        if any(thinking):
            final["message"]["thinking"] = "".join(thinking)
        final["wall_seconds"] = time.monotonic() - start
        return final


def metrics(response):
    seconds = response.get("eval_duration", 0) / 1e9
    return {
        "wall_seconds": round(response.get("wall_seconds", 0), 3),
        "prompt_tokens": response.get("prompt_eval_count", 0),
        "output_tokens": response.get("eval_count", 0),
        "tokens_per_second": round(response.get("eval_count", 0) / seconds, 2) if seconds else 0,
        "load_seconds": round(response.get("load_duration", 0) / 1e9, 3),
    }
