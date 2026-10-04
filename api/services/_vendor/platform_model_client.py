"""Backend-agnostic model client.

``ModelClient`` is the only interface the rest of the app uses. Swapping Ollama
for vLLM (or any OpenAI-compatible server) is a change of ``LLM_BACKEND``.
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

import httpx

Role = Literal["system", "user", "assistant"]


class ModelError(RuntimeError):
    """The model backend failed or returned something unusable."""


class ModelUnavailable(ModelError):
    """The model backend could not be reached."""


@dataclass(frozen=True)
class Message:
    """One chat turn."""

    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class GenerationParams:
    """Sampling parameters shared by every backend."""

    temperature: float = 0.2
    max_tokens: int = 512


@dataclass(frozen=True)
class Usage:
    """Token accounting for one generation."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.completion_tokens / self.seconds if self.seconds > 0 else 0.0


@dataclass(frozen=True)
class GenerationResult:
    """Complete, non-streamed generation."""

    text: str
    usage: Usage


@dataclass(frozen=True)
class StreamChunk:
    """Incremental piece of a streamed generation; ``usage`` is set on the last one."""

    text: str = ""
    usage: Usage | None = None


class ModelClient(ABC):
    """Interface every serving backend implements."""

    def __init__(self, base_url: str, model: str, embedding_model: str,
                 timeout: float = 120.0, client: httpx.AsyncClient | None = None, think: bool = False) -> None:
        self.model = model
        self.think = think  # allow reasoning/"thinking" output (Ollama only); off keeps answers within max_tokens
        self.embedding_model = embedding_model
        self._owns_client = client is None
        self._http = client or httpx.AsyncClient(base_url=base_url, timeout=timeout)

    @abstractmethod
    async def generate(self, messages: list[Message], params: GenerationParams) -> GenerationResult:
        """Return the full completion for ``messages``."""

    @abstractmethod
    def stream(self, messages: list[Message], params: GenerationParams) -> AsyncIterator[StreamChunk]:
        """Yield completion pieces as they are produced."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in order."""

    @abstractmethod
    async def health(self) -> bool:
        """True if the backend is reachable."""

    async def aclose(self) -> None:
        """Release the HTTP connection pool."""
        if self._owns_client:
            await self._http.aclose()

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = await self._http.post(path, json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.TransportError as exc:
            raise ModelUnavailable(f"cannot reach model backend: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise ModelError(f"backend returned {exc.response.status_code}: {exc.response.text[:300]}") from exc


class OllamaClient(ModelClient):
    """Ollama native API (``/api/chat``, ``/api/embed``)."""

    def _chat_payload(self, messages: list[Message], params: GenerationParams, stream: bool) -> dict[str, Any]:
        return {
            "model": self.model, "stream": stream, "think": self.think,
            "messages": [m.to_dict() for m in messages],
            "options": {"temperature": params.temperature, "num_predict": params.max_tokens},
        }

    async def generate(self, messages: list[Message], params: GenerationParams) -> GenerationResult:
        start = time.perf_counter()
        data = await self._post("/api/chat", self._chat_payload(messages, params, False))
        elapsed = time.perf_counter() - start
        return GenerationResult(
            text=data.get("message", {}).get("content", ""),
            usage=Usage(data.get("prompt_eval_count", 0), data.get("eval_count", 0), elapsed))

    async def stream(self, messages: list[Message], params: GenerationParams) -> AsyncIterator[StreamChunk]:
        start = time.perf_counter()
        try:
            async with self._http.stream("POST", "/api/chat", json=self._chat_payload(messages, params, True)) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode(errors="replace")[:300]
                    raise ModelError(f"backend returned {resp.status_code}: {body}")
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("done"):
                        yield StreamChunk(usage=Usage(
                            data.get("prompt_eval_count", 0), data.get("eval_count", 0),
                            time.perf_counter() - start))
                        return
                    yield StreamChunk(text=data.get("message", {}).get("content", ""))
        except httpx.TransportError as exc:
            raise ModelUnavailable(f"cannot reach model backend: {exc}") from exc

    async def embed(self, texts: list[str]) -> list[list[float]]:
        data = await self._post("/api/embed", {"model": self.embedding_model, "input": texts})
        vectors = data.get("embeddings")
        if not vectors or len(vectors) != len(texts):
            raise ModelError("embedding response missing or wrong length")
        return vectors

    async def health(self) -> bool:
        try:
            return (await self._http.get("/api/tags", timeout=3.0)).status_code == 200
        except httpx.HTTPError:
            return False


class OpenAICompatClient(ModelClient):
    """OpenAI-compatible API, as served by vLLM, llama.cpp server, etc."""

    def _chat_payload(self, messages: list[Message], params: GenerationParams, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model, "stream": stream,
            "messages": [m.to_dict() for m in messages],
            "temperature": params.temperature, "max_tokens": params.max_tokens,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    async def generate(self, messages: list[Message], params: GenerationParams) -> GenerationResult:
        start = time.perf_counter()
        data = await self._post("/v1/chat/completions", self._chat_payload(messages, params, False))
        elapsed = time.perf_counter() - start
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise ModelError("malformed chat completion response") from exc
        usage = data.get("usage", {})
        return GenerationResult(text, Usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), elapsed))

    async def stream(self, messages: list[Message], params: GenerationParams) -> AsyncIterator[StreamChunk]:
        start = time.perf_counter()
        usage: dict[str, int] = {}
        try:
            async with self._http.stream("POST", "/v1/chat/completions",
                                         json=self._chat_payload(messages, params, True)) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode(errors="replace")[:300]
                    raise ModelError(f"backend returned {resp.status_code}: {body}")
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        break
                    data = json.loads(body)
                    usage = data.get("usage") or usage
                    for choice in data.get("choices", []):
                        if piece := choice.get("delta", {}).get("content"):
                            yield StreamChunk(text=piece)
        except httpx.TransportError as exc:
            raise ModelUnavailable(f"cannot reach model backend: {exc}") from exc
        yield StreamChunk(usage=Usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
                                      time.perf_counter() - start))

    async def embed(self, texts: list[str]) -> list[list[float]]:
        data = await self._post("/v1/embeddings", {"model": self.embedding_model, "input": texts})
        try:
            rows = sorted(data["data"], key=lambda r: r["index"])
            vectors = [r["embedding"] for r in rows]
        except (KeyError, TypeError) as exc:
            raise ModelError("malformed embeddings response") from exc
        if len(vectors) != len(texts):
            raise ModelError("embedding response wrong length")
        return vectors

    async def health(self) -> bool:
        try:
            return (await self._http.get("/v1/models", timeout=3.0)).status_code == 200
        except httpx.HTTPError:
            return False


_BACKENDS: dict[str, type[ModelClient]] = {"ollama": OllamaClient, "vllm": OpenAICompatClient}


def create_model_client(backend: str, base_url: str, model: str, embedding_model: str,
                        timeout: float = 120.0, client: httpx.AsyncClient | None = None,
                        think: bool = False) -> ModelClient:
    """Factory: build the client for ``backend`` ('ollama' or 'vllm')."""
    try:
        cls = _BACKENDS[backend]
    except KeyError:
        raise ValueError(f"unknown backend '{backend}', expected one of {sorted(_BACKENDS)}") from None
    return cls(base_url, model, embedding_model, timeout, client, think)
