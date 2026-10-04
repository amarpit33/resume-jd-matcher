"""Structured-output LLM access built on local-llm-platform's ``ModelClient``.

The platform's ``model_client.py`` is loaded straight from the sibling checkout
(``LOCAL_LLM_PLATFORM_DIR``) so both projects share one backend abstraction. If
the platform is not present, a vendored copy of the same file is used, so this
project still works standalone.

On top of it, :class:`StructuredLLM` adds schema-constrained generation:
Ollama's ``format`` (JSON-schema grammar) or the OpenAI-compatible
``response_format`` (vLLM guided decoding). Responses are validated with
pydantic; free text is never parsed heuristically.
"""
from __future__ import annotations

import asyncio
import copy
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from api.config import Settings

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_PLATFORM_CLIENT = Path("services") / "api" / "services" / "model_client.py"
_MODULE_NAME = "_local_llm_platform_model_client"


def load_base_module(platform_dir: Path) -> tuple[ModuleType, str]:
    """Return the platform's ``model_client`` module and where it came from.

    Loaded by file path (not as a package) because both projects have an ``api``
    package and would otherwise collide on ``sys.path``.
    """
    path = platform_dir / _PLATFORM_CLIENT
    if path.is_file():
        if _MODULE_NAME in sys.modules:
            return sys.modules[_MODULE_NAME], str(path)
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[_MODULE_NAME] = module  # dataclasses resolve annotations via sys.modules
            try:
                spec.loader.exec_module(module)
                return module, str(path)
            except Exception:  # broken checkout: fall back rather than refuse to start
                sys.modules.pop(_MODULE_NAME, None)
                log.warning("could not load %s; using vendored model client", path, exc_info=True)
    from api.services._vendor import platform_model_client
    return platform_model_client, "vendored copy"


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve ``$ref``/``$defs`` from a pydantic JSON schema into one self-contained tree.

    Grammar converters in serving engines handle local refs unevenly; an inlined
    schema works everywhere.
    """
    defs = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].rsplit("/", 1)[-1]
                return resolve(copy.deepcopy(defs[name]))
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


class StructuredLLM:
    """Schema-constrained JSON generation over a platform ``ModelClient``."""

    def __init__(self, client: Any, base: ModuleType, backend: str, num_ctx: int = 8192,
                 source: str = "", max_concurrency: int = 2) -> None:
        self.client = client
        # Callers fan out (one call per responsibility/bullet pair). Queue here, not inside the
        # server, so waiting time doesn't count against the HTTP timeout.
        self._slots = asyncio.Semaphore(max_concurrency)
        self.ModelError: type[Exception] = base.ModelError
        self.ModelUnavailable: type[Exception] = base.ModelUnavailable
        self._base = base
        self.backend = backend
        self.num_ctx = num_ctx
        self.source = source

    @property
    def model(self) -> str:
        return self.client.model

    @property
    def embedding_model(self) -> str:
        return self.client.embedding_model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embedding vectors via the platform client.

        Raises:
            ModelError: no embedding model configured, or the backend failed.
        """
        if not self.embedding_model:
            raise self._base.ModelError("no embedding model configured")
        async with self._slots:
            return await self.client.embed(texts)

    async def generate_json(self, system: str, user: str, schema: type[T], max_tokens: int = 1024) -> T:
        """Generate an instance of ``schema``.

        Retries once on invalid output; a truncated answer is retried with double
        the token budget.

        Raises:
            ModelError: the backend failed or never produced valid output.
        """
        json_schema = inline_refs(schema.model_json_schema())
        messages = [self._base.Message("system", system), self._base.Message("user", user)]
        last_error: Exception | None = None
        for attempt in range(2):
            params = self._base.GenerationParams(temperature=0.0 if attempt == 0 else 0.3, max_tokens=max_tokens)
            async with self._slots:
                text, truncated = await self._complete(messages, params, json_schema, schema.__name__)
            try:
                return schema.model_validate_json(text)
            except ValidationError as exc:
                last_error = exc
                log.warning("invalid %s from model (attempt %d, truncated=%s): %s",
                            schema.__name__, attempt + 1, truncated, exc.errors()[:3])
                if truncated:
                    max_tokens *= 2
        raise self._base.ModelError(f"model did not return a valid {schema.__name__}: {last_error}")

    async def _complete(self, messages: list[Any], params: Any, json_schema: dict[str, Any],
                        name: str) -> tuple[str, bool]:
        payload = self.client._chat_payload(messages, params, False)
        if self.backend == "ollama":
            payload["format"] = json_schema
            payload["options"]["num_ctx"] = self.num_ctx
            data = await self.client._post("/api/chat", payload)
            return data.get("message", {}).get("content", ""), data.get("done_reason") == "length"
        payload["response_format"] = {"type": "json_schema",
                                      "json_schema": {"name": name, "schema": json_schema}}
        data = await self.client._post("/v1/chat/completions", payload)
        try:
            choice = data["choices"][0]
            return choice["message"]["content"] or "", choice.get("finish_reason") == "length"
        except (KeyError, IndexError) as exc:
            raise self._base.ModelError("malformed chat completion response") from exc

    async def health(self) -> bool:
        return await self.client.health()

    async def aclose(self) -> None:
        await self.client.aclose()


def create_llm(settings: Settings, http_client: Any = None) -> StructuredLLM:
    """Build a :class:`StructuredLLM` for the configured backend."""
    base, source = load_base_module(settings.local_llm_platform_dir)
    client = base.create_model_client(
        settings.llm_backend, settings.llm_base_url, settings.llm_model,
        embedding_model=settings.embedding_model or "", timeout=settings.llm_timeout_seconds, client=http_client, think=False)
    return StructuredLLM(client, base, settings.llm_backend, settings.llm_num_ctx, source,
                         settings.llm_max_concurrency)
