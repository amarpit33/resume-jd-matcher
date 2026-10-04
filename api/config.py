"""Application settings.

Model resolution order (highest wins):
1. environment variables / ``.env``
2. ``config/detected.json`` of a sibling ``local-llm-platform`` checkout
   (written by its ``scripts/detect_environment.py``)
3. built-in defaults (Ollama on localhost; the model must then be set explicitly)
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigError(RuntimeError):
    """Raised when the model cannot be resolved from env, the platform, or defaults."""


def load_platform_selection(platform_dir: Path) -> dict[str, Any]:
    """Return the ``selection`` block of the platform's detection report, or ``{}``."""
    try:
        return json.loads((platform_dir / "config" / "detected.json").read_text()).get("selection", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


class Settings(BaseSettings):
    """Typed, env-overridable configuration."""

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore", protected_namespaces=())

    # reuse of local-llm-platform (model client + detected model); optional
    local_llm_platform_dir: Path = PROJECT_ROOT.parent / "local-llm-platform"

    # model serving; None means "take it from the platform's detection report"
    llm_backend: Literal["ollama", "vllm"] | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    embedding_model: str | None = None  # ranks candidate bullets; optional (lexical fallback)
    llm_timeout_seconds: float = 300.0
    llm_max_concurrency: int = 2  # in-flight model requests; more just queue inside Ollama
    llm_num_ctx: int = 8192  # Ollama context window; resume + JD prompts need ~3-5k tokens

    # storage
    db_path: Path = PROJECT_ROOT / "data" / "matcher.db"
    max_upload_bytes: int = 5 * 1024 * 1024

    # input limits (keep prompts inside the context window)
    max_resume_chars: int = 12_000
    max_jd_chars: int = 10_000
    min_jd_chars: int = 200

    # JD fetching from URLs
    jd_fetch_timeout_seconds: float = 10.0
    jd_fetch_respect_robots: bool = True   # LinkedIn's robots.txt disallows all bots: set false to fetch it
    jd_fetch_cache_ttl_seconds: float = 3600.0
    jd_fetch_use_playwright: bool = True   # only used if installed, and only for JS-shell pages

    # scoring weights (normalised at use, so they need not sum to 1)
    weight_skills: float = 0.45
    weight_experience: float = 0.35
    weight_seniority: float = 0.20
    max_rewrite_bullets: int = 4

    @model_validator(mode="after")
    def _fill_from_platform(self) -> "Settings":
        detected = load_platform_selection(self.local_llm_platform_dir)
        self.llm_backend = self.llm_backend or detected.get("backend") or "ollama"
        self.llm_base_url = self.llm_base_url or detected.get("base_url") or "http://localhost:11434"
        self.llm_model = self.llm_model or detected.get("chat_model")
        self.embedding_model = self.embedding_model or detected.get("embedding_model") or ""
        if not self.llm_model:
            raise ConfigError(
                "No model configured. Set LLM_MODEL in .env (e.g. `ollama pull phi3.5` then "
                "LLM_MODEL=phi3.5), or point LOCAL_LLM_PLATFORM_DIR at a local-llm-platform checkout.")
        return self


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
