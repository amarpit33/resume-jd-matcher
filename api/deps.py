"""FastAPI dependencies; tests override these with fakes."""
from __future__ import annotations

from fastapi import Request

from api.config import Settings
from api.services.jd_fetcher import JDFetcher
from api.services.model_client import StructuredLLM
from api.services.storage import Storage


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_llm(request: Request) -> StructuredLLM:
    return request.app.state.llm


def get_storage(request: Request) -> Storage:
    return request.app.state.storage


def get_fetcher(request: Request) -> JDFetcher:
    return request.app.state.fetcher
