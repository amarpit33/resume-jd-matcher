"""FastAPI app: ``uvicorn api.main:app --port 8010``."""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from api.config import get_settings
from api.routers import analyze, history
from api.services.jd_fetcher import JDFetcher, playwright_render
from api.services.model_client import create_llm
from api.services.storage import Storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.settings = settings
    app.state.llm = create_llm(settings)
    app.state.storage = Storage(settings.db_path)
    app.state.fetcher = JDFetcher(
        timeout=settings.jd_fetch_timeout_seconds, min_chars=settings.min_jd_chars, max_chars=settings.max_jd_chars,
        cache_ttl=settings.jd_fetch_cache_ttl_seconds, respect_robots=settings.jd_fetch_respect_robots,
        renderer=playwright_render if settings.jd_fetch_use_playwright else None)
    logging.getLogger(__name__).info("model %s via %s (client: %s)", settings.llm_model, settings.llm_backend,
                                     app.state.llm.source)
    try:
        yield
    finally:
        await app.state.fetcher.aclose()
        await app.state.llm.aclose()
        app.state.storage.close()


app = FastAPI(title="resume-jd-matcher", version="0.1.0", lifespan=lifespan,
              description="Match a resume against a job description with a local LLM.")
app.include_router(analyze.router)
app.include_router(history.router)


@app.get("/healthz")
async def healthz(request: Request) -> dict[str, object]:
    """Model reachability plus which model/client is in use."""
    llm = request.app.state.llm
    return {"model_ok": await llm.health(), "model": llm.model, "backend": llm.backend, "client": llm.source}
