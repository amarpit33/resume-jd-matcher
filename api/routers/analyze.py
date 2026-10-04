"""POST /analyze (resume + JD text or URL) and POST /jd/fetch (URL -> editable JD preview)."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from api.config import Settings
from api.deps import get_fetcher, get_llm, get_settings_dep, get_storage
from api.schemas import AnalysisResult
from api.services.jd_fetcher import JDFetcher, JDFetchError
from api.services.jd_parser import JDTextError, clean_jd_text
from api.services.matcher import PipelineInput, analyze
from api.services.model_client import StructuredLLM
from api.services.resume_parser import ResumeParseError, content_hash, extract_text
from api.services.storage import Storage

router = APIRouter(tags=["analyze"])


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status, detail={"code": code, "message": message})


class FetchRequest(BaseModel):
    url: str


class FetchResponse(BaseModel):
    url: str
    text: str
    title: str
    method: str
    renderer: str
    cached: bool
    chars: int


class AnalyzeResponse(AnalysisResult):
    analysis_id: int
    jd_source: Literal["text", "url"]
    jd_url: str | None = None


@router.post("/jd/fetch", response_model=FetchResponse)
async def fetch_jd(body: FetchRequest, fetcher: JDFetcher = Depends(get_fetcher)) -> FetchResponse:
    """Fetch a job posting and return the extracted text for the user to check/edit before analysis."""
    try:
        jd = await fetcher.fetch(body.url)
    except JDFetchError as exc:
        raise _error(422, exc.code, exc.message) from None
    return FetchResponse(url=jd.url, text=jd.text, title=jd.title, method=jd.method, renderer=jd.renderer,
                         cached=jd.cached, chars=len(jd.text))


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_resume(
    resume: UploadFile = File(..., description="PDF or DOCX resume"),
    jd_text: str | None = Form(None, description="pasted JD text (exclusive with jd_url)"),
    jd_url: str | None = Form(None, description="job posting URL to fetch (exclusive with jd_text)"),
    source_url: str | None = Form(None, description="where pasted/edited jd_text came from; stored only"),
    company: str | None = Form(None),
    role: str | None = Form(None),
    settings: Settings = Depends(get_settings_dep),
    llm: StructuredLLM = Depends(get_llm),
    storage: Storage = Depends(get_storage),
    fetcher: JDFetcher = Depends(get_fetcher),
) -> AnalyzeResponse:
    """Run the full analysis; the response includes the JD text actually used (``jd_text``)."""
    jd_text = (jd_text or "").strip() or None
    jd_url = (jd_url or "").strip() or None
    if bool(jd_text) == bool(jd_url):
        raise _error(422, "jd_input", "Provide exactly one of jd_text or jd_url.")

    data = await resume.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise _error(413, "resume_too_large", f"Resume is larger than {settings.max_upload_bytes // 1024 // 1024} MB.")
    try:
        resume_text = extract_text(resume.filename or "resume", data)
    except ResumeParseError as exc:
        raise _error(422, "resume_unreadable", str(exc)) from None

    raw = jd_text
    if jd_url:
        try:
            raw = (await fetcher.fetch(jd_url)).text
        except JDFetchError as exc:
            raise _error(422, exc.code, exc.message) from None
    try:
        text = clean_jd_text(raw or "", settings.min_jd_chars, settings.max_jd_chars)
    except JDTextError as exc:
        raise _error(422, "jd_too_short", str(exc)) from None

    try:
        result = await analyze(llm, storage, PipelineInput(resume_text[: settings.max_resume_chars], text), settings)
    except llm.ModelUnavailable as exc:
        raise _error(503, "model_unavailable", f"The local model isn't reachable: {exc}") from None
    except llm.ModelError as exc:
        raise _error(502, "model_error", str(exc)) from None

    url = jd_url or source_url
    payload = result.model_dump() | {"jd_url": url}
    analysis_id = storage.save_analysis(
        payload, resume_hash=content_hash(resume_text), jd_hash=content_hash(text), resume_name=resume.filename,
        company=(company or "").strip() or result.jd.company or None,
        role=(role or "").strip() or result.jd.title or None)
    return AnalyzeResponse(**payload, analysis_id=analysis_id, jd_source="url" if jd_url else "text")
