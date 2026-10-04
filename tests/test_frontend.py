"""Streamlit UI flow tests (AppTest, API mocked at httpx.request)."""
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "frontend" / "app.py")
JD = "Senior Backend Engineer\nResponsibilities: build Python services. Requirements: 5+ years of experience. " * 4


def fake_api(fetch_ok: bool = True, history: list | None = None):
    def request(method, url, **kw):
        req = httpx.Request(method, url)
        if url.endswith("/healthz"):
            return httpx.Response(200, json={"model_ok": True, "model": "m", "backend": "ollama", "client": "x"},
                                  request=req)
        if url.endswith("/jd/fetch"):
            if fetch_ok:
                return httpx.Response(200, request=req, json={
                    "url": kw["json"]["url"], "text": JD, "title": "SBE", "method": "json-ld",
                    "renderer": "http", "cached": False, "chars": len(JD)})
            return httpx.Response(422, request=req, json={"detail": {
                "code": "login_wall", "message": "This page needs you to sign in to see the full job."}})
        if url.endswith("/history"):
            return httpx.Response(200, json=history or [], request=req)
        return httpx.Response(404, request=req)
    return request


@pytest.fixture
def app(monkeypatch):
    def make(**kw):
        monkeypatch.setattr(httpx, "request", fake_api(**kw))
        return AppTest.from_file(APP, default_timeout=60)
    return make


def analyze_button(at):
    return next(b for b in at.button if b.label == "Analyze")


def test_paste_mode_gates_analyze_on_resume_and_length(app):
    at = app().run()
    assert not at.exception
    assert analyze_button(at).disabled
    at.text_area(key="jd_paste").input("too short").run()
    assert analyze_button(at).disabled
    assert any("at least 200 characters" in c.value for c in at.caption)


def test_link_mode_fetch_fills_editable_preview(app):
    at = app()
    at.session_state["jd_mode"] = "Job posting link"
    at.run()
    at.text_input(key="jd_url").input("https://jobs.lever.co/acme/123").run()
    next(b for b in at.button if b.label == "Fetch & Preview").click().run()
    assert not at.exception and not at.error
    assert at.text_area(key="jd_preview").value == JD
    assert any("Extracted" in c.value and "json-ld" in c.value for c in at.caption)


def test_link_mode_fetch_failure_shows_banner_with_paste_hint(app):
    at = app(fetch_ok=False)
    at.session_state["jd_mode"] = "Job posting link"
    at.run()
    at.text_input(key="jd_url").input("https://www.linkedin.com/jobs/view/1/").run()
    next(b for b in at.button if b.label == "Fetch & Preview").click().run()
    assert at.error and "sign in" in at.error[0].value and "Paste JD" in at.error[0].value
    assert analyze_button(at).disabled


def test_history_tab_lists_rows(app):
    rows = [{"id": 1, "created_at": "2026-10-04T10:00:00+00:00", "company": "A", "role": "SWE", "resume_name": "v1.pdf",
             "overall_score": 55.0, "skills_score": 50.0, "experience_score": 60.0, "seniority_score": 100.0},
            {"id": 2, "created_at": "2026-10-04T11:00:00+00:00", "company": "B", "role": "SWE", "resume_name": "v2.pdf",
             "overall_score": 71.0, "skills_score": 70.0, "experience_score": 60.0, "seniority_score": 100.0}]
    at = app(history=rows).run()
    assert not at.exception
    df = at.dataframe[0].value
    assert list(df["id"]) == [2, 1]  # default sort: overall score, best first
