"""API tests: real routers and pipeline, fake LLM, in-memory storage, fixture-backed fetcher."""
import httpx
import pytest
from fastapi.testclient import TestClient

from api.config import Settings
from api.deps import get_fetcher, get_llm, get_settings_dep, get_storage
from api.main import app
from api.services.storage import Storage
from tests.conftest import SAMPLES
from tests.fakes import FakeLLM, FakeModelUnavailable
from tests.test_jd_fetcher import GH_URL, fetcher_for

RESUME = ("sample_resume.pdf", (SAMPLES / "sample_resume.pdf").read_bytes(), "application/pdf")


@pytest.fixture
def client():
    llm, storage = FakeLLM(), Storage(":memory:")
    fetcher, _ = fetcher_for({GH_URL: "greenhouse_lead_backend.html",
                              "https://blocked.example.com/jobs/12345": httpx.Response(403)})
    app.dependency_overrides.update({get_llm: lambda: llm, get_storage: lambda: storage,
                                     get_fetcher: lambda: fetcher, get_settings_dep: lambda: Settings()})
    yield TestClient(app), llm
    app.dependency_overrides.clear()


def test_analyze_with_pasted_text_then_history(client, jd_text):
    c, _ = client
    r = c.post("/analyze", files={"resume": RESUME}, data={"jd_text": jd_text, "company": "Orbitpay"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["jd_source"] == "text" and body["jd_url"] is None
    assert body["jd_text"].startswith("Senior Backend Engineer (Python)") and "Apply now" not in body["jd_text"]
    assert {"overall_match_score", "skills_matched", "skills_missing", "section_scores",
            "bullet_suggestions", "summary"} <= body.keys()

    hist = c.get("/history").json()
    assert hist[0]["id"] == body["analysis_id"] and hist[0]["company"] == "Orbitpay"
    assert hist[0]["role"] == "Senior Backend Engineer"            # filled from the parsed JD
    full = c.get(f"/history/{body['analysis_id']}").json()
    assert full["result"]["overall_match_score"] == body["overall_match_score"]
    assert c.get("/history/9999").status_code == 404


def test_analyze_with_url_returns_extracted_text(client):
    c, _ = client
    r = c.post("/analyze", files={"resume": RESUME}, data={"jd_url": GH_URL})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["jd_source"] == "url" and body["jd_url"] == GH_URL
    assert body["jd_text"].startswith("Lead Backend Developer")


@pytest.mark.parametrize("data", [{}, {"jd_text": "x" * 300, "jd_url": GH_URL}, {"jd_text": "   "}])
def test_analyze_requires_exactly_one_jd_input(client, data):
    c, llm = client
    r = c.post("/analyze", files={"resume": RESUME}, data=data)
    assert r.status_code == 422 and r.json()["detail"]["code"] == "jd_input"
    assert llm.calls == []


def test_short_jd_never_reaches_the_llm(client):
    c, llm = client
    r = c.post("/analyze", files={"resume": RESUME}, data={"jd_text": "Python dev needed. Apply now."})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "jd_too_short"
    assert llm.calls == []


def test_fetch_failure_is_a_clear_error(client):
    c, llm = client
    r = c.post("/analyze", files={"resume": RESUME}, data={"jd_url": "https://blocked.example.com/jobs/12345"})
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "blocked" and "pasting" in r.json()["detail"]["message"]
    assert llm.calls == []


def test_bad_resume_rejected(client, jd_text):
    c, _ = client
    r = c.post("/analyze", files={"resume": ("r.png", b"\x89PNG", "image/png")}, data={"jd_text": jd_text})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "resume_unreadable"


def test_model_down_is_503(client, jd_text):
    c, llm = client

    async def down(*a, **kw):
        raise FakeModelUnavailable("connection refused")

    llm.generate_json = down
    r = c.post("/analyze", files={"resume": RESUME}, data={"jd_text": jd_text})
    assert r.status_code == 503 and r.json()["detail"]["code"] == "model_unavailable"


def test_jd_fetch_preview(client):
    c, _ = client
    r = c.post("/jd/fetch", json={"url": GH_URL})
    assert r.status_code == 200
    body = r.json()
    assert body["title"] == "Lead Backend Developer" and body["chars"] == len(body["text"]) > 200
    assert c.post("/jd/fetch", json={"url": GH_URL}).json()["cached"] is True
    bad = c.post("/jd/fetch", json={"url": "ftp://nope"})
    assert bad.status_code == 422 and bad.json()["detail"]["code"] == "invalid_url"
