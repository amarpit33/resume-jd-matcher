"""Fetcher tests against saved HTML fixtures (real pages saved locally); no live network."""
from pathlib import Path

import httpx
import pytest

from api.services.jd_fetcher import (JDFetcher, JDFetchError, extract_jd, html_fragment_to_text,
                                     looks_like_js_shell, normalize_url, redirected_away, validate_extracted)

HTML = Path(__file__).parent / "fixtures" / "html"
GH_URL = "https://job-boards.greenhouse.io/definitivehcindia/jobs/6151223004"
LEVER_URL = "https://jobs.lever.co/zimperium/6e0239b1-9fa6-4967-89a3-f90889dde23b"
LI_URL = "https://www.linkedin.com/jobs/view/4448186752/"


def page(name: str) -> str:
    return (HTML / name).read_text(errors="replace")


def fetcher_for(routes: dict, renderer=None, **kw) -> tuple[JDFetcher, list[str]]:
    """JDFetcher over a MockTransport. ``routes`` maps URL -> fixture name | httpx.Response | Exception."""
    hits: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        hits.append(url)
        target = routes.get(url)
        if target is None:
            return httpx.Response(404)
        if isinstance(target, Exception):
            raise target
        if isinstance(target, httpx.Response):
            return target
        return httpx.Response(200, text=page(target), headers={"content-type": "text/html; charset=utf-8"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)
    return JDFetcher(client, renderer=renderer, **kw), hits


# ---------------------------------------------------------------- pure extraction on real pages

def test_greenhouse_page_extracts_via_trafilatura():
    text, title, method = extract_jd(page("greenhouse_lead_backend.html"), GH_URL)
    assert method == "trafilatura" and title == "Lead Backend Developer"
    cleaned = validate_extracted(text, 200, 10_000)
    assert "Kubernetes" in cleaned and "Python" in cleaned
    assert "Create a Job Alert" not in cleaned and "indicates a required field" not in cleaned


def test_lever_page_extracts_via_jsonld():
    text, title, method = extract_jd(page("lever_zimperium.html"), LEVER_URL)
    assert method == "json-ld"
    assert title.startswith("Senior AI Engineer/Python Developer")
    assert text.startswith(title) and "Zimperium" in text.split("\n")[0]
    assert "Django" in text and "<" not in text


def test_linkedin_guest_page_extracts():
    text, _, method = extract_jd(page("linkedin_guest_job.html"), LI_URL)
    cleaned = validate_extracted(text, 200, 10_000)
    assert "Python" in cleaned and len(cleaned) > 1000
    assert "has hired for this role" not in cleaned


def test_careers_homepage_is_rejected_not_passed_through():
    """A closed posting's redirect target (a careers homepage) must not be treated as a JD."""
    text, _, _ = extract_jd(page("scopely_careers_home.html"), "https://careers.scopely.com/us/en")
    with pytest.raises(JDFetchError) as e:
        validate_extracted(text, 200, 10_000)
    assert e.value.code == "extraction_failed" and "pasting the jd text" in e.value.message.lower()


def test_login_wall_detected():
    text, _, _ = extract_jd(page("linkedin_authwall.html"), LI_URL)
    with pytest.raises(JDFetchError) as e:
        validate_extracted(text, 200, 10_000)
    assert e.value.code == "login_wall"


def test_js_shell_detection():
    assert looks_like_js_shell(page("js_shell.html"))
    assert not looks_like_js_shell(page("greenhouse_lead_backend.html"))


def test_html_fragment_to_text_keeps_list_structure():
    assert html_fragment_to_text("<p>About</p><ul><li>Python</li><li>Go</li></ul>") == "About\n- Python\n- Go"


@pytest.mark.parametrize("raw,expected", [
    ("https://www.linkedin.com/jobs/search/?currentJobId=4448186752&keywords=python", LI_URL),
    ("https://in.linkedin.com/jobs/view/senior-backend-developer-python-at-mishipay-4448186752", LI_URL),
    ("jobs.lever.co/acme/123#apply", "https://jobs.lever.co/acme/123"),
    ("HTTPS://Job-Boards.Greenhouse.io/x/jobs/1?gh_jid=1", "https://job-boards.greenhouse.io/x/jobs/1?gh_jid=1"),
])
def test_normalize_url(raw, expected):
    assert normalize_url(raw) == expected


@pytest.mark.parametrize("raw", ["ftp://x.com/job", "not a url", "javascript:alert(1)", ""])
def test_normalize_url_rejects(raw):
    with pytest.raises(JDFetchError) as e:
        normalize_url(raw)
    assert e.value.code == "invalid_url"


def test_redirected_away():
    gh = "https://job-boards.greenhouse.io/scopely/jobs/5298509008?gh_jid=5298509008"
    assert redirected_away(gh, "https://careers.scopely.com/us/en")
    assert redirected_away("https://job-boards.greenhouse.io/d/jobs/5751581004",
                           "https://job-boards.greenhouse.io/d?error=true")
    assert not redirected_away(gh, gh)
    assert not redirected_away("https://acme.com/careers/backend", "https://acme.com/careers/backend/")


# ---------------------------------------------------------------- fetch flow (mocked HTTP)

async def test_fetch_success_and_cache_by_url():
    f, hits = fetcher_for({GH_URL: "greenhouse_lead_backend.html"})
    jd1 = await f.fetch(GH_URL)
    jd2 = await f.fetch(GH_URL + "#apply")      # same URL after normalisation
    assert (jd1.renderer, jd1.cached, jd2.cached) == ("http", False, True)
    assert jd1.text == jd2.text
    assert hits.count(GH_URL) == 1              # not fetched twice
    assert hits.count("https://job-boards.greenhouse.io/robots.txt") == 1


async def test_failures_are_cached_briefly_too():
    f, hits = fetcher_for({GH_URL: httpx.Response(429, headers={"Retry-After": "30"})})
    for _ in range(2):
        with pytest.raises(JDFetchError) as e:
            await f.fetch(GH_URL)
        assert e.value.code == "rate_limited" and "30s" in e.value.message
    assert hits.count(GH_URL) == 1


@pytest.mark.parametrize("response,code", [
    (httpx.Response(403), "blocked"), (httpx.Response(999), "blocked"), (httpx.Response(429), "rate_limited"),
    (httpx.Response(410), "not_found"), (httpx.Response(500), "http_error"),
    (httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"}), "not_html"),
    (httpx.ReadTimeout("slow"), "timeout"), (httpx.ConnectError("dns"), "network"),
])
async def test_http_failures_map_to_clear_codes(response, code):
    f, _ = fetcher_for({GH_URL: response})
    with pytest.raises(JDFetchError) as e:
        await f.fetch(GH_URL)
    assert e.value.code == code
    if code not in ("not_found",):
        assert "pasting the jd text" in e.value.message.lower()


async def test_closed_posting_redirect_is_reported():
    closed = "https://job-boards.greenhouse.io/scopely/jobs/5298509008"
    f, _ = fetcher_for({closed: httpx.Response(302, headers={"location": "https://careers.scopely.com/us/en"}),
                        "https://careers.scopely.com/us/en": "scopely_careers_home.html"})
    with pytest.raises(JDFetchError) as e:
        await f.fetch(closed)
    assert e.value.code == "not_found" and "closed" in e.value.message


async def test_login_redirect_is_login_wall():
    f, _ = fetcher_for({LI_URL: httpx.Response(302, headers={"location": "https://www.linkedin.com/authwall?x=1"}),
                        "https://www.linkedin.com/authwall?x=1": "linkedin_authwall.html"}, respect_robots=False)
    with pytest.raises(JDFetchError) as e:
        await f.fetch(LI_URL)
    assert e.value.code == "login_wall"


async def test_robots_disallow_is_respected_and_optional():
    robots = httpx.Response(200, text="User-agent: *\nDisallow: /", headers={"content-type": "text/plain"})
    routes = {"https://www.linkedin.com/robots.txt": robots, LI_URL: "linkedin_guest_job.html"}
    f, hits = fetcher_for(routes)
    with pytest.raises(JDFetchError) as e:
        await f.fetch(LI_URL)
    assert e.value.code == "robots_disallowed" and LI_URL not in hits
    f2, _ = fetcher_for(routes, respect_robots=False)
    assert "Python" in (await f2.fetch(LI_URL)).text


async def test_js_shell_falls_back_to_renderer():
    rendered = []

    async def fake_render(url, timeout):
        rendered.append(url)
        return page("greenhouse_lead_backend.html")

    url = "https://careers.example.com/jobs/12345"
    f, _ = fetcher_for({url: "js_shell.html"}, renderer=fake_render)
    jd = await f.fetch(url)
    assert jd.renderer == "playwright" and rendered == [url] and "Kubernetes" in jd.text


async def test_js_shell_without_playwright_degrades_with_message(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_playwright(name, *a, **kw):
        if name.startswith("playwright"):
            raise ImportError(name)
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_playwright)
    from api.services.jd_fetcher import playwright_render

    url = "https://careers.example.com/jobs/12345"
    f, _ = fetcher_for({url: "js_shell.html"}, renderer=playwright_render)
    with pytest.raises(JDFetchError) as e:
        await f.fetch(url)
    assert e.value.code == "js_required" and "playwright install" in e.value.message


async def test_renderer_not_used_for_normal_pages():
    async def boom(url, timeout):
        raise AssertionError("renderer should not run")

    f, _ = fetcher_for({GH_URL: "scopely_careers_home.html"}, renderer=boom)
    with pytest.raises(JDFetchError) as e:    # big page, not a JS shell: no browser fallback
        await f.fetch(GH_URL)
    assert e.value.code == "extraction_failed"
