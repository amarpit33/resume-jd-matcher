"""Job posting URL -> clean JD text. Nothing else: parsing stays in :mod:`jd_parser`.

Order of attempts:
1. Plain HTTP fetch with a browser User-Agent (10 s timeout), after a robots.txt check.
2. Extraction: schema.org ``JobPosting`` JSON-LD if the page has it (LinkedIn guest pages,
   Greenhouse, Lever, Naukri, most ATSs publish it for Google Jobs), else ``trafilatura``
   main-content extraction. No per-site CSS selectors.
3. Only if the page is an obvious JS shell / near-empty: headless Chromium via Playwright,
   which is optional and imported lazily.

Every failure becomes a :class:`JDFetchError` with a code and a user-facing message.
Garbage is never returned: the text must pass login-wall detection, a JD-vocabulary check
and the same minimum-length gate as pasted text.
"""
from __future__ import annotations

import json
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import parse_qs, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx

from api.services.jd_parser import JDTextError, clean_jd_text

USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")
ROBOTS_AGENT = "resume-jd-matcher"  # product token matched against robots.txt groups
_HEADERS = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9"}
_MAX_BYTES = 4 * 1024 * 1024
PASTE_HINT = "Try pasting the JD text instead."

_LOGIN_WALL = re.compile(
    r"sign in to (view|see|apply)|join (linkedin|now) to see|log ?in to (continue|view)|"
    r"you must be (logged|signed) in|create an account to (view|continue)|authwall|"
    r"please (log|sign) in|session has expired|verify you are (a )?human|are you a robot|captcha",
    re.IGNORECASE)
# A real JD hits at least three of these; a careers homepage or blog post rarely does.
_JD_VOCAB = re.compile(
    r"\b(responsibilit|requirement|qualification|years? of (?:\w+ )?experience|you will|you'll|what you|"
    r"we(?: are|'re) looking for|nice to have|preferred|must have|skills|about the (?:role|job|position)|"
    r"job description|proficien|familiar(?:ity)? with|knowledge of|experience (?:with|in)|benefits)\w*",
    re.IGNORECASE)
_JOB_ID = re.compile(r"(?<![0-9a-f])([0-9]{5,}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(?![0-9a-f])",
                     re.IGNORECASE)
_JS_SHELL = re.compile(
    r"enable javascript|javascript is (required|disabled)|<div id=\"(root|app|__next)\"></div>|"
    r"you need to enable javascript", re.IGNORECASE)


class JDFetchError(Exception):
    """Fetching or extracting failed; ``message`` is safe to show to the user."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class FetchedJD:
    url: str
    text: str
    title: str
    method: str          # 'json-ld' | 'trafilatura'
    renderer: str        # 'http' | 'playwright'
    cached: bool = False


Renderer = Callable[[str, float], Awaitable[str]]


# ---------------------------------------------------------------- URL handling

def normalize_url(url: str) -> str:
    """Validate and canonicalise a job URL (drop fragment; LinkedIn search links -> job view).

    Raises:
        JDFetchError: not an http(s) URL.
    """
    url = url.strip()
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url if re.match(r"^[\w.-]+\.[a-z]{2,}(/|$)", url, re.IGNORECASE) else url
    parts = urlsplit(url)
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        raise JDFetchError("invalid_url", "That doesn't look like a job posting link (need an http(s) URL).")
    host = parts.hostname.lower()
    if host.endswith("linkedin.com"):
        job_id = parse_qs(parts.query).get("currentJobId", [None])[0]
        m = re.search(r"/jobs/view/(?:[^/]*?-)?(\d+)", parts.path)
        if job_id or m:  # search/collection links carry the job in ?currentJobId=
            return f"https://www.linkedin.com/jobs/view/{job_id or m.group(1)}/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", parts.query, ""))


# ---------------------------------------------------------------- extraction (pure)

def html_fragment_to_text(fragment: str) -> str:
    """Convert a JD's HTML description to text, keeping list items and paragraphs on their own lines."""
    import lxml.html

    if "<" not in fragment:
        return fragment
    root = lxml.html.fragment_fromstring(fragment, create_parent="div")
    for br in root.iter("br"):
        br.tail = "\n" + (br.tail or "")
    for el in root.iter("li"):
        el.text = "- " + (el.text or "")
    for el in root.iter("p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr"):
        el.tail = "\n" + (el.tail or "")
    text = root.text_content()
    return re.sub(r"\n\s*\n+", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()


def _iter_jsonld(node: Any):
    if isinstance(node, list):
        for item in node:
            yield from _iter_jsonld(item)
    elif isinstance(node, dict):
        yield node
        if "@graph" in node:
            yield from _iter_jsonld(node["@graph"])


def extract_jsonld_jobposting(html: str) -> tuple[str, str] | None:
    """Return ``(text, title)`` from a schema.org JobPosting block, or None."""
    import html as htmlmod

    for raw in re.findall(r"<script[^>]+application/ld\+json[^>]*>(.*?)</script>", html, re.S | re.I):
        try:
            data = json.loads(raw.strip())
        except json.JSONDecodeError:
            continue
        for node in _iter_jsonld(data):
            types = node.get("@type")
            if "JobPosting" not in (types if isinstance(types, list) else [types]):
                continue
            desc = node.get("description") or ""
            if not isinstance(desc, str) or not desc.strip():
                continue
            title = str(node.get("title") or "").strip()
            org = node.get("hiringOrganization")
            org_name = org.get("name", "") if isinstance(org, dict) else str(org or "")
            header = " - ".join(x for x in (title, org_name) if x)
            # some sites double-escape the description
            body = html_fragment_to_text(htmlmod.unescape(desc) if "&lt;" in desc else desc)
            return (f"{header}\n\n{body}" if header else body), title
    return None


def extract_main_content(html: str, url: str) -> tuple[str, str]:
    """``(text, title)`` via trafilatura; ('', '') when it finds nothing."""
    import trafilatura

    text = trafilatura.extract(html, url=url, include_comments=False, include_tables=True,
                               favor_recall=True, deduplicate=True) or ""
    meta = trafilatura.extract_metadata(html)
    title = (meta.title if meta and meta.title else "") or ""
    if title and text and not text.startswith(title):
        text = f"{title}\n\n{text}"
    return text, title


def extract_jd(html: str, url: str) -> tuple[str, str, str]:
    """``(text, title, method)``: JSON-LD JobPosting first, trafilatura otherwise."""
    if found := extract_jsonld_jobposting(html):
        return found[0], found[1], "json-ld"
    text, title = extract_main_content(html, url)
    return text, title, "trafilatura"


def redirected_away(requested: str, final: str) -> bool:
    """True if a request for a specific job ended up on a page that no longer names that job.

    ATSs answer closed postings with a redirect to the board or careers homepage (often 200 OK).
    """
    ids = _JOB_ID.findall(urlsplit(requested).path + "?" + urlsplit(requested).query)
    if not ids or requested == final:
        return False
    return not any(i.lower() in final.lower() for i in ids) or "error=true" in final


def visible_text_length(html: str) -> int:
    """Rough length of human-visible text (scripts/styles stripped)."""
    body = re.sub(r"<(script|style|noscript|template)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    return len(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body)).strip())


def looks_like_js_shell(html: str) -> bool:
    """Near-empty body or an explicit 'enable JavaScript' app shell."""
    return visible_text_length(html) < 400 or bool(_JS_SHELL.search(html) and visible_text_length(html) < 2000)


def validate_extracted(text: str, min_chars: int, max_chars: int) -> str:
    """Gate extracted text: no login walls, must read like a JD, min length. Returns cleaned text.

    Raises:
        JDFetchError: login_wall / extraction_failed.
    """
    head = text[:1500]
    try:
        cleaned = clean_jd_text(text, min_chars, max_chars)
    except JDTextError:
        if _LOGIN_WALL.search(head):
            raise JDFetchError("login_wall", f"This page needs you to sign in to see the full job. {PASTE_HINT}")
        raise JDFetchError("extraction_failed", f"Couldn't extract this page. {PASTE_HINT}") from None
    if _LOGIN_WALL.search(head) and len(cleaned) < 1500:
        raise JDFetchError("login_wall", f"This page needs you to sign in to see the full job. {PASTE_HINT}")
    if len(set(m.group(1).lower() for m in _JD_VOCAB.finditer(cleaned))) < 3:
        raise JDFetchError("extraction_failed",
                           f"Couldn't extract this page: the text doesn't look like a job description. {PASTE_HINT}")
    return cleaned


# ---------------------------------------------------------------- fetching

async def playwright_render(url: str, timeout: float) -> str:
    """Render ``url`` in headless Chromium and return the HTML.

    Raises:
        JDFetchError: Playwright is not installed or rendering failed.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise JDFetchError(
            "js_required",
            "This page builds its content with JavaScript, and the optional headless browser isn't installed "
            "(pip install playwright && playwright install chromium). " + PASTE_HINT) from None
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                page = await browser.new_page(user_agent=USER_AGENT)
                await page.goto(url, timeout=timeout * 1000, wait_until="domcontentloaded")
                try:
                    await page.wait_for_load_state("networkidle", timeout=timeout * 1000)
                except Exception:  # long-polling pages never go idle; use what rendered
                    pass
                return await page.content()
            finally:
                await browser.close()
    except JDFetchError:
        raise
    except Exception as exc:
        raise JDFetchError("js_required", f"The headless browser couldn't load this page ({exc}). {PASTE_HINT}")


class JDFetcher:
    """Fetches and extracts JD text; caches results (and failures, briefly) per URL for the process lifetime."""

    def __init__(self, client: httpx.AsyncClient | None = None, *, timeout: float = 10.0, min_chars: int = 200,
                 max_chars: int = 10_000, cache_ttl: float = 3600.0, error_ttl: float = 60.0,
                 respect_robots: bool = True, renderer: Renderer | None = playwright_render) -> None:
        self._owns_client = client is None
        self._http = client or httpx.AsyncClient(headers=_HEADERS, follow_redirects=True, timeout=timeout)
        self.timeout = timeout
        self.min_chars, self.max_chars = min_chars, max_chars
        self.cache_ttl, self.error_ttl = cache_ttl, error_ttl
        self.respect_robots = respect_robots
        self.renderer = renderer
        self._cache: dict[str, tuple[float, FetchedJD | JDFetchError]] = {}
        self._robots: dict[str, RobotFileParser | None] = {}

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    async def fetch(self, url: str) -> FetchedJD:
        """URL -> validated JD text.

        Raises:
            JDFetchError: with ``code`` one of invalid_url, robots_disallowed, timeout, network, blocked,
                rate_limited, not_found, http_error, not_html, login_wall, js_required, extraction_failed.
        """
        url = normalize_url(url)
        now = time.monotonic()
        if (hit := self._cache.get(url)) and hit[0] > now:
            if isinstance(hit[1], JDFetchError):
                raise hit[1]
            return replace(hit[1], cached=True)
        try:
            result = await self._fetch_uncached(url)
        except JDFetchError as exc:
            if exc.code not in ("invalid_url",):
                self._cache[url] = (now + self.error_ttl, exc)
            raise
        self._cache[url] = (now + self.cache_ttl, result)
        return result

    async def _fetch_uncached(self, url: str) -> FetchedJD:
        if self.respect_robots and not await self._robots_allows(url):
            raise JDFetchError("robots_disallowed",
                               f"This site's robots.txt asks automated tools not to fetch this page. {PASTE_HINT}")
        html, final_url = await self._get(url)
        if re.search(r"authwall|/login|/signin|/uas/|accounts\.google", final_url, re.I):
            raise JDFetchError("login_wall", f"This page needs you to sign in to see the full job. {PASTE_HINT}")
        if redirected_away(url, final_url):
            raise JDFetchError("not_found", f"That posting seems to be closed: the link now redirects to {final_url}.")

        text, title, method = extract_jd(html, final_url)
        try:
            return FetchedJD(url, validate_extracted(text, self.min_chars, self.max_chars), title, method, "http")
        except JDFetchError as exc:
            if exc.code == "login_wall" or not looks_like_js_shell(html) or self.renderer is None:
                raise
        rendered = await self.renderer(url, max(self.timeout, 15.0))
        text, title, method = extract_jd(rendered, url)
        return FetchedJD(url, validate_extracted(text, self.min_chars, self.max_chars), title, method, "playwright")

    async def _get(self, url: str) -> tuple[str, str]:
        try:
            resp = await self._http.get(url, headers=_HEADERS, timeout=self.timeout)
        except httpx.TimeoutException:
            raise JDFetchError("timeout", f"The site didn't respond within {self.timeout:g}s. {PASTE_HINT}") from None
        except httpx.HTTPError as exc:
            raise JDFetchError("network", f"Couldn't reach the site ({type(exc).__name__}). {PASTE_HINT}") from None
        status = resp.status_code
        if status == 429:
            wait = resp.headers.get("Retry-After")
            raise JDFetchError("rate_limited", "The site is rate-limiting requests"
                               + (f" (retry after {wait}s)" if wait else "") + f". {PASTE_HINT}")
        if status in (401, 403, 999):  # 999: LinkedIn's bot block
            raise JDFetchError("blocked", f"The site blocked the request (HTTP {status}). {PASTE_HINT}")
        if status in (404, 410):
            raise JDFetchError("not_found", "That posting wasn't found; it may have expired.")
        if status >= 400:
            raise JDFetchError("http_error", f"The site returned HTTP {status}. {PASTE_HINT}")
        ctype = resp.headers.get("content-type", "text/html")
        if "html" not in ctype and "xml" not in ctype:
            raise JDFetchError("not_html", f"That link isn't a web page ({ctype.split(';')[0]}). {PASTE_HINT}")
        return resp.content[:_MAX_BYTES].decode(resp.encoding or "utf-8", errors="replace"), str(resp.url)

    async def _robots_allows(self, url: str) -> bool:
        """robots.txt check, cached per host. Unreachable/missing robots.txt means allowed."""
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            parser: RobotFileParser | None = None
            try:
                resp = await self._http.get(f"{origin}/robots.txt", headers=_HEADERS, timeout=5.0)
                if resp.status_code == 200 and "html" not in resp.headers.get("content-type", ""):
                    parser = RobotFileParser()
                    parser.parse(resp.text.splitlines())
            except httpx.HTTPError:
                parser = None
            self._robots[origin] = parser
        parser = self._robots[origin]
        return parser is None or parser.can_fetch(ROBOTS_AGENT, url)
