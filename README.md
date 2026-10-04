# resume-jd-matcher

[![ci](https://github.com/amarpit33/resume-jd-matcher/actions/workflows/ci.yml/badge.svg)](https://github.com/amarpit33/resume-jd-matcher/actions/workflows/ci.yml)

Match your resume against a job description with a **local, open-source LLM**: a match score with a
breakdown, the skills you're missing, and before/after rewrites of weak bullets in the job's own language
(ATS optimisation). Every rewrite is checked for claims your resume doesn't support.

- **Private by design.** The model runs in [Ollama](https://ollama.com) (or vLLM) on your machine, history is a local
  SQLite file, and both servers bind to localhost. No API keys, no resume leaves your laptop.
- **Paste a JD or drop a link.** Greenhouse, Lever, LinkedIn public pages and most careers sites. You see
  (and can edit) the extracted text before anything is analysed.
- **Not one big prompt.** Structured, multi-step pipeline. Skill overlap and seniority are computed in code;
  the LLM only does what needs language understanding, always with schema-constrained JSON output.
- **Track your tailoring.** Every analysis is saved; the History tab shows how each resume version scores per company.

Contents: [Demo](#demo) · [Quick start](#quick-start) · [User guide](#user-guide) ·
[Reading the results](#reading-the-results) · [CLI](#cli) · [API](#api) · [JD from a URL](#jd-from-a-url) ·
[How it works](#how-it-works) · [Configuration](#configuration) · [Performance](#performance) ·
[Troubleshooting](#troubleshooting) · [Development](#development) · [Limitations](#limitations)

---

## Demo

> **Screenshot placeholder:** the UI's result view (score gauge, missing-skill tags, side-by-side bullet
> rewrites). Drop a capture at `docs/screenshot.png` and replace this note with `![UI](docs/screenshot.png)`.

The output below is **real**: the fictional sample resume (`data/sample_resumes/`) against the sample JD
(`data/sample_jds/backend_senior.txt`), run on an 8 GB M1 with `qwen3.5:4b`. Full report:
[docs/demo.md](docs/demo.md). Raw API-shaped JSON: [docs/sample_response.json](docs/sample_response.json).

```text
=== Senior Backend Engineer @ Orbitpay ===
OVERALL 68%   skills 45 | experience 79 | seniority 100 (aligned: you 6.3y, JD senior, min 5.0)

SKILLS MATCHED (6): Python, Django, PostgreSQL, Redis, Docker, AWS
  required missing: FastAPI, Kafka, Kubernetes, CI/CD
  nice-to-have missing: Go, Terraform, Prometheus, Grafana

EXPERIENCE vs RESPONSIBILITIES
  [STRONG ] Optimize PostgreSQL schemas and queries for scale.  (tuned PostgreSQL queries for reports)
  [PARTIAL] Build event-driven workflows on Kafka for settlement and reconciliation.  (Migrated
  batch reconciliation jobs to a task queue)
  ...

BULLET REWRITES
  BEFORE: Moved nightly reconciliation jobs from cron scripts to Celery workers, cutting failures by 60%
    AFTER 1: Designed and operated Python microservices using Celery for nightly reconciliation workflows.
    AFTER 2: Improved service reliability by migrating cron scripts to asynchronous Python workers,
    reducing failures by 60%.
    ! rewrite 1: uses JD terms your resume doesn't support (operated, microservices, workflows); keep only if true
    ! rewrite 2: uses JD terms your resume doesn't support (reliability); keep only if true
```

Note the `!` lines: rewrite 1 over-sells ("microservices", "operated"), and the tool flags it.

Reproduce it yourself in one command once [set up](#quick-start):

```bash
.venv/bin/python -m api.cli data/sample_resumes/sample_resume.pdf --jd-file data/sample_jds/backend_senior.txt
```

## Quick start

**You need:** macOS or Linux, Python 3.10+, [Ollama](https://ollama.com/download) running, ~4 GB free RAM for
a 4B model (8 GB machine is fine).

```bash
git clone https://github.com/amarpit33/resume-jd-matcher.git
cd resume-jd-matcher

# 1. Python env
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
#    (or with uv: uv venv .venv && uv pip install -p .venv -r requirements.txt)

# 2. Models: a chat model, plus an embedding model (optional, improves experience matching)
ollama pull qwen3.5:4b          # 8 GB RAM. With 16 GB+, an 8B model (e.g. llama3.1:8b) gives better rewrites
ollama pull nomic-embed-text

# 3. Config
cp .env.example .env
echo "LLM_MODEL=qwen3.5:4b" >> .env
echo "EMBEDDING_MODEL=nomic-embed-text" >> .env

# 4. Run
./start.sh
```

Open **http://localhost:8501** (UI). The API is on http://localhost:8010 with interactive docs at `/docs`.
Ctrl-C stops both.

> **Using [local-llm-platform](https://github.com/amarpit33/local-llm-platform)?** If it's checked out next to
> this repo (`../local-llm-platform`), skip step 3: its hardware detection (`config/detected.json`) picks the
> model, and its `model_client.py` is loaded and reused directly. Point elsewhere with `LOCAL_LLM_PLATFORM_DIR`.

## User guide

### 1. Analyze a resume against a job

1. **Upload your resume**: PDF or DOCX, text-based (not a scanned image). Company and role are optional;
   if left blank they're taken from the JD.
2. **Give it the job description**, either way:
   - **Paste JD**: paste the full posting. Include requirements *and* responsibilities.
   - **Job posting link**: paste the URL and click **Fetch & Preview**. The extracted text appears in an
     editable box. Skim it: if the scrape picked up something extra (or missed a section), fix it right there.
     If the site can't be read (login wall, blocked, JS-only page) you get a red banner saying why. Switch
     to **Paste JD** and paste it instead.
3. Click **Analyze**. It stays disabled until there's a resume and at least 200 characters of JD text,
   so junk never reaches the model. A run takes **3-6 minutes** on a laptop-class model; the page shows a spinner.

### 2. Tailor and re-check

The intended loop:

1. Analyze, then read the missing skills and bullet rewrites.
2. Edit your resume (keep only what's true, see [warnings](#-warnings-on-rewrites)).
3. Upload the new version against the **same JD**. The JD isn't re-parsed (it's cached by text), so the
   second run is faster, and the score change reflects your edits.
4. Repeat per company. **History** lists every run; sort by score, click a row to reopen the full result.

### 3. Tips

- Keep real resumes in `data/my_resumes/` and saved JDs in `data/my_jds/`. Both are git-ignored, as is the
  history database `data/matcher.db`.
- To check what a link extracts without running the model:
  `.venv/bin/python scripts/fetch_jd.py <url> [<url> ...] --save-dir data/my_jds`

## Reading the results

| Part | What it means | Computed by |
|---|---|---|
| **Overall %** | 0.45 × skills + 0.35 × experience + 0.20 × seniority (weights configurable) | code |
| **Skills** | 75% coverage of the JD's required skills + 25% of nice-to-haves. Name-based with aliases (`k8s` = `Kubernetes`), the way ATS keyword filters work | code |
| **Experience relevance** | For each JD responsibility: does any bullet show you doing it (even in different words)? strong = 1, partial = ½, none = 0, averaged | LLM verdicts, code score |
| **Seniority** | Your years (computed from role dates) vs the JD's minimum or level. Under-qualified loses 20 points per missing year; well over the band loses a little | code |
| **Missing skills** (red/amber tags) | Required / nice-to-have skills not found anywhere in your resume | code |
| **"Found in your bullets but not your skills section"** | You mention it in experience; listing it in Skills helps ATS matching | code |
| **Bullet rewrites** | 2-3 rewrites for bullets that are relevant but weakly phrased for this JD | LLM, checked by code |
| **Experience vs responsibilities** | Per responsibility: verdict, the bullet that supports it, and why | LLM |
| **Summary / qualification signals** | Plain-English verdict and over/under-qualification flags | LLM (years flag: code) |

### ⚠ Warnings on rewrites

A small model will happily over-sell you. Each rewrite is scanned, and flagged if it:

- adds **numbers** your original bullet doesn't have (placeholders like `[X%]` are fine: fill them with real figures),
- uses **terms from the JD that appear nowhere in your resume** (e.g. "observability", "microservices", "Kafka"),
- adds **leadership/ownership verbs** ("led", "owned", "spearheaded") the original doesn't use.

A flag doesn't mean wrong. It means *only keep this if it's true*. Interviewers ask about every line.

## CLI

Runs the same pipeline without the servers. Results go to the same history (`--no-save` to skip).

```bash
.venv/bin/python -m api.cli path/to/resume.pdf --jd-file path/to/jd.txt --company Acme --role "Backend Engineer"
.venv/bin/python -m api.cli resume.docx --jd-text "$(pbpaste)"
.venv/bin/python -m api.cli resume.pdf --jd-file jd.txt --json > result.json     # raw JSON
```

## API

| Endpoint | |
|---|---|
| `POST /analyze` | multipart: `resume` file + **exactly one** of `jd_text` / `jd_url`; optional `company`, `role`, `source_url` (metadata for edited text). Returns the full analysis plus `jd_text` (exactly what was analysed), `analysis_id`, `jd_source` |
| `POST /jd/fetch` | `{"url": "..."}` returns extracted `text`, `title`, `method` (`json-ld` / `trafilatura`), `renderer` (`http` / `playwright`), `cached`, `chars`. Preview/edit before analysing |
| `GET /history?limit=100` | newest-first summaries: id, date, company, role, resume name, four scores |
| `GET /history/{id}` | one stored analysis with its full result |
| `GET /healthz` | model reachability and which model/client is in use |

```bash
curl -s localhost:8010/analyze -F resume=@data/my_resumes/me.pdf \
     -F jd_url=https://jobs.lever.co/acme/123 | jq '{overall_match_score, skills_missing, summary}'
```

Errors are always `{"detail": {"code": "...", "message": "..."}}`; `message` is written for end users.
Codes: `jd_input` (not exactly one JD input), `jd_too_short`, `resume_unreadable`, `resume_too_large`,
`model_unavailable` (503), `model_error` (502), plus the fetch codes below.

<details><summary>Response shape</summary>

```json
{
  "overall_match_score": 58.3,
  "section_scores": {"skills": 46.2, "experience": 50.0, "seniority": 100.0},
  "skills_matched": ["Python", "PostgreSQL"],
  "skills_missing": ["Kafka", "Kubernetes"],
  "skills_breakdown": {"required_matched": [], "required_missing": [], "nice_matched": [], "nice_missing": [], "matched_via_text": []},
  "missing_skill_advice": [{"skill": "Kafka", "advice": "..."}],
  "experience_matches": [{"responsibility": "...", "verdict": "partial", "evidence": ["..."], "reason": "..."}],
  "seniority": {"jd_level": "senior", "jd_min_years": 5, "candidate_years": 6.3, "verdict": "aligned", "score": 100},
  "bullet_suggestions": [{"original": "...", "role": "...", "suggestions": ["...", "..."], "why": "...", "warnings": ["rewrite 2: ..."]}],
  "qualification_signals": [{"kind": "under", "signal": "..."}],
  "summary": "2-3 sentence verdict",
  "jd": {"title": "...", "company": "...", "seniority": "senior", "min_years_experience": 5,
         "required_skills": [], "nice_to_have_skills": [], "tech_stack": [], "responsibilities": []},
  "jd_text": "the exact JD text analysed",
  "jd_cached": true, "resume_cached": false, "model": "qwen3.5:4b",
  "timings_s": {"parse": 58.4, "experience": 120.1, "gap_analysis": 95.2, "total": 273.7},
  "analysis_id": 12, "jd_source": "url", "jd_url": "https://..."
}
```
</details>

## JD from a URL

`api/services/jd_fetcher.py` only turns a URL into clean text. Parsing is the same `jd_parser` pipeline as pasted text.

1. **robots.txt** is checked (cached per host), then a plain `httpx` GET with a browser User-Agent and a 10 s timeout.
2. **Extraction:** schema.org `JobPosting` JSON-LD when the page has it (Lever, LinkedIn guest pages, many
   ATSs publish it for Google Jobs), otherwise [`trafilatura`](https://trafilatura.readthedocs.io) main-content
   extraction. No per-site CSS selectors to rot.
3. **JS-only pages** (near-empty HTML / "enable JavaScript" shells) are rendered in headless Chromium via
   Playwright, **only if it's installed**: `pip install playwright && playwright install chromium`.
   Without it you get a clear message instead of a crash.
4. **Gates:** login-wall detection; a closed posting that redirects to the careers homepage is reported as
   closed; the text must read like a JD (requirements/responsibilities vocabulary); then the same 200-char
   minimum as pasted text. **Garbage is never sent to the model.**

| Failure | Code | Shown to user |
|---|---|---|
| robots.txt disallows | `robots_disallowed` | asks to paste instead |
| no answer in 10 s | `timeout` | |
| 401/403/999 | `blocked` | |
| 429 | `rate_limited` | includes Retry-After when sent |
| 404/410 or redirect away from the job | `not_found` | "posting seems to be closed" |
| sign-in page | `login_wall` | |
| JS-only page, Playwright missing | `js_required` | how to install it |
| page isn't a JD | `extraction_failed` | |

Results are cached per URL for the server's lifetime (failures for 60 s), so retrying doesn't hammer a site.

**LinkedIn:** its robots.txt disallows all automated clients, so with the default `JD_FETCH_RESPECT_ROBOTS=true`
LinkedIn links fail with `robots_disallowed`. Public LinkedIn job pages extract fine if you set it to
`false` (search links with `?currentJobId=` are normalised to the job page). Logged-in-only pages are detected as
`login_wall`. **Naukri** and other JS-rendered boards need Playwright.

## How it works

```
resume.pdf/.docx --pdfplumber/python-docx--> text --LLM--> skills, tools, roles, verbatim bullets   (cached)
JD text or URL ---------jd_fetcher---------> text --LLM--> required / nice-to-have, level, years,  (cached)
                                                           responsibilities
                                      |
     skills score        <- code: normalised set comparison + whole-word search of resume text
     seniority score     <- code: years from role dates vs JD minimum/level band
     experience score    <- embeddings pick top-3 bullets per responsibility
                            -> LLM judges each (responsibility, bullet) pair: strong / partial / none
                            -> code: best verdict per responsibility, averaged
     gap analysis        <- code picks weak-but-relevant bullets (by id)
                            -> LLM: missing-skill advice, 2-3 rewrites each, signals, summary
     honesty checks      <- code: flag invented numbers, unsupported JD terms, leadership verbs
```

Design choices, and the evidence behind them:

- **Schema-constrained output everywhere.** Pydantic models become JSON schemas passed to Ollama's `format`
  (vLLM: `response_format`), so output is always parseable. Invalid output is retried once, a truncated
  answer with double the token budget.
- **Code where code is reliable.** The model got total experience as 4.83 years for roles spanning 6.3, so
  years come from date parsing. Skill overlap is a set comparison, deterministic and instant.
- **Small, single judgments.** Asking a 4B model to grade 7 responsibilities against 10 bullets in one prompt
  gave 29%, 79% and 14% on the same input across three runs. Pairwise judgments on embedding-retrieved
  candidates gave identical verdicts on repeat runs.
- **The model never quotes your resume.** Bullets get ids (`B3`); the model refers to ids, so the "before" in
  a rewrite is always your real text and hallucinated references are dropped.
- **Caching.** Resume and JD parses are cached by content hash + model + parser version: testing several resume
  versions against one JD parses the JD once.

### Project layout

```
api/
  main.py               FastAPI app (lifespan wires model client, storage, fetcher)
  cli.py                command-line runner
  config.py             settings: env > local-llm-platform detection > defaults
  schemas.py            pydantic models: LLM output schemas + API responses
  routers/analyze.py    POST /analyze, POST /jd/fetch
  routers/history.py    GET /history, GET /history/{id}
  services/
    model_client.py     loads local-llm-platform's ModelClient (or vendored copy) + schema-constrained JSON
    resume_parser.py    PDF/DOCX -> text -> structured resume
    jd_parser.py        clean/gate JD text -> structured JD
    jd_fetcher.py       URL -> clean JD text (robots, httpx, JSON-LD/trafilatura, optional Playwright)
    matcher.py          scoring, pairwise experience matching, gap analysis, honesty checks
    storage.py          SQLite: history + parse caches
frontend/app.py         Streamlit UI
scripts/                fetch_jd.py (extraction check), make_sample_resumes.py
data/                   fictional sample resume + JD (your own files go in git-ignored my_* folders)
tests/                  pytest suite; fixtures/html holds real job pages saved locally
```

## Configuration

All optional. Set in `.env` (see [.env.example](.env.example)) or the environment.

| Variable | Default | |
|---|---|---|
| `LLM_MODEL` | from local-llm-platform | chat model name, e.g. `qwen3.5:4b`. Required without the platform |
| `LLM_BACKEND` | `ollama` | `ollama` or `vllm` (any OpenAI-compatible server) |
| `LLM_BASE_URL` | `http://localhost:11434` | |
| `EMBEDDING_MODEL` | from platform | e.g. `nomic-embed-text`; without it, candidate bullets are ranked by word overlap |
| `LLM_MAX_CONCURRENCY` | `2` | in-flight model requests (match `OLLAMA_NUM_PARALLEL`) |
| `LLM_TIMEOUT_SECONDS` | `300` | per model request |
| `LLM_NUM_CTX` | `8192` | Ollama context window |
| `LOCAL_LLM_PLATFORM_DIR` | `../local-llm-platform` | where to find the platform checkout |
| `DB_PATH` | `data/matcher.db` | history + caches |
| `MIN_JD_CHARS` / `MAX_JD_CHARS` / `MAX_RESUME_CHARS` | `200` / `10000` / `12000` | input gates |
| `JD_FETCH_TIMEOUT_SECONDS` | `10` | |
| `JD_FETCH_RESPECT_ROBOTS` | `true` | set `false` to fetch LinkedIn public pages |
| `JD_FETCH_CACHE_TTL_SECONDS` | `3600` | |
| `JD_FETCH_USE_PLAYWRIGHT` | `true` | only used if installed, only for JS-shell pages |
| `WEIGHT_SKILLS` / `WEIGHT_EXPERIENCE` / `WEIGHT_SENIORITY` | `0.45` / `0.35` / `0.20` | overall score weights |
| `MAX_REWRITE_BULLETS` | `4` | bullets that get rewrite suggestions |
| `API_URL` | `http://localhost:8010` | where the UI finds the API |
| `API_PORT` | `8010` | used by `start.sh` |

## Performance

Measured on an 8 GB Apple M1 with `qwen3.5:4b` (~14 tokens/s):

| Run | Time |
|---|---|
| cold (new resume, new JD) | 5-11 min, depending on resume length |
| same resume, new JD | ~5-6 min |
| resume and JD both cached | ~4 min |

Time goes to ~20 small pairwise judgments (~2-5 min) and the gap analysis (~1.5-2 min). A 7-9B model on a
16 GB+ machine or a GPU gives better rewrites. vLLM on an NVIDIA GPU makes a run take seconds.

## Troubleshooting

| Symptom | Fix |
|---|---|
| UI: "Can't reach the API" | Start with `./start.sh`, or check `API_URL` |
| Sidebar: "model backend isn't reachable" | `ollama serve` / open the Ollama app; `curl localhost:11434/api/tags` |
| `No model configured` on startup | set `LLM_MODEL` in `.env` (or check out local-llm-platform next to this repo) |
| `model_unavailable` mid-run | Ollama crashed or ran out of memory. Close other apps or use a smaller model |
| `resume_unreadable` "almost no text" | scanned/image PDF. Export a text PDF or DOCX from your editor |
| link fails with `robots_disallowed` | the site opts out of bots: paste the JD (or for LinkedIn, see [above](#jd-from-a-url)) |
| link fails with `js_required` | `pip install playwright && playwright install chromium`, or paste |
| results look different after changing models | expected: caches are per model, so the JD and resume are re-parsed |

## Development

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q          # 100+ tests, no model or network needed
```

Tests use a fake LLM and `httpx.MockTransport`. They cover:
- PDF/DOCX extraction on the sample resumes
- JD cleaning and caching
- skill normalisation and matching
- date and seniority maths
- pairwise verdict combining, rewrite selection and honesty warnings
- the model client's schema, retry and concurrency limits
- the API's validation, gating, error mapping and history
- the Streamlit flow, via `streamlit.testing.AppTest`
- the URL fetcher, against **real job pages saved in `tests/fixtures/html/`** (Greenhouse, Lever,
  LinkedIn guest page, a closed posting's redirect target) plus synthetic JS-shell and login-wall pages,
  so tests don't break when sites change their markup

CI runs the suite on every push. `scripts/make_sample_resumes.py` regenerates the fictional sample resume.

## Limitations

- A 4B model is sometimes plainly wrong: counting "reviewed pull requests" as "led design reviews", labelling
  a strength as a qualification gap, or giving vague missing-skill advice. Read the reasons and warnings,
  and treat the score as a guide, not a verdict.
- Skill matching is name-based with aliases. `Jenkins` won't count as `CI/CD`. That's deliberate (it's how ATS
  keyword filters behave), and it's why "missing" skills are worth listing explicitly when you have them.
- Scanned (image-only) PDFs aren't supported (no OCR).
- History has no multi-user separation: it's a single-person local tool.

## License

MIT, see [LICENSE](LICENSE).
