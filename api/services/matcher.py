"""Core matching pipeline.

Steps (each LLM step is schema-constrained):
1. structure resume (LLM, cached by text hash)      } run concurrently
2. structure JD (LLM, cached by text hash)          }
3. skill overlap: programmatic normalised set comparison, no LLM
4. experience relevance: LLM labels each JD responsibility strong/partial/none with cited
   bullet ids; the score is computed here from those labels, never asked of the model
5. seniority: programmatic, from years and the JD's level
6. gap analysis: LLM suggests missing-skill advice and bullet rewrites for bullets chosen here;
   rewrites are checked for invented numbers/skills before being returned
"""
from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from api.config import Settings
from api.schemas import (AnalysisResult, BulletSuggestion, ExperienceItem, GapAnalysis,
                         JDStructured, QualificationSignal, PairJudgment, ResponsibilityMatch, ResumeStructured, SectionScores, Seniority,
                         SeniorityResult, SkillsBreakdown)
from api.services.jd_parser import parse_jd
from api.services.model_client import StructuredLLM
from api.services.resume_parser import Bullet, bullets_with_ids, structure_resume
from api.services.storage import Storage

# ---------------------------------------------------------------- skill normalisation

_ALIASES: dict[str, str] = {
    "js": "javascript", "ecmascript": "javascript", "ts": "typescript",
    "k8s": "kubernetes", "postgres": "postgresql", "psql": "postgresql", "mongo": "mongodb",
    "golang": "go", "node": "node.js", "nodejs": "node.js", "reactjs": "react", "react.js": "react",
    "vuejs": "vue", "vue.js": "vue", "nextjs": "next.js", "py": "python", "python3": "python",
    "amazon web services": "aws", "google cloud": "gcp", "google cloud platform": "gcp",
    "microsoft azure": "azure", "ml": "machine learning", "llms": "llm", "large language models": "llm",
    "large language model": "llm", "natural language processing": "nlp", "ci cd": "ci/cd", "cicd": "ci/cd",
    "continuous integration": "ci/cd", "rest": "rest api", "restful": "rest api", "rest apis": "rest api",
    "restful api": "rest api", "restful apis": "rest api", "genai": "generative ai", "gen ai": "generative ai",
    "sklearn": "scikit-learn", "scikit learn": "scikit-learn", "tf": "tensorflow", "c sharp": "c#",
    "cpp": "c++", "dotnet": ".net", ".net core": ".net", "springboot": "spring boot",
    "github action": "github actions", "mssql": "sql server", "ms sql": "sql server",
    "retrieval augmented generation": "rag", "retrieval-augmented generation": "rag",
    "microservice": "microservices", "microservices architecture": "microservices",
    "elastic search": "elasticsearch", "apache kafka": "kafka", "apache spark": "spark",
    "pyspark": "spark", "hugging face": "huggingface", "llm ops": "llmops",
}
_SUFFIXES = (" programming", " language", " framework", " scripting", " development", " experience",
             " skills", " (preferred)")
_BOUND_L = r"(?<![a-z0-9+#])"
_BOUND_R = r"(?![a-z0-9+#])"


def normalize_skill(skill: str) -> str:
    """Canonical lowercase form used for set comparison ('ReactJS' -> 'react', 'K8s' -> 'kubernetes')."""
    s = skill.lower().replace("’", "'")
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" .,;:-/")
    if s in _ALIASES:
        return _ALIASES[s]
    for suffix in _SUFFIXES:
        if s.endswith(suffix) and len(s) > len(suffix):
            s = s[: -len(suffix)].strip()
    return _ALIASES.get(s, s)


def _variants(canonical: str) -> set[str]:
    """Every surface form that normalises to ``canonical``."""
    return {canonical} | {alias for alias, target in _ALIASES.items() if target == canonical}


def contains_phrase(haystack: str, needle: str) -> bool:
    """Whole-token containment that treats '+', '#' as word chars (so 'c' is not in 'c++')."""
    return bool(needle) and re.search(_BOUND_L + re.escape(needle) + _BOUND_R, haystack) is not None


def _dedupe(items: Iterable[str]) -> dict[str, str]:
    """Map normalised -> first display form, dropping blanks."""
    out: dict[str, str] = {}
    for item in items:
        key = normalize_skill(item)
        if key and key not in out:
            out[key] = item.strip()
    return out


# ---------------------------------------------------------------- scoring (pure functions)

REQUIRED_SHARE = 0.75  # nice-to-haves can move the score by at most 25 points


def match_skills(resume_skills: Iterable[str], resume_text: str, required: Iterable[str],
                 nice: Iterable[str]) -> tuple[SkillsBreakdown, float]:
    """Programmatic skill overlap.

    A JD skill counts as present if its normalised form equals, or appears as a
    whole phrase inside, a resume skill ('aws' in 'aws lambda'), or failing that
    appears in the resume text (only for names of 3+ chars, so 'go'/'r' don't match prose).

    Returns the breakdown and a 0-100 score: 75% required coverage + 25% nice-to-have coverage
    (whichever list exists, if only one does), so a long nice-to-have list can't swamp the requirements.
    """
    have = set(_dedupe(resume_skills))
    text = resume_text.lower()
    req = _dedupe(required)
    nice_map = {k: v for k, v in _dedupe(nice).items() if k not in req}

    def where(key: str) -> str | None:
        if key in have or any(contains_phrase(h, key) for h in have):
            return "list"
        if any(len(v) >= 3 and contains_phrase(text, v) for v in _variants(key)):
            return "text"
        return None

    bd = SkillsBreakdown(required_matched=[], required_missing=[], nice_matched=[], nice_missing=[])
    for mapping, hit_list, miss_list in ((req, bd.required_matched, bd.required_missing),
                                         (nice_map, bd.nice_matched, bd.nice_missing)):
        for key, display in mapping.items():
            found = where(key)
            if found:
                hit_list.append(display)
                if found == "text":
                    bd.matched_via_text.append(display)
            else:
                miss_list.append(display)
    req_cov = len(bd.required_matched) / len(req) if req else None
    nice_cov = len(bd.nice_matched) / len(nice_map) if nice_map else None
    if req_cov is None and nice_cov is None:
        return bd, 100.0
    if req_cov is None or nice_cov is None:
        return bd, round(100 * (req_cov if nice_cov is None else nice_cov), 1)
    return bd, round(100 * (REQUIRED_SHARE * req_cov + (1 - REQUIRED_SHARE) * nice_cov), 1)


VERDICT_POINTS = {"strong": 1.0, "partial": 0.5, "none": 0.0}
_RANK = {"none": 0, "partial": 1, "strong": 2}


def combine_pair_verdicts(responsibility_id: str,
                          pairs: list[tuple[str, PairJudgment]]) -> ResponsibilityMatch:
    """A responsibility gets the best verdict any of its candidate bullets earned."""
    best = max((_RANK[j.verdict] for _, j in pairs), default=0)
    if best == 0:
        return ResponsibilityMatch(responsibility_id=responsibility_id, verdict="none", evidence_bullet_ids=[],
                                   reason=_tidy(pairs[0][1].reason) if pairs else "no candidate bullets")
    winners = [(bid, j) for bid, j in pairs if _RANK[j.verdict] == best]
    return ResponsibilityMatch(responsibility_id=responsibility_id, verdict=winners[0][1].verdict,
                               evidence_bullet_ids=[bid for bid, _ in winners], reason=_tidy(winners[0][1].reason))


def _tidy(reason: str, cap: int = 120) -> str:
    """Mark reasons cut off by the schema's length cap."""
    capped = len(reason) >= cap
    reason = reason.strip()
    return reason.rstrip(" ,;:") + "…" if capped else reason


def experience_score(n_responsibilities: int, matches: Iterable[ResponsibilityMatch]) -> float:
    """Mean of strong=1 / partial=0.5 / none=0 over all responsibilities (unlabelled = none)."""
    if n_responsibilities == 0:
        return 50.0  # nothing to compare against: neutral
    by_id = {m.responsibility_id.upper(): m.verdict for m in matches}
    points = sum(VERDICT_POINTS[by_id.get(f"R{i}", "none")] for i in range(1, n_responsibilities + 1))
    return round(100 * points / n_responsibilities, 1)


# Typical experience band per level, used only when the JD gives no explicit minimum.
LEVEL_YEARS: dict[str, tuple[float, float]] = {
    "intern": (0, 1), "junior": (0, 2), "mid": (2, 5), "senior": (5, 9), "staff": (8, 14),
    "principal": (10, 20), "manager": (5, 12), "director": (10, 25),
}


def seniority_alignment(level: Seniority, min_years: float | None, candidate_years: float) -> SeniorityResult:
    """Score how the candidate's years fit the JD's level / minimum.

    Under-qualified loses 20 points per missing year; over-qualified (more than
    2 years past the band's top) loses 5 per year, floored at 60.
    """
    band = LEVEL_YEARS.get(level)
    if band is None and min_years is None:
        return SeniorityResult(jd_level=level, jd_min_years=None, candidate_years=candidate_years,
                               verdict="unknown", score=100.0)
    lo = min_years if min_years is not None else band[0]
    hi = band[1] if band else lo + 6
    if hi < lo:  # explicit minimum above the level's usual band
        hi = lo + 4
    if candidate_years < lo:
        verdict, score = "under", max(0.0, 100 - 20 * (lo - candidate_years))
    elif candidate_years > hi + 2:
        verdict, score = "over", max(60.0, 100 - 5 * (candidate_years - hi - 2))
    else:
        verdict, score = "aligned", 100.0
    return SeniorityResult(jd_level=level, jd_min_years=min_years, candidate_years=candidate_years,
                           verdict=verdict, score=round(score, 1))


def overall_score(scores: SectionScores, settings: Settings) -> float:
    """Weighted mean of section scores (weights normalised)."""
    w = (settings.weight_skills, settings.weight_experience, settings.weight_seniority)
    total = sum(w) or 1.0
    return round((scores.skills * w[0] + scores.experience * w[1] + scores.seniority * w[2]) / total, 1)


def select_rewrite_candidates(bullets: list[Bullet], matches: Iterable[ResponsibilityMatch],
                              jd_terms: Iterable[str], limit: int) -> list[Bullet]:
    """Pick the bullets worth rewriting.

    Priority: bullets cited only as *partial* evidence (relevant but weakly
    phrased), then uncited bullets from the two most recent roles, most
    JD-keyword overlap first. Bullets already cited as strong evidence are left alone.
    """
    by_id = {b.id: b for b in bullets}
    strong, partial = set(), []
    for m in matches:
        for bid in m.evidence_bullet_ids:
            bid = bid.upper()
            if m.verdict == "strong":
                strong.add(bid)
            elif m.verdict == "partial" and bid in by_id and bid not in partial:
                partial.append(bid)
    chosen = [by_id[b] for b in partial if b not in strong]
    recent_roles = list(dict.fromkeys(b.role for b in bullets))[:2]
    terms = [t for t in _dedupe(jd_terms) if len(t) >= 2]
    cited = strong | set(partial)
    uncited = [b for b in bullets if b.role in recent_roles and b.id not in cited]
    uncited.sort(key=lambda b: -sum(contains_phrase(b.text.lower(), t) for t in terms))
    chosen.extend(uncited)
    return chosen[:limit]


_MONTHS = {m: i for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}
_PRESENT = re.compile(r"present|current|now|till date|to date|ongoing", re.IGNORECASE)


def parse_month(value: str, today: date) -> tuple[int, int] | None:
    """'Jun 2022' / '06/2022' / '2022-06' / '2022' / 'Present' -> (year, month); None if unparseable."""
    v = value.strip().lower()
    if not v:
        return None
    if _PRESENT.search(v):
        return today.year, today.month
    year = re.search(r"(19|20)\d{2}", v)
    if not year:
        return None
    y = int(year.group())
    if m := re.search(r"[a-z]{3,}", v):
        if (month := _MONTHS.get(m.group()[:3])) is not None:
            return y, month
    if m := re.search(r"\b(0?[1-9]|1[0-2])\s*[/.-]\s*(?:19|20)\d{2}|(?:19|20)\d{2}\s*[/.-]\s*(0?[1-9]|1[0-2])\b", v):
        return y, int(m.group(1) or m.group(2))
    return y, 1  # year only: assume January


def years_from_roles(resume: ResumeStructured, today: date) -> float | None:
    """Total professional years from role dates, overlaps merged.

    Small models are unreliable at date arithmetic, so this replaces the model's
    own ``total_years_experience`` whenever every role's dates parse.
    """
    spans = []
    for role in resume.experience:
        start, end = parse_month(role.start, today), parse_month(role.end, today)
        if not start or not end:
            return None
        a, b = start[0] * 12 + start[1], end[0] * 12 + end[1] + 1  # end month inclusive
        if b > a:
            spans.append((a, b))
    if not spans:
        return None
    spans.sort()
    months, (cur_a, cur_b) = 0, spans[0]
    for a, b in spans[1:]:
        if a <= cur_b:
            cur_b = max(cur_b, b)
        else:
            months += cur_b - cur_a
            cur_a, cur_b = a, b
    months += cur_b - cur_a
    return round(months / 12, 1)


_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_WORD = re.compile(r"[a-z][a-z0-9+#]*(?:-[a-z0-9+#]+)*")
# Words common enough in any JD/resume pair that borrowing them is not a new claim.
_GENERIC = set("""with that this from into your their they them using across ensure ensuring build
built design designed develop developed work worked working team teams engineer engineers engineering
service services system systems support supported strong experience including improve improved
deliver delivered make made also more other help helped create created provide provided product
products code within while well high ability based like such each over through about
optimize implement maintain enhance streamline utilize leverage collaborate contribute handle perform
write wrote reduce increase migrate deploy partner""".split())


def _stem(word: str) -> str:
    if word.endswith("ies") and len(word) >= 6:
        return word[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s", "er"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            word = word[: -len(suffix)]
            break
    return word[:-1] if word.endswith("e") and len(word) >= 5 else word


_GENERIC_STEMS = {_stem(w) for w in _GENERIC}
# Scope/leadership verbs: a rewrite may only use them if the original bullet does.
_CLAIM_VERBS = {"led", "lead", "leading", "owned", "own", "owning", "headed", "spearheaded", "architected",
                "managed", "directed", "championed", "drove", "oversaw"}


def _stems(text: str) -> set[str]:
    return {st for w in _WORD.findall(text.lower()) if len(w) >= 4 and (st := _stem(w)) not in _GENERIC_STEMS}


def rewrite_warnings(original: str, suggestions: Iterable[str], resume_text: str,
                     jd_text: str, jd_terms: Iterable[str]) -> list[str]:
    """Flag rewrites that claim more than the resume supports.

    Per rewrite, flags (a) numbers not in the original bullet, and (b) terms lifted
    from the JD that appear nowhere in the resume ('observability', 'high-throughput',
    'Kafka'). Placeholders in [brackets] are allowed: they ask for a real number.
    """
    orig_nums = set(_NUMBER.findall(original))
    resume_low = resume_text.lower()
    resume_stems = _stems(f"{resume_text}\n{original}")
    jd_stems = _stems(jd_text)
    jd_skills = _dedupe(jd_terms)
    out = []
    for i, s in enumerate(suggestions, 1):
        body = re.sub(r"\[[^\]]*\]", "", s)
        low = body.lower()
        nums = list(dict.fromkeys(n for n in _NUMBER.findall(body) if n not in orig_nums))
        terms: list[str] = []
        for key, display in jd_skills.items():
            variants = [v for v in _variants(key) if len(v) >= 2]
            if any(contains_phrase(low, v) for v in variants) and not any(
                    contains_phrase(resume_low, v) for v in variants):
                terms.append(display)
        for w in _WORD.findall(low):
            st = _stem(w)
            if (len(w) >= 4 and st not in _GENERIC_STEMS and st in jd_stems and st not in resume_stems
                    and not any(contains_phrase(t.lower(), w) for t in terms)):
                terms.append(w)
        terms = list(dict.fromkeys(terms))
        orig_low = original.lower()
        claims = [w for w in dict.fromkeys(_WORD.findall(low))
                  if w in _CLAIM_VERBS and not contains_phrase(orig_low, w)]
        if claims:
            out.append(f"rewrite {i}: claims leadership/ownership ({', '.join(claims)}) the original doesn't; "
                       f"keep only if true")
        if nums:
            out.append(f"rewrite {i}: adds figures not in your original ({', '.join(nums)}); keep only if true")
        if terms:
            out.append(f"rewrite {i}: uses JD terms your resume doesn't support ({', '.join(terms)}); "
                       f"keep only if true")
    return out


# ---------------------------------------------------------------- LLM steps

_PAIR_SYSTEM = """You check whether ONE resume bullet shows the candidate doing ONE job responsibility,
even if worded differently.
strong = the bullet shows the core of this responsibility, or very close work. If the responsibility
has several parts ("X and Y"), strong only if the bullet shows most of them.
partial = related or smaller-scope work, or only one part of the responsibility.
none = different work. Similar-sounding but different work (e.g. task queues vs observability) is none.
reason: say what the bullet shows in a few words (e.g. "tuned PostgreSQL queries for reports")."""


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = sum(x * x for x in a) ** 0.5, sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


async def rank_bullets(llm: StructuredLLM, responsibilities: list[str], bullets: list[Bullet],
                       k: int) -> list[list[Bullet]]:
    """Top-``k`` candidate bullets per responsibility.

    Uses embeddings when an embedding model is available (catches paraphrases),
    otherwise word-stem overlap.
    """
    sims: list[list[float]]
    try:
        q_prefix, d_prefix = ("search_query: ", "search_document: ") if "nomic" in llm.embedding_model else ("", "")
        vecs = await llm.embed([q_prefix + r for r in responsibilities] + [d_prefix + b.text for b in bullets])
        rv, bv = vecs[: len(responsibilities)], vecs[len(responsibilities):]
        sims = [[_cosine(r, b) for b in bv] for r in rv]
    except Exception:  # no embedding model / backend error: lexical fallback
        bstems = [_stems(b.text) for b in bullets]
        sims = []
        for r in responsibilities:
            rs = _stems(r)
            sims.append([len(rs & bs) / (len(rs | bs) or 1) for bs in bstems])
    out = []
    for row in sims:
        order = sorted(range(len(bullets)), key=lambda i: -row[i])[:k]
        out.append([bullets[i] for i in order])
    return out


async def _match_experience(llm: StructuredLLM, jd: JDStructured, bullets: list[Bullet],
                            k: int = 3) -> list[ResponsibilityMatch]:
    """Judge each responsibility against its top-k bullets, one small call per pair.

    A 4B model judging 7 responsibilities x 10 bullets at once swung between 14% and 79% on the same
    input; single (responsibility, bullet) judgments are far more stable.
    """
    if not bullets or not jd.responsibilities:
        return []
    candidates = await rank_bullets(llm, jd.responsibilities, bullets, k)

    async def judge(responsibility: str, b: Bullet) -> tuple[str, PairJudgment]:
        user = f"RESPONSIBILITY: {responsibility}\nBULLET: {b.text}"
        return b.id, await llm.generate_json(_PAIR_SYSTEM, user, PairJudgment, max_tokens=120)

    pair_lists = await asyncio.gather(*(asyncio.gather(*(judge(r, b) for b in cands))
                                        for r, cands in zip(jd.responsibilities, candidates)))
    return [combine_pair_verdicts(f"R{i}", list(pairs)) for i, pairs in enumerate(pair_lists, 1)]


_GAP_SYSTEM = """You are an honest resume coach. You help a candidate mirror a job description's language
for ATS screening WITHOUT fabricating anything. Rules:
- A rewrite must describe the same work and facts as the original bullet. Never invent numbers,
  tools, team sizes or outcomes. If a metric would help but is unknown, write a placeholder like [X%].
- Prefer the job description's exact terms where the original bullet genuinely supports them.
- Start rewrites with a strong past-tense verb; keep each under 35 words.
- For a missing skill, say add it only if the candidate has really used it, and where it would fit.
- Do not add claims of scale, leadership, ownership or tooling ("high-throughput", "led", "owned",
  "observability") unless the original bullet states them.
- For a missing skill the candidate has not clearly used, say so plainly: learn it before claiming it,
  or leave it out. Never suggest implying experience they don't have.
- qualification_signals: concrete over- or under-qualification signals about scope, leadership or
  domain (NOT years of experience, that is handled separately); empty list if none."""


async def _gap_analysis(llm: StructuredLLM, jd: JDStructured, resume: ResumeStructured,
                        skills: SkillsBreakdown, scores: SectionScores, overall: float,
                        seniority: SeniorityResult, candidates: list[Bullet]) -> GapAnalysis:
    missing = (skills.required_missing or skills.nice_missing)[:5]  # advice for required gaps first
    years_note = {"under": "fewer years than required", "over": "well above the expected years",
                  "aligned": "years fit the level", "unknown": "no stated requirement"}[seniority.verdict]
    bl = "\n".join(f"{b.id} [{b.role}]: {b.text}" for b in candidates) or "(none)"
    user = f"""TARGET ROLE: {jd.title}{' at ' + jd.company if jd.company else ''} \
(level: {jd.seniority}, min years: {jd.min_years_experience if jd.min_years_experience is not None else 'not stated'})
REQUIRED SKILLS: {', '.join(jd.required_skills) or '-'}
NICE TO HAVE: {', '.join(jd.nice_to_have_skills) or '-'}
RESPONSIBILITIES:
{chr(10).join('- ' + r for r in jd.responsibilities) or '-'}

CANDIDATE: {resume.total_years_experience:g} years; skills: {', '.join(resume.skills + resume.tools)}
FIT: overall {overall:.0f}/100; experience years: {years_note}
MISSING SKILLS: {', '.join(missing) or 'none'}

BULLETS TO IMPROVE:
{bl}

Produce:
1. missing_skill_advice: one entry per missing skill listed above, one sentence each.
2. bullet_rewrites: one entry for EVERY bullet listed above (use its id), 2-3 rewrites each.
3. qualification_signals.
4. summary: 2-3 sentences: overall fit, the biggest gap, the single most useful change."""
    return await llm.generate_json(_GAP_SYSTEM, user, GapAnalysis,
                                   max_tokens=400 + 250 * len(candidates) + 60 * len(missing))


# ---------------------------------------------------------------- orchestration

@dataclass(frozen=True)
class PipelineInput:
    resume_text: str
    jd_text: str  # already cleaned / length-gated by jd_parser.clean_jd_text


async def analyze(llm: StructuredLLM, storage: Storage | None, inp: PipelineInput, settings: Settings,
                  today: date | None = None) -> AnalysisResult:
    """Run the full resume-vs-JD pipeline and return the structured analysis."""
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    (resume, resume_cached), (jd, jd_cached) = await asyncio.gather(
        structure_resume(llm, storage, inp.resume_text, (today or date.today()).isoformat()),
        parse_jd(llm, storage, inp.jd_text))
    timings["parse"] = time.perf_counter() - t0

    if (years := years_from_roles(resume, today or date.today())) is not None:
        resume.total_years_experience = years
    jd_terms = jd.required_skills + jd.nice_to_have_skills + jd.tech_stack
    skills_bd, skills_score = match_skills(
        resume.skills + resume.tools + [s.skill for s in resume.skill_years], inp.resume_text,
        jd.required_skills, jd.nice_to_have_skills + jd.tech_stack)

    t = time.perf_counter()
    bullets = bullets_with_ids(resume)
    matches = await _match_experience(llm, jd, bullets)
    timings["experience"] = time.perf_counter() - t
    known = {b.id for b in bullets}
    for m in matches:  # drop hallucinated citations
        m.evidence_bullet_ids = [b.upper() for b in m.evidence_bullet_ids if b.upper() in known]
    exp_score = experience_score(len(jd.responsibilities), matches)

    seniority = seniority_alignment(jd.seniority, jd.min_years_experience, resume.total_years_experience)
    scores = SectionScores(skills=skills_score, experience=exp_score, seniority=seniority.score)
    overall = overall_score(scores, settings)

    t = time.perf_counter()
    candidates = select_rewrite_candidates(bullets, matches, jd_terms, settings.max_rewrite_bullets)
    gap = await _gap_analysis(llm, jd, resume, skills_bd, scores, overall, seniority, candidates)
    timings["gap_analysis"] = time.perf_counter() - t

    by_id = {b.id: b for b in bullets}
    suggestions: list[BulletSuggestion] = []
    seen: set[str] = set()
    for rw in gap.bullet_rewrites:
        bid = rw.bullet_id.upper()
        if bid not in by_id or bid in seen:
            continue  # unknown id: the model referred to a bullet that doesn't exist
        seen.add(bid)
        original = by_id[bid].text
        texts = [s.strip() for s in rw.suggestions if s.strip() and s.strip() != original]
        if texts:
            suggestions.append(BulletSuggestion(
                original=original, role=by_id[bid].role, suggestions=texts, why=rw.why,
                warnings=rewrite_warnings(original, texts, inp.resume_text, inp.jd_text, jd_terms)))

    by_resp = {m.responsibility_id.upper(): m for m in matches}
    exp_items = []
    for i, r in enumerate(jd.responsibilities, 1):
        m = by_resp.get(f"R{i}")
        exp_items.append(ExperienceItem(
            responsibility=r, verdict=m.verdict if m else "none",
            evidence=[by_id[b].text for b in (m.evidence_bullet_ids if m else [])],
            reason=m.reason if m else "not assessed by the model"))

    signals = [q for q in gap.qualification_signals
               if not re.search(r"\byears?\b|score|^\W*(none|n/?a)\b", q.signal, re.I)]
    for q in signals:  # small models label gaps as "over"; a lack of something is never over-qualification
        if q.kind == "over" and re.search(r"\b(no|not|lacks?|lacking|missing|without|limited)\b", q.signal, re.I):
            q.kind = "under"
    if seniority.verdict in ("under", "over"):
        need = f"{seniority.jd_min_years:g}+ years" if seniority.jd_min_years is not None else f"{jd.seniority} level"
        signals.insert(0, QualificationSignal(
            kind=seniority.verdict, signal=f"{seniority.candidate_years:g} years of experience vs {need} expected"))
    missing_known = {normalize_skill(s) for s in skills_bd.required_missing + skills_bd.nice_missing}
    timings["total"] = time.perf_counter() - t0
    return AnalysisResult(
        overall_match_score=overall,
        section_scores=scores,
        skills_matched=skills_bd.required_matched + skills_bd.nice_matched,
        skills_missing=skills_bd.required_missing + skills_bd.nice_missing,
        skills_breakdown=skills_bd,
        missing_skill_advice=[a for a in gap.missing_skill_advice if normalize_skill(a.skill) in missing_known],
        experience_matches=exp_items,
        seniority=seniority,
        bullet_suggestions=suggestions,
        qualification_signals=signals,
        summary=gap.summary.strip(),
        jd=jd,
        jd_text=inp.jd_text,
        jd_cached=jd_cached,
        resume_cached=resume_cached,
        model=llm.model,
        timings_s={k: round(v, 1) for k, v in timings.items()},
    )
