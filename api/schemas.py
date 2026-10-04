"""Pydantic models: LLM output schemas (constrained generation) and API responses.

Field descriptions are part of the JSON schema sent to the model, so they double
as extraction instructions. LLM schemas stay small: every output token costs
~70 ms on an 8 GB laptop.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Seniority = Literal["intern", "junior", "mid", "senior", "staff", "principal", "manager", "director", "unspecified"]
Verdict = Literal["strong", "partial", "none"]


# ---------- LLM output: resume ----------

class Role(BaseModel):
    title: str
    company: str
    start: str = Field(description="start date as written, e.g. 'Jan 2021'")
    end: str = Field(description="end date as written, or 'Present'")
    bullets: list[str] = Field(description="achievement/responsibility bullets, copied verbatim")


class SkillYears(BaseModel):
    skill: str
    years: float


class ResumeStructured(BaseModel):
    """Structured resume, as extracted by the model."""

    skills: list[str] = Field(description="technical skills, languages, frameworks, methods")
    tools: list[str] = Field(description="tools, platforms, cloud services, databases")
    experience: list[Role] = Field(description="roles, most recent first")
    total_years_experience: float = Field(description="total professional years, from role dates")
    skill_years: list[SkillYears] = Field(
        description="years per major skill, only where inferable from role dates")


# ---------- LLM output: job description ----------

class JDStructured(BaseModel):
    """Structured job description, as extracted by the model."""

    title: str
    company: str = Field(description="company name, or '' if not stated")
    seniority: Seniority
    min_years_experience: float | None = Field(description="minimum years required, null if not stated")
    required_skills: list[str] = Field(description="hard requirements: must-have technical skills")
    nice_to_have_skills: list[str] = Field(description="preferred / bonus / nice-to-have skills")
    tech_stack: list[str] = Field(description="technologies mentioned anywhere in the posting")
    responsibilities: list[str] = Field(description="key responsibilities, at most 8, short")


# ---------- LLM output: experience relevance ----------

class PairJudgment(BaseModel):
    """Does one bullet demonstrate one responsibility? ``reason`` first, so the model reasons before the verdict."""

    reason: str = Field(max_length=120,  # grammar-enforced cap
                        description="what the bullet itself shows, max 15 words; do not restate the responsibility")
    verdict: Verdict


class ResponsibilityMatch(BaseModel):
    responsibility_id: str
    verdict: Verdict
    evidence_bullet_ids: list[str]
    reason: str


# ---------- LLM output: gap analysis ----------

class BulletRewrite(BaseModel):
    bullet_id: str = Field(description="id like 'B3'")
    suggestions: list[str] = Field(min_length=2, max_length=3)
    why: str = Field(description="what JD language/requirement the rewrite targets, one sentence")


class MissingSkillAdvice(BaseModel):
    skill: str
    advice: str = Field(description="add only if genuinely used, and where; else learn it first. One sentence")


class QualificationSignal(BaseModel):
    kind: Literal["over", "under"]
    signal: str


class GapAnalysis(BaseModel):
    missing_skill_advice: list[MissingSkillAdvice]
    bullet_rewrites: list[BulletRewrite]
    qualification_signals: list[QualificationSignal]
    summary: str = Field(description="2-3 sentence plain-English verdict")


# ---------- API response ----------

class SectionScores(BaseModel):
    skills: float
    experience: float
    seniority: float


class SkillsBreakdown(BaseModel):
    required_matched: list[str]
    required_missing: list[str]
    nice_matched: list[str]
    nice_missing: list[str]
    matched_via_text: list[str] = Field(
        default_factory=list, description="JD skills not in the resume's skill list but found in its text")


class ExperienceItem(BaseModel):
    responsibility: str
    verdict: Verdict
    evidence: list[str]
    reason: str


class SeniorityResult(BaseModel):
    jd_level: Seniority
    jd_min_years: float | None
    candidate_years: float
    verdict: Literal["aligned", "under", "over", "unknown"]
    score: float


class BulletSuggestion(BaseModel):
    original: str
    role: str
    suggestions: list[str]
    why: str
    warnings: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    overall_match_score: float
    section_scores: SectionScores
    skills_matched: list[str]
    skills_missing: list[str]
    skills_breakdown: SkillsBreakdown
    missing_skill_advice: list[MissingSkillAdvice]
    experience_matches: list[ExperienceItem]
    seniority: SeniorityResult
    bullet_suggestions: list[BulletSuggestion]
    qualification_signals: list[QualificationSignal]
    summary: str
    jd: JDStructured
    jd_text: str
    jd_cached: bool
    resume_cached: bool
    model: str
    timings_s: dict[str, float]
