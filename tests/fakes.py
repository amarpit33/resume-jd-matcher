"""Fake LLM returning canned schema instances; records every call."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from api.schemas import (BulletRewrite, GapAnalysis, JDStructured, MissingSkillAdvice,
                         PairJudgment, QualificationSignal, ResumeStructured, Role, SkillYears)

RESUME = ResumeStructured(
    skills=["Python", "SQL", "Django", "Flask"],
    tools=["PostgreSQL", "Redis", "Docker", "AWS"],
    experience=[
        Role(title="Software Engineer", company="Finlytics", start="Jun 2022", end="Present",
             bullets=["Built Django REST endpoints used by 40 partner banks",
                      "Moved reconciliation jobs to Celery workers, cutting failures by 60%",
                      "Tuned PostgreSQL queries for reporting"]),
        Role(title="Associate Engineer", company="DataNest", start="Jul 2020", end="May 2022",
             bullets=["Wrote Flask services to ingest CSV uploads"]),
    ],
    total_years_experience=4.0,  # deliberately wrong: pipeline recomputes from dates
    skill_years=[SkillYears(skill="Python", years=6)],
)
JD = JDStructured(
    title="Senior Backend Engineer", company="Orbitpay", seniority="senior", min_years_experience=5,
    required_skills=["Python", "PostgreSQL", "Kafka", "Kubernetes"], nice_to_have_skills=["Go", "Redis"],
    tech_stack=["Python", "Kafka", "Terraform"],
    responsibilities=["Build Python microservices", "Build Kafka workflows", "Optimize PostgreSQL queries"],
)


def experience_response(user: str) -> PairJudgment:
    """Pair judge: strong for the Django bullet vs microservices, partial for PostgreSQL tuning, else none."""
    responsibility, bullet = user.split("\n", 1)
    if "microservices" in responsibility and "Django" in bullet:
        return PairJudgment(reason="APIs in Python", verdict="strong")
    if "PostgreSQL" in responsibility and "PostgreSQL" in bullet:
        return PairJudgment(reason="query tuning only", verdict="partial")
    return PairJudgment(reason="different work", verdict="none")


GAP = GapAnalysis(
    missing_skill_advice=[MissingSkillAdvice(skill="Kafka", advice="Learn it first."),
                          MissingSkillAdvice(skill="Rust", advice="hallucinated skill, should be dropped")],
    bullet_rewrites=[
        BulletRewrite(bullet_id="B3", why="mirror JD",
                      suggestions=["Optimized PostgreSQL queries for reporting pages",
                                   "Optimized PostgreSQL queries, cutting latency by 45%"]),
        BulletRewrite(bullet_id="B42", why="unknown id", suggestions=["x", "y"]),
    ],
    qualification_signals=[QualificationSignal(kind="over", signal="Seniority score 96 is high"),
                           QualificationSignal(kind="over", signal="No evidence of leading design reviews")],
    summary="Decent fit. Kafka is the main gap.",
)


class FakeModelError(RuntimeError):
    pass


class FakeModelUnavailable(FakeModelError):
    pass


class FakeLLM:
    ModelError = FakeModelError
    ModelUnavailable = FakeModelUnavailable
    model = "fake-model"
    embedding_model = ""
    backend = "ollama"
    source = "fake"

    def __init__(self, responses: dict[type, Any] | None = None) -> None:
        self.responses = responses or {ResumeStructured: RESUME, JDStructured: JD,
                                       PairJudgment: experience_response, GapAnalysis: GAP}
        self.calls: list[dict[str, Any]] = []

    async def generate_json(self, system: str, user: str, schema: type, max_tokens: int = 1024):
        self.calls.append({"schema": schema, "user": user})
        r = self.responses[schema]
        return (r(user) if callable(r) and not isinstance(r, BaseModel) else r).model_copy(deep=True)

    async def embed(self, texts):
        raise RuntimeError("no embedding model configured")

    def count(self, schema: type) -> int:
        return sum(c["schema"] is schema for c in self.calls)
