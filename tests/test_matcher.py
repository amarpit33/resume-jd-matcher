import re
from datetime import date

import pytest

from api.config import Settings
from api.schemas import PairJudgment, ResponsibilityMatch, SectionScores
from api.services.matcher import (PipelineInput, analyze, combine_pair_verdicts, contains_phrase, experience_score, match_skills,
                                  normalize_skill, overall_score, parse_month, rank_bullets, rewrite_warnings,
                                  select_rewrite_candidates, seniority_alignment, years_from_roles)
from api.services.resume_parser import bullets_with_ids
from api.services.storage import Storage
from tests.fakes import JD, RESUME, FakeLLM

TODAY = date(2026, 10, 4)


@pytest.mark.parametrize("raw,expected", [
    ("ReactJS", "react"), ("K8s", "kubernetes"), ("Postgres", "postgresql"), ("Python programming", "python"),
    ("Node.js", "node.js"), ("RESTful APIs", "rest api"), ("AWS (EC2, S3)", "aws"), ("  Go. ", "go"),
])
def test_normalize_skill(raw, expected):
    assert normalize_skill(raw) == expected


def test_contains_phrase_respects_token_boundaries():
    assert contains_phrase("aws lambda", "aws")
    assert not contains_phrase("c++ and c#", "c")
    assert not contains_phrase("javascript", "java")
    assert not contains_phrase("nosql", "sql")


def test_match_skills_set_comparison_and_text_fallback():
    bd, score = match_skills(
        ["Python", "Postgres", "AWS Lambda"], "Built services on k8s and deployed with Terraform.",
        required=["python", "PostgreSQL", "AWS", "Kubernetes", "Kafka"], nice=["Terraform", "Go", "Python"])
    assert bd.required_matched == ["python", "PostgreSQL", "AWS", "Kubernetes"]
    assert bd.required_missing == ["Kafka"]
    assert bd.nice_matched == ["Terraform"] and bd.nice_missing == ["Go"]  # 'Python' deduped into required
    assert bd.matched_via_text == ["Kubernetes", "Terraform"]
    assert score == round(100 * (0.75 * 4 / 5 + 0.25 * 1 / 2), 1)


def test_match_skills_short_names_never_match_prose():
    bd, _ = match_skills([], "I go to the office and r&d is fun", required=["Go", "R"], nice=[])
    assert bd.required_missing == ["Go", "R"]


def test_match_skills_empty_jd_is_full_score():
    assert match_skills(["Python"], "", [], [])[1] == 100.0


def test_long_nice_to_have_list_cannot_swamp_required():
    """Regression: a JD parse with 20 domain-ish nice-to-haves scored a 2/3 required match at 15%."""
    nice = [f"domain thing {i}" for i in range(20)]
    _, score = match_skills(["Python", "REST API"], "", ["Python", "REST APIs", "Kafka"], nice)
    assert score == 50.0
    assert match_skills(["Python"], "", [], ["Python", "Go"])[1] == 50.0      # only nice-to-haves
    assert match_skills(["Python"], "", ["Python", "Go"], [])[1] == 50.0      # only required


def test_experience_score_missing_labels_count_as_none():
    m = [ResponsibilityMatch(responsibility_id="r1", verdict="strong", evidence_bullet_ids=[], reason=""),
         ResponsibilityMatch(responsibility_id="R2", verdict="partial", evidence_bullet_ids=[], reason="")]
    assert experience_score(4, m) == 37.5
    assert experience_score(0, []) == 50.0


def test_combine_pair_verdicts_takes_best_and_its_evidence():
    pj = lambda v, r="": PairJudgment(reason=r, verdict=v)  # noqa: E731
    m = combine_pair_verdicts("R1", [("B1", pj("partial", "p")), ("B2", pj("strong", "s")), ("B3", pj("strong"))])
    assert (m.verdict, m.evidence_bullet_ids, m.reason) == ("strong", ["B2", "B3"], "s")
    m = combine_pair_verdicts("R2", [("B1", pj("none", "unrelated"))])
    assert (m.verdict, m.evidence_bullet_ids, m.reason) == ("none", [], "unrelated")
    assert combine_pair_verdicts("R3", []).verdict == "none"
    capped = combine_pair_verdicts("R4", [("B1", pj("partial", "x" * 118 + ": "))]).reason
    assert capped.endswith("x…")


class _EmbedLLM:
    embedding_model = "nomic-embed-text"

    def __init__(self):
        self.seen = []

    async def embed(self, texts):
        self.seen = texts
        axis = lambda t: [1.0, 0.0] if re.search("api|rest", t.lower()) else [0.0, 1.0]  # noqa: E731
        return [axis(t) for t in texts]


async def test_rank_bullets_uses_embeddings_with_nomic_prefixes():
    bullets = bullets_with_ids(RESUME)
    llm = _EmbedLLM()
    top = await rank_bullets(llm, ["Own public APIs", "Tune databases"], bullets[:2], k=1)
    assert llm.seen[0] == "search_query: Own public APIs" and llm.seen[2].startswith("search_document: ")
    assert top[1][0].id == "B2"  # 'Tune databases' sits on the non-API axis, like B2


async def test_rank_bullets_falls_back_to_lexical_overlap():
    top = await rank_bullets(FakeLLM(), ["Optimize PostgreSQL queries"], bullets_with_ids(RESUME), k=2)
    assert top[0][0].text == "Tuned PostgreSQL queries for reporting"


@pytest.mark.parametrize("level,min_years,years,verdict,score", [
    ("senior", 5, 6, "aligned", 100), ("senior", 5, 3.5, "under", 70), ("senior", None, 1, "under", 20),
    ("junior", None, 10, "over", 70), ("junior", None, 20, "over", 60), ("mid", None, 8, "over", 95), ("unspecified", None, 3, "unknown", 100),
    ("unspecified", 3, 2, "under", 80),
])
def test_seniority_alignment(level, min_years, years, verdict, score):
    r = seniority_alignment(level, min_years, years)
    assert (r.verdict, r.score) == (verdict, score)


def test_overall_score_normalises_weights():
    s = Settings(weight_skills=2, weight_experience=1, weight_seniority=1)
    assert overall_score(SectionScores(skills=80, experience=40, seniority=100), s) == 75.0


@pytest.mark.parametrize("raw,expected", [
    ("Jun 2022", (2022, 6)), ("06/2022", (2022, 6)), ("2022-06", (2022, 6)), ("2019", (2019, 1)),
    ("Present", (2026, 10)), ("September 2018", (2018, 9)), ("soon", None),
])
def test_parse_month(raw, expected):
    assert parse_month(raw, TODAY) == expected


def test_years_from_roles_merges_and_overrides_model_value():
    assert years_from_roles(RESUME, TODAY) == 6.3  # Jul 2020 - Oct 2026 inclusive
    overlapping = RESUME.model_copy(deep=True)
    overlapping.experience[1].end = "Dec 2023"   # overlaps the current role
    assert years_from_roles(overlapping, TODAY) == 6.3
    broken = RESUME.model_copy(deep=True)
    broken.experience[0].start = "a while ago"
    assert years_from_roles(broken, TODAY) is None


def test_select_rewrite_candidates_prefers_partial_then_recent_uncited():
    bullets = bullets_with_ids(RESUME)
    m = [ResponsibilityMatch(responsibility_id="R1", verdict="strong", evidence_bullet_ids=["B1"], reason=""),
         ResponsibilityMatch(responsibility_id="R2", verdict="partial", evidence_bullet_ids=["b3"], reason="")]
    picked = [b.id for b in select_rewrite_candidates(bullets, m, ["PostgreSQL", "Celery"], limit=3)]
    assert picked[0] == "B3"          # partial evidence first
    assert "B1" not in picked         # strong evidence left alone
    assert picked == ["B3", "B2", "B4"]


def test_stem_matches_inflections():
    from api.services.matcher import _stem
    assert _stem("optimized") == _stem("optimize") == _stem("optimizing")
    assert _stem("services") == _stem("service")


def test_rewrite_warnings_flag_invented_numbers_and_jd_terms():
    jd = "Improve reliability with observability and alerting. Kafka. High-throughput microservices."
    w = rewrite_warnings(
        "Tuned PostgreSQL queries for reporting",
        ["Optimized PostgreSQL queries for reporting pages [X%] faster",
         "Cut latency by 45% with observability and Kafka-based alerting on high-throughput microservices"],
        resume_text="Tuned PostgreSQL queries for reporting", jd_text=jd, jd_terms=["Kafka", "PostgreSQL"])
    assert len(w) == 2 and all(x.startswith("rewrite 2:") for x in w)  # 'Optimized' vs 'Tuned' is fine
    assert "45" in w[0]
    for term in ("Kafka", "observability", "alerting", "high-throughput", "microservices"):
        assert term in w[1]


def test_rewrite_warnings_flag_leadership_claims_and_handle_plurals():
    w = rewrite_warnings("Reviewed pull requests and fixed slow query pages",
                         ["Led design reviews and fixed slow queries"], resume_text="slow query pages",
                         jd_text="Lead design reviews. Optimize queries.", jd_terms=[])
    assert w == ["rewrite 1: claims leadership/ownership (led) the original doesn't; keep only if true"]


async def test_analyze_end_to_end_with_fake_llm(jd_text):
    llm, storage = FakeLLM(), Storage(":memory:")
    settings = Settings(max_rewrite_bullets=2)
    resume_text = "Python Django Flask PostgreSQL Redis Docker AWS Celery\n" + "\n".join(
        b for role in RESUME.experience for b in role.bullets)
    r = await analyze(llm, storage, PipelineInput(resume_text, jd_text), settings, today=TODAY)

    assert r.seniority.candidate_years == 6.3 and r.seniority.verdict == "aligned"
    assert r.skills_breakdown.required_missing == ["Kafka", "Kubernetes"]
    assert r.section_scores.experience == 50.0                  # strong + none + partial
    assert r.experience_matches[0].evidence == ["Built Django REST endpoints used by 40 partner banks"]
    assert r.experience_matches[1].evidence == []
    assert [b.original for b in r.bullet_suggestions] == ["Tuned PostgreSQL queries for reporting"]  # B42 dropped
    assert "45" in r.bullet_suggestions[0].warnings[0]
    assert [a.skill for a in r.missing_skill_advice] == ["Kafka"]  # 'Rust' isn't missing: dropped
    assert [(q.kind, q.signal) for q in r.qualification_signals] == [
        ("under", "No evidence of leading design reviews")]          # score-parroting dropped, mislabel fixed
    assert 0 <= r.overall_match_score <= 100

    r2 = await analyze(llm, storage, PipelineInput(resume_text, jd_text), settings, today=TODAY)
    assert r2.jd_cached and r2.resume_cached
    assert llm.count(PairJudgment) == 18  # 3 responsibilities x top-3 bullets, per run
