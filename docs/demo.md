# Demo: real output

Fictional sample resume [`data/sample_resumes/sample_resume.pdf`](../data/sample_resumes/sample_resume.pdf)
against the sample JD [`data/sample_jds/backend_senior.txt`](../data/sample_jds/backend_senior.txt), produced by

```bash
.venv/bin/python -m api.cli data/sample_resumes/sample_resume.pdf --jd-file data/sample_jds/backend_senior.txt
```

on an 8 GB Apple M1 with `qwen3.5:4b` via Ollama. Unedited: this is what the tool printed, small-model
quirks included. Lines starting with `!` are the tool's own honesty warnings on a rewrite.
The same run as JSON (the `POST /analyze` shape, minus `analysis_id`): [sample_response.json](sample_response.json).

```text
=== Senior Backend Engineer @ Orbitpay ===
OVERALL 68%   skills 45 | experience 79 | seniority 100 (aligned: you 6.3y, JD senior, min 5.0)

SUMMARY
    Your Python, PostgreSQL, and AWS experience provides a strong foundation for backend
    engineering, but the missing Kafka and Kubernetes skills create a significant gap for senior-
    level event-driven architecture. The most useful change is to explicitly mention your Celery
    usage as asynchronous task management while learning Kafka terminology to avoid under-
    qualification signals.

SKILLS MATCHED (6): Python, Django, PostgreSQL, Redis, Docker, AWS
  required missing: FastAPI, Kafka, Kubernetes, CI/CD
  nice-to-have missing: Go, Terraform, Prometheus, Grafana

MISSING-SKILL ADVICE
    - FastAPI: You have built Flask services; consider learning FastAPI to match the target role's
    preferred framework without fabricating new experience.
    - Kafka: Your background with Celery for asynchronous tasks is relevant, but you must learn
    Kafka before claiming event-driven workflow expertise.
    - Kubernetes: You have Docker experience; study Kubernetes orchestration concepts before
    implying container management at scale.
    - CI/CD: You used Jenkins for automation; review modern CI/CD pipelines to align your language
    with the job description's requirements.

EXPERIENCE vs RESPONSIBILITIES
  [STRONG ] Design, build and operate high-throughput Python microservices for payment processing.
  (Built and maintained Django REST endpoints)
  [STRONG ] Own REST and gRPC APIs consumed by web, mobile and partner integrations.  (Built and
  maintained Django REST API)
  [PARTIAL] Build event-driven workflows on Kafka for settlement and reconciliation.  (Migrated
  batch reconciliation jobs to a task queue)
  [PARTIAL] Improve reliability: observability, alerting and on-call for your services.  (Migrated
  job execution from cron to Celery)
  [STRONG ] Optimize PostgreSQL schemas and queries for scale.  (tuned PostgreSQL queries for
  reports)
  [STRONG ] Mentor engineers and lead technical design reviews.  (Reviewed PRs and mentored interns)
  [PARTIAL] Work with product to break down ambiguous problems into shippable increments.  (Improved
  job reliability via task queue)

QUALIFICATION SIGNALS
    - under: Missing Kafka experience limits ability to claim event-driven workflow design; Celery
    is a different paradigm.
    - under: Lack of Kubernetes and CI/CD specific terminology may hinder screening against senior-
    level infrastructure requirements.

BULLET REWRITES
  BEFORE: Moved nightly reconciliation jobs from cron scripts to Celery workers, cutting failures by
  60%
    why: Uses 'designed/operated' and 'improved reliability' which aligns with responsibilities
    while keeping the exact 60% metric.
    AFTER 1: Designed and operated Python microservices using Celery for nightly reconciliation
    workflows.
    AFTER 2: Improved service reliability by migrating cron scripts to asynchronous Python workers,
    reducing failures by 60%.
    ! rewrite 1: uses JD terms your resume doesn't support (operated, microservices, workflows);
    keep only if true
    ! rewrite 2: uses JD terms your resume doesn't support (reliability); keep only if true

  BEFORE: Wrote Flask services to ingest CSV uploads from clients into PostgreSQL
    why: Maps 'Flask services' to 'RESTful APIs' and keeps the core action of ingesting data into
    PostgreSQL.
    AFTER 1: Built Python services to ingest client CSV uploads into PostgreSQL databases.
    AFTER 2: Developed RESTful APIs using Flask and Django for data ingestion from external clients.
    ! rewrite 2: uses JD terms your resume doesn't support (apis); keep only if true

  BEFORE: Fixed production bugs and wrote unit tests with pytest
    why: Focuses on fixing bugs and testing without inflating the scope beyond 'Associate Engineer'
    duties.
    AFTER 1: Fixed production bugs in Python services and implemented unit tests using pytest.
    AFTER 2: Enhanced code quality by writing comprehensive unit tests for critical production
    fixes.

  BEFORE: Helped the team set up Docker-based local development
    why: Uses 'assisted' and 'configured' to accurately reflect help rather than ownership, matching
    the original intent.
    AFTER 1: Assisted the team in setting up Docker-based local development environments.
    AFTER 2: Configured Docker containers to streamline local development workflows for the
    engineering team.
    ! rewrite 2: uses JD terms your resume doesn't support (workflows); keep only if true

model qwen3.5:4b | jd cached True | resume cached True | timings {'parse': 0.0, 'experience': 136.4, 'gap_analysis': 80.8, 'total': 217.3}
```

## What to notice

- **Seniority is computed, not guessed:** 6.3 years comes from the role dates
  (Jul 2020 to present), not the model's arithmetic.
- **Skills are a set comparison:** the missing list is literally what the JD asks for that the resume never mentions.
- **Rewrites are checked:** every `!` line is a rewrite borrowing JD language (or numbers, or "led"/"owned")
  that the resume doesn't support. Keep those only if they're true.
