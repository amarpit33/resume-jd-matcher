"""Command-line runner for the matching pipeline (no API server needed).

    python -m api.cli data/sample_resumes/sample_resume.pdf --jd-file data/sample_jds/backend_senior.txt
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import textwrap
from pathlib import Path

from api.config import get_settings
from api.schemas import AnalysisResult
from api.services.jd_parser import JDTextError, clean_jd_text
from api.services.matcher import PipelineInput, analyze
from api.services.model_client import create_llm
from api.services.resume_parser import ResumeParseError, content_hash, extract_text
from api.services.storage import Storage


def render(r: AnalysisResult) -> str:
    """Human-readable report."""
    s = r.section_scores
    w = lambda t, ind="    ": textwrap.fill(t, 100, initial_indent=ind, subsequent_indent=ind)  # noqa: E731
    out = [f"=== {r.jd.title}{' @ ' + r.jd.company if r.jd.company else ''} ===",
           f"OVERALL {r.overall_match_score:.0f}%   skills {s.skills:.0f} | experience {s.experience:.0f} | "
           f"seniority {s.seniority:.0f} ({r.seniority.verdict}: you {r.seniority.candidate_years:g}y, "
           f"JD {r.seniority.jd_level}, min {r.seniority.jd_min_years})",
           "", "SUMMARY", w(r.summary), "",
           f"SKILLS MATCHED ({len(r.skills_matched)}): {', '.join(r.skills_matched) or '-'}",
           f"  required missing: {', '.join(r.skills_breakdown.required_missing) or '-'}",
           f"  nice-to-have missing: {', '.join(r.skills_breakdown.nice_missing) or '-'}"]
    if r.skills_breakdown.matched_via_text:
        out.append(f"  found in text but not in your skills list: {', '.join(r.skills_breakdown.matched_via_text)}")
    if r.missing_skill_advice:
        out += ["", "MISSING-SKILL ADVICE"] + [w(f"- {a.skill}: {a.advice}") for a in r.missing_skill_advice]
    out += ["", "EXPERIENCE vs RESPONSIBILITIES"]
    for e in r.experience_matches:
        out.append(w(f"[{e.verdict.upper():7}] {e.responsibility}  ({e.reason})", "  "))
    if r.qualification_signals:
        out += ["", "QUALIFICATION SIGNALS"] + [w(f"- {q.kind}: {q.signal}") for q in r.qualification_signals]
    out += ["", "BULLET REWRITES"]
    for b in r.bullet_suggestions:
        out += [w(f"BEFORE: {b.original}", "  "), w(f"why: {b.why}", "    ")]
        out += [w(f"AFTER {i}: {t}", "    ") for i, t in enumerate(b.suggestions, 1)]
        out += [w(f"! {x}", "    ") for x in b.warnings]
        out.append("")
    out.append(f"model {r.model} | jd cached {r.jd_cached} | resume cached {r.resume_cached} | timings {r.timings_s}")
    return "\n".join(out)


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    try:
        path = Path(args.resume)
        resume_text = extract_text(path.name, path.read_bytes())
        jd_raw = Path(args.jd_file).read_text() if args.jd_file else args.jd_text
        jd_text = clean_jd_text(jd_raw, settings.min_jd_chars, settings.max_jd_chars)
    except (ResumeParseError, JDTextError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    llm = create_llm(settings)
    storage = Storage(settings.db_path)
    try:
        print(f"model {llm.model} via {llm.backend} (client: {llm.source})", file=sys.stderr)
        result = await analyze(llm, storage, PipelineInput(resume_text[: settings.max_resume_chars], jd_text),
                               settings)
        if not args.no_save:
            storage.save_analysis(result.model_dump(), resume_hash=content_hash(resume_text),
                                  jd_hash=content_hash(jd_text), resume_name=path.name,
                                  company=args.company or result.jd.company or None,
                                  role=args.role or result.jd.title or None)
        print(json.dumps(result.model_dump(), indent=2) if args.json else render(result))
        return 0
    finally:
        await llm.aclose()
        storage.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("resume", help="resume file (.pdf, .docx, .txt, .md)")
    jd = p.add_mutually_exclusive_group(required=True)
    jd.add_argument("--jd-file")
    jd.add_argument("--jd-text")
    p.add_argument("--company")
    p.add_argument("--role")
    p.add_argument("--json", action="store_true", help="print the raw JSON result")
    p.add_argument("--no-save", action="store_true", help="don't write to history")
    sys.exit(asyncio.run(_run(p.parse_args())))


if __name__ == "__main__":
    main()
