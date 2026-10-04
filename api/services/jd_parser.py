"""Job description ingestion: pasted text -> cleaned text -> structured JD (LLM, cached)."""
from __future__ import annotations

import re

from api.schemas import JDStructured
from api.services.model_client import StructuredLLM
from api.services.resume_parser import content_hash
from api.services.storage import Storage


PARSER_VERSION = "jd2"  # bump when the prompt/schema changes so cached parses are redone


class JDTextError(ValueError):
    """The JD text is empty, too short, or otherwise unusable."""


# Lines that are pure page chrome when a JD is copied from a job board.
_NOISE_LINE = re.compile(
    r"^(apply( now)?|save( job)?|share( this job)?|report( this)? job|easy apply|show (more|less)|"
    r"see more|back to (search|jobs)|sign in|join now|\d+ (applicants|views)|promoted|actively recruiting|"
    r"create a job alert|apply for this job|\*|indicates a required field|see who .{1,80} has hired for this role|"
    r"interested in building your career at .{1,80}|get future opportunities sent straight to your email\.?)$",
    re.IGNORECASE)


def clean_jd_text(text: str, min_chars: int = 200, max_chars: int = 10_000) -> str:
    """Normalise pasted JD text and gate on length.

    Raises:
        JDTextError: fewer than ``min_chars`` characters remain after cleaning.
    """
    text = text.replace(" ", " ").replace("​", "").replace("\r\n", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    lines = [line for line in lines if not _NOISE_LINE.match(line)]
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    if len(cleaned) < min_chars:
        raise JDTextError(
            f"job description is too short ({len(cleaned)} chars, need {min_chars}+); paste the full posting")
    return cleaned[:max_chars]


_SYSTEM = """You extract structured data from job descriptions. Rules:
- required_skills: technical skills stated as required / must-have / "you have". Short canonical names
  (e.g. "Python", "Kubernetes", "REST APIs"), no soft skills, no sentences.
- nice_to_have_skills: technical skills marked preferred / bonus / nice to have / plus.
- tech_stack: every technology named anywhere, including the two lists above.
- Skills lists hold technologies only (languages, frameworks, databases, cloud services, tools,
  engineering practices like CI/CD). NOT business domains, products or features ("payments", "casino",
  "wallet", "KYC", "geolocation"), and not generic phrases ("database design", "production deployment").
- seniority: infer from title and years ("Senior", "Staff", "5+ years" ...); 'unspecified' if unclear.
- min_years_experience: the smallest number of years explicitly required, else null.
- responsibilities: at most 8, each under 20 words, in the posting's own wording.
- Never add anything not in the text."""


async def parse_jd(llm: StructuredLLM, storage: Storage | None, text: str) -> tuple[JDStructured, bool]:
    """Return the structured JD for already-cleaned ``text`` and whether it came from cache."""
    key = content_hash(text)
    cache_model = f"{llm.model}|{PARSER_VERSION}"
    if storage and (hit := storage.get_cached_parse("jd", key, cache_model)):
        return JDStructured.model_validate(hit), True
    jd = await llm.generate_json(_SYSTEM, f"JOB DESCRIPTION:\n{text}", JDStructured, max_tokens=1536)
    if storage:
        storage.put_cached_parse("jd", key, cache_model, jd.model_dump())
    return jd, False
