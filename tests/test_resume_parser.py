import pytest

from api.services.resume_parser import (ResumeParseError, bullets_with_ids, clean_text, content_hash,
                                        extract_text, structure_resume)
from api.services.storage import Storage
from tests.conftest import SAMPLES
from tests.fakes import RESUME, FakeLLM


@pytest.mark.parametrize("name", ["sample_resume.pdf", "sample_resume.docx"])
def test_extracts_pdf_and_docx(name):
    text = extract_text(name, (SAMPLES / name).read_bytes())
    assert "Finlytics" in text and "Celery workers" in text
    assert "- Moved nightly reconciliation" in text  # bullets keep a marker in both formats


def test_rejects_unsupported_and_empty():
    with pytest.raises(ResumeParseError, match="unsupported"):
        extract_text("resume.png", b"...")
    with pytest.raises(ResumeParseError, match="scanned"):
        extract_text("resume.txt", b"hi")
    with pytest.raises(ResumeParseError, match="could not read"):
        extract_text("resume.pdf", b"not a pdf at all" * 20)


def test_clean_text_normalises_bullets_and_space():
    assert clean_text("• one\n\n\n ●  two  ") == "- one\n\n- two"


def test_content_hash_ignores_whitespace_and_case():
    assert content_hash("Senior  Engineer\n") == content_hash("senior engineer")


def test_bullet_ids_are_sequential_across_roles():
    ids = [(b.id, b.role) for b in bullets_with_ids(RESUME)]
    assert ids[0] == ("B1", "Software Engineer @ Finlytics")
    assert ids[-1] == ("B4", "Associate Engineer @ DataNest")


async def test_structure_resume_is_cached_per_model():
    llm, storage = FakeLLM(), Storage(":memory:")
    _, cached1 = await structure_resume(llm, storage, "resume text", "2026-10-04")
    _, cached2 = await structure_resume(llm, storage, "resume  TEXT", "2026-10-04")
    assert (cached1, cached2) == (False, True)
    llm.model = "other-model"
    _, cached3 = await structure_resume(llm, storage, "resume text", "2026-10-04")
    assert cached3 is False
