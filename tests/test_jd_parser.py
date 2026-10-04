import pytest

from api.schemas import JDStructured
from api.services.jd_parser import JDTextError, clean_jd_text, parse_jd
from api.services.storage import Storage
from tests.fakes import FakeLLM


def test_clean_strips_job_board_chrome(jd_text):
    cleaned = clean_jd_text(jd_text)
    assert "Apply now" not in cleaned and "Save job" not in cleaned
    assert cleaned.startswith("Senior Backend Engineer (Python)")
    assert "\n\n\n" not in cleaned


def test_clean_enforces_min_length_and_truncates():
    with pytest.raises(JDTextError, match="too short"):
        clean_jd_text("Python dev wanted. Apply now", min_chars=200)
    assert len(clean_jd_text("x" * 500, min_chars=10, max_chars=300)) == 300


async def test_parse_jd_cache_hit_skips_llm(jd_text):
    llm, storage = FakeLLM(), Storage(":memory:")
    text = clean_jd_text(jd_text)
    jd1, c1 = await parse_jd(llm, storage, text)
    jd2, c2 = await parse_jd(llm, storage, text + "  ")
    assert (c1, c2) == (False, True)
    assert jd1 == jd2
    assert llm.count(JDStructured) == 1
