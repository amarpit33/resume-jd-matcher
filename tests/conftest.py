import os
from pathlib import Path

import pytest

# Settings must resolve without a local-llm-platform checkout or a running model.
os.environ.setdefault("LLM_BACKEND", "ollama")
os.environ.setdefault("LLM_BASE_URL", "http://model.test")
os.environ.setdefault("LLM_MODEL", "test-model")
os.environ.setdefault("LOCAL_LLM_PLATFORM_DIR", "/nonexistent")

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLES = Path(__file__).resolve().parents[1] / "data" / "sample_resumes"


@pytest.fixture
def jd_text() -> str:
    return (FIXTURES / "sample_jd.txt").read_text()
