import json

import httpx
import pytest

from api.config import Settings
from api.schemas import JDStructured
from api.services.model_client import create_llm, inline_refs, load_base_module

GOOD = {"title": "SWE", "company": "", "seniority": "mid", "min_years_experience": None,
        "required_skills": ["Python"], "nice_to_have_skills": [], "tech_stack": ["Python"],
        "responsibilities": ["Build APIs"]}


def _llm(handler, backend="ollama"):
    http = httpx.AsyncClient(base_url="http://model.test", transport=httpx.MockTransport(handler))
    return create_llm(Settings(llm_backend=backend), http_client=http)


def test_inline_refs_removes_defs():
    schema = inline_refs({"$defs": {"A": {"type": "string"}},
                          "properties": {"a": {"$ref": "#/$defs/A"}, "b": {"items": {"$ref": "#/$defs/A"}}}})
    assert schema == {"properties": {"a": {"type": "string"}, "b": {"items": {"type": "string"}}}}
    assert "$ref" not in json.dumps(inline_refs(JDStructured.model_json_schema()))


def test_falls_back_to_vendored_client_without_platform(tmp_path):
    module, source = load_base_module(tmp_path)
    assert source == "vendored copy" and hasattr(module, "OllamaClient")


async def test_ollama_sends_schema_and_validates():
    sent = []

    def handler(req):
        sent.append(json.loads(req.content))
        return httpx.Response(200, json={"message": {"content": json.dumps(GOOD)}, "done_reason": "stop"})

    jd = await _llm(handler).generate_json("sys", "user", JDStructured)
    assert jd.title == "SWE"
    body = sent[0]
    assert body["format"]["properties"]["seniority"]["enum"][0] == "intern"
    assert body["think"] is False and body["options"]["temperature"] == 0.0 and body["options"]["num_ctx"] == 8192


async def test_retries_invalid_then_doubles_budget_when_truncated():
    sent = []
    replies = iter([{"message": {"content": '{"title": "SW'}, "done_reason": "length"},
                    {"message": {"content": json.dumps(GOOD)}, "done_reason": "stop"}])

    def handler(req):
        sent.append(json.loads(req.content))
        return httpx.Response(200, json=next(replies))

    await _llm(handler).generate_json("s", "u", JDStructured, max_tokens=100)
    assert [b["options"]["num_predict"] for b in sent] == [100, 200]


async def test_raises_after_two_invalid_answers():
    llm = _llm(lambda req: httpx.Response(200, json={"message": {"content": "{}"}}))
    with pytest.raises(Exception, match="valid JDStructured"):
        await llm.generate_json("s", "u", JDStructured)


async def test_vllm_uses_response_format():
    sent = []

    def handler(req):
        sent.append((req.url.path, json.loads(req.content)))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(GOOD)},
                                                      "finish_reason": "stop"}]})

    await _llm(handler, backend="vllm").generate_json("s", "u", JDStructured)
    path, body = sent[0]
    assert path == "/v1/chat/completions"
    assert body["response_format"]["type"] == "json_schema"


async def test_concurrency_is_capped_client_side():
    import asyncio
    in_flight = peak = 0

    async def handler(req):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return httpx.Response(200, json={"message": {"content": json.dumps(GOOD)}, "done_reason": "stop"})

    http = httpx.AsyncClient(base_url="http://model.test", transport=httpx.MockTransport(handler))
    llm = create_llm(Settings(llm_max_concurrency=2), http_client=http)
    await asyncio.gather(*(llm.generate_json("s", "u", JDStructured) for _ in range(6)))
    assert peak == 2
