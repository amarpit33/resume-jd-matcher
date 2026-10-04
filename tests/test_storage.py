from api.services.storage import Storage


def _result(score):
    return {"overall_match_score": score, "section_scores": {"skills": 1, "experience": 2, "seniority": 3}}


def test_history_roundtrip_newest_first():
    s = Storage(":memory:")
    a = s.save_analysis(_result(55), resume_hash="r1", jd_hash="j", company="Acme", role="SWE")
    b = s.save_analysis(_result(70), resume_hash="r2", jd_hash="j", resume_name="v2.pdf")
    rows = s.list_analyses()
    assert [r["id"] for r in rows] == [b, a]
    assert rows[1]["company"] == "Acme" and "result_json" not in rows[0]
    assert s.get_analysis(a)["result"]["overall_match_score"] == 55
    assert s.get_analysis(999) is None


def test_parse_cache_keyed_by_kind_hash_model():
    s = Storage(":memory:")
    s.put_cached_parse("jd", "h", "m1", {"x": 1})
    assert s.get_cached_parse("jd", "h", "m1") == {"x": 1}
    assert s.get_cached_parse("jd", "h", "m2") is None
    assert s.get_cached_parse("resume", "h", "m1") is None
