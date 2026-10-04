"""Streamlit UI for resume-jd-matcher: ``streamlit run frontend/app.py``.

Talks to the local API (``API_URL``, default http://localhost:8010). Nothing here calls a model directly.
"""
from __future__ import annotations

import html
import os
from typing import Any

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8010").rstrip("/")
MIN_JD_CHARS = 200
ANALYZE_TIMEOUT = 900.0  # a cold run on an 8 GB laptop takes ~5 min

st.set_page_config(page_title="Resume-JD Matcher", page_icon="📄", layout="wide")
st.markdown("""
<style>
.tag {display:inline-block; padding:2px 10px; margin:2px 4px 2px 0; border-radius:12px; font-size:0.85rem;}
.tag-miss {background:#fde8e8; color:#9b1c1c; border:1px solid #f8b4b4;}
.tag-nice {background:#fdf6e3; color:#8a6d0b; border:1px solid #f3d98b;}
.tag-hit {background:#e6f4ea; color:#1e6b34; border:1px solid #a8d5b5;}
.before {background:rgba(220,38,38,0.06); border-left:4px solid #dc2626; padding:10px 12px; border-radius:4px;}
.after {background:rgba(22,163,74,0.06); border-left:4px solid #16a34a; padding:8px 12px; border-radius:4px;
        margin-bottom:6px;}
.gauge {height:22px; border-radius:11px; background:rgba(128,128,128,0.18); overflow:hidden;}
.gauge > div {height:100%; border-radius:11px;}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- API helpers

def api_error(resp: httpx.Response) -> str:
    try:
        detail = resp.json().get("detail")
    except ValueError:
        return f"HTTP {resp.status_code}"
    if isinstance(detail, dict):
        return detail.get("message") or str(detail)
    return str(detail)


def call(method: str, path: str, timeout: float = 30.0, **kw: Any) -> tuple[Any | None, str | None]:
    """(json, None) on success, (None, message) on failure."""
    try:
        resp = httpx.request(method, f"{API_URL}{path}", timeout=timeout, **kw)
    except httpx.TimeoutException:
        return None, "The API took too long to answer."
    except httpx.HTTPError:
        return None, f"Can't reach the API at {API_URL}. Start it with `./start.sh` (or `uvicorn api.main:app --port 8010`)."
    if resp.status_code >= 400:
        return None, api_error(resp)
    return resp.json(), None


# ---------------------------------------------------------------- rendering

def color_for(score: float) -> str:
    return "#16a34a" if score >= 75 else "#d97706" if score >= 50 else "#dc2626"


def gauge(label: str, score: float, big: bool = False) -> None:
    size = "2.6rem" if big else "1.1rem"
    st.markdown(
        f"<div style='display:flex;justify-content:space-between;align-items:baseline'>"
        f"<span>{html.escape(label)}</span><span style='font-size:{size};font-weight:700;color:{color_for(score)}'>"
        f"{score:.0f}%</span></div><div class='gauge'><div style='width:{max(2, min(score, 100))}%;"
        f"background:{color_for(score)}'></div></div>", unsafe_allow_html=True)


def tags(items: list[str], cls: str) -> str:
    return "".join(f"<span class='tag {cls}'>{html.escape(i)}</span>" for i in items) or "<i>none</i>"


def render_result(r: dict[str, Any]) -> None:
    jd = r["jd"]
    st.subheader(f"{jd.get('title') or 'Role'}{' at ' + jd['company'] if jd.get('company') else ''}")
    left, right = st.columns([1, 1.4])
    with left:
        gauge("Overall match", r["overall_match_score"], big=True)
        st.write("")
        s = r["section_scores"]
        sen = r["seniority"]
        gauge("Skills", s["skills"])
        gauge("Experience relevance", s["experience"])
        gauge(f"Seniority ({sen['verdict']}: {sen['candidate_years']:g} yrs vs "
              f"{jd['seniority']}{', min ' + format(sen['jd_min_years'], 'g') if sen['jd_min_years'] is not None else ''})",
              s["seniority"])
    with right:
        st.info(r["summary"])
        bd = r["skills_breakdown"]
        st.markdown("**Missing required skills**<br>" + tags(bd["required_missing"], "tag-miss"), unsafe_allow_html=True)
        st.markdown("**Missing nice-to-haves**<br>" + tags(bd["nice_missing"], "tag-nice"), unsafe_allow_html=True)
        st.markdown("**Matched**<br>" + tags(r["skills_matched"], "tag-hit"), unsafe_allow_html=True)
        if bd.get("matched_via_text"):
            st.caption("Found in your bullets but not your skills section (consider listing them): "
                       + ", ".join(bd["matched_via_text"]))

    st.markdown("### Bullet rewrites")
    st.caption("Rewrites mirror the JD's wording. ⚠ marks claims your resume doesn't support. Keep only what's true.")
    if not r["bullet_suggestions"]:
        st.write("No rewrite suggestions.")
    for b in r["bullet_suggestions"]:
        before, after = st.columns(2)
        with before:
            st.markdown(f"<div class='before'><small>BEFORE · {html.escape(b['role'])}</small><br>"
                        f"{html.escape(b['original'])}</div>", unsafe_allow_html=True)
            st.caption(b["why"])
        with after:
            for i, text in enumerate(b["suggestions"], 1):
                warn = [w.split(": ", 1)[1] for w in b["warnings"] if w.startswith(f"rewrite {i}:")]
                flag = f"<br><small>⚠ {html.escape('; '.join(warn))}</small>" if warn else ""
                st.markdown(f"<div class='after'><small>AFTER {i}</small><br>{html.escape(text)}{flag}</div>",
                            unsafe_allow_html=True)
        st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Missing-skill advice")
        for a in r["missing_skill_advice"] or []:
            st.markdown(f"- **{a['skill']}**: {a['advice']}")
        if r["qualification_signals"]:
            st.markdown("### Qualification signals")
            for q in r["qualification_signals"]:
                st.markdown(f"- {'⬆ over' if q['kind'] == 'over' else '⬇ under'}: {q['signal']}")
    with c2:
        st.markdown("### Experience vs responsibilities")
        icon = {"strong": "🟢", "partial": "🟡", "none": "🔴"}
        for e in r["experience_matches"]:
            with st.expander(f"{icon[e['verdict']]} {e['responsibility']}"):
                st.write(e["reason"])
                for ev in e["evidence"]:
                    st.caption(f"• {ev}")
    with st.expander("JD text that was analysed"):
        st.text(r["jd_text"])
    st.caption(f"model {r['model']} · JD parse cached: {r['jd_cached']} · resume parse cached: {r['resume_cached']}"
               f" · timings {r['timings_s']}")


# ---------------------------------------------------------------- page

health, health_err = call("GET", "/healthz", timeout=5)
with st.sidebar:
    st.markdown("### Resume-JD Matcher")
    st.caption("Everything runs locally: no resume or JD leaves this machine.")
    if health_err:
        st.error(health_err)
    elif not health["model_ok"]:
        st.warning(f"API is up but the model backend isn't reachable ({health['backend']}). Is Ollama running?")
    else:
        st.success(f"Model: {health['model']} ({health['backend']})")

tab_analyze, tab_history = st.tabs(["Analyze", "History"])

with tab_analyze:
    col_resume, col_jd = st.columns([1, 1.6])
    with col_resume:
        resume_file = st.file_uploader("Resume (PDF or DOCX)", type=["pdf", "docx"])
        company = st.text_input("Company (optional)")
        role = st.text_input("Role (optional)")
    with col_jd:
        mode = st.segmented_control("Job description", ["Paste JD", "Job posting link"], default="Paste JD",
                                    key="jd_mode") or "Paste JD"
        if mode == "Paste JD":
            jd_value = st.text_area("Paste the full job description", height=320, key="jd_paste")
            source_url = None
        else:
            url_col, btn_col = st.columns([4, 1])
            with url_col:
                url = st.text_input("Job posting URL", placeholder="https://jobs.lever.co/...", key="jd_url",
                                    label_visibility="collapsed")
            with btn_col:
                fetch_clicked = st.button("Fetch & Preview", disabled=not url.strip(), width="stretch")
            if fetch_clicked:
                with st.spinner("Fetching the posting..."):
                    fetched, err = call("POST", "/jd/fetch", timeout=40, json={"url": url})
                if err:
                    st.session_state.pop("jd_preview", None)
                    st.session_state["fetch_error"] = err
                else:
                    st.session_state["fetch_error"] = None
                    st.session_state["jd_preview"] = fetched["text"]
                    st.session_state["jd_preview_url"] = fetched["url"]
                    st.session_state["jd_preview_meta"] = (
                        f"Extracted {fetched['chars']} characters via {fetched['method']}"
                        f"{' (headless browser)' if fetched['renderer'] == 'playwright' else ''}"
                        f"{' · cached' if fetched['cached'] else ''}. Check it and edit if anything is off.")
            if st.session_state.get("fetch_error"):
                st.error(f"Couldn't use that link: {st.session_state['fetch_error']}  \n"
                         "Switch to **Paste JD** and paste the text instead.")
            if "jd_preview" in st.session_state:
                st.caption(st.session_state.get("jd_preview_meta", ""))
            jd_value = st.text_area("Extracted JD (editable)", height=320, key="jd_preview",
                                    placeholder="Fetch a posting to preview its text here, or type/paste it.")
            source_url = st.session_state.get("jd_preview_url")

    jd_len = len((jd_value or "").strip())
    ready = resume_file is not None and jd_len >= MIN_JD_CHARS
    hint = ("Upload a resume. " if resume_file is None else "") + (
        f"JD needs at least {MIN_JD_CHARS} characters ({jd_len} now)." if jd_len < MIN_JD_CHARS else "")
    if st.button("Analyze", type="primary", disabled=not ready):
        files = {"resume": (resume_file.name, resume_file.getvalue(), resume_file.type or "application/octet-stream")}
        data = {"jd_text": jd_value, "company": company, "role": role}
        if source_url:
            data["source_url"] = source_url
        with st.spinner("Analyzing: parsing, matching, gap analysis. Takes 3-5 minutes on a laptop-sized model."):
            result, err = call("POST", "/analyze", timeout=ANALYZE_TIMEOUT, files=files, data=data)
        if err:
            st.error(err)
        else:
            st.session_state["result"] = result
    elif not ready:
        st.caption(hint)

    if st.session_state.get("result"):
        st.divider()
        render_result(st.session_state["result"])

with tab_history:
    rows, err = call("GET", "/history", params={"limit": 500})
    if err:
        st.error(err)
    elif not rows:
        st.write("No analyses yet.")
    else:
        df = pd.DataFrame(rows).rename(columns={
            "created_at": "date", "resume_name": "resume", "overall_score": "overall",
            "skills_score": "skills", "experience_score": "experience", "seniority_score": "seniority"})
        df["date"] = pd.to_datetime(df["date"]).dt.tz_convert(None).dt.strftime("%Y-%m-%d %H:%M")
        sort_by = st.selectbox("Sort by", ["overall", "date", "skills", "experience", "seniority", "company"])
        df = df.sort_values(sort_by, ascending=sort_by == "company", kind="stable")
        st.caption("Click a column header to re-sort; select a row to open it.")
        event = st.dataframe(
            df[["id", "date", "company", "role", "resume", "overall", "skills", "experience", "seniority"]],
            hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
            column_config={c: st.column_config.ProgressColumn(c, min_value=0, max_value=100, format="%.0f")
                           for c in ("overall", "skills", "experience", "seniority")})
        picked = event.selection.rows if event and event.selection else []
        if picked:
            item, err = call("GET", f"/history/{int(df.iloc[picked[0]]['id'])}")
            if err:
                st.error(err)
            else:
                st.divider()
                render_result(item["result"])
