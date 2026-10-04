"""SQLite persistence: analysis history plus parse caches.

Parse caches are keyed by (content hash, model), so testing several resume
versions against one JD parses the JD once, and switching models re-parses.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    company TEXT,
    role TEXT,
    resume_name TEXT,
    resume_hash TEXT NOT NULL,
    jd_hash TEXT NOT NULL,
    overall_score REAL NOT NULL,
    skills_score REAL NOT NULL,
    experience_score REAL NOT NULL,
    seniority_score REAL NOT NULL,
    result_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS parse_cache (
    kind TEXT NOT NULL,          -- 'jd' | 'resume'
    content_hash TEXT NOT NULL,
    model TEXT NOT NULL,
    parsed_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (kind, content_hash, model)
);
"""

_SUMMARY_COLS = ("id", "created_at", "company", "role", "resume_name", "overall_score",
                 "skills_score", "experience_score", "seniority_score")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Storage:
    """Thread-safe wrapper around one SQLite connection."""

    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    # ----- parse cache -----

    def get_cached_parse(self, kind: str, content_hash: str, model: str) -> dict[str, Any] | None:
        """Return the cached parse for this content and model, if any."""
        with self._lock:
            row = self._conn.execute(
                "SELECT parsed_json FROM parse_cache WHERE kind=? AND content_hash=? AND model=?",
                (kind, content_hash, model)).fetchone()
        return json.loads(row["parsed_json"]) if row else None

    def put_cached_parse(self, kind: str, content_hash: str, model: str, parsed: dict[str, Any]) -> None:
        """Insert or replace a cached parse."""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO parse_cache VALUES (?, ?, ?, ?, ?)",
                (kind, content_hash, model, json.dumps(parsed), _now()))

    # ----- history -----

    def save_analysis(self, result: dict[str, Any], *, resume_hash: str, jd_hash: str,
                      resume_name: str | None = None, company: str | None = None,
                      role: str | None = None) -> int:
        """Persist one analysis; returns its id."""
        scores = result["section_scores"]
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO analyses (created_at, company, role, resume_name, resume_hash, jd_hash, "
                "overall_score, skills_score, experience_score, seniority_score, result_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (_now(), company, role, resume_name, resume_hash, jd_hash, result["overall_match_score"],
                 scores["skills"], scores["experience"], scores["seniority"], json.dumps(result)))
            return int(cur.lastrowid)

    def list_analyses(self, limit: int = 100) -> list[dict[str, Any]]:
        """Newest-first summaries (no full result payload)."""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {', '.join(_SUMMARY_COLS)} FROM analyses ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_analysis(self, analysis_id: int) -> dict[str, Any] | None:
        """Full stored result for one analysis."""
        with self._lock:
            row = self._conn.execute(
                f"SELECT {', '.join(_SUMMARY_COLS)}, result_json FROM analyses WHERE id=?",
                (analysis_id,)).fetchone()
        if not row:
            return None
        out = dict(row)
        out["result"] = json.loads(out.pop("result_json"))
        return out
