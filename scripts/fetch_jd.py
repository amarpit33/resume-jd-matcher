"""Fetch + extract JD text from job URLs (no LLM), printing what the analyzer would receive.

    python scripts/fetch_jd.py URL [URL ...] [--save-dir data/my_jds] [--ignore-robots]
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.services.jd_fetcher import JDFetcher, JDFetchError  # noqa: E402


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("urls", nargs="+")
    p.add_argument("--save-dir", type=Path)
    p.add_argument("--ignore-robots", action="store_true")
    p.add_argument("--chars", type=int, default=600, help="preview length")
    args = p.parse_args()
    fetcher = JDFetcher(respect_robots=not args.ignore_robots)
    try:
        for url in args.urls:
            print(f"\n=== {url}")
            try:
                jd = await fetcher.fetch(url)
            except JDFetchError as exc:
                print(f"FAILED [{exc.code}] {exc.message}")
                continue
            print(f"ok: {len(jd.text)} chars via {jd.method}/{jd.renderer} | title: {jd.title!r}")
            print(jd.text[: args.chars] + ("..." if len(jd.text) > args.chars else ""))
            if args.save_dir:
                args.save_dir.mkdir(parents=True, exist_ok=True)
                name = re.sub(r"[^a-z0-9]+", "_", (jd.title or url).lower()).strip("_")[:60]
                (args.save_dir / f"{name}.txt").write_text(jd.text)
    finally:
        await fetcher.aclose()


if __name__ == "__main__":
    asyncio.run(main())
