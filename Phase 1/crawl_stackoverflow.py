"""Crawl Stack Overflow C++ questions from the Stack Exchange API.

The cloud CI environment blocks outbound access to api.stackexchange.com, so run
this script **locally** (where the network is open) to (re)generate or grow the
dataset. It reproduces the exact column layout of
``Phase 1/stackoverflow_questions.csv`` so the rest of the Phase 2 pipeline keeps
working unchanged after you scale the data up.

Examples
--------
    pip install requests pandas
    # Grow to 50k questions (an API key lifts the daily quota from 300 to 10000):
    python crawl_stackoverflow.py --target 50000 --key YOUR_APP_KEY
    # Smaller smoke test without a key:
    python crawl_stackoverflow.py --target 1000

Notes
-----
* Without a key the API allows ~300 requests/day per IP (≈30k questions/day at
  pagesize 100). With a free registered key the quota is 10,000 requests/day.
* The script honors the API ``backoff`` field and stops cleanly when the quota
  is exhausted, saving whatever it has gathered so far.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

logger = logging.getLogger("crawl")

API_URL = "https://api.stackexchange.com/2.3/questions"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "stackoverflow_questions.csv"
# Returns the question body alongside the default fields, matching the original crawl.
WITHBODY_FILTER = "withbody"


def fetch_page(params: dict) -> dict:
    """Fetch a single API page, transparently handling gzip responses."""
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def crawl(target: int, tag: str = "c++", key: str | None = None,
          page_size: int = 100, from_date: int | None = None) -> list[dict]:
    """Page through the API until ``target`` questions are collected or the quota ends."""
    items: list[dict] = []
    page = 1
    while len(items) < target:
        params = {
            "order": "desc", "sort": "creation", "tagged": tag,
            "site": "stackoverflow", "filter": WITHBODY_FILTER,
            "pagesize": page_size, "page": page,
        }
        if key:
            params["key"] = key
        if from_date:
            params["fromdate"] = from_date

        payload = fetch_page(params)
        items.extend(payload.get("items", []))
        logger.info("page %d: +%d items (total %d), quota_remaining=%s",
                    page, len(payload.get("items", [])), len(items), payload.get("quota_remaining"))

        if payload.get("backoff"):
            logger.warning("API requested backoff of %ss", payload["backoff"])
            time.sleep(payload["backoff"] + 1)
        if not payload.get("has_more"):
            logger.info("No more pages available from the API.")
            break
        if payload.get("quota_remaining", 1) <= 0:
            logger.warning("API quota exhausted; stopping early.")
            break
        page += 1
        time.sleep(0.2)  # be polite

    return items[:target]


def flatten_items(items: list[dict]) -> pd.DataFrame:
    """Flatten nested API items exactly as the original crawl did (json_normalize)."""
    return pd.json_normalize(items)


def ensure_schema(df: pd.DataFrame, reference_columns: list[str] | None) -> pd.DataFrame:
    """Reindex to the canonical column order so the import step never breaks."""
    df = df.drop_duplicates(subset=["question_id"])
    if reference_columns:
        df = df.reindex(columns=reference_columns)
    return df.reset_index(drop=True)


def reference_columns_from(output_path: Path) -> list[str] | None:
    """Reuse the existing CSV's columns as the canonical schema, if it exists."""
    if output_path.exists():
        return list(pd.read_csv(output_path, nrows=0).columns)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl Stack Overflow C++ questions.")
    parser.add_argument("--target", type=int, default=50000, help="number of questions to collect")
    parser.add_argument("--tag", default="c++", help="tag to crawl (default: c++)")
    parser.add_argument("--key", default=None, help="Stack Apps API key (raises the daily quota)")
    parser.add_argument("--page-size", type=int, default=100, help="API page size (max 100)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output CSV path")
    args = parser.parse_args()

    reference = reference_columns_from(args.output)
    items = crawl(args.target, tag=args.tag, key=args.key, page_size=args.page_size)
    if not items:
        logger.error("No items collected; nothing written.")
        return

    df = ensure_schema(flatten_items(items), reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info("Wrote %d questions to %s", len(df), args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
