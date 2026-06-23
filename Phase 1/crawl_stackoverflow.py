"""Crawl Stack Overflow C++ questions from the Stack Exchange API.

Based on the team's original scraper: it queries the ``/search/advanced``
endpoint with a full-text query and ``sort=activity`` (so the collected
questions match the original dataset's distribution), using ``requests`` with a
retry strategy and an optional local proxy.

The cloud/CI environment blocks outbound access to api.stackexchange.com, so run
this **locally**. It reproduces the exact column layout of
``Phase 1/stackoverflow_questions.csv`` so the rest of the Phase 2 pipeline keeps
working unchanged after you scale the data up.

Examples
--------
    pip install requests pandas
    # Grow to ~50k questions (500 pages x 100). An API key lifts the daily quota.
    python crawl_stackoverflow.py --pages 500 --key YOUR_APP_KEY
    # Behind a local proxy (e.g. a VPN client), as in the original script:
    python crawl_stackoverflow.py --pages 500 --proxy http://127.0.0.1:10808

Notes
-----
* Without a key the API allows ~300 requests/day per IP (100 questions/request).
  A free Stack Apps key raises the quota to 10,000 requests/day.
* The script honors the API ``backoff`` field and stops cleanly when the quota
  is exhausted, saving whatever it has gathered so far.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger("crawl")

API_URL = "https://api.stackexchange.com/2.3/search/advanced"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "stackoverflow_questions.csv"
WITHBODY_FILTER = "withbody"  # includes the question body, matching the original crawl


def build_session():
    """Create a requests session with the original retry strategy (lazy import)."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def crawl(query: str, pages: int, key: str | None = None, proxy: str | None = None,
          target: int | None = None) -> list[dict]:
    """Page through ``/search/advanced`` until ``pages`` / ``target`` / quota ends."""
    session = build_session()
    proxies = {"http": proxy, "https": proxy} if proxy else None
    items: list[dict] = []

    for page in range(1, pages + 1):
        params = {
            "order": "desc", "sort": "activity", "site": "stackoverflow",
            "q": query, "pagesize": 100, "page": page, "filter": WITHBODY_FILTER,
        }
        if key:
            params["key"] = key

        try:
            response = session.get(API_URL, params=params, proxies=proxies, timeout=15)
            data = response.json()
        except Exception as exc:  # network/JSON errors: log and continue
            logger.warning("Error on page %d: %s", page, exc)
            time.sleep(5)
            continue

        items.extend(data.get("items", []))
        logger.info("page %d: +%d items (total %d), quota_remaining=%s",
                    page, len(data.get("items", [])), len(items), data.get("quota_remaining"))

        if data.get("backoff"):
            logger.warning("API requested backoff of %ss", data["backoff"])
            time.sleep(data["backoff"] + 1)
        if target and len(items) >= target:
            break
        if not data.get("has_more", False):
            logger.info("No more pages available from the API.")
            break
        if data.get("quota_remaining", 1) <= 0:
            logger.warning("API quota exhausted; stopping early.")
            break
        time.sleep(1)

    return items[:target] if target else items


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
    parser.add_argument("--query", default="c++", help="full-text search query (default: c++)")
    parser.add_argument("--pages", type=int, default=500, help="max pages to fetch (100 questions each)")
    parser.add_argument("--target", type=int, default=None, help="optional cap on total questions")
    parser.add_argument("--key", default=None, help="Stack Apps API key (raises the daily quota)")
    parser.add_argument("--proxy", default=None, help="optional proxy URL, e.g. http://127.0.0.1:10808")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output CSV path")
    args = parser.parse_args()

    reference = reference_columns_from(args.output)
    items = crawl(args.query, pages=args.pages, key=args.key, proxy=args.proxy, target=args.target)
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
