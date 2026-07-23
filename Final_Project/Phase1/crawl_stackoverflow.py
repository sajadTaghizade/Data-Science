"""Crawl Stack Overflow C++ questions from the Stack Exchange API.

Based on the team's original scraper (``requests`` + retry strategy + optional
local proxy, ``/search/advanced`` with ``filter=withbody``), but crawls by
**tag** (``tagged=c++``) instead of a full-text query. The original ``q='c++'``
full-text search only matches a few thousand questions — which is why the first
dataset was small — whereas the ``c++`` tag covers ~800k questions.

The cloud/CI environment blocks outbound access to api.stackexchange.com, so run
this **locally**. It reproduces the exact column layout of
``Phase1/stackoverflow_questions.csv`` so the rest of the Phase 2 pipeline keeps
working unchanged after you scale the data up.

Works WITHOUT an API key
------------------------
Without a key the API allows ~300 requests/day per IP (100 questions/request),
so a single run can collect up to ~30,000 questions. To go beyond that, this
crawler is **resumable**: it loads the existing CSV, skips question IDs already
saved, and appends only new ones — so you can run it again tomorrow (or after
the quota resets) and the dataset keeps growing. It also saves incrementally
every few pages, so an interrupted/quota-stopped run never loses progress.

Examples
--------
    pip install requests pandas
    # Keyless: grab as much C++-tagged data as today's quota allows (~30k).
    python crawl_stackoverflow.py --pages 300
    # Run it again another day to accumulate more (dedup is automatic):
    python crawl_stackoverflow.py --pages 300
    # Behind a local proxy/VPN, as in the original script:
    python crawl_stackoverflow.py --pages 300 --proxy http://127.0.0.1:10808
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


def flatten_items(items: list[dict]) -> pd.DataFrame:
    """Flatten nested API items exactly as the original crawl did (json_normalize)."""
    return pd.json_normalize(items)


def ensure_schema(df: pd.DataFrame, reference_columns: list[str] | None) -> pd.DataFrame:
    """De-duplicate by question_id and reindex to the canonical column order."""
    df = df.drop_duplicates(subset=["question_id"])
    if reference_columns:
        df = df.reindex(columns=reference_columns)
    return df.reset_index(drop=True)


def reference_columns_from(output_path: Path) -> list[str] | None:
    """Reuse the existing CSV's columns as the canonical schema, if it exists."""
    if output_path.exists():
        return list(pd.read_csv(output_path, nrows=0).columns)
    return None


def load_existing(output_path: Path) -> tuple[pd.DataFrame | None, set]:
    """Load the existing dataset (for resuming) and its known question IDs."""
    if not output_path.exists():
        return None, set()
    existing = pd.read_csv(output_path)
    ids = set(existing["question_id"].dropna().astype("int64")) if "question_id" in existing else set()
    return existing, ids


def combine(existing_df: pd.DataFrame | None, new_items: list[dict],
            reference_columns: list[str] | None) -> pd.DataFrame:
    """Append newly crawled items to the existing dataset and normalize the schema."""
    frames = [df for df in (existing_df, flatten_items(new_items)) if df is not None and not df.empty]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return ensure_schema(combined, reference_columns)


def crawl(tagged: str | None = "c++", pages: int = 300, key: str | None = None,
          proxy: str | None = None, target: int | None = None, sort: str = "creation",
          query: str | None = None, output_path: Path | None = None, save_every: int = 20) -> list[dict]:
    """Page through ``/search/advanced``; skip already-saved IDs and save incrementally.

    Use ``tagged='c++'`` (default) to access the full ~800k C++ questions. A
    full-text ``query`` (the original scraper's ``q='c++'``) only matches a few
    thousand questions, which is why the original dataset was small.
    """
    session = build_session()
    proxies = {"http": proxy, "https": proxy} if proxy else None

    existing_df, seen = (load_existing(output_path) if output_path else (None, set()))
    reference = reference_columns_from(output_path) if output_path else None
    starting_count = len(seen)
    new_items: list[dict] = []

    for page in range(1, pages + 1):
        params = {
            "order": "desc", "sort": sort, "site": "stackoverflow",
            "pagesize": 100, "page": page, "filter": WITHBODY_FILTER,
        }
        if tagged:
            params["tagged"] = tagged
        if query:
            params["q"] = query
        if key:
            params["key"] = key

        try:
            response = session.get(API_URL, params=params, proxies=proxies, timeout=15)
            data = response.json()
        except Exception as exc:  # network/JSON errors: log and continue
            logger.warning("Error on page %d: %s", page, exc)
            time.sleep(5)
            continue

        fresh = [item for item in data.get("items", []) if item.get("question_id") not in seen]
        for item in fresh:
            seen.add(item.get("question_id"))
        new_items.extend(fresh)
        logger.info("page %d: +%d new (dataset %d), quota_remaining=%s",
                    page, len(fresh), starting_count + len(new_items), data.get("quota_remaining"))

        if output_path and new_items and page % save_every == 0:
            combine(existing_df, new_items, reference).to_csv(output_path, index=False)
            logger.info("Saved progress: %d questions -> %s", starting_count + len(new_items), output_path)

        if data.get("backoff"):
            logger.warning("API requested backoff of %ss", data["backoff"])
            time.sleep(data["backoff"] + 1)
        if target and starting_count + len(new_items) >= target:
            break
        if not data.get("has_more", False):
            logger.info("No more pages available from the API.")
            break
        if data.get("quota_remaining", 1) <= 0:
            logger.warning("API quota exhausted; stopping (progress is saved).")
            break
        time.sleep(1)

    return new_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl Stack Overflow C++ questions (resumable, no key needed).")
    parser.add_argument("--tagged", default="c++", help="tag to crawl (default: c++ — the full ~800k pool)")
    parser.add_argument("--query", default=None, help="optional full-text query (the original scraper's q='c++')")
    parser.add_argument("--pages", type=int, default=300, help="max pages to fetch (100 questions each)")
    parser.add_argument("--target", type=int, default=None, help="optional cap on total dataset size")
    parser.add_argument("--key", default=None, help="optional Stack Apps API key (raises the daily quota)")
    parser.add_argument("--proxy", default=None, help="optional proxy URL, e.g. http://127.0.0.1:10808")
    parser.add_argument("--sort", default="creation", choices=["activity", "creation", "votes"],
                        help="API sort order (default: creation — stable for deep pagination)")
    parser.add_argument("--save-every", type=int, default=20, help="save progress every N pages")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output CSV path")
    args = parser.parse_args()

    existing_df, existing_ids = load_existing(args.output)
    reference = reference_columns_from(args.output)
    new_items = crawl(tagged=args.tagged, pages=args.pages, key=args.key, proxy=args.proxy,
                      target=args.target, sort=args.sort, query=args.query,
                      output_path=args.output, save_every=args.save_every)

    if not new_items and existing_df is None:
        logger.error("No items collected; nothing written.")
        return

    final = combine(existing_df, new_items, reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.output, index=False)
    logger.info("Done. Added %d new questions; dataset now has %d (was %d).",
                len(new_items), len(final), len(existing_ids))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
