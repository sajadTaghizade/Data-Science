"""Unit tests for the local crawler's pure data-shaping functions (no network)."""

import importlib.util
from pathlib import Path

import pandas as pd

CRAWLER_PATH = Path(__file__).resolve().parents[3] / "Phase 1" / "crawl_stackoverflow.py"
spec = importlib.util.spec_from_file_location("crawl_stackoverflow", CRAWLER_PATH)
crawler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(crawler)


FAKE_ITEMS = [
    {"question_id": 1, "title": "t1", "body": "b1", "tags": ["c++", "qt"],
     "score": 5, "owner": {"user_id": 9, "display_name": "alice"}},
    {"question_id": 1, "title": "dup", "body": "dup", "tags": ["c++"],
     "score": 1, "owner": {"user_id": 9, "display_name": "alice"}},
    {"question_id": 2, "title": "t2", "body": "b2", "tags": ["c++"],
     "score": 3, "owner": {"user_id": 10, "display_name": "bob"}},
]


def test_flatten_items_expands_nested_owner():
    df = crawler.flatten_items(FAKE_ITEMS)
    assert "owner.user_id" in df.columns
    assert "owner.display_name" in df.columns
    # The tag list is preserved as a Python list (later parsed by import_to_db).
    assert df.loc[0, "tags"] == ["c++", "qt"]


def test_ensure_schema_dedupes_and_reindexes():
    df = crawler.flatten_items(FAKE_ITEMS)
    reference = ["question_id", "title", "tags", "owner.user_id", "closed_date"]
    result = crawler.ensure_schema(df, reference)
    assert list(result.columns) == reference          # exact canonical schema
    assert len(result) == 2                            # duplicate question_id 1 removed
    assert result["closed_date"].isna().all()          # absent field filled with NaN
