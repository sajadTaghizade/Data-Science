"""Shared pytest fixtures: a tiny, deterministic synthetic corpus.

The fixture mimics the columns the real preprocessed questions have (cleaned
text, tags, creation date, engagement counts) so the recommender models,
metrics, and inference can be tested without touching the database or the full
2.5k dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make ``scripts/`` importable when tests run from the project root.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# Three clear topics so tag-overlap relevance and text similarity are meaningful.
_ROWS = [
    (1, "std vector resize", "vector", "2020-01-01"),
    (2, "std vector push back capacity", "vector", "2020-02-01"),
    (3, "vector of pointers memory", "vector|memory", "2020-03-01"),
    (4, "template specialization example", "templates", "2020-04-01"),
    (5, "variadic template parameter pack", "templates", "2020-05-01"),
    (6, "template sfinae enable if", "templates|metaprogramming", "2020-06-01"),
    (7, "std thread join detach", "multithreading", "2020-07-01"),
    (8, "mutex lock guard thread safety", "multithreading", "2020-08-01"),
    (9, "condition variable thread wait", "multithreading", "2020-09-01"),
    (10, "std vector emplace back move", "vector", "2020-10-01"),
    (11, "template class member function", "templates", "2020-11-01"),
    (12, "thread pool worker queue", "multithreading|concurrency", "2020-12-01"),
]


@pytest.fixture
def synthetic_questions() -> pd.DataFrame:
    """A 12-row preprocessed-like DataFrame with three topical clusters."""
    df = pd.DataFrame(_ROWS, columns=["question_id", "text", "tags", "creation_at"])
    # The real pipeline up-weights the title inside document_clean; emulate that.
    df["title_clean"] = df["text"] + " c++"
    df["document_clean"] = (df["text"] + " ") * 2 + "c++"
    df["tag_list"] = df["tags"].map(lambda value: value.split("|"))
    df["creation_at"] = pd.to_datetime(df["creation_at"], utc=True)
    df["view_count"] = df["question_id"] * 10
    df["score"] = df["question_id"]
    return df.drop(columns="text")
