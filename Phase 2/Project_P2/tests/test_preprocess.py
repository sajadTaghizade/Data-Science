"""Unit tests for text cleaning, tag handling, and the temporal split."""

import pandas as pd

from scripts.feature_engineering import temporal_split, weight_grid
from scripts.preprocess import (
    normalize_tags,
    normalize_technical_text,
    secondary_tags,
    strip_html_keep_code_markers,
)


def test_normalize_technical_text_preserves_cpp_tokens():
    cleaned = normalize_technical_text("How to use C++ std::vector and nullptr?")
    assert "c++" in cleaned
    assert "std::vector" in cleaned
    assert "nullptr" in cleaned
    assert cleaned == cleaned.lower()


def test_strip_html_keeps_code_marker_and_counts_blocks():
    text, count = strip_html_keep_code_markers("<p>hello</p><pre>int main(){}</pre>")
    assert count == 1
    assert "CODE_BLOCK" in text
    assert "<" not in text  # tags removed


def test_strip_html_handles_missing_value():
    text, count = strip_html_keep_code_markers(None)
    assert text == ""
    assert count == 0


def test_normalize_tags_dedupes_sorts_lowercases():
    assert normalize_tags("C++|Qt|qt|") == ["c++", "qt"]
    assert normalize_tags("boost, c++") == ["boost", "c++"]
    assert normalize_tags(None) == []


def test_secondary_tags_drops_global():
    assert secondary_tags(["c++", "qt", "cpp"]) == {"qt"}


def test_temporal_split_is_ordered_and_disjoint():
    df = pd.DataFrame({
        "question_id": range(100),
        "creation_at": pd.date_range("2010-01-01", periods=100, freq="D", tz="UTC"),
    })
    train, val, test = temporal_split(df, train_frac=0.7, val_frac=0.15)
    assert len(train) + len(val) + len(test) == len(df)
    # No leakage across time: train is strictly older than val, which is older than test.
    assert train["creation_at"].max() <= val["creation_at"].min()
    assert val["creation_at"].max() <= test["creation_at"].min()
    # No overlapping questions between splits.
    ids = set(train["question_id"]) | set(val["question_id"]) | set(test["question_id"])
    assert len(ids) == len(df)


def test_weight_grid_entries_sum_to_one():
    for weights in weight_grid():
        assert abs(sum(weights.values()) - 1.0) < 1e-9
