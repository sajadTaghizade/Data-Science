"""Perform deterministic text cleaning and preprocessing for question retrieval."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

try:  # Supports both direct script execution and imports from the EDA notebook.
    from .database_connection import PROJECT_ROOT
    from .load_data import load_question_dataframe
except ImportError:  # pragma: no cover - direct script execution
    from database_connection import PROJECT_ROOT
    from load_data import load_question_dataframe


PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessed_questions.csv"
REPORT_PATH = PROJECT_ROOT / "data" / "reports" / "preprocessing_report.json"
CORE_TEXT_COLUMNS = ["title", "body_html"]


def html_to_text(value: Any) -> str:
    """Decode HTML and normalize whitespace without removing technical punctuation."""
    if pd.isna(value):
        return ""
    soup = BeautifulSoup(html.unescape(str(value)), "html.parser")
    for element in soup(["script", "style"]):
        element.decompose()
    text = soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def normalise_document(value: Any) -> str:
    """Create a stable text representation while preserving tokens such as C++ and std::vector."""
    return html_to_text(value).lower()


def preprocess_dataframe(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean model inputs and record every data-quality decision."""
    df = raw_df.copy()
    report: dict[str, Any] = {"input_rows": int(len(df))}
    report["duplicate_question_ids_removed"] = int(df.duplicated(subset=["question_id"]).sum())
    df = df.drop_duplicates(subset=["question_id"], keep="first")

    missing_core = df[CORE_TEXT_COLUMNS].isna().any(axis=1)
    blank_core = df[CORE_TEXT_COLUMNS].fillna("").apply(
        lambda column: column.astype(str).str.strip().eq("")
    ).any(axis=1)
    invalid_text = missing_core | blank_core
    report["missing_or_blank_core_text_removed"] = int(invalid_text.sum())
    df = df.loc[~invalid_text].copy()

    df["title_text"] = df["title"].map(normalise_document)
    df["body_text"] = df["body_html"].map(normalise_document)
    df["document_text"] = (df["title_text"] + " " + df["body_text"]).str.strip()
    empty_documents = df["document_text"].eq("")
    report["empty_documents_removed"] = int(empty_documents.sum())
    df = df.loc[~empty_documents].copy()

    df["tags"] = df["tags"].fillna("").map(
        lambda value: "|".join(sorted({tag.strip() for tag in str(value).split("|") if tag.strip()}))
    )
    df["tag_count"] = df["tags"].map(lambda value: len(value.split("|")) if value else 0)
    report["missing_tag_lists"] = int((df["tag_count"] == 0).sum())

    for column in ["creation_at", "last_activity_at", "last_edit_at", "closed_at"]:
        df[column] = pd.to_datetime(df[column], utc=True, errors="coerce")

    numeric_columns = ["view_count", "answer_count", "score", "tag_count"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    report["numeric_missing_after_coercion"] = {
        column: int(df[column].isna().sum()) for column in numeric_columns
    }
    df[numeric_columns] = df[numeric_columns].fillna(0)

    correlation_matrix = df[numeric_columns].corr().round(4)
    high_correlation_pairs = []
    for index, first_feature in enumerate(numeric_columns):
        for second_feature in numeric_columns[index + 1:]:
            correlation = float(correlation_matrix.loc[first_feature, second_feature])
            if abs(correlation) >= 0.90:
                high_correlation_pairs.append({
                    "feature_1": first_feature,
                    "feature_2": second_feature,
                    "correlation": correlation,
                })
    report["numeric_correlation_matrix"] = correlation_matrix.to_dict()
    report["high_correlation_pairs_abs_ge_0_90"] = high_correlation_pairs
    report["outlier_strategy"] = (
        "Retain legitimate high-engagement questions, then apply log/signed-log transformations "
        "and standardization in feature_engineering.py instead of deleting informative records."
    )

    report["output_rows"] = int(len(df))
    report["excluded_from_similarity_model"] = [
        "question_url", "accepted_answer_id", "closed_reason", "raw owner fields",
        "sparse migration metadata", "raw engagement counts",
    ]
    return df, report


def main() -> None:
    raw_df = load_question_dataframe()  # Required: this script loads directly from SQLite.
    processed_df, report = preprocess_dataframe(raw_df)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(PROCESSED_PATH, index=False)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Preprocessed {len(processed_df):,} questions into {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
