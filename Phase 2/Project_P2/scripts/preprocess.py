"""Advanced, leakage-safe text cleaning and preprocessing for question retrieval.

This module loads the question data directly from SQLite (as required by the
brief) and produces a cleaned, model-ready DataFrame. The cleaning preserves
technical C++ tokens (``std::vector``, ``nullptr``, ``c++``) and code markers,
because they carry strong similarity signal for the recommender.
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd

try:  # Supports both direct script execution and imports from the EDA notebook.
    from .database_connection import PROJECT_ROOT
    from .load_data import load_question_dataframe
except ImportError:  # pragma: no cover - direct script execution
    from database_connection import PROJECT_ROOT
    from load_data import load_question_dataframe


PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessed_questions.csv"
REPORT_PATH = PROJECT_ROOT / "data" / "reports" / "preprocessing_report.json"
CORE_TEXT_COLUMNS = ["title", "body_html"]

# Tags shared by (almost) every question carry no discriminative signal.
GLOBAL_TAGS = {"c++", "cpp", "cplusplus"}

CODE_BLOCK_RE = re.compile(r"<pre.*?>.*?</pre>|<code.*?>.*?</code>", flags=re.I | re.S)
HTML_TAG_RE = re.compile(r"<[^>]+>")
NON_TEXT_RE = re.compile(r"[^a-zA-Z0-9_+#.:-]+")
SPACE_RE = re.compile(r"\s+")

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - bs4 is declared in requirements.txt
    BeautifulSoup = None


def strip_html_keep_code_markers(html_value: Any) -> tuple[str, int]:
    """Remove HTML but keep a ``CODE_BLOCK`` marker and the count of code blocks."""
    html_text = "" if pd.isna(html_value) else str(html_value)
    code_block_count = len(CODE_BLOCK_RE.findall(html_text))
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html_text, "html.parser")
        for tag in soup.find_all(["pre", "code"]):
            tag.replace_with(" CODE_BLOCK " + tag.get_text(" ") + " ")
        text = soup.get_text(" ")
    else:
        text = CODE_BLOCK_RE.sub(" CODE_BLOCK ", html_text)
        text = HTML_TAG_RE.sub(" ", text)
    return text, code_block_count


def normalize_technical_text(value: Any) -> str:
    """Lowercase and normalize while preserving technical tokens such as ``std::vector``."""
    text = "" if pd.isna(value) else str(value).lower()
    text = text.replace("c ++", "c++").replace("cplusplus", "c++")
    text = text.replace("std ::", "std::")
    text = NON_TEXT_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def normalize_tags(value: Any) -> list[str]:
    """Return a sorted, de-duplicated, lower-cased tag list from a ``|``/``,`` string or list."""
    if isinstance(value, list):
        tags = value
    elif pd.isna(value):
        return []
    else:
        tags = [tag.strip().lower() for tag in str(value).replace(",", "|").split("|")]
    return sorted({tag for tag in tags if tag})


def secondary_tags(tags: list[str]) -> set[str]:
    """Drop the universal C++ tags so only discriminative tags remain."""
    return {tag for tag in tags if tag not in GLOBAL_TAGS}


def preprocess_dataframe(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean model inputs and record every data-quality decision."""
    df = raw_df.copy()
    report: dict[str, Any] = {"input_rows": int(len(df))}

    report["duplicate_question_ids_removed"] = int(df.duplicated(subset=["question_id"]).sum())
    df = df.drop_duplicates(subset=["question_id"], keep="first").copy()

    df["title"] = df["title"].fillna("").astype(str)
    df["body_html"] = df["body_html"].fillna("").astype(str)
    invalid_text = df["title"].str.strip().eq("") | df["body_html"].str.strip().eq("")
    report["missing_or_blank_core_text_removed"] = int(invalid_text.sum())
    df = df.loc[~invalid_text].copy()

    body_result = df["body_html"].apply(strip_html_keep_code_markers)
    df["body_text"] = body_result.apply(lambda result: result[0])
    df["code_block_count"] = body_result.apply(lambda result: result[1])

    df["title_clean"] = df["title"].apply(normalize_technical_text)
    df["body_clean"] = df["body_text"].apply(normalize_technical_text)
    # Title is repeated so it weighs more heavily than the (much longer) body.
    df["document_clean"] = (df["title_clean"] + " " + df["title_clean"] + " " + df["body_clean"]).str.strip()
    empty_documents = df["document_clean"].eq("")
    report["empty_documents_removed"] = int(empty_documents.sum())
    df = df.loc[~empty_documents].copy()

    df["tag_list"] = df["tags"].apply(normalize_tags)
    df["tags"] = df["tag_list"].apply(lambda tags: "|".join(tags))
    df["tag_count"] = df["tag_list"].apply(len)
    report["missing_tag_lists"] = int((df["tag_count"] == 0).sum())

    for column in ["creation_at", "last_activity_at", "last_edit_at", "closed_at"]:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], utc=True, errors="coerce")

    numeric_columns = ["view_count", "answer_count", "score", "tag_count"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    report["numeric_missing_after_coercion"] = {
        column: int(df[column].isna().sum()) for column in numeric_columns
    }
    df[numeric_columns] = df[numeric_columns].fillna(0)

    df["title_word_count"] = df["title_clean"].str.split().apply(len)
    df["body_word_count"] = df["body_clean"].str.split().apply(len)
    df["document_word_count"] = df["document_clean"].str.split().apply(len)
    df["log_view_count"] = np.log1p(df["view_count"].clip(lower=0))
    df["signed_log_score"] = np.sign(df["score"]) * np.log1p(df["score"].abs())
    df["creation_year"] = df["creation_at"].dt.year
    df["creation_month"] = df["creation_at"].dt.month

    correlation_matrix = df[numeric_columns].corr().round(4)
    report["numeric_correlation_matrix"] = correlation_matrix.to_dict()
    report["high_correlation_pairs_abs_ge_0_90"] = [
        {"feature_1": first, "feature_2": second,
         "correlation": float(correlation_matrix.loc[first, second])}
        for index, first in enumerate(numeric_columns)
        for second in numeric_columns[index + 1:]
        if abs(correlation_matrix.loc[first, second]) >= 0.90
    ]
    report["outlier_strategy"] = (
        "Retain legitimate high-engagement questions; apply log/signed-log transforms instead "
        "of deleting informative records. Engagement is not the semantic-similarity target."
    )
    report["output_rows"] = int(len(df))
    return df.reset_index(drop=True), report


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
