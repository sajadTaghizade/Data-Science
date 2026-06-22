"""Create text and metadata features for the similar-question recommender."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

try:  # Supports both direct script execution and imports from the EDA notebook.
    from .database_connection import PROJECT_ROOT
    from .load_data import load_question_dataframe
    from .preprocess import PROCESSED_PATH, preprocess_dataframe
except ImportError:  # pragma: no cover - direct script execution
    from database_connection import PROJECT_ROOT
    from load_data import load_question_dataframe
    from preprocess import PROCESSED_PATH, preprocess_dataframe


FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "question_features.csv"
VECTORIZER_PATH = PROJECT_ROOT / "data" / "models" / "tfidf_vectorizer.joblib"
MATRIX_PATH = PROJECT_ROOT / "data" / "models" / "question_tfidf_matrix.npz"
METADATA_PATH = PROJECT_ROOT / "data" / "reports" / "feature_engineering_report.json"

# Tag multi-hot encoding settings. Tags present in every row (e.g. ``c++``) carry
# no signal, so they are dropped; only sufficiently frequent, informative tags
# become binary indicator columns.
TAG_MIN_QUESTIONS = 30
TAG_MAX_FEATURES = 20


def load_preprocessed_features() -> pd.DataFrame:
    """Return the cleaned dataset for feature building.

    In a full ``pipeline.py`` run, ``preprocess.py`` has already written the
    cleaned artefact, so we reuse it instead of recomputing the cleaning step.
    When this script is run standalone it falls back to loading **directly from
    the database** (as required by the brief) and preprocessing in-memory.
    """
    if PROCESSED_PATH.exists():
        return pd.read_csv(PROCESSED_PATH)
    raw_df = load_question_dataframe()  # Direct SQLite load for standalone runs.
    preprocessed_df, _ = preprocess_dataframe(raw_df)
    return preprocessed_df


def count_code_blocks(html_text: str) -> int:
    soup = BeautifulSoup(str(html_text), "html.parser")
    return len(soup.find_all("pre"))


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", str(text)))


def tag_column_name(tag: str) -> str:
    """Turn a raw tag into a safe binary-feature column name (e.g. ``smart-pointers`` -> ``tag_smart_pointers``)."""
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", tag).strip("_").lower()
    return f"tag_{slug}"


def build_tag_multihot(tag_series: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    """One-hot/multi-hot encode the most frequent, informative tags."""
    tag_lists = tag_series.fillna("").map(lambda value: [tag for tag in str(value).split("|") if tag])
    counts = Counter(tag for tags in tag_lists for tag in tags)
    total_rows = len(tag_series)
    selected_tags = [
        tag for tag, count in counts.most_common()
        if count >= TAG_MIN_QUESTIONS and count < total_rows  # drop constant tags such as c++
    ][:TAG_MAX_FEATURES]

    encoded = pd.DataFrame(index=tag_series.index)
    for tag in selected_tags:
        encoded[tag_column_name(tag)] = tag_lists.map(lambda tags, tag=tag: int(tag in tags))
    return encoded, selected_tags


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, TfidfVectorizer, sparse.csr_matrix, dict]:
    """Generate explainable numeric features and a sparse TF-IDF text representation."""
    features = df.copy()
    features["title_word_count"] = features["title_text"].map(word_count)
    features["body_word_count"] = features["body_text"].map(word_count)
    features["document_word_count"] = features["document_text"].map(word_count)
    features["unique_token_ratio"] = features["document_text"].map(
        lambda value: len(set(str(value).split())) / max(len(str(value).split()), 1)
    )
    features["code_block_count"] = features["body_html"].map(count_code_blocks)

    features["creation_at"] = pd.to_datetime(features["creation_at"], utc=True, errors="coerce")
    features["creation_year"] = features["creation_at"].dt.year
    features["creation_month"] = features["creation_at"].dt.month
    # Deterministic age: measured against the most recent question in the dataset
    # (the crawl snapshot) instead of the wall clock, so reruns are reproducible.
    age_reference_date = features["creation_at"].max()
    features["question_age_days"] = (age_reference_date - features["creation_at"]).dt.days

    features["view_count_log1p"] = np.log1p(features["view_count"].clip(lower=0))
    features["answer_count_log1p"] = np.log1p(features["answer_count"].clip(lower=0))
    features["score_signed_log1p"] = np.sign(features["score"]) * np.log1p(features["score"].abs())
    scale_columns = [
        "view_count_log1p", "answer_count_log1p", "score_signed_log1p",
        "tag_count", "document_word_count", "code_block_count",
    ]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features[scale_columns])
    zscore_columns = []
    for index, column in enumerate(scale_columns):
        zscore_column = f"{column}_zscore"
        features[zscore_column] = scaled[:, index]
        zscore_columns.append(zscore_column)

    tag_multihot, encoded_tags = build_tag_multihot(features["tags"])
    features = pd.concat([features, tag_multihot], axis=1)
    tag_feature_columns = list(tag_multihot.columns)

    vectorizer = TfidfVectorizer(
        lowercase=False,  # The text was already normalized in preprocess.py.
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=12_000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(features["document_text"].fillna("")).tocsr()

    # Columns that are the actual model inputs vs. columns retained only for auditing.
    model_numeric_features = zscore_columns + tag_feature_columns
    audit_only_columns = [
        "view_count", "answer_count", "score",  # raw skewed counts
        "view_count_log1p", "answer_count_log1p", "score_signed_log1p",  # pre-standardization
    ]

    metadata = {
        "rows": int(len(features)),
        "tfidf_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "tfidf_vocabulary_size": int(len(vectorizer.vocabulary_)),
        "tfidf_settings": {
            "ngram_range": [1, 2], "min_df": 2, "max_df": 0.95,
            "max_features": 12_000, "sublinear_tf": True,
        },
        "age_reference_date": None if pd.isna(age_reference_date) else age_reference_date.isoformat(),
        "scaled_metadata_features": scale_columns,
        "tag_multihot_columns": tag_feature_columns,
        "encoded_tags": encoded_tags,
        "model_text_input": "data/models/question_tfidf_matrix.npz (TF-IDF of title + cleaned body)",
        "model_numeric_features": model_numeric_features,
        "audit_only_columns": audit_only_columns,
        "similarity_input": "TF-IDF vectors of title + cleaned body; cosine similarity in Phase 3.",
    }
    return features, vectorizer, matrix, metadata


def main() -> None:
    preprocessed_df = load_preprocessed_features()
    features, vectorizer, matrix, metadata = build_features(preprocessed_df)

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    VECTORIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_PATH, index=False)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    sparse.save_npz(MATRIX_PATH, matrix)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Engineered {len(features):,} feature rows and a {matrix.shape[0]}x{matrix.shape[1]} TF-IDF matrix.")


if __name__ == "__main__":
    main()
