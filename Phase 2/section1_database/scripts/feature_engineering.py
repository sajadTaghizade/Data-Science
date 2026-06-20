"""Create text and metadata features for the similar-question recommender."""

from __future__ import annotations

import json
import re
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
    from .preprocess import preprocess_dataframe
except ImportError:  # pragma: no cover - direct script execution
    from database_connection import PROJECT_ROOT
    from load_data import load_question_dataframe
    from preprocess import preprocess_dataframe


FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "question_features.csv"
VECTORIZER_PATH = PROJECT_ROOT / "data" / "models" / "tfidf_vectorizer.joblib"
MATRIX_PATH = PROJECT_ROOT / "data" / "models" / "question_tfidf_matrix.npz"
METADATA_PATH = PROJECT_ROOT / "data" / "reports" / "feature_engineering_report.json"


def count_code_blocks(html_text: str) -> int:
    soup = BeautifulSoup(str(html_text), "html.parser")
    return len(soup.find_all("pre"))


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, TfidfVectorizer, sparse.csr_matrix, dict]:
    """Generate explainable numeric features and a sparse TF-IDF text representation."""
    features = df.copy()
    features["title_word_count"] = features["title_text"].map(word_count)
    features["body_word_count"] = features["body_text"].map(word_count)
    features["document_word_count"] = features["document_text"].map(word_count)
    features["unique_token_ratio"] = features["document_text"].map(
        lambda value: len(set(value.split())) / max(len(value.split()), 1)
    )
    features["code_block_count"] = features["body_html"].map(count_code_blocks)

    features["creation_at"] = pd.to_datetime(features["creation_at"], utc=True, errors="coerce")
    features["creation_year"] = features["creation_at"].dt.year
    features["creation_month"] = features["creation_at"].dt.month
    features["question_age_days"] = (
        pd.Timestamp.now(tz="UTC") - features["creation_at"]
    ).dt.days

    features["view_count_log1p"] = np.log1p(features["view_count"].clip(lower=0))
    features["answer_count_log1p"] = np.log1p(features["answer_count"].clip(lower=0))
    features["score_signed_log1p"] = np.sign(features["score"]) * np.log1p(features["score"].abs())
    scale_columns = [
        "view_count_log1p", "answer_count_log1p", "score_signed_log1p",
        "tag_count", "document_word_count", "code_block_count",
    ]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features[scale_columns])
    for index, column in enumerate(scale_columns):
        features[f"{column}_zscore"] = scaled[:, index]

    vectorizer = TfidfVectorizer(
        lowercase=False,  # The text was already normalized in preprocess.py.
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=12_000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(features["document_text"]).tocsr()
    metadata = {
        "rows": int(len(features)),
        "tfidf_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "tfidf_vocabulary_size": int(len(vectorizer.vocabulary_)),
        "tfidf_settings": {
            "ngram_range": [1, 2], "min_df": 2, "max_df": 0.95,
            "max_features": 12_000, "sublinear_tf": True,
        },
        "scaled_metadata_features": scale_columns,
        "similarity_input": "TF-IDF vectors of title + cleaned body; cosine similarity in Phase 3.",
    }
    return features, vectorizer, matrix, metadata


def main() -> None:
    raw_df = load_question_dataframe()  # Required: this script loads directly from SQLite.
    preprocessed_df, _ = preprocess_dataframe(raw_df)
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
