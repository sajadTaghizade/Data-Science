"""Leakage-safe feature engineering and retrieval modeling for the recommender.

This stage takes the cleaned questions, builds several TF-IDF text
representations, evaluates them with proper retrieval metrics under a temporal
train/validation/test split, tunes a hybrid ensemble on validation only, reports
test metrics once, and finally refits the chosen model on all data for
deployment. All vectorizers are fit on the training corpus only.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

try:  # Supports both direct script execution and imports from the EDA notebook.
    from .database_connection import PROJECT_ROOT
    from .load_data import load_question_dataframe
    from .preprocess import GLOBAL_TAGS, PROCESSED_PATH, preprocess_dataframe, secondary_tags
except ImportError:  # pragma: no cover - direct script execution
    from database_connection import PROJECT_ROOT
    from load_data import load_question_dataframe
    from preprocess import GLOBAL_TAGS, PROCESSED_PATH, preprocess_dataframe, secondary_tags


logger = logging.getLogger(__name__)


FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "question_features.csv"
ARTIFACT_PATH = PROJECT_ROOT / "data" / "models" / "best_hybrid_recommender.joblib"
SINGLE_METRICS_PATH = PROJECT_ROOT / "data" / "reports" / "section2_single_model_validation_metrics.csv"
HYBRID_SEARCH_PATH = PROJECT_ROOT / "data" / "reports" / "section2_hybrid_weight_search_validation.csv"
TEST_METRICS_PATH = PROJECT_ROOT / "data" / "reports" / "section2_best_hybrid_test_metrics.csv"
BASELINE_METRICS_PATH = PROJECT_ROOT / "data" / "reports" / "section2_baseline_vs_model_test_metrics.csv"
METADATA_PATH = PROJECT_ROOT / "data" / "reports" / "feature_engineering_report.json"

RANDOM_STATE = 42
K_VALUES = (5, 10, 20)
TOKEN_PATTERN = r"(?u)\b[\w+#.:-]{2,}\b"
# Cap the number of query rows used for metric computation so the validation/test
# evaluation stays fast even when the corpus is scaled to tens of thousands of
# rows. The candidate corpus itself is never sub-sampled. With the current 2.5k
# dataset the splits are far below this cap, so behaviour is unchanged.
MAX_EVAL_QUERIES = 3000
# Tags that appear on at least this many questions become optional multi-hot
# metadata columns (kept for traceability, NOT used for similarity scoring).
TAG_MIN_QUESTIONS = 30
TAG_MAX_FEATURES = 20


# --------------------------------------------------------------------------- #
# Retrieval metrics
# --------------------------------------------------------------------------- #
def average_precision_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return np.nan
    hits, score = 0, 0.0
    for rank, idx in enumerate(ranked[:k], start=1):
        if idx in relevant:
            hits += 1
            score += hits / rank
    return score / min(len(relevant), k)


def reciprocal_rank_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return np.nan
    for rank, idx in enumerate(ranked[:k], start=1):
        if idx in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return np.nan
    return len(set(ranked[:k]) & relevant) / len(relevant)


def hit_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return np.nan
    return float(len(set(ranked[:k]) & relevant) > 0)


def ndcg_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return np.nan
    dcg = sum(1.0 / np.log2(rank + 1) for rank, idx in enumerate(ranked[:k], start=1) if idx in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def relevant_indices_for_query(query_tags: list[str], candidate_tags: list[set[str]]) -> set[int]:
    """Weak relevance: a candidate is relevant if it shares a non-global tag with the query."""
    query_secondary = secondary_tags(query_tags)
    if not query_secondary:
        return set()
    return {i for i, tags in enumerate(candidate_tags) if query_secondary & tags}


def evaluate_rankings(rankings, query_tag_lists, candidate_tag_sets, k_values=K_VALUES, relevant_sets=None) -> dict:
    # Relevance is fixed for a given query set + candidate pool, so it can be
    # precomputed once and reused across every hybrid-weight combination — this
    # keeps the weight search tractable when the corpus is scaled to tens of
    # thousands of questions.
    if relevant_sets is None:
        relevant_sets = [relevant_indices_for_query(tags, candidate_tag_sets) for tags in query_tag_lists]
    total = len(relevant_sets)
    valid = [(ranked, relevant) for ranked, relevant in zip(rankings, relevant_sets, strict=True) if relevant]
    results = {
        "evaluated_queries": len(valid),
        "coverage": len(valid) / total if total else 0.0,
    }
    for k in k_values:
        results[f"Hit@{k}"] = np.mean([hit_at_k(r, rel, k) for r, rel in valid]) if valid else np.nan
        results[f"Recall@{k}"] = np.mean([recall_at_k(r, rel, k) for r, rel in valid]) if valid else np.nan
        results[f"MAP@{k}"] = np.mean([average_precision_at_k(r, rel, k) for r, rel in valid]) if valid else np.nan
        results[f"MRR@{k}"] = np.mean([reciprocal_rank_at_k(r, rel, k) for r, rel in valid]) if valid else np.nan
        results[f"nDCG@{k}"] = np.mean([ndcg_at_k(r, rel, k) for r, rel in valid]) if valid else np.nan
    return results


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
@dataclass
class VectorModel:
    name: str
    vectorizer: TfidfVectorizer
    train_matrix: sparse.csr_matrix
    field: str


MODEL_SPECS = [
    {"name": "title_word_tfidf", "field": "title_clean",
     "analyzer": "word", "ngram_range": (1, 3), "min_df": 1, "max_df": 0.95,
     "max_features": 20000, "token_pattern": TOKEN_PATTERN},
    {"name": "document_word_tfidf", "field": "document_clean",
     "analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "max_df": 0.90,
     "max_features": 60000, "token_pattern": TOKEN_PATTERN},
    {"name": "char_wb_tfidf", "field": "document_clean",
     # 3-4 char n-grams (not 3-5) keep the character TF-IDF fit fast and
     # memory-bounded on the full 26k corpus of long documents.
     "analyzer": "char_wb", "ngram_range": (3, 4), "min_df": 3, "max_df": 0.95,
     "max_features": 40000},
]
# LSA (latent semantic analysis): TruncatedSVD over a document TF-IDF gives a
# dense, semantically-aware representation that captures synonymy/co-occurrence
# beyond exact lexical overlap — a lightweight "embedding" with no extra deps.
LSA_NAME = "lsa_document"
LSA_FIELD = "document_clean"
LSA_COMPONENTS = 200


def fit_models(corpus_df: pd.DataFrame) -> list[VectorModel]:
    models = []
    for raw_spec in MODEL_SPECS:
        spec = dict(raw_spec)
        name, field = spec.pop("name"), spec.pop("field")
        vectorizer = TfidfVectorizer(lowercase=False, norm="l2", sublinear_tf=True, **spec)
        matrix = vectorizer.fit_transform(corpus_df[field].fillna(""))
        models.append(VectorModel(name=name, vectorizer=vectorizer, train_matrix=matrix, field=field))

    # Dense LSA representation; Normalizer makes the dot product a cosine similarity.
    n_components = min(LSA_COMPONENTS, max(2, len(corpus_df) - 1))
    lsa = make_pipeline(
        TfidfVectorizer(lowercase=False, sublinear_tf=True, ngram_range=(1, 2),
                        min_df=2, max_df=0.90, max_features=60000, token_pattern=TOKEN_PATTERN),
        TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE),
        Normalizer(copy=False),
    )
    lsa_matrix = lsa.fit_transform(corpus_df[LSA_FIELD].fillna(""))
    models.append(VectorModel(name=LSA_NAME, vectorizer=lsa, train_matrix=lsa_matrix, field=LSA_FIELD))
    return models


def models_to_dicts(models: list[VectorModel]) -> list[dict]:
    """Serialize models as plain dicts so the artefact does not pickle a custom class."""
    return [{"name": m.name, "vectorizer": m.vectorizer, "train_matrix": m.train_matrix, "field": m.field}
            for m in models]


def models_from_dicts(model_dicts: list[dict]) -> list[VectorModel]:
    """Rebuild VectorModel objects from the artefact dicts."""
    return [VectorModel(**d) for d in model_dicts]


def score_matrix(model: VectorModel, query_df: pd.DataFrame) -> np.ndarray:
    """Dense (n_queries x n_candidates) cosine-similarity matrix for one model.

    Works for both the sparse TF-IDF models and the dense LSA model; all
    representations are L2-normalized, so the dot product is a cosine similarity.
    """
    query_vectors = model.vectorizer.transform(query_df[model.field].fillna(""))
    scores = query_vectors @ model.train_matrix.T
    return scores.toarray() if sparse.issparse(scores) else np.asarray(scores)


def _rank_from_scores(scores: np.ndarray, top_k: int) -> list[list[int]]:
    rankings = []
    for row in scores:
        if top_k >= len(row):
            order = np.argsort(-row)
        else:
            partial = np.argpartition(-row, top_k)[:top_k]
            order = partial[np.argsort(-row[partial])]
        rankings.append(order.tolist())
    return rankings


def rank_single(model: VectorModel, query_df: pd.DataFrame, top_k: int = 50) -> list[list[int]]:
    return _rank_from_scores(score_matrix(model, query_df), top_k)


def rank_hybrid(models: list[VectorModel], query_df: pd.DataFrame, weights: dict[str, float], top_k: int = 50):
    total = None
    for model in models:
        weight = weights.get(model.name, 0.0)
        if weight == 0:
            continue
        weighted = weight * score_matrix(model, query_df)
        total = weighted if total is None else total + weighted
    if total is None:
        raise ValueError("All model weights are zero.")
    return _rank_from_scores(total, top_k)


def sample_queries(df: pd.DataFrame, max_n: int = MAX_EVAL_QUERIES, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Deterministically cap the number of evaluation queries for tractable metrics at scale."""
    if len(df) <= max_n:
        return df
    return df.sample(n=max_n, random_state=seed).sort_values("creation_at").reset_index(drop=True)


def temporal_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    df_sorted = df.sort_values("creation_at").reset_index(drop=True)
    n = len(df_sorted)
    train_end, val_end = int(n * train_frac), int(n * (train_frac + val_frac))
    return (df_sorted.iloc[:train_end].reset_index(drop=True),
            df_sorted.iloc[train_end:val_end].reset_index(drop=True),
            df_sorted.iloc[val_end:].reset_index(drop=True))


WEIGHT_KEYS = ("title_word_tfidf", "document_word_tfidf", "char_wb_tfidf", LSA_NAME)


def weight_grid():
    grid = []
    for title_w in (0.2, 0.3):
        for doc_w in (0.3, 0.4, 0.5):
            for char_w in (0.1, 0.2):
                for lsa_w in (0.1, 0.2, 0.3):
                    total = title_w + doc_w + char_w + lsa_w
                    grid.append({
                        "title_word_tfidf": title_w / total,
                        "document_word_tfidf": doc_w / total,
                        "char_wb_tfidf": char_w / total,
                        LSA_NAME: lsa_w / total,
                    })
    return grid


def rank_random(n_queries: int, n_candidates: int, top_k: int, seed: int = RANDOM_STATE) -> list[list[int]]:
    """Random-order baseline: a shuffled candidate list per query."""
    rng = np.random.default_rng(seed)
    return [rng.permutation(n_candidates)[:top_k].tolist() for _ in range(n_queries)]


def rank_popularity(corpus_df: pd.DataFrame, n_queries: int, top_k: int) -> list[list[int]]:
    """Popularity baseline: the same most-viewed/highest-scored candidates for every query."""
    order = np.lexsort((corpus_df["view_count"].to_numpy(), corpus_df["score"].to_numpy()))[::-1]
    ranking = order[:top_k].tolist()
    return [ranking for _ in range(n_queries)]


# --------------------------------------------------------------------------- #
# Optional metadata: multi-hot tag encoding (NOT used for scoring)
# --------------------------------------------------------------------------- #
def build_tag_multihot(tag_lists: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    counts = Counter(tag for tags in tag_lists for tag in tags if tag not in GLOBAL_TAGS)
    total = len(tag_lists)
    selected = [tag for tag, count in counts.most_common()
                if count >= TAG_MIN_QUESTIONS and count < total][:TAG_MAX_FEATURES]
    encoded = pd.DataFrame(index=tag_lists.index)
    for tag in selected:
        column = "tag_" + re.sub(r"[^0-9a-zA-Z]+", "_", tag).strip("_").lower()
        encoded[column] = tag_lists.map(lambda tags, tag=tag: int(tag in tags))
    return encoded, selected


# --------------------------------------------------------------------------- #
# Pipeline entry point
# --------------------------------------------------------------------------- #
def load_preprocessed() -> pd.DataFrame:
    """Reuse the preprocess.py artefact in a pipeline run; fall back to a direct DB load."""
    if PROCESSED_PATH.exists():
        df = pd.read_csv(PROCESSED_PATH)
    else:
        df, _ = preprocess_dataframe(load_question_dataframe())  # Direct SQLite load when standalone.
    df["creation_at"] = pd.to_datetime(df["creation_at"], utc=True, errors="coerce")
    df["tag_list"] = df["tags"].fillna("").map(lambda value: [t for t in str(value).split("|") if t])
    return df


def main() -> None:
    np.random.seed(RANDOM_STATE)
    questions = load_preprocessed()

    train_df, val_df, test_df = temporal_split(questions)
    # The candidate corpus stays the full training set; only the query rows are capped.
    val_df = sample_queries(val_df)
    test_df = sample_queries(test_df)
    train_secondary = [secondary_tags(tags) for tags in train_df["tag_list"]]
    models = fit_models(train_df)

    # Precompute relevance once per query set (reused across all models and every
    # hybrid-weight combination) so the search stays fast on a large corpus.
    val_relevant = [relevant_indices_for_query(tags, train_secondary) for tags in val_df["tag_list"]]
    test_relevant = [relevant_indices_for_query(tags, train_secondary) for tags in test_df["tag_list"]]

    # Single-model validation metrics.
    single_rows = []
    for model in models:
        metrics = evaluate_rankings(rank_single(model, val_df), None, None, relevant_sets=val_relevant)
        metrics["model"] = model.name
        single_rows.append(metrics)
    single_df = pd.DataFrame(single_rows).set_index("model").sort_values("nDCG@10", ascending=False)

    # Hybrid weight search on validation only.
    hybrid_rows = []
    for weights in weight_grid():
        metrics = evaluate_rankings(rank_hybrid(models, val_df, weights), None, None, relevant_sets=val_relevant)
        metrics.update(weights)
        hybrid_rows.append(metrics)
    hybrid_df = pd.DataFrame(hybrid_rows).sort_values(["nDCG@10", "MAP@10", "MRR@10"], ascending=False)
    best_weights = {k: float(hybrid_df.iloc[0][k]) for k in WEIGHT_KEYS}

    # Final test evaluation (once); candidate corpus = train only.
    test_rankings = rank_hybrid(models, test_df, best_weights)
    test_metrics = evaluate_rankings(test_rankings, None, None, relevant_sets=test_relevant)

    # Baselines for context: prove the model beats trivial rankings on the same test set.
    n_test, n_train = len(test_df), len(train_df)
    baseline_metrics = {
        "random": evaluate_rankings(rank_random(n_test, n_train, 50), None, None, relevant_sets=test_relevant),
        "popularity": evaluate_rankings(rank_popularity(train_df, n_test, 50), None, None, relevant_sets=test_relevant),
        "hybrid_model": test_metrics,
    }
    baseline_df = pd.DataFrame(baseline_metrics).T

    # Refit the chosen representation for deployment on train+validation. Fitting the
    # character n-gram model on all 26k documents at once is memory-heavy; train+validation
    # keeps the artefact tractable and matches the Phase 3 reference setup. The reported
    # metrics above are unchanged — they come from the train-only fit evaluated on test.
    deploy_corpus = pd.concat([train_df, val_df], ignore_index=True)
    final_models = fit_models(deploy_corpus)

    for path in (FEATURES_PATH, ARTIFACT_PATH, METADATA_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)

    tag_multihot, encoded_tags = build_tag_multihot(questions["tag_list"])
    feature_columns = [
        "question_id", "title", "title_clean", "document_clean", "tags", "tag_count",
        "score", "view_count", "answer_count", "log_view_count", "signed_log_score",
        "title_word_count", "body_word_count", "document_word_count", "code_block_count",
        "creation_year", "creation_month", "creation_at",
    ]
    features = pd.concat(
        [questions[[c for c in feature_columns if c in questions.columns]].reset_index(drop=True),
         tag_multihot.reset_index(drop=True)],
        axis=1,
    )
    features.to_csv(FEATURES_PATH, index=False)

    artifact_questions_cols = ["question_id", "title", "tags", "creation_at"]
    if "question_url" in deploy_corpus.columns:
        artifact_questions_cols.insert(2, "question_url")
    # The stored candidate list must align row-for-row with the fitted models'
    # matrices, so it uses the same train+validation corpus they were fit on.
    joblib.dump(
        {"models": models_to_dicts(final_models), "weights": best_weights,
         "questions": deploy_corpus[artifact_questions_cols].copy()},
        ARTIFACT_PATH,
    )
    single_df.to_csv(SINGLE_METRICS_PATH)
    hybrid_df.to_csv(HYBRID_SEARCH_PATH, index=False)
    pd.DataFrame([test_metrics]).to_csv(TEST_METRICS_PATH, index=False)
    baseline_df.to_csv(BASELINE_METRICS_PATH)

    metadata = {
        "task": "semantic_similar_question_recommendation",
        "split": {"type": "temporal", "train_rows": len(train_df),
                  "validation_rows": len(val_df), "test_rows": len(test_df)},
        "models": [spec["name"] for spec in MODEL_SPECS] + [LSA_NAME],
        "best_validation_weights": best_weights,
        "test_metrics": {k: (float(v) if pd.notna(v) else None) for k, v in test_metrics.items()},
        "baseline_test_metrics": {
            name: {k: (float(v) if pd.notna(v) else None) for k, v in metrics.items()}
            for name, metrics in baseline_metrics.items()
        },
        "tag_multihot_columns": list(tag_multihot.columns),
        "encoded_tags_metadata_only": encoded_tags,
        "model_text_input": "hybrid of TF-IDF (title word, document word, char n-gram) + LSA; cosine similarity.",
        "audit_only_columns": ["view_count", "answer_count", "score", "log_view_count", "signed_log_score"],
        "anti_overfit_controls": [
            "Temporal train/validation/test split",
            "TF-IDF vectorizers fitted only on the training corpus",
            "Hybrid weights selected only on validation",
            "Test set evaluated only once",
            "Generic c++ tag removed from weak relevance labels",
            "Tags kept as weak labels / optional metadata, not as default scoring features",
        ],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info("Engineered features for %d questions.", len(questions))
    logger.info("Best validation weights: %s", best_weights)
    logger.info("Test    Hit@10=%.3f  MRR@10=%.3f  nDCG@10=%.3f",
                test_metrics["Hit@10"], test_metrics["MRR@10"], test_metrics["nDCG@10"])
    logger.info("Random  Hit@10=%.3f  nDCG@10=%.3f",
                baseline_metrics["random"]["Hit@10"], baseline_metrics["random"]["nDCG@10"])
    logger.info("Popular Hit@10=%.3f  nDCG@10=%.3f",
                baseline_metrics["popularity"]["Hit@10"], baseline_metrics["popularity"]["nDCG@10"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
