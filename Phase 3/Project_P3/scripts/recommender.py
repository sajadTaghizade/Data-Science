"""Reusable similar-question recommender models, metrics, and inference helpers.

Phase 3 turns the Phase 2 exploratory recommender into a proper, reusable
component that both the **training** pipeline (fit, compare, select, evaluate)
and the **prediction** pipeline (load, retrieve, save) share.

Design goals
------------
* **Several comparable candidate models** so Section 1's "test multiple models
  and pick the best" is a real experiment, not a formality:
    - ``title_word_tfidf``    - TF-IDF cosine over the (up-weighted) title,
    - ``document_word_tfidf`` - TF-IDF cosine over the whole question,
    - ``char_wb_tfidf``       - character n-gram TF-IDF (robust to C++ syntax),
    - ``lsa_document``        - Truncated SVD (LSA) latent-semantic cosine,
    - ``bm25_document``       - Okapi BM25, a classic probabilistic IR ranker
                                (new in Phase 3, implemented dependency-free),
    - ``hybrid``              - a weighted blend whose weights are tuned on
                                validation only.
* Every model exposes the **same interface** (``fit`` / ``score``) so the
  hybrid, the metrics, and the inference code treat them uniformly.
* Everything is fit on the **training corpus only** and is pickle-friendly
  (no lambdas / custom closures) so the trained artefact loads cleanly in the
  separate prediction process.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import field as dataclass_field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import Normalizer

try:  # Works both as ``python scripts/train_model.py`` and as a package import.
    from .preprocess import secondary_tags
except ImportError:  # pragma: no cover - direct script execution
    from preprocess import secondary_tags

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration shared across the training and prediction pipelines
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
K_VALUES = (5, 10, 20)
PRIMARY_METRIC = "nDCG@10"
# Keep the same technical-token pattern as Phase 2 so ``std::vector``, ``c++``,
# ``-fno-rtti`` and similar stay single tokens instead of being split apart.
TOKEN_PATTERN = r"(?u)\b[\w+#.:-]{2,}\b"
# Cap the number of *query* rows used when computing metrics so validation/test
# evaluation stays fast if the corpus is scaled up. The candidate pool is never
# sub-sampled. With the 2.5k dataset the splits are far below this cap.
MAX_EVAL_QUERIES = 3000

CANDIDATE_MODEL_NAMES = (
    "title_word_tfidf",
    "document_word_tfidf",
    "char_wb_tfidf",
    "lsa_document",
    "bm25_document",
)


# --------------------------------------------------------------------------- #
# Retrieval metrics (ranking quality) - identical definitions to Phase 2
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
    """Weak relevance label: a candidate is relevant if it shares a non-global tag."""
    query_secondary = secondary_tags(query_tags)
    if not query_secondary:
        return set()
    return {i for i, tags in enumerate(candidate_tags) if query_secondary & tags}


def evaluate_rankings(rankings, query_tag_lists, candidate_tag_sets, k_values=K_VALUES) -> dict:
    """Average the ranking metrics over every query that has at least one relevant candidate."""
    relevant_sets = [relevant_indices_for_query(tags, candidate_tag_sets) for tags in query_tag_lists]
    valid = [(ranked, relevant) for ranked, relevant in zip(rankings, relevant_sets, strict=True) if relevant]
    results = {
        "evaluated_queries": len(valid),
        "coverage": len(valid) / len(query_tag_lists) if len(query_tag_lists) else 0.0,
    }
    for k in k_values:
        results[f"Hit@{k}"] = np.mean([hit_at_k(r, rel, k) for r, rel in valid]) if valid else np.nan
        results[f"Recall@{k}"] = np.mean([recall_at_k(r, rel, k) for r, rel in valid]) if valid else np.nan
        results[f"MAP@{k}"] = np.mean([average_precision_at_k(r, rel, k) for r, rel in valid]) if valid else np.nan
        results[f"MRR@{k}"] = np.mean([reciprocal_rank_at_k(r, rel, k) for r, rel in valid]) if valid else np.nan
        results[f"nDCG@{k}"] = np.mean([ndcg_at_k(r, rel, k) for r, rel in valid]) if valid else np.nan
    return results


# --------------------------------------------------------------------------- #
# Candidate models - all share the fit / score interface
# --------------------------------------------------------------------------- #
@dataclass
class TfidfCosineModel:
    """TF-IDF vector space scored by cosine similarity (L2-normalised dot product)."""

    name: str
    field: str
    params: dict
    vectorizer: TfidfVectorizer | None = None
    candidate_matrix: sparse.csr_matrix | None = None

    def fit(self, corpus_df: pd.DataFrame) -> TfidfCosineModel:
        self.vectorizer = TfidfVectorizer(lowercase=False, norm="l2", sublinear_tf=True, **self.params)
        self.candidate_matrix = self.vectorizer.fit_transform(corpus_df[self.field].fillna(""))
        return self

    def score(self, query_df: pd.DataFrame) -> np.ndarray:
        query_vectors = self.vectorizer.transform(query_df[self.field].fillna(""))
        scores = query_vectors @ self.candidate_matrix.T
        return scores.toarray() if sparse.issparse(scores) else np.asarray(scores)

    @property
    def candidate_count(self) -> int:
        return self.candidate_matrix.shape[0]


@dataclass
class LsaCosineModel:
    """Latent Semantic Analysis: TruncatedSVD over a document TF-IDF, cosine scored.

    Captures synonymy / co-occurrence beyond exact lexical overlap - a
    lightweight, dependency-free "embedding". The ``Normalizer`` makes the dot
    product a cosine similarity.
    """

    name: str
    field: str
    n_components: int = 200
    vectorizer: TfidfVectorizer | None = None
    svd: TruncatedSVD | None = None
    normalizer: Normalizer | None = None
    candidate_matrix: np.ndarray | None = None

    def fit(self, corpus_df: pd.DataFrame) -> LsaCosineModel:
        self.vectorizer = TfidfVectorizer(lowercase=False, sublinear_tf=True, ngram_range=(1, 2),
                                          min_df=2, max_df=0.90, max_features=60000, token_pattern=TOKEN_PATTERN)
        tfidf = self.vectorizer.fit_transform(corpus_df[self.field].fillna(""))
        # TruncatedSVD needs n_components < n_features and <= n_samples - 1; cap by both
        # so the model also works on very small corpora (e.g. the unit-test fixtures).
        n_components = max(1, min(self.n_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
        self.normalizer = Normalizer(copy=False)
        reduced = self.svd.fit_transform(tfidf)
        self.candidate_matrix = self.normalizer.fit_transform(reduced)
        return self

    def score(self, query_df: pd.DataFrame) -> np.ndarray:
        reduced = self.svd.transform(self.vectorizer.transform(query_df[self.field].fillna("")))
        query_vectors = self.normalizer.transform(reduced)
        return np.asarray(query_vectors @ self.candidate_matrix.T)

    @property
    def candidate_count(self) -> int:
        return self.candidate_matrix.shape[0]


@dataclass
class BM25Model:
    """Okapi BM25 - the classic probabilistic ranking function used by search engines.

    Unlike cosine TF-IDF, BM25 saturates term frequency (a word appearing 20
    times is not 20x more relevant) and explicitly penalises long documents.
    It is implemented here directly on top of a ``CountVectorizer`` so it needs
    no extra dependency. The per-document BM25 weights are precomputed once at
    fit time, so scoring a batch of queries is a single sparse matrix product,
    exactly like the cosine models.
    """

    name: str
    field: str
    k1: float = 1.5
    b: float = 0.75
    min_df: int = 2
    vectorizer: CountVectorizer | None = None
    weight_matrix: sparse.csr_matrix | None = None  # (n_candidates x n_terms) BM25 weights

    def fit(self, corpus_df: pd.DataFrame) -> BM25Model:
        self.vectorizer = CountVectorizer(lowercase=False, min_df=self.min_df, token_pattern=TOKEN_PATTERN)
        counts = self.vectorizer.fit_transform(corpus_df[self.field].fillna("")).tocsr()
        counts.eliminate_zeros()

        n_docs = counts.shape[0]
        document_frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
        idf = np.log(1.0 + (n_docs - document_frequency + 0.5) / (document_frequency + 0.5))

        doc_lengths = np.asarray(counts.sum(axis=1)).ravel()
        avg_doc_length = doc_lengths.mean() if n_docs else 0.0
        length_norm = self.k1 * (1.0 - self.b + self.b * doc_lengths / (avg_doc_length or 1.0))

        term_freq = counts.data.astype(np.float64)
        row_index = np.repeat(np.arange(n_docs), np.diff(counts.indptr))
        col_index = counts.indices
        numerator = term_freq * (self.k1 + 1.0)
        denominator = term_freq + length_norm[row_index]
        weights = idf[col_index] * (numerator / denominator)

        self.weight_matrix = sparse.csr_matrix((weights, counts.indices, counts.indptr), shape=counts.shape)
        return self

    def score(self, query_df: pd.DataFrame) -> np.ndarray:
        query_counts = self.vectorizer.transform(query_df[self.field].fillna("")).tocsr()
        query_presence = query_counts.copy()
        query_presence.data[:] = 1.0  # BM25 treats each query term as present (binary).
        scores = query_presence @ self.weight_matrix.T
        return scores.toarray() if sparse.issparse(scores) else np.asarray(scores)

    @property
    def candidate_count(self) -> int:
        return self.weight_matrix.shape[0]


# --------------------------------------------------------------------------- #
# Model factory
# --------------------------------------------------------------------------- #
def build_candidate_models() -> list:
    """Instantiate (unfitted) the full set of candidate recommender models."""
    return [
        TfidfCosineModel("title_word_tfidf", "title_clean", {
            "analyzer": "word", "ngram_range": (1, 3), "min_df": 1, "max_df": 0.95,
            "max_features": 20000, "token_pattern": TOKEN_PATTERN}),
        TfidfCosineModel("document_word_tfidf", "document_clean", {
            "analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "max_df": 0.90,
            "max_features": 60000, "token_pattern": TOKEN_PATTERN}),
        TfidfCosineModel("char_wb_tfidf", "document_clean", {
            "analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "max_df": 0.95,
            "max_features": 50000}),
        LsaCosineModel("lsa_document", "document_clean", n_components=200),
        BM25Model("bm25_document", "document_clean", k1=1.5, b=0.75),
    ]


def fit_candidate_models(corpus_df: pd.DataFrame) -> list:
    models = build_candidate_models()
    for model in models:
        model.fit(corpus_df)
    return models


# --------------------------------------------------------------------------- #
# Ranking helpers
# --------------------------------------------------------------------------- #
def rank_from_scores(scores: np.ndarray, top_k: int) -> list[list[int]]:
    """Return, per query row, the indices of the ``top_k`` highest scores (descending)."""
    rankings = []
    for row in scores:
        if top_k >= len(row):
            order = np.argsort(-row)
        else:
            partial = np.argpartition(-row, top_k)[:top_k]
            order = partial[np.argsort(-row[partial])]
        rankings.append(order.tolist())
    return rankings


def rank_single(model, query_df: pd.DataFrame, top_k: int = 50) -> list[list[int]]:
    return rank_from_scores(model.score(query_df), top_k)


def normalize_rows(scores: np.ndarray) -> np.ndarray:
    """Per-query min-max scaling to [0, 1] so heterogeneous models blend fairly.

    Cosine similarities live in roughly [0, 1] but BM25 scores are unbounded, so
    the hybrid normalises every model's scores per query before the weighted sum.
    Single-model rankings are unaffected because scaling preserves order.
    """
    row_min = scores.min(axis=1, keepdims=True)
    row_max = scores.max(axis=1, keepdims=True)
    span = row_max - row_min
    span[span == 0] = 1.0
    return (scores - row_min) / span


def combine_scores(models: list, query_df: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    """Weighted sum of per-query-normalised model scores (the hybrid score)."""
    total = None
    for model in models:
        weight = weights.get(model.name, 0.0)
        if weight == 0:
            continue
        contribution = weight * normalize_rows(model.score(query_df))
        total = contribution if total is None else total + contribution
    if total is None:
        raise ValueError("All model weights are zero; cannot build a hybrid score.")
    return total


def rank_hybrid(models: list, query_df: pd.DataFrame, weights: dict[str, float], top_k: int = 50):
    return rank_from_scores(combine_scores(models, query_df, weights), top_k)


# --------------------------------------------------------------------------- #
# Splitting, sampling, hyperparameter grid, and baselines
# --------------------------------------------------------------------------- #
def temporal_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    """Order by creation date and cut into train / validation / test.

    A time-ordered split is the honest choice for a recommender that suggests
    *historical* questions for *newer* ones: it never lets the future leak into
    the past.
    """
    df_sorted = df.sort_values("creation_at").reset_index(drop=True)
    n = len(df_sorted)
    train_end, val_end = int(n * train_frac), int(n * (train_frac + val_frac))
    return (df_sorted.iloc[:train_end].reset_index(drop=True),
            df_sorted.iloc[train_end:val_end].reset_index(drop=True),
            df_sorted.iloc[val_end:].reset_index(drop=True))


def sample_queries(df: pd.DataFrame, max_n: int = MAX_EVAL_QUERIES, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Deterministically cap evaluation queries so metrics stay tractable at scale."""
    if len(df) <= max_n:
        return df
    return df.sample(n=max_n, random_state=seed).sort_values("creation_at").reset_index(drop=True)


def hybrid_weight_grid() -> list[dict[str, float]]:
    """Normalised weight combinations for the five candidate models (tuned on validation)."""
    grid = []
    for title_w in (0.2, 0.3):
        for doc_w in (0.3, 0.4):
            for char_w in (0.1, 0.2):
                for lsa_w in (0.1, 0.2):
                    for bm25_w in (0.2, 0.3, 0.4):
                        total = title_w + doc_w + char_w + lsa_w + bm25_w
                        grid.append({
                            "title_word_tfidf": title_w / total,
                            "document_word_tfidf": doc_w / total,
                            "char_wb_tfidf": char_w / total,
                            "lsa_document": lsa_w / total,
                            "bm25_document": bm25_w / total,
                        })
    return grid


def rank_random(n_queries: int, n_candidates: int, top_k: int, seed: int = RANDOM_STATE) -> list[list[int]]:
    """Random-order baseline: a shuffled candidate list per query."""
    rng = np.random.default_rng(seed)
    return [rng.permutation(n_candidates)[:top_k].tolist() for _ in range(n_queries)]


def rank_popularity(corpus_df: pd.DataFrame, n_queries: int, top_k: int) -> list[list[int]]:
    """Popularity baseline: the same most-viewed / highest-scored candidates for every query."""
    order = np.lexsort((corpus_df["view_count"].to_numpy(), corpus_df["score"].to_numpy()))[::-1]
    ranking = order[:top_k].tolist()
    return [ranking for _ in range(n_queries)]


# --------------------------------------------------------------------------- #
# Trained artefact + inference (shared by the prediction pipeline)
# --------------------------------------------------------------------------- #
CANDIDATE_META_COLUMNS = ["question_id", "title", "question_url", "tags", "creation_at"]


@dataclass
class RecommenderArtifact:
    """Everything the prediction pipeline needs to score new questions.

    Stored with :func:`joblib.dump`. It holds the fitted models, the metadata of
    the candidate pool they were fit on, the selected best model, and (for the
    hybrid) its tuned weights.
    """

    models: list
    candidates: pd.DataFrame
    best_model_name: str
    hybrid_weights: dict[str, float]
    text_fields: list[str] = dataclass_field(default_factory=list)


def _score_with_selection(artifact: RecommenderArtifact, query_df: pd.DataFrame) -> np.ndarray:
    if artifact.best_model_name == "hybrid":
        return combine_scores(artifact.models, query_df, artifact.hybrid_weights)
    model = next(m for m in artifact.models if m.name == artifact.best_model_name)
    return model.score(query_df)


def recommend(artifact: RecommenderArtifact, query_df: pd.DataFrame, top_k: int = 10) -> list[list[dict]]:
    """Return, per query row, the ``top_k`` most similar candidate questions.

    A candidate that is the query itself (same ``question_id``) is skipped so a
    question is never recommended to itself. Each recommendation is a dict with
    the candidate ``question_id``, its ``rank`` (1-based), the ``score``, and the
    candidate ``title``/``question_url`` for readability.
    """
    scores = _score_with_selection(artifact, query_df)
    candidate_ids = artifact.candidates["question_id"].to_numpy()
    candidate_titles = artifact.candidates["title"].astype(str).to_numpy()
    has_url = "question_url" in artifact.candidates.columns
    candidate_urls = artifact.candidates["question_url"].astype(str).to_numpy() if has_url else None
    query_ids = query_df["question_id"].to_numpy() if "question_id" in query_df.columns else [None] * len(query_df)

    # Retrieve a few extra so we can drop a self-match and still return top_k.
    ranked = rank_from_scores(scores, min(top_k + 1, scores.shape[1]))
    results = []
    for row_position, ranking in enumerate(ranked):
        query_id = query_ids[row_position]
        recommendations = []
        for candidate_index in ranking:
            if candidate_ids[candidate_index] == query_id:
                continue  # never recommend the question to itself
            entry = {
                "recommended_question_id": int(candidate_ids[candidate_index]),
                "rank": len(recommendations) + 1,
                "score": float(scores[row_position, candidate_index]),
                "recommended_title": candidate_titles[candidate_index],
            }
            if candidate_urls is not None:
                entry["recommended_url"] = candidate_urls[candidate_index]
            recommendations.append(entry)
            if len(recommendations) >= top_k:
                break
        results.append(recommendations)
    return results
