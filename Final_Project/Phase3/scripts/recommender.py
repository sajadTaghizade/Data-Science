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
    "document_prf",
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


def relevance_grades_for_query(query_tags: list[str], candidate_tags: list[set[str]]) -> dict[int, int]:
    """Graded relevance label: a candidate's grade is how many non-global tags it shares.

    Sharing two tags (e.g. ``templates`` **and** ``constexpr``) is stronger evidence of
    similarity than sharing one, so the grade rewards it. Binary relevance is just
    ``grade > 0``; graded nDCG uses the grades themselves.
    """
    query_secondary = secondary_tags(query_tags)
    if not query_secondary:
        return {}
    grades = {}
    for i, tags in enumerate(candidate_tags):
        shared = len(query_secondary & tags)
        if shared:
            grades[i] = shared
    return grades


def relevant_indices_for_query(query_tags: list[str], candidate_tags: list[set[str]]) -> set[int]:
    """Weak (binary) relevance label: a candidate is relevant if it shares a non-global tag."""
    return set(relevance_grades_for_query(query_tags, candidate_tags))


def ndcg_graded_at_k(ranked: list[int], grades: dict[int, int], k: int) -> float:
    """nDCG with *graded* gains (2**grade - 1), rewarding higher-grade hits ranked higher."""
    if not grades:
        return np.nan
    dcg = sum((2 ** grades.get(idx, 0) - 1) / np.log2(rank + 1)
              for rank, idx in enumerate(ranked[:k], start=1))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2 ** grade - 1) / np.log2(rank + 1) for rank, grade in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def evaluate_rankings(rankings, query_tag_lists, candidate_tag_sets, k_values=K_VALUES) -> dict:
    """Average the ranking metrics over every query that has at least one relevant candidate.

    Reports both the strict **binary** metrics (Hit/Recall/MAP/MRR/nDCG) and a fairer
    **graded** nDCG (``gnDCG@k``) that credits candidates sharing more tags.
    """
    grade_maps = [relevance_grades_for_query(tags, candidate_tag_sets) for tags in query_tag_lists]
    valid = [(ranked, grades) for ranked, grades in zip(rankings, grade_maps, strict=True) if grades]
    results = {
        "evaluated_queries": len(valid),
        "coverage": len(valid) / len(query_tag_lists) if len(query_tag_lists) else 0.0,
    }
    for k in k_values:
        results[f"Hit@{k}"] = np.mean([hit_at_k(r, set(g), k) for r, g in valid]) if valid else np.nan
        results[f"Recall@{k}"] = np.mean([recall_at_k(r, set(g), k) for r, g in valid]) if valid else np.nan
        results[f"MAP@{k}"] = np.mean([average_precision_at_k(r, set(g), k) for r, g in valid]) if valid else np.nan
        results[f"MRR@{k}"] = np.mean([reciprocal_rank_at_k(r, set(g), k) for r, g in valid]) if valid else np.nan
        results[f"nDCG@{k}"] = np.mean([ndcg_at_k(r, set(g), k) for r, g in valid]) if valid else np.nan
        results[f"gnDCG@{k}"] = np.mean([ndcg_graded_at_k(r, g, k) for r, g in valid]) if valid else np.nan
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


@dataclass
class PRFTfidfModel:
    """TF-IDF cosine with **pseudo-relevance feedback** (Rocchio query expansion).

    A classic information-retrieval trick that needs no embeddings: do a first
    retrieval, *assume* the top-M results are relevant, add their average vector
    to the query (``expanded = alpha*query + beta*centroid``), then retrieve
    again. This pulls in questions that use related vocabulary the original query
    did not mention, improving recall. Fully vectorised via a sparse feedback
    matrix, so it scores a whole batch of queries at once.
    """

    name: str
    field: str
    top_m: int = 10
    alpha: float = 1.0
    beta: float = 0.6
    params: dict | None = None
    vectorizer: TfidfVectorizer | None = None
    candidate_matrix: sparse.csr_matrix | None = None

    def fit(self, corpus_df: pd.DataFrame) -> PRFTfidfModel:
        params = self.params or {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2,
                                 "max_df": 0.90, "max_features": 60000, "token_pattern": TOKEN_PATTERN}
        self.vectorizer = TfidfVectorizer(lowercase=False, norm="l2", sublinear_tf=True, **params)
        self.candidate_matrix = self.vectorizer.fit_transform(corpus_df[self.field].fillna(""))
        return self

    def score(self, query_df: pd.DataFrame) -> np.ndarray:
        from sklearn.preprocessing import normalize
        queries = self.vectorizer.transform(query_df[self.field].fillna(""))
        first = queries @ self.candidate_matrix.T
        first = first.toarray() if sparse.issparse(first) else np.asarray(first)

        n_queries, n_candidates = first.shape
        top_m = min(self.top_m, n_candidates)
        # Build a sparse feedback matrix F (n_queries x n_candidates): 1/top_m on the
        # top-M candidates of each query. F @ candidate_matrix = per-query centroid.
        top_idx = np.concatenate([np.argpartition(-row, top_m - 1)[:top_m] for row in first])
        rows = np.repeat(np.arange(n_queries), top_m)
        feedback = sparse.csr_matrix((np.full(n_queries * top_m, 1.0 / top_m), (rows, top_idx)),
                                     shape=(n_queries, n_candidates))
        centroids = feedback @ self.candidate_matrix
        expanded = normalize(self.alpha * queries + self.beta * centroids)
        final = expanded @ self.candidate_matrix.T
        return final.toarray() if sparse.issparse(final) else np.asarray(final)

    @property
    def candidate_count(self) -> int:
        return self.candidate_matrix.shape[0]


# --------------------------------------------------------------------------- #
# Model factory
# --------------------------------------------------------------------------- #
DOCUMENT_TFIDF_PARAMS = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "max_df": 0.90,
                         "max_features": 60000, "token_pattern": TOKEN_PATTERN}


def build_candidate_models(bm25_params: dict | None = None) -> list:
    """Instantiate (unfitted) the full set of candidate recommender models.

    ``bm25_params`` lets the training pipeline pass BM25's tuned ``k1``/``b``.
    """
    bm25_params = bm25_params or {"k1": 1.5, "b": 0.75}
    return [
        TfidfCosineModel("title_word_tfidf", "title_clean", {
            "analyzer": "word", "ngram_range": (1, 3), "min_df": 1, "max_df": 0.95,
            "max_features": 20000, "token_pattern": TOKEN_PATTERN}),
        TfidfCosineModel("document_word_tfidf", "document_clean", dict(DOCUMENT_TFIDF_PARAMS)),
        TfidfCosineModel("char_wb_tfidf", "document_clean", {
            "analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "max_df": 0.95,
            "max_features": 50000}),
        LsaCosineModel("lsa_document", "document_clean", n_components=200),
        BM25Model("bm25_document", "document_clean", **bm25_params),
        PRFTfidfModel("document_prf", "document_clean", top_m=10, alpha=1.0, beta=0.6),
    ]


def fit_candidate_models(corpus_df: pd.DataFrame, bm25_params: dict | None = None) -> list:
    models = build_candidate_models(bm25_params)
    for model in models:
        model.fit(corpus_df)
    return models


def tune_bm25(train_df: pd.DataFrame, val_df: pd.DataFrame, candidate_secondary,
              metric: str = PRIMARY_METRIC) -> tuple[dict, pd.DataFrame]:
    """Grid-search BM25's ``k1`` and ``b`` on validation; return the best params + the search table."""
    rows = []
    for k1 in (1.0, 1.5, 2.0):
        for b in (0.5, 0.75, 0.9):
            model = BM25Model("bm25_document", "document_clean", k1=k1, b=b).fit(train_df)
            metrics = evaluate_rankings(rank_single(model, val_df), val_df["tag_list"], candidate_secondary)
            rows.append({"k1": k1, "b": b, metric: metrics[metric]})
    table = pd.DataFrame(rows).sort_values(metric, ascending=False).reset_index(drop=True)
    best = {"k1": float(table.iloc[0]["k1"]), "b": float(table.iloc[0]["b"])}
    return best, table


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


def mmr_rerank(relevance: np.ndarray, similarity: np.ndarray, top_k: int, lambda_: float = 0.7) -> list[int]:
    """Maximal Marginal Relevance: re-order candidates to balance relevance and diversity.

    Each pick maximises ``lambda*relevance - (1-lambda)*max_similarity_to_already_picked``,
    so near-duplicate results are pushed down and the top-K covers more ground.
    ``relevance`` and ``similarity`` are over the same candidate subset (both indexed
    0..n-1); returns local indices in MMR order. ``lambda_=1`` reduces to plain ranking.
    """
    n = len(relevance)
    top_k = min(top_k, n)
    span = relevance.max() - relevance.min()
    rel = (relevance - relevance.min()) / span if span > 0 else np.zeros(n)
    selected: list[int] = []
    remaining = list(range(n))
    while len(selected) < top_k and remaining:
        if not selected:
            best = max(remaining, key=lambda c: rel[c])
        else:
            best = max(remaining,
                       key=lambda c: lambda_ * rel[c] - (1 - lambda_) * max(similarity[c, s] for s in selected))
        selected.append(best)
        remaining.remove(best)
    return selected


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


def sample_hybrid_weights(model_names, n_random: int = 60, seed: int = RANDOM_STATE) -> list[dict[str, float]]:
    """Candidate weight vectors for the hybrid, tuned on validation.

    Scales to any number of models: it mixes structured seeds (uniform, and each
    single model dominant) with random simplex points drawn from a Dirichlet
    distribution - a standard randomised hyperparameter search that avoids the
    combinatorial blow-up of a hand-built grid.
    """
    names = list(model_names)
    n = len(names)
    combos: list[dict[str, float]] = [dict.fromkeys(names, 1.0 / n)]  # uniform blend
    for i in range(n):  # each-model-dominant seeds
        weights = {name: 0.1 / (n - 1) for name in names}
        weights[names[i]] = 0.9
        combos.append(weights)
    rng = np.random.default_rng(seed)
    for sample in rng.dirichlet(np.ones(n), size=n_random):
        combos.append({name: float(w) for name, w in zip(names, sample, strict=True)})
    return combos


def hybrid_weight_grid() -> list[dict[str, float]]:  # backwards-compatible helper
    """Default hybrid weight candidates over the standard candidate model set."""
    return sample_hybrid_weights(CANDIDATE_MODEL_NAMES)


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


def _diversity_vectors(artifact: RecommenderArtifact):
    """L2-normalised candidate vectors used for MMR similarity (document TF-IDF if available)."""
    for model in artifact.models:
        if model.name == "document_word_tfidf" and getattr(model, "candidate_matrix", None) is not None:
            return model.candidate_matrix
    return None


def recommend(artifact: RecommenderArtifact, query_df: pd.DataFrame, top_k: int = 10,
              diversity: float = 1.0, mmr_pool: int = 50) -> list[list[dict]]:
    """Return, per query row, the ``top_k`` most similar candidate questions.

    A candidate that is the query itself (same ``question_id``) is skipped so a
    question is never recommended to itself. When ``diversity < 1.0`` the top
    candidates are re-ranked with **MMR** (``diversity`` is the MMR ``lambda``) to
    reduce near-duplicate recommendations. Each recommendation is a dict with the
    candidate ``question_id``, ``rank`` (1-based), ``score``, and the candidate
    ``title``/``question_url``.
    """
    scores = _score_with_selection(artifact, query_df)
    candidate_ids = artifact.candidates["question_id"].to_numpy()
    candidate_titles = artifact.candidates["title"].astype(str).to_numpy()
    has_url = "question_url" in artifact.candidates.columns
    candidate_urls = artifact.candidates["question_url"].astype(str).to_numpy() if has_url else None
    query_ids = query_df["question_id"].to_numpy() if "question_id" in query_df.columns else [None] * len(query_df)

    use_mmr = diversity < 1.0
    div_vectors = _diversity_vectors(artifact) if use_mmr else None
    n_candidates = scores.shape[1]
    # Retrieve a pool wide enough to drop self-matches and give MMR room to diversify.
    pool_size = min(max(mmr_pool, top_k + 1), n_candidates) if use_mmr else min(top_k + 1, n_candidates)
    ranked = rank_from_scores(scores, pool_size)

    results = []
    for row_position, ranking in enumerate(ranked):
        query_id = query_ids[row_position]
        pool = [c for c in ranking if candidate_ids[c] != query_id]  # drop self up front

        if use_mmr and div_vectors is not None and len(pool) > 1:
            vectors = div_vectors[pool]
            similarity = vectors @ vectors.T
            similarity = similarity.toarray() if sparse.issparse(similarity) else np.asarray(similarity)
            order = mmr_rerank(scores[row_position, pool], similarity, top_k, lambda_=diversity)
            chosen = [pool[i] for i in order]
        else:
            chosen = pool[:top_k]

        recommendations = []
        for rank, candidate_index in enumerate(chosen, start=1):
            entry = {
                "recommended_question_id": int(candidate_ids[candidate_index]),
                "rank": rank,
                "score": float(scores[row_position, candidate_index]),
                "recommended_title": candidate_titles[candidate_index],
            }
            if candidate_urls is not None:
                entry["recommended_url"] = candidate_urls[candidate_index]
            recommendations.append(entry)
        results.append(recommendations)
    return results
