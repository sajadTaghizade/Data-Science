"""Unit tests for the dependency-free BM25 implementation."""

from __future__ import annotations

import numpy as np
import recommender as R


def test_bm25_exact_match_ranks_first(synthetic_questions):
    """A query identical to a candidate document should rank that document top-1."""
    model = R.BM25Model("bm25", "document_clean").fit(synthetic_questions)
    query = synthetic_questions.iloc[[3]]  # the 'template specialization example' row
    ranking = R.rank_single(model, query, top_k=3)[0]
    assert ranking[0] == 3


def test_bm25_topical_neighbour_beats_unrelated(synthetic_questions):
    """A thread question should score other thread questions above vector ones."""
    model = R.BM25Model("bm25", "document_clean").fit(synthetic_questions)
    scores = model.score(synthetic_questions.iloc[[6]])[0]  # 'std thread join detach'
    thread_indices = [7, 8, 11]      # other multithreading rows
    vector_indices = [0, 1, 2, 9]    # vector rows
    assert min(scores[i] for i in thread_indices) > max(scores[i] for i in vector_indices)


def test_bm25_score_shape_and_non_negative(synthetic_questions):
    model = R.BM25Model("bm25", "document_clean").fit(synthetic_questions)
    scores = model.score(synthetic_questions)
    assert scores.shape == (len(synthetic_questions), len(synthetic_questions))
    assert np.all(scores >= 0)       # BM25 idf uses log(1 + ...) so weights stay non-negative


def test_bm25_saturation_penalises_repetition():
    """Term-frequency saturation: 20 repeats must not score ~20x a single occurrence."""
    import pandas as pd
    corpus = pd.DataFrame({
        "document_clean": ["alpha", "alpha " * 20, "beta gamma"],
    })
    model = R.BM25Model("bm25", "document_clean", min_df=1).fit(corpus)
    scores = model.score(pd.DataFrame({"document_clean": ["alpha"]}))[0]
    # The 20x document scores higher, but far below 20x the single-occurrence doc.
    assert scores[1] > scores[0]
    assert scores[1] < 3 * scores[0]
