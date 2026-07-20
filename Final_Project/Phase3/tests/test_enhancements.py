"""Unit tests for the Phase 3 enhancements: graded nDCG, MMR, PRF, BM25 tuning."""

from __future__ import annotations

import math

import numpy as np
import pytest
import recommender as R


# --------------------------- graded relevance --------------------------- #
def test_relevance_grades_count_shared_tags():
    candidates = [{"templates", "constexpr", "c++"}, {"templates"}, {"threads"}]
    grades = R.relevance_grades_for_query(["templates", "constexpr", "c++"], candidates)
    assert grades == {0: 2, 1: 1}          # 2 shared, 1 shared, none


def test_ndcg_graded_rewards_higher_grade_first():
    grades = {0: 2, 1: 1}
    assert R.ndcg_graded_at_k([0, 1], grades, 10) == 1.0            # ideal order
    # Swapped order scores less than 1 but stays positive.
    swapped = R.ndcg_graded_at_k([1, 0], grades, 10)
    assert 0 < swapped < 1
    expected = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))       # gains 2^g-1
    assert swapped == pytest.approx(expected)


def test_evaluate_rankings_includes_graded_ndcg():
    metrics = R.evaluate_rankings([[0, 1]], [["vector"]], [{"vector"}, {"threads"}])
    assert "gnDCG@10" in metrics and "gnDCG@5" in metrics


# --------------------------------- MMR ---------------------------------- #
def test_mmr_prefers_diverse_over_near_duplicate():
    # Items 0 and 1 are near-identical; item 2 is different but slightly less relevant.
    relevance = np.array([1.0, 0.9, 0.8])
    similarity = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    order = R.mmr_rerank(relevance, similarity, top_k=2, lambda_=0.5)
    assert order[0] == 0          # most relevant picked first
    assert order[1] == 2          # diverse item beats the near-duplicate of item 0


def test_mmr_lambda_one_is_plain_relevance_order():
    relevance = np.array([0.2, 0.9, 0.5])
    similarity = np.eye(3)
    assert R.mmr_rerank(relevance, similarity, top_k=3, lambda_=1.0) == [1, 2, 0]


def test_recommend_with_diversity_runs_and_excludes_self(synthetic_questions):
    models = R.fit_candidate_models(synthetic_questions)
    artifact = R.RecommenderArtifact(
        models=models, candidates=synthetic_questions.rename(columns={"title_clean": "title"}),
        best_model_name="document_word_tfidf", hybrid_weights={})
    query = synthetic_questions[synthetic_questions["question_id"] == 1]
    recs = R.recommend(artifact, query, top_k=3, diversity=0.6)[0]
    assert len(recs) == 3
    assert 1 not in [r["recommended_question_id"] for r in recs]


# --------------------------------- PRF ---------------------------------- #
def test_prf_model_interface_and_shape(synthetic_questions):
    model = R.PRFTfidfModel("document_prf", "document_clean").fit(synthetic_questions)
    assert model.candidate_count == len(synthetic_questions)
    scores = model.score(synthetic_questions.iloc[:3])
    assert scores.shape == (3, len(synthetic_questions))


def test_prf_retrieves_topical_neighbours(synthetic_questions):
    model = R.PRFTfidfModel("document_prf", "document_clean").fit(synthetic_questions)
    ranking = R.rank_single(model, synthetic_questions.iloc[[6]], top_k=3)[0]  # a thread question
    top_tags = [set(synthetic_questions.iloc[i]["tag_list"]) for i in ranking]
    assert any("multithreading" in tags for tags in top_tags)


# ----------------------------- BM25 tuning ------------------------------ #
def test_tune_bm25_returns_valid_params(synthetic_questions):
    train, val, _ = R.temporal_split(synthetic_questions)
    candidate_secondary = [R.secondary_tags(t) for t in train["tag_list"]]
    best, table = R.tune_bm25(train, val, candidate_secondary)
    assert best["k1"] in (1.0, 1.5, 2.0) and best["b"] in (0.5, 0.75, 0.9)
    assert len(table) == 9                                   # 3 x 3 grid
    assert table.iloc[0][R.PRIMARY_METRIC] >= table.iloc[-1][R.PRIMARY_METRIC]  # sorted best-first


# ------------------------- hybrid weight search ------------------------- #
def test_sample_hybrid_weights_are_valid_simplex():
    names = list(R.CANDIDATE_MODEL_NAMES)
    combos = R.sample_hybrid_weights(names, n_random=5)
    assert len(combos) == 1 + len(names) + 5                 # uniform + per-model + random
    for weights in combos:
        assert set(weights) == set(names)
        assert sum(weights.values()) == pytest.approx(1.0)
        assert all(w >= 0 for w in weights.values())
