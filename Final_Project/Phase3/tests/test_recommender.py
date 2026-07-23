"""Unit tests for the model interface, splitting, hybrid, and inference."""

from __future__ import annotations

import make_predictions as MP
import numpy as np
import recommender as R


def test_temporal_split_is_ordered_and_disjoint(synthetic_questions):
    train, val, test = R.temporal_split(synthetic_questions)
    assert len(train) + len(val) + len(test) == len(synthetic_questions)
    ids = set(train["question_id"]) | set(val["question_id"]) | set(test["question_id"])
    assert len(ids) == len(synthetic_questions)                 # no row lost or duplicated
    # No future leaks into the past: every train date <= every test date.
    assert train["creation_at"].max() <= test["creation_at"].min()


def test_fit_candidate_models_shapes(synthetic_questions):
    models = R.fit_candidate_models(synthetic_questions)
    assert [m.name for m in models] == list(R.CANDIDATE_MODEL_NAMES)
    for model in models:
        assert model.candidate_count == len(synthetic_questions)
        scores = model.score(synthetic_questions.iloc[:3])
        assert scores.shape == (3, len(synthetic_questions))


def test_normalize_rows_scales_to_unit_interval_preserving_order():
    scores = np.array([[1.0, 3.0, 2.0], [10.0, 10.0, 10.0]])
    normalised = R.normalize_rows(scores)
    assert normalised.min() >= 0.0 and normalised.max() <= 1.0
    assert np.argmax(normalised[0]) == np.argmax(scores[0])     # order preserved
    assert np.allclose(normalised[1], 0.0)                      # flat row -> no div-by-zero


def test_rank_from_scores_orders_descending():
    scores = np.array([[0.1, 0.9, 0.5, 0.7]])
    assert R.rank_from_scores(scores, top_k=2)[0] == [1, 3]


def test_hybrid_combines_all_models(synthetic_questions):
    models = R.fit_candidate_models(synthetic_questions)
    weights = R.hybrid_weight_grid()[0]
    combined = R.combine_scores(models, synthetic_questions.iloc[:2], weights)
    assert combined.shape == (2, len(synthetic_questions))


def test_recommend_excludes_self_and_returns_top_k(synthetic_questions):
    models = R.fit_candidate_models(synthetic_questions)
    artifact = R.RecommenderArtifact(
        models=models,
        candidates=synthetic_questions[["question_id", "title_clean", "tags", "creation_at"]].rename(
            columns={"title_clean": "title"}),
        best_model_name="document_word_tfidf",
        hybrid_weights=R.hybrid_weight_grid()[0],
    )
    recs = R.recommend(artifact, synthetic_questions, top_k=3)
    assert len(recs) == len(synthetic_questions)
    for position, rec_list in enumerate(recs):
        assert len(rec_list) == 3
        query_id = synthetic_questions.iloc[position]["question_id"]
        assert query_id not in [r["recommended_question_id"] for r in rec_list]   # no self-recommendation
        assert [r["rank"] for r in rec_list] == [1, 2, 3]


def test_recommend_returns_topically_relevant_neighbours(synthetic_questions):
    """A vector question should get other vector questions recommended first."""
    models = R.fit_candidate_models(synthetic_questions)
    artifact = R.RecommenderArtifact(
        models=models, candidates=synthetic_questions.rename(columns={"title_clean": "title"}),
        best_model_name="document_word_tfidf", hybrid_weights={})
    query = synthetic_questions[synthetic_questions["question_id"] == 1]   # 'std vector resize'
    top = R.recommend(artifact, query, top_k=3)[0]
    top_tags = [set(synthetic_questions.set_index("question_id").loc[r["recommended_question_id"], "tag_list"])
                for r in top]
    assert any("vector" in tags for tags in top_tags)


def test_recommendations_to_frame_flattens(synthetic_questions):
    query_df = synthetic_questions.iloc[:2]
    recommendations = [
        [{"recommended_question_id": 5, "rank": 1, "score": 0.9, "recommended_title": "a"}],
        [{"recommended_question_id": 7, "rank": 1, "score": 0.8, "recommended_title": "b"}],
    ]
    frame = MP.recommendations_to_frame(query_df, recommendations, "document_word_tfidf")
    assert list(frame["query_question_id"]) == [1, 2]
    assert set(frame["model_name"]) == {"document_word_tfidf"}
    assert "generated_at" in frame.columns
