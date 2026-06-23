"""Unit tests for the retrieval metrics used to evaluate the recommender."""

import math

import numpy as np
import pytest

from scripts.feature_engineering import (
    average_precision_at_k,
    evaluate_rankings,
    hit_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
    relevant_indices_for_query,
)


RANKED = [0, 1, 2, 3, 4]
RELEVANT = {2, 4}


def test_hit_at_k():
    assert hit_at_k(RANKED, RELEVANT, 5) == 1.0
    assert hit_at_k(RANKED, {2}, 2) == 0.0  # relevant item is at rank 3, outside top 2
    assert hit_at_k(RANKED, {0}, 1) == 1.0


def test_recall_at_k():
    assert recall_at_k(RANKED, RELEVANT, 5) == 1.0
    assert recall_at_k(RANKED, RELEVANT, 3) == 0.5  # only index 2 retrieved out of {2, 4}


def test_reciprocal_rank():
    assert reciprocal_rank_at_k(RANKED, RELEVANT, 5) == pytest.approx(1 / 3)  # first hit at rank 3
    assert reciprocal_rank_at_k(RANKED, {0}, 5) == 1.0
    assert reciprocal_rank_at_k(RANKED, {99}, 5) == 0.0  # relevant item absent


def test_average_precision():
    # hits at ranks 3 and 5 -> (1/3 + 2/5) / min(2, 5)
    expected = (1 / 3 + 2 / 5) / 2
    assert average_precision_at_k(RANKED, RELEVANT, 5) == pytest.approx(expected)


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k([2, 4, 0, 1, 3], RELEVANT, 5) == pytest.approx(1.0)


def test_ndcg_matches_manual_value():
    dcg = 1 / math.log2(3 + 1) + 1 / math.log2(5 + 1)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(RANKED, RELEVANT, 5) == pytest.approx(dcg / idcg)


def test_empty_relevant_returns_nan():
    for metric in (hit_at_k, recall_at_k, reciprocal_rank_at_k, average_precision_at_k, ndcg_at_k):
        assert np.isnan(metric(RANKED, set(), 5))


def test_relevant_indices_drop_global_tag():
    # The universal c++ tag must not create relevance; only the shared 'qt' tag counts.
    candidates = [{"qt"}, {"boost"}, set()]
    relevant = relevant_indices_for_query(["c++", "qt"], candidates)
    assert relevant == {0}
    # A query with only the global tag has no usable relevance signal.
    assert relevant_indices_for_query(["c++"], candidates) == set()


def test_evaluate_rankings_skips_unlabeled_queries():
    rankings = [[0, 1, 2], [0, 1, 2]]
    query_tags = [["qt"], ["c++"]]  # second query has no secondary tag -> skipped
    candidate_tags = [{"qt"}, {"boost"}, {"qt"}]
    results = evaluate_rankings(rankings, query_tags, candidate_tags, k_values=(3,))
    assert results["evaluated_queries"] == 1
    assert results["Hit@3"] == 1.0
