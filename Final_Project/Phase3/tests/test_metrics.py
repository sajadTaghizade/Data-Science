"""Unit tests for the ranking metrics (hand-computed expected values)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import recommender as R


def test_hit_at_k():
    assert R.hit_at_k([3, 1, 2], {1}, 1) == 0.0   # idx 3 first, not relevant
    assert R.hit_at_k([3, 1, 2], {1}, 2) == 1.0   # idx 1 within top-2
    assert R.hit_at_k([3, 1, 2], {9}, 3) == 0.0   # no relevant candidate present


def test_recall_at_k():
    assert R.recall_at_k([1, 5, 2], {1, 2}, 1) == 0.5   # 1 of 2 relevant retrieved
    assert R.recall_at_k([1, 5, 2], {1, 2}, 3) == 1.0   # both retrieved


def test_average_precision_at_k():
    # ranked [1,5,2], relevant {1,2}: hits at rank 1 (1/1) and rank 3 (2/3); /min(2,3).
    expected = (1.0 + 2.0 / 3.0) / 2
    assert R.average_precision_at_k([1, 5, 2], {1, 2}, 3) == pytest.approx(expected)


def test_reciprocal_rank_at_k():
    assert R.reciprocal_rank_at_k([1, 5, 2], {1}, 3) == 1.0    # first position
    assert R.reciprocal_rank_at_k([5, 1, 2], {1}, 3) == 0.5    # second position
    assert R.reciprocal_rank_at_k([5, 4, 3], {1}, 3) == 0.0    # not found


def test_ndcg_at_k():
    assert R.ndcg_at_k([1], {1}, 1) == 1.0                      # perfect
    assert R.ndcg_at_k([5, 1], {1}, 2) == pytest.approx(1.0 / math.log2(3))  # relevant at rank 2


def test_empty_relevant_is_nan():
    assert np.isnan(R.hit_at_k([1, 2, 3], set(), 3))
    assert np.isnan(R.recall_at_k([1, 2, 3], set(), 3))
    assert np.isnan(R.average_precision_at_k([1, 2, 3], set(), 3))


def test_relevant_indices_ignore_global_tag():
    # The universal 'c++' tag must not create relevance; only secondary tags count.
    candidate_tags = [{"c++", "vector"}, {"c++"}, {"c++", "templates"}]
    relevant = R.relevant_indices_for_query(["c++", "vector"], candidate_tags)
    assert relevant == {0}                                     # only the 'vector' candidate


def test_evaluate_rankings_coverage_and_perfect_scores():
    # Two queries, candidates share their secondary tag at index 0 -> perfect ranking.
    candidate_tags = [{"vector"}, {"threads"}]
    rankings = [[0, 1], [0, 1]]
    query_tags = [["vector"], ["c++"]]  # second query has no secondary tag -> skipped
    metrics = R.evaluate_rankings(rankings, query_tags, candidate_tags)
    assert metrics["evaluated_queries"] == 1
    assert metrics["coverage"] == 0.5
    assert metrics["Hit@5"] == 1.0
