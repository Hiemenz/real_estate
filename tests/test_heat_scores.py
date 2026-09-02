"""compute_heat_scores and rank_clusters_by_heat are the two pure functions the
rest of the pipeline (mine, history, dashboard) all lean on — worth locking
their behavior down directly rather than only through an end-to-end run.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline import compute_heat_scores, rank_clusters_by_heat

WEIGHTS = {"a": 1.0, "b": -1.0}


def test_higher_positive_weight_feature_scores_higher():
    agg = pd.DataFrame({"a": [0.0, 5.0, 10.0], "b": [0.0, 0.0, 0.0]})
    scores = compute_heat_scores(agg, WEIGHTS)
    assert list(scores) == sorted(scores)


def test_negative_weight_feature_scores_inversely():
    agg = pd.DataFrame({"a": [0.0, 0.0, 0.0], "b": [0.0, 5.0, 10.0]})
    scores = compute_heat_scores(agg, WEIGHTS)
    assert list(scores) == sorted(scores, reverse=True)


def test_missing_required_feature_is_nan_not_dropped():
    agg = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1.0, 1.0, 1.0]})
    scores = compute_heat_scores(agg, WEIGHTS)
    assert len(scores) == len(agg)
    assert pd.isna(scores.iloc[1])
    assert scores.notna().sum() == 2


def test_output_is_0_to_100():
    # b runs opposite to a so both terms reinforce (a: +weight, b: -weight)
    # instead of canceling out.
    agg = pd.DataFrame({"a": np.linspace(0, 1000, 20), "b": np.linspace(1, 0, 20)})
    scores = compute_heat_scores(agg, WEIGHTS)
    assert scores.min() == pytest.approx(0.0, abs=1e-6)
    assert scores.max() == pytest.approx(100.0, abs=1e-6)


def test_empty_valid_rows_returns_all_nan():
    agg = pd.DataFrame({"a": [np.nan, np.nan], "b": [1.0, 2.0]})
    scores = compute_heat_scores(agg, WEIGHTS)
    assert scores.isna().all()


def test_winsorizing_limits_a_single_outliers_influence():
    """A near-zero base price can make one ZIP's YoY% a 6-figure outlier and
    (pre-winsorizing) flatten every other ZIP's score toward the low end of the
    range. With clipping, the well-behaved majority should still spread out.

    Uses 199 normal rows + 1 outlier (outlier = 0.5% of the population) so the
    outlier sits safely past the 99th-percentile clip boundary — at exactly 1
    outlier per 100 rows the interpolated 99th percentile itself gets dragged
    partway toward the outlier, which is a real (separate) edge case, not what
    this test is checking.
    """
    values = list(np.linspace(0, 20, 199)) + [1_000_000.0]
    agg = pd.DataFrame({"a": values, "b": [0.0] * 200})
    scores = compute_heat_scores(agg, WEIGHTS)
    normal_scores = scores.iloc[:199]
    # Without clipping, the outlier would compress the other 199 rows into a
    # sliver near 0. Clipping keeps them spread across most of the range.
    assert normal_scores.max() - normal_scores.min() > 50


def test_rank_clusters_hottest_gets_id_zero():
    raw_labels = np.array([2, 2, 0, 0, 1, 1])
    index = pd.RangeIndex(6)
    # cluster 0 has the highest heat, cluster 2 lowest
    heat = pd.Series([10, 10, 90, 90, 50, 50], index=index)
    ranked = rank_clusters_by_heat(raw_labels, index, heat)
    assert ranked[2] == 0
    assert ranked[3] == 0
    assert ranked[0] == 2
    assert ranked[1] == 2
