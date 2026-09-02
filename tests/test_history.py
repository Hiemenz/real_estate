"""history.append_snapshot must be idempotent per data-month, and
compute_momentum must compare two periods on a shared normalization basis
(see the module docstring / [[heat_score_winsorizing]] memory) rather than
diffing each period's independently-normalized heat_score.
"""
import pandas as pd

from history import append_snapshot, compute_momentum
from pipeline import compute_heat_scores

WEIGHTS = {"a": 1.0}


def _agg(a_values):
    return pd.DataFrame({
        "_zip": [f"{i:05d}" for i in range(len(a_values))],
        "zip_name": [f"City {i}" for i in range(len(a_values))],
        "a": a_values,
        "heat_score": compute_heat_scores(pd.DataFrame({"a": a_values}), WEIGHTS),
    })


def test_rerunning_same_month_replaces_not_duplicates(tmp_path):
    path = tmp_path / "history.parquet"
    append_snapshot(_agg([1, 2, 3]), "2026-06-01", ["a"], path, max_snapshots=10)
    hist = append_snapshot(_agg([1, 2, 3]), "2026-06-01", ["a"], path, max_snapshots=10)
    assert hist["data_through"].nunique() == 1
    assert len(hist) == 3


def test_retention_keeps_only_max_snapshots(tmp_path):
    path = tmp_path / "history.parquet"
    for month in ["2026-01-01", "2026-02-01", "2026-03-01"]:
        hist = append_snapshot(_agg([1, 2]), month, ["a"], path, max_snapshots=2)
    assert hist["data_through"].nunique() == 2
    assert pd.Timestamp("2026-01-01") not in set(hist["data_through"])


def test_momentum_empty_with_one_snapshot(tmp_path):
    path = tmp_path / "history.parquet"
    hist = append_snapshot(_agg([1, 2, 3]), "2026-06-01", ["a"], path, max_snapshots=10)
    movers, prev, latest = compute_momentum(hist, ["a"], WEIGHTS, compute_heat_scores)
    assert movers.empty
    assert prev is None and latest is None


def test_momentum_reflects_relative_move_not_population_shift():
    """Same ZIP, same raw feature value, in two periods with different peer
    populations. Per-period normalization would show a fake swing; comparing
    on a shared basis (as compute_momentum does) should not."""
    # Period 1: zip 00000 sits in the middle of a wide population.
    prev = pd.DataFrame({
        "_zip": ["00000", "00001", "00002"],
        "a": [5.0, 0.0, 10.0],
        "data_through": pd.Timestamp("2026-05-01"),
    })
    # Period 2: same zip 00000, identical raw value, but now the *population*
    # shrank toward it — naive per-period MinMax would push its score toward
    # the middle/high end even though nothing about this ZIP itself changed.
    latest = pd.DataFrame({
        "_zip": ["00000", "00001"],
        "a": [5.0, 4.0],
        "data_through": pd.Timestamp("2026-06-01"),
    })
    hist = pd.concat([prev, latest], ignore_index=True)

    movers, prev_date, latest_date = compute_momentum(hist, ["a"], WEIGHTS, compute_heat_scores)
    row = movers[movers["_zip"] == "00000"].iloc[0]
    # Raw value for 00000 didn't move at all, so its comparable heat score
    # (computed jointly over both periods) shouldn't move either.
    assert row["delta"] == 0.0
