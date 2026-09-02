"""Append-only heat-score history, so market movement can be tracked over time.

heat_score is min-max normalized against the population of a single mine run
(see compute_heat_scores in pipeline.py / [[heat_score_winsorizing]] memory), so
a ZIP's score can shift purely because *other* ZIPs shifted, not because that
ZIP itself changed. Snapshotting the score alone would make cross-run deltas
meaningless — a ZIP could show a "+8 point" move that is entirely an artifact of
this month's population.

The fix: persist the raw (pre-normalization) features every run, and when
comparing two periods, recompute heat_score jointly over the union of both
periods' rows. That gives the two periods a shared normalization basis, so the
resulting delta reflects the ZIP moving relative to a fixed frame, not the frame
itself shifting under it.
"""
from pathlib import Path

import pandas as pd


def append_snapshot(
    agg: pd.DataFrame,
    data_through,
    feature_cols: list[str],
    history_path: Path,
    max_snapshots: int,
) -> pd.DataFrame:
    """Add one dated snapshot of raw features + heat_score to the history file.

    Snapshots are keyed by `data_through` (the source data's month), not the run
    date — re-running `mine` on data that hasn't moved replaces that snapshot
    in place instead of appending a duplicate/phantom observation.
    """
    keep = ["_zip", "zip_name", *feature_cols, "heat_score"]
    keep = [c for c in dict.fromkeys(keep) if c in agg.columns]
    snap = agg[keep].copy()
    snap.insert(0, "data_through", pd.Timestamp(data_through))
    snap["snapshot_at"] = pd.Timestamp.now(tz="UTC")

    if history_path.exists():
        hist = pd.read_parquet(history_path)
        hist = hist[hist["data_through"] != snap["data_through"].iloc[0]]
        hist = pd.concat([hist, snap], ignore_index=True) if len(hist) else snap
    else:
        hist = snap

    keep_dates = sorted(hist["data_through"].unique())[-max_snapshots:]
    hist = hist[hist["data_through"].isin(keep_dates)].reset_index(drop=True)

    hist.to_parquet(history_path, index=False)
    return hist


def compute_momentum(
    hist: pd.DataFrame,
    feature_cols: list[str],
    heat_weights: dict[str, float],
    compute_heat_scores_fn,
) -> tuple[pd.DataFrame, "pd.Timestamp | None", "pd.Timestamp | None"]:
    """Compare the two most recent snapshots on a shared normalization basis.

    Returns (movers_df, prev_date, latest_date). movers_df is empty (with
    prev_date/latest_date None) if fewer than two snapshots exist yet.
    """
    dates = sorted(hist["data_through"].unique())
    if len(dates) < 2:
        return pd.DataFrame(), None, None

    prev_date, latest_date = dates[-2], dates[-1]
    pair = hist[hist["data_through"].isin([prev_date, latest_date])].copy()
    pair["heat_score_comparable"] = compute_heat_scores_fn(pair, heat_weights)

    prev = pair[pair["data_through"] == prev_date][["_zip", "heat_score_comparable"]]
    prev = prev.rename(columns={"heat_score_comparable": "heat_score_prev"})
    latest_cols = ["_zip", "heat_score_comparable"]
    if "zip_name" in pair.columns:
        latest_cols.append("zip_name")
    latest = pair[pair["data_through"] == latest_date][latest_cols]
    latest = latest.rename(columns={"heat_score_comparable": "heat_score_latest"})

    movers = prev.merge(latest, on="_zip", how="inner")
    movers["delta"] = movers["heat_score_latest"] - movers["heat_score_prev"]
    movers = movers.sort_values("delta", ascending=False).reset_index(drop=True)
    return movers, prev_date, latest_date
