#!/usr/bin/env python3
"""FRED macro series ingestion.

Housing bulk data is monthly; these are the only genuinely daily-cadence inputs
in the project. Rates in particular drive the demand signals (pending ratio,
days on market, price reductions) that the heat score is built from, so having
them alongside gives the dashboard something that actually moves day to day.

Uses the keyless fredgraph.csv endpoint — no API key, no registration.
"""
import logging
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)


def fetch_series(series_id: str, base_url: str, timeout: int = 60) -> pd.DataFrame:
    """Download one series as tidy rows: series_id, date, value."""
    r = requests.get(base_url, params={"id": series_id}, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))

    # fredgraph returns [observation_date | DATE, <SERIES_ID>]; the date column
    # has been spelled both ways over time, so key off position rather than name.
    date_col, value_col = df.columns[0], df.columns[-1]
    out = pd.DataFrame({
        "series_id": series_id,
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        # Missing observations come through as "." in FRED's CSV.
        "value": pd.to_numeric(df[value_col].replace(".", pd.NA), errors="coerce"),
    })
    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def fetch_all(cfg: dict, data_dir: Path) -> pd.DataFrame:
    """Fetch every configured series into one tidy parquet.

    A single failing series is logged and skipped rather than aborting the run —
    FRED occasionally discontinues an id, and that should not cost us the other
    seventeen.
    """
    series_cfg = cfg["series"]
    base_url = cfg["base_url"]
    timeout = cfg.get("timeout", 60)

    frames, failed = [], []
    for series_id, meta in series_cfg.items():
        try:
            df = fetch_series(series_id, base_url, timeout=timeout)
        except Exception as exc:
            log.warning(f"  {series_id}: failed — {exc}")
            failed.append(series_id)
            continue

        if df.empty:
            log.warning(f"  {series_id}: no observations returned — skipping")
            failed.append(series_id)
            continue

        df["label"] = meta.get("label", series_id)
        df["freq"] = meta.get("freq", "")
        df["units"] = meta.get("units", "")
        frames.append(df)
        last = df.dropna(subset=["value"]).tail(1)
        through = f"{last['date'].iloc[0]:%Y-%m-%d}" if len(last) else "—"
        log.info(f"  {series_id:<13} {len(df):>6,} obs  through {through}")

    if not frames:
        raise RuntimeError("No FRED series could be fetched")

    out = pd.concat(frames, ignore_index=True)
    out["fetched_at"] = datetime.now(timezone.utc)

    path = data_dir / cfg["filename"]
    out.to_parquet(path, index=False)
    log.info(
        f"Saved → {path}  ({out['series_id'].nunique()} series, {len(out):,} rows"
        + (f", {len(failed)} failed: {', '.join(failed)}" if failed else "") + ")"
    )
    return out


def latest_values(df: pd.DataFrame) -> pd.DataFrame:
    """Most recent non-null observation per series, with its prior-period change."""
    rows = []
    for series_id, g in df.dropna(subset=["value"]).groupby("series_id"):
        g = g.sort_values("date")
        latest = g.iloc[-1]
        prev = g.iloc[-2] if len(g) > 1 else None
        # Year-ago comparison uses the nearest observation at or before the
        # anniversary, which keeps it meaningful across mixed frequencies.
        yr_ago_rows = g[g["date"] <= latest["date"] - pd.DateOffset(years=1)]
        yr_ago = yr_ago_rows.iloc[-1] if len(yr_ago_rows) else None
        rows.append({
            "series_id": series_id,
            "label": latest["label"],
            "freq": latest["freq"],
            "units": latest["units"],
            "date": latest["date"],
            "value": latest["value"],
            "change": None if prev is None else latest["value"] - prev["value"],
            "change_1y": None if yr_ago is None else latest["value"] - yr_ago["value"],
        })
    return pd.DataFrame(rows).sort_values("series_id").reset_index(drop=True)
