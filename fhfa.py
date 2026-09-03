"""FHFA House Price Index ingestion.

A third, independent price signal alongside Realtor.com (listing prices) and
Redfin (closed sale prices): FHFA's HPI is a repeat-sales index built from
Fannie Mae / Freddie Mac conforming-mortgage transaction data, so it moves on
a different methodology and a different slice of the market than either
listing- or closing-price medians. Useful as a genuine cross-check, not just
another opinion on the same underlying transactions.

No ZIP-level file is published — state and metro (MSA) are the finest
official grain (a tract-level file exists but at ~90MB with no ZIP mapping of
its own, so it's left out; see the [[fhfa]] config comment for CSV shapes).
"""
import logging
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

_UA = "Mozilla/5.0 (compatible; real-estate-pipeline/1.0)"


def _get_csv(url: str, timeout: int) -> str:
    # FHFA's CDN 404s a plain requests default User-Agent; a browser-like one
    # is required even though no auth/key is involved.
    r = requests.get(url, headers={"User-Agent": _UA}, timeout=timeout)
    r.raise_for_status()
    return r.text


def _fetch_state(cfg: dict) -> pd.DataFrame:
    text = _get_csv(cfg["state_url"], cfg.get("timeout", 30))
    df = pd.read_csv(StringIO(text), header=None, names=["place", "year", "quarter", "hpi"])
    df["level"] = "state"
    df["place_id"] = df["place"]
    df["yoy_pct"] = pd.Series(float("nan"), index=df.index, dtype="float64")
    return df


def _parse_paren_pct(s: pd.Series) -> pd.Series:
    """FHFA wraps the YoY %-change column in parens regardless of sign, e.g.
    '( 6.05)' or '(-6.05)'; '-' alone means not-yet-available."""
    cleaned = s.astype(str).str.strip().str.replace(r"[()]", "", regex=True).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def _fetch_metro(cfg: dict) -> pd.DataFrame:
    text = _get_csv(cfg["metro_url"], cfg.get("timeout", 30))
    df = pd.read_csv(
        StringIO(text), header=None,
        names=["place", "place_id", "year", "quarter", "hpi", "yoy_pct_raw"],
    )
    df["level"] = "metro"
    df["hpi"] = pd.to_numeric(df["hpi"], errors="coerce")
    df["yoy_pct"] = _parse_paren_pct(df["yoy_pct_raw"])
    return df.drop(columns=["yoy_pct_raw"])


def fetch_all(cfg: dict, data_dir: Path) -> pd.DataFrame:
    log.info("Fetching FHFA House Price Index (state + metro) …")
    state = _fetch_state(cfg)
    metro = _fetch_metro(cfg)
    out = pd.concat([state, metro], ignore_index=True)
    # state's place_id is a string abbreviation, metro's is an integer MSA
    # code — mixed dtypes in one column breaks the parquet write.
    out["place_id"] = out["place_id"].astype(str)
    out["period"] = pd.to_datetime(
        out["year"].astype(str) + "-" + ((out["quarter"] - 1) * 3 + 1).astype(str) + "-01"
    )
    out = out[["level", "place", "place_id", "year", "quarter", "period", "hpi", "yoy_pct"]]

    path = data_dir / cfg["filename"]
    out.to_parquet(path, index=False)
    log.info(
        f"Saved → {path}  "
        f"({(out['level'] == 'state').sum():,} state rows, {(out['level'] == 'metro').sum():,} metro rows, "
        f"through {out['period'].max():%Y-%m})"
    )
    return out


def latest_metro_movers(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Most recent quarter's metro rows with a usable YoY change, sorted hottest first."""
    metro = df[df["level"] == "metro"].dropna(subset=["yoy_pct"])
    if metro.empty:
        return metro
    latest_period = metro["period"].max()
    latest = metro[metro["period"] == latest_period]
    return latest.nlargest(top_n, "yoy_pct")
