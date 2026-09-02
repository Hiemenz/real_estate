#!/usr/bin/env python3
"""
Real Estate data pipeline: fetch → mine → forecast → report

Commands:
  python pipeline.py fetch [--force]         change-detected download (Realtor.com + Redfin)
  python pipeline.py fred                    fetch daily/weekly/monthly FRED macro series
  python pipeline.py mine [--lookback N]     compute market heat scores, clusters & history snapshot
  python pipeline.py history                 report biggest heat-score movers since last snapshot
  python pipeline.py forecast [--top N]      run Prophet for top N hot ZIPs
  python pipeline.py backtest [--top N]      hold-out accuracy (MAPE) for the forecast model
  python pipeline.py report                  render data/dashboard.html
  python pipeline.py all [--top N]           fetch + fred + (mine + forecast + backtest if changed) + report

`all` is safe to run daily: fetch only re-downloads a source whose ETag/
Last-Modified actually changed, and mine/forecast/backtest are skipped on days
where nothing new landed. A failed run is pushed to the alert channel
configured in config.toml's [alerts].

Output files (in data/):
  market_scores.parquet      — Realtor.com heat score + cluster per ZIP
  county_scores.parquet      — county-level rollup (price levels & YoY trend)
  redfin_zip_scores.parquet  — independent Redfin-derived heat score per ZIP
  forecasts.parquet          — 12-month Prophet forecasts for top ZIPs
  backtest.parquet           — held-out actual vs. forecast, per ZIP/month
  score_history.parquet      — append-only heat-score snapshots for momentum
  fred_series.parquet        — daily/weekly/monthly macro series (rates, starts, ...)
  fetch_meta.json            — per-source fetch timestamps (freshness indicator)
  fetch_state.json           — per-source ETag/Last-Modified for change detection
  dashboard.html             — static, self-contained HTML report

Configuration (weights, cluster count, source URLs, FRED series, alerts, ...)
lives in config.toml.
"""
import argparse
import json
import logging
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

import backtest as backtest_mod
import fetching
import fred as fred_mod
import history as history_mod
import notify

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config.toml"
LOCAL_CONFIG_PATH = Path(__file__).parent / "config.local.toml"


def _deep_merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Path = CONFIG_PATH, local_path: Path = LOCAL_CONFIG_PATH) -> dict:
    with open(path, "rb") as f:
        config = tomllib.load(f)
    # This repo is public — real secrets/private endpoints (an ntfy topic, a
    # webhook URL) go in a gitignored config.local.toml instead of committed
    # config.toml, merged in here when present.
    if local_path.exists():
        with open(local_path, "rb") as f:
            _deep_merge(config, tomllib.load(f))
    return config


CONFIG = load_config()

DATA_DIR = Path(CONFIG["data"]["dir"])
DATA_DIR.mkdir(exist_ok=True)

SOURCES = {
    name: {
        "url": src["url"],
        "path": DATA_DIR / src["filename"],
        "required_columns": src["required_columns"],
        "sep": src.get("sep", ","),
    }
    for name, src in CONFIG["sources"].items()
}

HEAT_WEIGHTS = CONFIG["heat_weights"]
REDFIN_HEAT_WEIGHTS = CONFIG["redfin_heat_weights"]
N_CLUSTERS = CONFIG["clusters"]["n_clusters"]
CLUSTER_LABELS = {int(k): v for k, v in CONFIG["clusters"]["labels"].items()}
CLUSTER_FEATURES = CONFIG["clusters"]["features"]

FORECAST_CFG = CONFIG["forecast"]
FETCH_CFG = CONFIG["fetch"]
ALERTS_CFG = CONFIG["alerts"]
FRED_CFG = CONFIG["fred"]
HISTORY_CFG = CONFIG["history"]
BACKTEST_CFG = CONFIG["backtest"]
FETCH_META_PATH = DATA_DIR / "fetch_meta.json"
FETCH_STATE_PATH = DATA_DIR / FETCH_CFG["state_file"]


# ─── Helpers ────────────────────────────────────────────────────────────────

class SchemaError(RuntimeError):
    """Raised when a downloaded CSV is missing columns the pipeline depends on."""


def _validate_columns(path: Path, required: list[str], sep: str = ",") -> None:
    header = pd.read_csv(path, sep=sep, nrows=0).columns.tolist()
    missing = [c for c in required if c not in header]
    if missing:
        raise SchemaError(
            f"{path.name} is missing expected column(s): {missing}. "
            f"The source may have changed their file schema — update "
            f"config.toml's required_columns / pipeline logic accordingly."
        )


def _validate_columns_for(path: Path, required: list[str], sep: str) -> None:
    _validate_columns(path, required, sep=sep)


def _record_fetch_meta(name: str) -> None:
    meta = {}
    if FETCH_META_PATH.exists():
        meta = json.loads(FETCH_META_PATH.read_text())
    meta[name] = datetime.now(timezone.utc).isoformat()
    FETCH_META_PATH.write_text(json.dumps(meta, indent=2))


def _parse_period(df: pd.DataFrame, col: str = "month_date_yyyymm") -> pd.Series:
    s = df[col].astype(str).str.replace(r"\.0$", "", regex=True)
    return pd.to_datetime(s + "01", format="%Y%m%d", errors="coerce")


def _load_zip_history() -> pd.DataFrame:
    path = SOURCES["zip_history"]["path"]
    if not path.exists():
        log.error(f"{path} not found — run 'fetch' first.")
        sys.exit(1)
    log.info(f"Loading {path} …")
    df = pd.read_csv(path, dtype={"postal_code": str}, low_memory=False)
    df["_zip"] = df["postal_code"].str.zfill(5)
    df["_period"] = _parse_period(df)
    log.info(
        f"  {len(df):,} rows · {df['_zip'].nunique():,} ZIPs · "
        f"{df['_period'].min():%Y-%m} – {df['_period'].max():%Y-%m}"
    )
    return df


def _load_county_history() -> pd.DataFrame | None:
    path = SOURCES["county_history"]["path"]
    if not path.exists():
        log.warning(f"{path} not found — skipping county rollup (run 'fetch' first).")
        return None
    log.info(f"Loading {path} …")
    df = pd.read_csv(path, dtype={"county_fips": str}, low_memory=False)
    df["_period"] = _parse_period(df)
    log.info(
        f"  {len(df):,} rows · {df['county_name'].nunique():,} counties · "
        f"{df['_period'].min():%Y-%m} – {df['_period'].max():%Y-%m}"
    )
    return df


def _load_redfin_zip_history() -> pd.DataFrame | None:
    """Stream + filter the Redfin ZIP tracker (multi-GB uncompressed) instead of
    loading it whole — only 'zip code' / 'All Residential' rows are kept."""
    path = SOURCES["redfin_zip"]["path"]
    if not path.exists():
        log.warning(f"{path} not found — skipping Redfin ZIP rollup (run 'fetch' first).")
        return None
    log.info(f"Loading {path} … (streaming/filtering — file is large)")
    usecols = [
        "PERIOD_END", "REGION_TYPE", "REGION", "PROPERTY_TYPE",
        "MEDIAN_SALE_PRICE", "MEDIAN_SALE_PRICE_YOY", "MEDIAN_DOM", "MEDIAN_DOM_YOY",
        "INVENTORY", "INVENTORY_YOY", "SOLD_ABOVE_LIST", "AVG_SALE_TO_LIST",
    ]
    chunks = []
    for chunk in pd.read_csv(path, sep=SOURCES["redfin_zip"]["sep"], usecols=usecols,
                              chunksize=500_000, low_memory=False):
        chunk = chunk[(chunk["REGION_TYPE"] == "zip code") & (chunk["PROPERTY_TYPE"] == "All Residential")]
        if len(chunk):
            chunks.append(chunk)

    if not chunks:
        log.warning("No matching Redfin ZIP-level rows found.")
        return None

    df = pd.concat(chunks, ignore_index=True)
    df["_zip"] = df["REGION"].str.extract(r"(\d{5})")
    df["_period"] = pd.to_datetime(df["PERIOD_END"], errors="coerce")
    df = df.dropna(subset=["_zip", "_period"])
    log.info(
        f"  {len(df):,} rows · {df['_zip'].nunique():,} ZIPs · "
        f"{df['_period'].min():%Y-%m} – {df['_period'].max():%Y-%m}"
    )
    return df


def compute_heat_scores(agg: pd.DataFrame, heat_weights: dict[str, float]) -> pd.Series:
    """Weighted, min-max-normalized 0-100 heat score. NaN where required features are missing.

    YoY/ratio columns can carry a handful of extreme outliers (e.g. a near-zero
    base price making a legitimate move look like a 49,000,000% YoY jump) that
    would otherwise dominate MinMaxScaler's range and flatten every other ZIP's
    score into a narrow band. Winsorizing to the 1st/99th percentile bounds
    their influence while preserving relative ranking.
    """
    heat_cols = list(heat_weights.keys())
    scaler = MinMaxScaler()
    valid_mask = agg[heat_cols].notna().all(axis=1)
    valid = agg[valid_mask]

    scores = pd.Series(np.nan, index=agg.index, dtype=float)
    if len(valid) == 0:
        return scores

    clipped = valid[heat_cols].clip(
        lower=valid[heat_cols].quantile(0.01), upper=valid[heat_cols].quantile(0.99), axis=1
    )
    normed = scaler.fit_transform(clipped)
    weights = np.array([heat_weights[c] for c in heat_cols])
    raw = normed @ weights
    denom = raw.max() - raw.min()
    scaled = 100.0 * (raw - raw.min()) / (denom + 1e-9)
    scores.loc[valid.index] = scaled
    return scores


def rank_clusters_by_heat(raw_labels: np.ndarray, index: pd.Index, heat_score: pd.Series) -> pd.Series:
    """Re-map arbitrary KMeans label ids so cluster 0 = hottest mean heat score."""
    temp = pd.Series(raw_labels, index=index).rename("_raw").to_frame()
    temp = temp.join(heat_score.rename("heat_score"))
    rank_map = {
        old: new
        for new, old in enumerate(
            temp.groupby("_raw")["heat_score"].mean().sort_values(ascending=False).index
        )
    }
    return pd.Series(raw_labels, index=index).map(rank_map)


# ─── Fetch ──────────────────────────────────────────────────────────────────

def cmd_fetch(force: bool = False) -> list[fetching.FetchResult]:
    """Probe every bulk source and download only the ones that actually changed.

    Previously this was all-or-nothing: plain `fetch` skipped anything already on
    disk (so a new month was never picked up) and `--force` re-pulled ~2.4 GB
    every time. Both origins expose ETag/Last-Modified, so a ~1 KB HEAD per
    source now decides.
    """
    results = []
    for name, src in SOURCES.items():
        res = fetching.sync_source(
            name, src, FETCH_STATE_PATH, FETCH_CFG,
            force=force, validate=_validate_columns_for,
        )
        results.append(res)
        if res.downloaded:
            _record_fetch_meta(name)

    downloaded = [r for r in results if r.downloaded]
    total_mb = sum(r.bytes_downloaded for r in downloaded) / 1e6
    log.info(
        f"Fetch complete: {len(downloaded)}/{len(results)} source(s) updated"
        + (f" ({total_mb:,.0f} MB downloaded)" if downloaded else " — everything already current")
    )
    return results


def cmd_fred() -> None:
    """Ingest FRED macro series (the only daily-cadence inputs in the project)."""
    log.info(f"Fetching {len(FRED_CFG['series'])} FRED series …")
    fred_mod.fetch_all(FRED_CFG, DATA_DIR)


# ─── Mine ───────────────────────────────────────────────────────────────────

def cmd_mine(lookback_months: int = 3) -> None:
    df = _load_zip_history()

    cutoff = df["_period"].max() - pd.DateOffset(months=lookback_months - 1)
    recent = df[df["_period"] >= cutoff].copy()
    log.info(f"Mining on {recent['_period'].nunique()} recent month(s) ({recent['_period'].min():%Y-%m} – {recent['_period'].max():%Y-%m})")

    agg = (
        recent.groupby("_zip")
        .agg(
            zip_name=("zip_name", "last"),
            median_listing_price=("median_listing_price", "mean"),
            median_listing_price_yy=("median_listing_price_yy", "mean"),
            active_listing_count=("active_listing_count", "mean"),
            active_listing_count_yy=("active_listing_count_yy", "mean"),
            median_days_on_market=("median_days_on_market", "mean"),
            median_days_on_market_yy=("median_days_on_market_yy", "mean"),
            pending_ratio=("pending_ratio", "mean"),
            price_reduced_share=("price_reduced_share", "mean"),
            price_increased_share=("price_increased_share", "mean"),
            data_months=("_period", "nunique"),
        )
        .reset_index()
    )
    agg = agg[agg["data_months"] >= max(1, lookback_months - 1)].copy()
    log.info(f"Aggregated to {len(agg):,} ZIPs")

    # ── Heat score ────────────────────────────────────────────────────────
    agg["heat_score"] = compute_heat_scores(agg, HEAT_WEIGHTS)

    # ── Market clusters ───────────────────────────────────────────────────
    cluster_mask = agg[CLUSTER_FEATURES].notna().all(axis=1)
    cluster_valid = agg[cluster_mask].copy()

    if len(cluster_valid) >= N_CLUSTERS:
        scaler = MinMaxScaler()
        km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
        raw_labels = km.fit_predict(scaler.fit_transform(cluster_valid[CLUSTER_FEATURES]))
        ranked = rank_clusters_by_heat(raw_labels, cluster_valid.index, agg["heat_score"])
        agg.loc[cluster_valid.index, "cluster_id"] = ranked
        agg["cluster_id"] = agg["cluster_id"].astype("Int64")
        agg["cluster_label"] = agg["cluster_id"].map(CLUSTER_LABELS)
        log.info("Clusters assigned: " + " | ".join(
            f"{v}={agg[agg['cluster_id'] == k].shape[0]}" for k, v in CLUSTER_LABELS.items()
        ))

    out = DATA_DIR / "market_scores.parquet"
    agg.to_parquet(out, index=False)
    log.info(f"Saved → {out}")

    meta = {
        "data_through": df["_period"].max().strftime("%Y-%m-%d"),
        "lookback_months": lookback_months,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (DATA_DIR / "mine_meta.json").write_text(json.dumps(meta, indent=2))

    history_mod.append_snapshot(
        agg, df["_period"].max(), HISTORY_CFG["zip_columns"],
        DATA_DIR / HISTORY_CFG["filename"], HISTORY_CFG["max_snapshots"],
    )

    cmd_mine_counties(lookback_months=lookback_months)
    cmd_mine_redfin(lookback_months=lookback_months)


def cmd_history() -> None:
    """Read-only: report the biggest heat-score movers since the prior snapshot."""
    path = DATA_DIR / HISTORY_CFG["filename"]
    if not path.exists():
        log.warning("No score_history.parquet yet — run 'mine' to create the first snapshot.")
        return

    hist = pd.read_parquet(path)
    movers, prev_date, latest_date = history_mod.compute_momentum(
        hist, HISTORY_CFG["zip_columns"], HEAT_WEIGHTS, compute_heat_scores,
    )
    if movers.empty:
        only = hist["data_through"].max()
        log.info(f"Only one snapshot on file ({only:%Y-%m-%d}) — need a second mine run on new data for momentum.")
        return

    log.info(f"Momentum {prev_date:%Y-%m-%d} → {latest_date:%Y-%m-%d} ({len(movers)} ZIPs, shared normalization basis):")
    gainers = movers.nlargest(10, "delta")
    losers = movers.nsmallest(10, "delta")
    label_col = "zip_name" if "zip_name" in movers.columns else "_zip"
    log.info("  Top gainers:")
    for _, r in gainers.iterrows():
        log.info(f"    +{r['delta']:5.1f}  {r['_zip']}  {r.get(label_col, '')}")
    log.info("  Top losers:")
    for _, r in losers.iterrows():
        log.info(f"    {r['delta']:6.1f}  {r['_zip']}  {r.get(label_col, '')}")


def cmd_mine_counties(lookback_months: int = 3) -> None:
    df = _load_county_history()
    if df is None:
        return

    cutoff = df["_period"].max() - pd.DateOffset(months=lookback_months - 1)
    recent = df[df["_period"] >= cutoff].copy()

    agg = (
        recent.groupby("county_fips")
        .agg(
            county_name=("county_name", "last"),
            median_listing_price=("median_listing_price", "mean"),
            median_listing_price_yy=("median_listing_price_yy", "mean"),
            active_listing_count=("active_listing_count", "mean"),
            active_listing_count_yy=("active_listing_count_yy", "mean"),
            median_days_on_market=("median_days_on_market", "mean"),
            median_days_on_market_yy=("median_days_on_market_yy", "mean"),
            pending_ratio=("pending_ratio", "mean"),
            price_reduced_share=("price_reduced_share", "mean"),
            data_months=("_period", "nunique"),
        )
        .reset_index()
    )
    agg = agg[agg["data_months"] >= max(1, lookback_months - 1)].copy()
    agg["heat_score"] = compute_heat_scores(agg, HEAT_WEIGHTS)

    out = DATA_DIR / "county_scores.parquet"
    agg.to_parquet(out, index=False)
    log.info(f"Saved → {out}  ({len(agg):,} counties)")


def cmd_mine_redfin(lookback_months: int = 3) -> None:
    """Independent ZIP-level heat score from Redfin's Data Center export —
    a second opinion alongside market_scores.parquet (Realtor.com-derived),
    not merged into it since the two providers' metrics aren't directly comparable."""
    df = _load_redfin_zip_history()
    if df is None:
        return

    cutoff = df["_period"].max() - pd.DateOffset(months=lookback_months - 1)
    recent = df[df["_period"] >= cutoff].copy()

    agg = (
        recent.groupby("_zip")
        .agg(
            median_sale_price=("MEDIAN_SALE_PRICE", "mean"),
            median_sale_price_yoy=("MEDIAN_SALE_PRICE_YOY", "mean"),
            median_dom=("MEDIAN_DOM", "mean"),
            median_dom_yoy=("MEDIAN_DOM_YOY", "mean"),
            inventory=("INVENTORY", "mean"),
            inventory_yoy=("INVENTORY_YOY", "mean"),
            sold_above_list=("SOLD_ABOVE_LIST", "mean"),
            avg_sale_to_list=("AVG_SALE_TO_LIST", "mean"),
            data_periods=("_period", "nunique"),
        )
        .reset_index()
    )
    agg["heat_score"] = compute_heat_scores(agg, REDFIN_HEAT_WEIGHTS)

    out = DATA_DIR / "redfin_zip_scores.parquet"
    agg.to_parquet(out, index=False)
    log.info(f"Saved → {out}  ({len(agg):,} ZIPs)")


# ─── Forecast ───────────────────────────────────────────────────────────────

def run_prophet(series: pd.DataFrame, periods: int = 12) -> pd.DataFrame:
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=FORECAST_CFG["changepoint_prior_scale"],
        interval_width=FORECAST_CFG["interval_width"],
    )
    m.fit(series)
    future = m.make_future_dataframe(periods=periods, freq="MS")
    return m.predict(future)


def cmd_forecast(top_n: int = 30, periods: int = 12) -> None:
    scores_path = DATA_DIR / "market_scores.parquet"
    if not scores_path.exists():
        log.error("market_scores.parquet not found — run 'mine' first.")
        sys.exit(1)

    scores = pd.read_parquet(scores_path)
    top_zips = (
        scores.dropna(subset=["heat_score"])
        .nlargest(top_n, "heat_score")["_zip"]
        .tolist()
    )
    log.info(f"Forecasting top {len(top_zips)} ZIPs × {periods} months …")

    df = _load_zip_history()
    results = []
    skipped = 0
    min_history = FORECAST_CFG["min_history_months"]

    for i, zip_code in enumerate(top_zips, 1):
        series = (
            df[df["_zip"] == zip_code][["_period", "median_listing_price"]]
            .dropna()
            .rename(columns={"_period": "ds", "median_listing_price": "y"})
            .sort_values("ds")
        )
        series["ds"] = pd.to_datetime(series["ds"]).dt.tz_localize(None)

        if len(series) < min_history:
            log.warning(f"  [{i}/{len(top_zips)}] {zip_code}: {len(series)} months — skip")
            skipped += 1
            continue

        try:
            fcst = run_prophet(series, periods=periods)
            fcst["_zip"] = zip_code
            fcst["is_forecast"] = fcst["ds"] > series["ds"].max()
            results.append(
                fcst[["_zip", "ds", "yhat", "yhat_lower", "yhat_upper", "is_forecast"]].copy()
            )
            log.info(f"  [{i}/{len(top_zips)}] {zip_code} ✓")
        except Exception as exc:
            log.warning(f"  [{i}/{len(top_zips)}] {zip_code}: failed — {exc}")
            skipped += 1

    if results:
        out_df = pd.concat(results, ignore_index=True)
        out = DATA_DIR / "forecasts.parquet"
        out_df.to_parquet(out, index=False)
        log.info(
            f"Saved → {out}  "
            f"({len(results)} ZIPs forecasted, {skipped} skipped, {len(out_df):,} rows)"
        )
    else:
        log.warning("No forecasts generated.")


# ─── Backtest ───────────────────────────────────────────────────────────────

def cmd_backtest(top_n: int | None = None) -> None:
    """Hold out the tail of each hot ZIP's history and score Prophet against it,
    so the forecast has an accuracy number attached instead of just a line."""
    scores_path = DATA_DIR / "market_scores.parquet"
    if not scores_path.exists():
        log.error("market_scores.parquet not found — run 'mine' first.")
        sys.exit(1)

    top_n = top_n or BACKTEST_CFG["default_top_n"]
    scores = pd.read_parquet(scores_path)
    top_zips = scores.dropna(subset=["heat_score"]).nlargest(top_n, "heat_score")["_zip"].tolist()
    log.info(f"Backtesting {len(top_zips)} ZIPs (holdout={BACKTEST_CFG['horizon_months']}mo) …")

    df = _load_zip_history()
    results, skipped = backtest_mod.run(df, top_zips, run_prophet, BACKTEST_CFG)

    if results.empty:
        log.warning(f"No backtest results — all {len(skipped)} ZIP(s) had insufficient history.")
        return

    out = DATA_DIR / BACKTEST_CFG["filename"]
    results.to_parquet(out, index=False)
    summary = backtest_mod.summarize(results)
    worst = summary.iloc[-1]
    log.info(
        f"Saved → {out}  ({results['_zip'].nunique()} ZIPs tested, {len(skipped)} skipped)  "
        f"median MAPE {summary['mape_pct'].median():.1f}%  worst {worst['_zip']} ({worst['mape_pct']:.1f}%)"
    )


# ─── Report ─────────────────────────────────────────────────────────────────

def cmd_report() -> None:
    from dashboard import build_dashboard

    out = DATA_DIR / "dashboard.html"
    build_dashboard(DATA_DIR, out)
    log.info(f"Saved → {out}")


# ─── Orchestration ──────────────────────────────────────────────────────────

def cmd_all(force: bool = False, lookback: int = 3, top: int | None = None,
            periods: int | None = None) -> None:
    """fetch → (fred) → mine/forecast/backtest (only if bulk data changed) → report.

    Fetch is now change-detected (see fetching.py), so it's safe to run this
    daily: an unchanged month costs a handful of HEAD requests, not a 2.4 GB
    re-download. mine/forecast/backtest are comparatively expensive and their
    output can't change unless the source data did, so they're skipped on days
    where nothing new landed — report still re-renders daily to pick up fresh
    FRED rates. Any failure is pushed through the configured alert channel
    (see [alerts] in config.toml) before being re-raised, so a broken run is
    never silent the way the poetry-PATH cron failures were for five weeks.
    """
    top = top if top is not None else FORECAST_CFG["default_top_n"]
    periods = periods if periods is not None else FORECAST_CFG["default_periods"]
    summary: dict = {}

    try:
        fetch_results = cmd_fetch(force=force)
        bulk_names = {"zip_history", "county_history", "redfin_zip"}
        bulk_changed = any(r.downloaded for r in fetch_results if r.name in bulk_names)
        summary["fetched"] = [r.name for r in fetch_results if r.downloaded]

        try:
            cmd_fred()
            summary["fred"] = "ok"
        except Exception as exc:
            # A flaky/discontinued FRED series must not take down the housing
            # pipeline that everything else depends on.
            log.warning(f"FRED fetch failed (non-fatal): {exc}")
            summary["fred"] = f"failed: {exc}"

        if bulk_changed or force:
            cmd_mine(lookback_months=lookback)
            cmd_forecast(top_n=top, periods=periods)
            try:
                cmd_backtest()
            except Exception as exc:
                log.warning(f"Backtest failed (non-fatal): {exc}")
                summary["backtest"] = f"failed: {exc}"
            else:
                summary["backtest"] = "ok"
            summary["mined"] = True
        else:
            log.info("No housing source changed — skipping mine/forecast/backtest, refreshing report only.")
            summary["mined"] = False

        cmd_report()
        summary["reported"] = True

    except Exception as exc:
        log.exception("Pipeline run failed")
        notify.notify_failure(ALERTS_CFG, stage="all", exc=exc)
        raise
    else:
        notify.notify_success(ALERTS_CFG, summary)


# ─── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real Estate pipeline: fetch → mine → forecast → report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser("fetch", help="Change-detected download of Realtor.com + Redfin sources")
    p_fetch.add_argument("--force", action="store_true", help="Re-download even if unchanged")

    sub.add_parser("fred", help="Fetch FRED macro series (rates, starts, permits, ...)")

    p_mine = sub.add_parser("mine", help="Compute market heat scores, clusters & history snapshot")
    p_mine.add_argument("--lookback", type=int, default=3, metavar="N",
                        help="Months to average over (default 3)")

    sub.add_parser("history", help="Report biggest heat-score movers since the last snapshot")

    p_fc = sub.add_parser("forecast", help="Run Prophet for top hot markets")
    p_fc.add_argument("--top", type=int, default=FORECAST_CFG["default_top_n"], metavar="N",
                      help=f"Number of ZIPs to forecast (default {FORECAST_CFG['default_top_n']})")
    p_fc.add_argument("--periods", type=int, default=FORECAST_CFG["default_periods"],
                      help=f"Forecast horizon in months (default {FORECAST_CFG['default_periods']})")

    p_bt = sub.add_parser("backtest", help="Hold-out accuracy (MAPE) for the forecast model")
    p_bt.add_argument("--top", type=int, default=BACKTEST_CFG["default_top_n"], metavar="N",
                      help=f"Number of ZIPs to backtest (default {BACKTEST_CFG['default_top_n']})")

    sub.add_parser("report", help="Render data/dashboard.html")

    p_all = sub.add_parser("all", help="fetch + fred + (mine + forecast + backtest if changed) + report")
    p_all.add_argument("--force", action="store_true")
    p_all.add_argument("--lookback", type=int, default=3, metavar="N")
    p_all.add_argument("--top", type=int, default=FORECAST_CFG["default_top_n"], metavar="N")
    p_all.add_argument("--periods", type=int, default=FORECAST_CFG["default_periods"])

    args = parser.parse_args()

    if args.cmd == "fetch":
        cmd_fetch(force=args.force)
    elif args.cmd == "fred":
        cmd_fred()
    elif args.cmd == "mine":
        cmd_mine(lookback_months=args.lookback)
    elif args.cmd == "history":
        cmd_history()
    elif args.cmd == "forecast":
        cmd_forecast(top_n=args.top, periods=args.periods)
    elif args.cmd == "backtest":
        cmd_backtest(top_n=args.top)
    elif args.cmd == "report":
        cmd_report()
    elif args.cmd == "all":
        cmd_all(force=args.force, lookback=args.lookback, top=args.top, periods=args.periods)


if __name__ == "__main__":
    main()
