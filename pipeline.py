#!/usr/bin/env python3
"""
Real Estate data pipeline: fetch → mine → forecast

Commands:
  python pipeline.py fetch                   download latest CSVs from Realtor.com
  python pipeline.py mine [--lookback N]     compute market heat scores & clusters
  python pipeline.py forecast [--top N]      run Prophet for top N hot ZIPs
  python pipeline.py all [--top N]           run all three steps

Output files (in data/):
  market_scores.parquet   — heat score + cluster per ZIP
  forecasts.parquet       — 12-month Prophet forecasts for top ZIPs
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from prophet import Prophet
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Realtor.com public research data (updated monthly)
SOURCES = {
    "zip_history": {
        "url": "https://econdata.s3-us-west-2.amazonaws.com/Reports/Core/RDC_Inventory_Core_Metrics_Zip_History.csv",
        "path": DATA_DIR / "RDC_Inventory_Core_Metrics_Zip_History.csv",
    },
    "zip_current": {
        "url": "https://econdata.s3-us-west-2.amazonaws.com/Reports/Core/RDC_Inventory_Core_Metrics_Zip.csv",
        "path": DATA_DIR / "RDC_Inventory_Core_Metrics_Zip.csv",
    },
    "county_history": {
        "url": "https://econdata.s3-us-west-2.amazonaws.com/Reports/Core/RDC_Inventory_Core_Metrics_County_History.csv",
        "path": DATA_DIR / "RDC_Inventory_Core_Metrics_County_History.csv",
    },
}

# ─── Heat score feature weights ────────────────────────────────────────────
# positive weight = higher value → hotter market
# negative weight = higher value → cooler market
HEAT_WEIGHTS = {
    "median_listing_price_yy": 1.5,    # price growth → hot
    "median_days_on_market_yy": -1.0,  # faster absorption → hot
    "pending_ratio": 1.2,              # demand signal → hot
    "price_reduced_share": -1.0,       # fewer reductions → hot
    "active_listing_count_yy": -0.5,   # tighter supply → hot
}

N_CLUSTERS = 5
CLUSTER_LABELS = {0: "Hot", 1: "Rising", 2: "Balanced", 3: "Cooling", 4: "Soft"}


# ─── Helpers ────────────────────────────────────────────────────────────────

def _download(url: str, path: Path) -> None:
    log.info(f"GET {url}")
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    bytes_written = 0
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bytes_written += len(chunk)
    log.info(f"  saved {path}  ({bytes_written / 1e6:.1f} MB)")


def _load_zip_history() -> pd.DataFrame:
    path = SOURCES["zip_history"]["path"]
    if not path.exists():
        log.error(f"{path} not found — run 'fetch' first.")
        sys.exit(1)
    log.info(f"Loading {path} …")
    df = pd.read_csv(path, dtype={"postal_code": str}, low_memory=False)
    df["_zip"] = df["postal_code"].str.zfill(5)
    s = df["month_date_yyyymm"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["_period"] = pd.to_datetime(s + "01", format="%Y%m%d", errors="coerce")
    log.info(
        f"  {len(df):,} rows · {df['_zip'].nunique():,} ZIPs · "
        f"{df['_period'].min():%Y-%m} – {df['_period'].max():%Y-%m}"
    )
    return df


# ─── Fetch ──────────────────────────────────────────────────────────────────

def cmd_fetch(force: bool = False) -> None:
    for name, src in SOURCES.items():
        path: Path = src["path"]
        if path.exists() and not force:
            log.info(f"Skip {name} (exists — use --force to re-download)")
            continue
        _download(src["url"], path)


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
    heat_cols = list(HEAT_WEIGHTS.keys())
    scaler = MinMaxScaler()
    valid_mask = agg[heat_cols].notna().all(axis=1)
    valid = agg[valid_mask].copy()

    if len(valid) == 0:
        log.error("No rows with all heat-score features — try increasing --lookback.")
        sys.exit(1)

    normed = scaler.fit_transform(valid[heat_cols])
    weights = np.array([HEAT_WEIGHTS[c] for c in heat_cols])
    raw = normed @ weights
    scaled = 100.0 * (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    agg.loc[valid.index, "heat_score"] = scaled

    # ── Market clusters ───────────────────────────────────────────────────
    cluster_cols = ["median_listing_price_yy", "pending_ratio", "price_reduced_share"]
    cluster_mask = agg[cluster_cols].notna().all(axis=1)
    cluster_valid = agg[cluster_mask].copy()

    if len(cluster_valid) >= N_CLUSTERS:
        km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
        raw_labels = km.fit_predict(scaler.fit_transform(cluster_valid[cluster_cols]))
        # Re-rank cluster IDs so 0 = hottest (by mean heat score)
        temp = pd.Series(raw_labels, index=cluster_valid.index).rename("_raw").to_frame()
        temp = temp.join(agg["heat_score"])
        rank_map = {
            old: new
            for new, old in enumerate(
                temp.groupby("_raw")["heat_score"].mean().sort_values(ascending=False).index
            )
        }
        ranked = pd.Series(raw_labels, index=cluster_valid.index).map(rank_map)
        agg.loc[cluster_valid.index, "cluster_id"] = ranked
        agg["cluster_id"] = agg["cluster_id"].astype("Int64")
        agg["cluster_label"] = agg["cluster_id"].map(CLUSTER_LABELS)
        log.info("Clusters assigned: " + " | ".join(
            f"{v}={agg[agg['cluster_id'] == k].shape[0]}" for k, v in CLUSTER_LABELS.items()
        ))

    out = DATA_DIR / "market_scores.parquet"
    agg.to_parquet(out, index=False)
    log.info(f"Saved → {out}")


# ─── Forecast ───────────────────────────────────────────────────────────────

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

    for i, zip_code in enumerate(top_zips, 1):
        series = (
            df[df["_zip"] == zip_code][["_period", "median_listing_price"]]
            .dropna()
            .rename(columns={"_period": "ds", "median_listing_price": "y"})
            .sort_values("ds")
        )
        series["ds"] = pd.to_datetime(series["ds"]).dt.tz_localize(None)

        if len(series) < 12:
            log.warning(f"  [{i}/{len(top_zips)}] {zip_code}: {len(series)} months — skip")
            skipped += 1
            continue

        try:
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.15,
                interval_width=0.90,
            )
            m.fit(series)
            future = m.make_future_dataframe(periods=periods, freq="MS")
            fcst = m.predict(future)
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


# ─── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real Estate pipeline: fetch → mine → forecast",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser("fetch", help="Download latest CSVs from Realtor.com")
    p_fetch.add_argument("--force", action="store_true", help="Re-download even if files exist")

    p_mine = sub.add_parser("mine", help="Compute market heat scores and clusters")
    p_mine.add_argument("--lookback", type=int, default=3, metavar="N",
                        help="Months to average over (default 3)")

    p_fc = sub.add_parser("forecast", help="Run Prophet for top hot markets")
    p_fc.add_argument("--top", type=int, default=30, metavar="N",
                      help="Number of ZIPs to forecast (default 30)")
    p_fc.add_argument("--periods", type=int, default=12,
                      help="Forecast horizon in months (default 12)")

    p_all = sub.add_parser("all", help="Run fetch + mine + forecast")
    p_all.add_argument("--force", action="store_true")
    p_all.add_argument("--lookback", type=int, default=3, metavar="N")
    p_all.add_argument("--top", type=int, default=30, metavar="N")
    p_all.add_argument("--periods", type=int, default=12)

    args = parser.parse_args()

    if args.cmd == "fetch":
        cmd_fetch(force=args.force)
    elif args.cmd == "mine":
        cmd_mine(lookback_months=args.lookback)
    elif args.cmd == "forecast":
        cmd_forecast(top_n=args.top, periods=args.periods)
    elif args.cmd == "all":
        cmd_fetch(force=args.force)
        cmd_mine(lookback_months=args.lookback)
        cmd_forecast(top_n=args.top, periods=args.periods)


if __name__ == "__main__":
    main()
