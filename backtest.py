"""Forecast backtesting: hold out the tail of each ZIP's history, forecast it,
and score against what actually happened.

Without this, the Prophet forecasts on the dashboard carry no accuracy
measurement at all — a 12-month projection is shown with the same visual
confidence whether the model tracks that ZIP well or badly. Holding out the
last `horizon_months` and comparing yhat to the actual gives each ZIP a MAPE
(mean absolute percentage error) grounded in that ZIP's own recent history.
"""
import numpy as np
import pandas as pd


def run(
    df_history: pd.DataFrame,
    zips: list[str],
    run_prophet_fn,
    cfg: dict,
) -> tuple[pd.DataFrame, list[str]]:
    """Backtest each ZIP: train on all but the last `horizon_months`, predict
    that horizon, and return per-month actual vs. predicted rows.

    `run_prophet_fn` is injected (pipeline.py's run_prophet) rather than
    imported, so this module carries no dependency on pipeline.py.

    A held-out month with an implausible near-zero actual price (a data glitch,
    not a real listing — Realtor.com's feed occasionally carries a literal $1
    placeholder) is dropped rather than scored: MAPE divides by the actual, so
    one such row can produce a "55,800,000% error" that swamps every real
    number in the summary. This mirrors compute_heat_scores' winsorizing of
    near-zero-base-price YoY outliers for the same underlying reason.
    """
    horizon = cfg["horizon_months"]
    min_history = cfg["min_history_months"]
    min_plausible_price = cfg.get("min_plausible_price", 10_000)

    rows, skipped = [], []
    for zip_code in zips:
        series = (
            df_history[df_history["_zip"] == zip_code][["_period", "median_listing_price"]]
            .dropna()
            .rename(columns={"_period": "ds", "median_listing_price": "y"})
            .sort_values("ds")
        )
        series["ds"] = pd.to_datetime(series["ds"]).dt.tz_localize(None)

        if len(series) < min_history + horizon:
            skipped.append(zip_code)
            continue

        cutoff = series["ds"].iloc[-(horizon + 1)]
        train = series[series["ds"] <= cutoff]
        test = series[series["ds"] > cutoff]
        if test.empty or len(train) < min_history:
            skipped.append(zip_code)
            continue

        try:
            fcst = run_prophet_fn(train, periods=len(test))
        except Exception:
            skipped.append(zip_code)
            continue

        fcst = fcst[fcst["ds"].isin(test["ds"])][["ds", "yhat", "yhat_lower", "yhat_upper"]]
        merged = test.merge(fcst, on="ds", how="inner").rename(columns={"y": "actual"})
        merged = merged[merged["actual"] >= min_plausible_price]
        if merged.empty:
            skipped.append(zip_code)
            continue

        merged["_zip"] = zip_code
        merged["abs_pct_error"] = (
            (merged["actual"] - merged["yhat"]).abs() / merged["actual"].replace(0, np.nan)
        )
        rows.append(merged[["_zip", "ds", "actual", "yhat", "yhat_lower", "yhat_upper", "abs_pct_error"]])

    if not rows:
        return pd.DataFrame(), skipped
    return pd.concat(rows, ignore_index=True), skipped


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Per-ZIP MAPE, sorted best (most trustworthy forecast) to worst."""
    if results.empty:
        return pd.DataFrame()
    g = (
        results.groupby("_zip")
        .agg(mape=("abs_pct_error", "mean"), months_tested=("abs_pct_error", "count"))
        .reset_index()
    )
    g["mape_pct"] = g["mape"] * 100
    return g.sort_values("mape_pct").reset_index(drop=True)
