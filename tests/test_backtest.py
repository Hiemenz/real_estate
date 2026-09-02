"""backtest.run/summarize sanity-checked against a synthetic, perfectly
predictable series (no Prophet involved — run_prophet_fn is injected) so the
holdout slicing and MAPE math are verified without needing network or a
multi-second model fit per test.
"""
import pandas as pd
import pytest

from backtest import run, summarize

CFG = {"horizon_months": 3, "min_history_months": 6, "min_plausible_price": 10_000}


def _perfect_forecaster(train: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Stand-in for run_prophet: continues the training series' linear trend
    exactly, so the backtest should score a ~0% MAPE."""
    step = train["y"].iloc[-1] - train["y"].iloc[-2]
    last_ds, last_y = train["ds"].iloc[-1], train["y"].iloc[-1]
    future_ds = pd.date_range(last_ds, periods=periods + 1, freq="MS")[1:]
    future_y = [last_y + step * (i + 1) for i in range(periods)]
    all_ds = pd.concat([train["ds"], pd.Series(future_ds)], ignore_index=True)
    all_y = pd.concat([train["y"], pd.Series(future_y)], ignore_index=True)
    return pd.DataFrame({"ds": all_ds, "yhat": all_y, "yhat_lower": all_y, "yhat_upper": all_y})


def _flat_forecaster(train: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Deliberately wrong for a trending series: predicts no change at all."""
    last_ds = train["ds"].iloc[-1]
    future_ds = pd.date_range(last_ds, periods=periods + 1, freq="MS")[1:]
    all_ds = pd.concat([train["ds"], pd.Series(future_ds)], ignore_index=True)
    flat_val = train["y"].iloc[-1]
    all_y = [flat_val] * len(all_ds)
    return pd.DataFrame({"ds": all_ds, "yhat": all_y, "yhat_lower": all_y, "yhat_upper": all_y})


def _linear_history(zip_code: str, n_months: int = 24, start: float = 100_000, step: float = 1_000):
    ds = pd.date_range("2024-01-01", periods=n_months, freq="MS")
    return pd.DataFrame({
        "_zip": zip_code,
        "_period": ds,
        "median_listing_price": [start + step * i for i in range(n_months)],
    })


def test_perfect_forecaster_scores_near_zero_mape():
    df = _linear_history("00000")
    results, skipped = run(df, ["00000"], _perfect_forecaster, CFG)
    assert not skipped
    assert (results["abs_pct_error"] < 1e-6).all()
    summary = summarize(results)
    assert summary.iloc[0]["mape_pct"] == pytest.approx(0.0, abs=1e-4)


def test_insufficient_history_is_skipped():
    df = _linear_history("00000", n_months=5)  # below min_history + horizon
    results, skipped = run(df, ["00000"], _perfect_forecaster, CFG)
    assert results.empty
    assert skipped == ["00000"]


def test_flat_forecast_scores_worse_than_perfect_on_a_trending_series():
    df = _linear_history("00000")
    perfect, _ = run(df, ["00000"], _perfect_forecaster, CFG)
    flat, _ = run(df, ["00000"], _flat_forecaster, CFG)
    assert summarize(flat)["mape_pct"].iloc[0] > summarize(perfect)["mape_pct"].iloc[0]


def test_implausible_actual_is_excluded_not_scored():
    """A held-out month with a $1 placeholder actual must not register as a
    multi-million-percent error and swamp the rest of that ZIP's holdout."""
    df = _linear_history("00000")
    glitched = df.copy()
    glitch_idx = glitched.index[glitched["_period"] == glitched["_period"].max()]
    glitched.loc[glitch_idx, "median_listing_price"] = 1.0

    results, skipped = run(glitched, ["00000"], _perfect_forecaster, CFG)
    assert not skipped
    # The glitched month is dropped; the rest of the holdout still scores near 0.
    assert (results["actual"] >= CFG["min_plausible_price"]).all()
    assert summarize(results).iloc[0]["mape_pct"] < 1.0


def test_summarize_ranks_best_first():
    df = pd.concat([
        _linear_history("00000", step=1_000),
        _linear_history("00001", step=5_000),
    ], ignore_index=True)
    results, _ = run(df, ["00000", "00001"], _flat_forecaster, CFG)
    summary = summarize(results)
    assert list(summary["mape_pct"]) == sorted(summary["mape_pct"])
