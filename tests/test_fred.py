"""fred.fetch_series's CSV parsing — including FRED's '.' missing-value marker
and its date column having been spelled both 'DATE' and 'observation_date'
over time — verified against canned responses, no network calls."""
from unittest.mock import patch

import pandas as pd
import pytest

from fred import fetch_series, latest_values


class _FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def _mock_get(text):
    return patch("fred.requests.get", return_value=_FakeResponse(text))


def test_parses_dot_as_missing_value():
    csv_text = "DATE,DGS10\n2026-01-01,4.5\n2026-01-02,.\n2026-01-03,4.6\n"
    with _mock_get(csv_text):
        df = fetch_series("DGS10", "http://fake")
    assert len(df) == 3
    assert df["value"].isna().sum() == 1
    assert df.loc[df["date"] == pd.Timestamp("2026-01-03"), "value"].iloc[0] == 4.6


def test_handles_observation_date_column_name():
    csv_text = "observation_date,MORTGAGE30US\n2026-01-01,6.5\n2026-01-08,6.6\n"
    with _mock_get(csv_text):
        df = fetch_series("MORTGAGE30US", "http://fake")
    assert list(df["value"]) == [6.5, 6.6]


def test_sorted_by_date_ascending():
    csv_text = "DATE,X\n2026-02-01,2\n2026-01-01,1\n"
    with _mock_get(csv_text):
        df = fetch_series("X", "http://fake")
    assert list(df["date"]) == sorted(df["date"])


def test_latest_values_computes_change_and_1y_lookback():
    df = pd.DataFrame({
        "series_id": ["X"] * 4,
        "date": pd.to_datetime(["2025-01-01", "2025-06-01", "2026-01-01", "2026-02-01"]),
        "value": [10.0, 12.0, 15.0, 16.0],
        "label": "Test Series",
        "freq": "monthly",
        "units": "percent",
    })
    out = latest_values(df)
    row = out[out["series_id"] == "X"].iloc[0]
    assert row["value"] == 16.0
    assert row["change"] == pytest.approx(1.0)   # 16 - 15
    assert row["change_1y"] == pytest.approx(6.0)  # 16 - 10 (nearest obs >= 1y prior)
