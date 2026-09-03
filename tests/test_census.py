"""census.fetch_acs's batching and sentinel-handling, mocked (no key/network
available in this environment — the live endpoint format was confirmed
manually against api.census.gov before writing this module)."""
from unittest.mock import patch

import pandas as pd
import pytest

from census import MissingApiKeyError, fetch_acs

CFG = {
    "base_url": "http://fake",
    "year": 2023,
    "dataset": "acs/acs5",
    "api_key": "",
    "batch_size": 2,
    "timeout": 30,
    "variables": {
        "B19013_001E": "median_household_income",
        "B25003_001E": "occupied_housing_units",
        "B25003_002E": "owner_occupied_units",
    },
}


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_raises_clear_error_without_api_key():
    with pytest.raises(MissingApiKeyError):
        fetch_acs(CFG, ["10001"])


def test_batches_requests_by_batch_size():
    cfg = {**CFG, "api_key": "fake-key"}
    header = ["NAME", "B19013_001E", "B25003_001E", "B25003_002E", "zip code tabulation area"]
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(params["for"])
        zcta = params["for"].split(":")[1]
        rows = [[f"ZCTA {z}", "60000", "100", "60", z] for z in zcta.split(",")]
        return _FakeResponse([header, *rows])

    with patch("census.requests.get", side_effect=fake_get):
        df = fetch_acs(cfg, ["10001", "10002", "10003"])

    assert len(calls) == 2  # batch_size=2 over 3 zips
    assert len(df) == 3
    assert set(df["_zip"]) == {"10001", "10002", "10003"}


def test_sentinel_values_become_nan_and_homeownership_rate_computed():
    cfg = {**CFG, "api_key": "fake-key", "batch_size": 10}
    header = ["NAME", "B19013_001E", "B25003_001E", "B25003_002E", "zip code tabulation area"]

    def fake_get(url, params=None, timeout=None):
        rows = [
            ["ZCTA 10001", "-666666666", "100", "60", "10001"],  # sentinel -> NaN
            ["ZCTA 10002", "75000", "200", "150", "10002"],
        ]
        return _FakeResponse([header, *rows])

    with patch("census.requests.get", side_effect=fake_get):
        df = fetch_acs(cfg, ["10001", "10002"])

    row1 = df[df["_zip"] == "10001"].iloc[0]
    assert pd.isna(row1["median_household_income"])

    row2 = df[df["_zip"] == "10002"].iloc[0]
    assert row2["median_household_income"] == 75000
    assert row2["homeownership_rate"] == pytest.approx(0.75)
