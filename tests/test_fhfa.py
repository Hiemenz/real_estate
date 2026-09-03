"""fhfa's CSV parsing verified against canned responses shaped like the real
files (state: no header, 4 cols; metro: no header, 6 cols with a
parens-wrapped YoY column) — no network calls."""
from unittest.mock import patch

import pandas as pd

from fhfa import _fetch_metro, _fetch_state, _parse_paren_pct, fetch_all

CFG = {"state_url": "http://fake/state.csv", "metro_url": "http://fake/metro.csv", "filename": "fhfa_hpi.parquet"}


def _mock_get(url_to_text):
    def fake_get(url, headers=None, timeout=None):
        class _Resp:
            text = url_to_text[url]

            def raise_for_status(self):
                pass

        return _Resp()
    return patch("fhfa.requests.get", side_effect=fake_get)


def test_parse_paren_pct_handles_signs_and_missing():
    s = pd.Series(["( 6.05)", "(-6.05)", "-", "( 0.00)"])
    out = _parse_paren_pct(s)
    assert out.iloc[0] == 6.05
    assert out.iloc[1] == -6.05
    assert pd.isna(out.iloc[2])
    assert out.iloc[3] == 0.0


def test_fetch_state_shape():
    text = "AK,1975,1,69.15\nAK,1975,2,69.06\n"
    with _mock_get({"http://fake/state.csv": text}):
        df = _fetch_state(CFG)
    assert set(["place", "year", "quarter", "hpi", "level", "place_id", "yoy_pct"]) <= set(df.columns)
    assert df["level"].eq("state").all()
    assert df.loc[0, "place_id"] == "AK"
    assert df["yoy_pct"].isna().all()


def test_fetch_metro_shape_and_yoy_parsing():
    text = '"Abilene, TX",10180,1975,1,-,-\n"Abilene, TX",10180,2026,2,250.12,( 4.20)\n'
    with _mock_get({"http://fake/metro.csv": text}):
        df = _fetch_metro(CFG)
    assert df["level"].eq("metro").all()
    assert df.loc[1, "place_id"] == 10180
    assert df.loc[1, "yoy_pct"] == 4.20
    assert pd.isna(df.loc[0, "hpi"])  # "-" placeholder


def test_fetch_all_combines_and_types_place_id_as_string(tmp_path):
    state_text = "AK,2026,1,100.0\n"
    metro_text = '"Abilene, TX",10180,2026,1,250.0,( 4.20)\n'
    with _mock_get({"http://fake/state.csv": state_text, "http://fake/metro.csv": metro_text}):
        out = fetch_all(CFG, tmp_path)
    assert (tmp_path / "fhfa_hpi.parquet").exists()
    assert set(out["level"]) == {"state", "metro"}
    assert out["place_id"].apply(type).eq(str).all()
    assert out.loc[out["level"] == "state", "period"].iloc[0] == pd.Timestamp("2026-01-01")
