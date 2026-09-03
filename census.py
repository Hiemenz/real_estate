"""Census ACS 5-year demographics, batched by ZCTA (ZIP Code Tabulation Area).

Economic context for why a ZIP scores hot, not just that it does — median
household income, rent burden, homeownership rate, population. Confirmed
against the live API (2026-09-03): every request now requires a key
(`X-DataWebAPI-KeyError: 1` / redirect to missing_key.html on an unkeyed
request), and the ZCTA geography has no wildcard support (confirmed via the
dataset's own examples.json — unlike state/county, "for=zip code tabulation
area:*" is not offered), so ZIPs must be named explicitly, batched to stay
under the endpoint's per-request limit.

ACS5 estimates only update once a year, unlike the monthly/daily sources
elsewhere in this pipeline — this module is meant to be run standalone
(`pipeline.py census`), not folded into the daily `all` gate.
"""
import logging
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

MISSING_KEY_MSG = (
    "No Census API key configured — get one free (instant signup, no cost) at "
    "https://api.census.gov/data/key_signup.html and set census.api_key in "
    "config.local.toml (never in the committed config.toml)."
)


class MissingApiKeyError(RuntimeError):
    pass


def fetch_acs(cfg: dict, zips: list[str]) -> pd.DataFrame:
    api_key = cfg.get("api_key", "").strip()
    if not api_key:
        raise MissingApiKeyError(MISSING_KEY_MSG)

    variables = cfg["variables"]
    var_codes = list(variables.keys())
    url = f"{cfg['base_url']}/{cfg['year']}/{cfg['dataset']}"
    batch_size = cfg.get("batch_size", 50)
    timeout = cfg.get("timeout", 30)

    frames = []
    zips = sorted(set(zips))
    for i in range(0, len(zips), batch_size):
        batch = zips[i:i + batch_size]
        params = {
            "get": ",".join(["NAME", *var_codes]),
            "for": f"zip code tabulation area:{','.join(batch)}",
            "key": api_key,
        }
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        header, *records = r.json()
        frames.append(pd.DataFrame(records, columns=header))
        log.info(f"  ACS batch {i // batch_size + 1}/{-(-len(zips) // batch_size)}: {len(batch)} ZCTAs")

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={**variables, "zip code tabulation area": "_zip"})
    for label in variables.values():
        # ACS uses large negative sentinels (e.g. -666666666) for "not computed"
        df[label] = pd.to_numeric(df[label], errors="coerce")
        df.loc[df[label] < -1_000_000, label] = pd.NA

    if {"owner_occupied_units", "occupied_housing_units"} <= set(df.columns):
        df["homeownership_rate"] = df["owner_occupied_units"] / df["occupied_housing_units"]

    return df.drop(columns=["NAME"], errors="ignore")


def fetch_and_save(cfg: dict, zips: list[str], data_dir: Path) -> pd.DataFrame:
    df = fetch_acs(cfg, zips)
    path = data_dir / cfg["filename"]
    df.to_parquet(path, index=False)
    log.info(f"Saved → {path}  ({len(df):,} ZCTAs)")
    return df
