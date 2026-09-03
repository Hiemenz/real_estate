# Real Estate Market Analytics

Pipeline and dashboard for mining U.S. housing market trends from Realtor.com's
and Redfin's public inventory data — identifying "hot" ZIP codes and
forecasting where prices are headed.

## What it does

1. **Fetch** — change-detected download of the latest ZIP/county inventory
   data from Realtor.com's public research bucket and Redfin's Data Center
   export (both republished monthly). Each source is HEAD-probed for its
   ETag/Last-Modified before downloading, so a daily run costs a handful of
   HTTP HEAD requests on days nothing changed instead of re-pulling ~2.4 GB.
2. **FRED** — pulls daily/weekly/monthly macro series (mortgage & Treasury
   rates, Fed funds, CPI, unemployment, housing starts/permits, Case-Shiller)
   from FRED's keyless CSV endpoint — the only genuinely daily-cadence inputs
   in the project, since the housing bulk sources are monthly.
3. **FHFA** — pulls the FHFA House Price Index (state + metro), a repeat-sales
   index built from Fannie/Freddie conforming-mortgage data — a third price
   signal, methodologically independent of Realtor's listing prices and
   Redfin's closed-sale prices.
4. **Census** *(optional, manual)* — pulls ACS demographics (income, rent
   burden, homeownership rate) per ZIP via `pipeline.py census`. Requires a
   free API key and only updates annually, so it isn't part of `all`.
5. **Mine** — computes a weighted "heat score" per ZIP (price growth, days on
   market, pending ratio, price reductions, etc.), clusters ZIPs with KMeans,
   rolls prices up to the county level, derives an independent Redfin-based
   heat score as a cross-check on the Realtor.com numbers, and appends a
   dated snapshot to `score_history.parquet` for tracking momentum over time.
6. **Forecast** — runs Prophet time-series forecasts (12-month horizon) for
   the hottest ZIPs.
7. **Backtest** — holds out the last few months of each hot ZIP's history,
   forecasts them, and scores the result (MAPE) against what actually
   happened — so the forecast carries an accuracy number instead of just a
   line on a chart.
8. **Report** — renders a static, self-contained `data/dashboard.html` (no
   server, no CDN dependency) summarizing all of the above.
9. **Explore** — a Streamlit app for digging into individual ZIPs, browsing
   market-mining results, comparing providers, and viewing batch forecasts.

Weights, cluster count, source URLs, FRED series, alert channel, and forecast
parameters are all tunable in `config.toml` without touching the code.

## Setup

Requires Python 3.13+ and [Poetry](https://python-poetry.org/).

```bash
poetry install
```

## Usage

Run the pipeline (outputs land in `data/`):

```bash
poetry run python pipeline.py fetch [--force]        # change-detected download (Realtor.com + Redfin)
poetry run python pipeline.py fred                   # fetch FRED macro series
poetry run python pipeline.py fhfa                    # fetch FHFA House Price Index (state + metro)
poetry run python pipeline.py census                  # fetch Census ACS demographics (annual, needs an API key — run manually)
poetry run python pipeline.py mine [--lookback N]     # heat scores, clusters, county/Redfin rollups, history snapshot
poetry run python pipeline.py history                 # report biggest heat-score movers since last snapshot
poetry run python pipeline.py forecast [--top N]      # Prophet forecasts for top N ZIPs
poetry run python pipeline.py backtest [--top N]      # held-out forecast accuracy (MAPE)
poetry run python pipeline.py report                  # render data/dashboard.html
poetry run python pipeline.py all [--top N]           # fetch + fred + fhfa + (mine+forecast+backtest if changed) + report
```

`all` is safe to run daily: `fetch` only re-downloads a source whose
ETag/Last-Modified actually changed, and the mine/forecast/backtest steps are
skipped on days where nothing new landed (the report still re-renders daily
to pick up fresh FRED/FHFA data). A failed run is pushed to whichever channel
is configured in `config.toml`'s `[alerts]` (ntfy, a webhook, or an arbitrary
shell command) before the process exits non-zero.

`census` is deliberately not part of `all`: ACS 5-year estimates only update
once a year, and every request needs a free API key (instant signup, no
cost, at https://api.census.gov/data/key_signup.html) — set it as
`census.api_key` in `config.local.toml` (gitignored; never in the committed
`config.toml`, since this repo is public). Without a key, `pipeline.py
census` logs a warning naming the signup URL and skips cleanly.

Launch the interactive dashboard:

```bash
poetry run streamlit run streamlit_app.py
```

For a recurring refresh, `scripts/daily_update.sh` runs `pipeline.py all`
under a flock (already installed as a daily cron job on this machine —
`crontab -l` to check). It invokes the project's `.venv/bin/python` directly
rather than `poetry`, since cron's non-interactive `PATH` doesn't include
`~/.local/bin`.

## Output files (in `data/`)

```
market_scores.parquet      Realtor.com heat score + cluster per ZIP
county_scores.parquet      county-level rollup (price levels & YoY trend)
redfin_zip_scores.parquet  independent Redfin-derived heat score per ZIP
forecasts.parquet          12-month Prophet forecasts for top ZIPs
backtest.parquet           held-out actual vs. forecast, per ZIP/month (accuracy check)
score_history.parquet      append-only heat-score snapshots, for momentum tracking
fred_series.parquet        daily/weekly/monthly macro series (rates, starts, permits, ...)
fhfa_hpi.parquet           FHFA repeat-sales price index (state + metro) — third, independent signal
census_acs.parquet         Census ACS demographics per ZIP (run `pipeline.py census` manually — needs a key)
fetch_meta.json            per-source fetch timestamps (freshness indicator)
fetch_state.json           per-source ETag/Last-Modified, used for change detection
mine_meta.json              data-through date / lookback used for the last mine run
dashboard.html             static HTML report generated by dashboard.py
```

Raw source downloads (`*.csv`, `*.gz`) and logs are gitignored and regenerated
via `pipeline.py fetch`; the derived outputs above are committed as a snapshot.

## Score history & momentum

`heat_score` is min-max normalized against the population of a single `mine`
run, so a ZIP's score can shift purely because *other* ZIPs shifted that
month — not because that ZIP itself changed. `score_history.parquet` therefore
stores the raw, pre-normalization features every run, and `pipeline.py
history` (or the dashboard's "Biggest movers" section) recomputes heat_score
jointly over the two periods being compared, so the resulting delta reflects
real relative movement rather than the normalization frame shifting under it.
See `history.py`'s module docstring for the full rationale.

## Alerting

Four consecutive monthly cron runs failed silently in August 2026 (`poetry:
command not found` — cron's `PATH` doesn't include `~/.local/bin`) before
anyone noticed, leaving the dashboard five weeks stale. `notify.py` pushes a
message to the channel configured in `config.toml`'s `[alerts]` — `ntfy`
(push notification, no signup), a `webhook` (JSON POST), or an arbitrary
shell `command` — whenever `pipeline.py all` fails, and optionally on success
too (`notify_on_success`).

## Testing

```bash
poetry run pytest
```

Covers the pure logic: heat-score weighting/winsorizing and cluster ranking
(`pipeline.py`), fetch change-detection (`fetching.py`), history snapshotting
and cross-period momentum (`history.py`), backtest holdout/MAPE scoring
including the near-zero-price data-glitch guard (`backtest.py`), FRED CSV
parsing (`fred.py`), FHFA CSV parsing including its parens-wrapped YoY column
(`fhfa.py`), and Census ACS batching/sentinel-handling (`census.py`).

## Project layout

```
pipeline.py     fetch → fred → fhfa → mine → forecast → backtest → report CLI (also: history, census)
fetching.py     change-detected download (ETag/Last-Modified HEAD probing)
fred.py         FRED macro series ingestion
fhfa.py         FHFA House Price Index ingestion (state + metro)
census.py       Census ACS demographics ingestion (batched by ZCTA, needs an API key)
history.py      append-only heat-score snapshots + cross-period momentum
backtest.py     held-out forecast accuracy (MAPE)
notify.py       failure/success alerting (ntfy / webhook / shell command)
dashboard.py    static HTML report generator (inline SVG, no dependencies)
streamlit_app.py  interactive dashboard (ZIP deep dive, market mining, county overview,
                  provider comparison, batch forecasts)
config.toml     source URLs, heat-score weights, cluster settings, FRED/FHFA/Census
                config, alert channel, forecast/backtest params (public — no secrets)
config.local.toml  gitignored local overrides (real ntfy topic, Census API key, ...)
scripts/        operational scripts (daily_update.sh for cron)
tests/          pytest suite for the pure logic in the modules above
data/           source files (gitignored, fetched via pipeline.py) and generated outputs
```
