# Real Estate Market Analytics

Pipeline and dashboard for mining U.S. housing market trends from Realtor.com's
public inventory data — identifying "hot" ZIP codes and forecasting where
prices are headed.

## What it does

1. **Fetch** — downloads the latest ZIP/county inventory CSVs from Realtor.com's
   public research data (updated monthly).
2. **Mine** — computes a weighted "heat score" per ZIP (price growth, days on
   market, pending ratio, price reductions, etc.) and clusters ZIPs with KMeans.
3. **Forecast** — runs Prophet time-series forecasts (12-month horizon) for the
   hottest ZIPs.
4. **Explore** — a Streamlit app for digging into individual ZIPs, browsing
   market-mining results, and viewing batch forecasts.

## Setup

Requires Python 3.13+ and [Poetry](https://python-poetry.org/).

```bash
poetry install
```

## Usage

Run the pipeline (outputs land in `data/`):

```bash
poetry run python pipeline.py fetch                # download source CSVs
poetry run python pipeline.py mine [--lookback N]   # compute heat scores & clusters
poetry run python pipeline.py forecast [--top N]    # Prophet forecasts for top N ZIPs
poetry run python pipeline.py all [--top N]         # run all three steps
```

Launch the dashboard:

```bash
poetry run streamlit run streamlit_app.py
```

## Project layout

```
pipeline.py         fetch → mine → forecast CLI
streamlit_app.py     interactive dashboard (ZIP deep dive, market mining, batch forecasts)
data/                source CSVs (gitignored, fetched via pipeline.py) and generated parquet outputs
```
