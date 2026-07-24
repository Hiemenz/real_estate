from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Real Estate Analytics", layout="wide", page_icon="🏠")

DATA_DIR = Path("data")


# ─── Data loaders ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_zip_history(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"postal_code": str}, low_memory=False)
    df["_zip"] = df["postal_code"].str.zfill(5)
    s = df["month_date_yyyymm"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["_period"] = pd.to_datetime(s + "01", format="%Y%m%d", errors="coerce")
    df = df.sort_values(["_zip", "_period"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_market_scores() -> Optional[pd.DataFrame]:
    p = DATA_DIR / "market_scores.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data(show_spinner=False)
def load_forecasts() -> Optional[pd.DataFrame]:
    p = DATA_DIR / "forecasts.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def _run_prophet(series: pd.DataFrame, periods: int = 12) -> pd.DataFrame:
    from prophet import Prophet
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.15,
        interval_width=0.90,
    )
    m.fit(series)
    future = m.make_future_dataframe(periods=periods, freq="MS")
    return m.predict(future)


def _pipeline_hint(cmd: str) -> None:
    st.info(
        f"Run the pipeline first:  `poetry run python pipeline.py {cmd}`  "
        f"then refresh this page.",
        icon="ℹ️",
    )


# ─── Load base data ──────────────────────────────────────────────────────────

zip_history_path = str(DATA_DIR / "RDC_Inventory_Core_Metrics_Zip_History.csv")
if not Path(zip_history_path).exists():
    st.error("data/RDC_Inventory_Core_Metrics_Zip_History.csv not found.")
    st.code("poetry run python pipeline.py fetch")
    st.stop()

with st.spinner("Loading ZIP history …"):
    df = load_zip_history(zip_history_path)

# ─── Tabs ────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📍 ZIP Deep Dive", "🔥 Market Mining", "📈 Batch Forecasts"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — ZIP Deep Dive
# ════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("ZIP Deep Dive")

    scores = load_market_scores()

    col_zip, col_metric, col_range = st.columns([1, 1, 2])

    all_zips = sorted(df["_zip"].dropna().unique().tolist())

    with col_zip:
        if scores is not None:
            top_zip = scores.dropna(subset=["heat_score"]).nlargest(1, "heat_score")["_zip"].iloc[0]
            default_idx = all_zips.index(top_zip) if top_zip in all_zips else 0
        else:
            default_idx = 0
        selected_zip = st.selectbox("ZIP code", options=all_zips, index=default_idx)

    numeric_cols = [
        c for c in df.columns
        if c not in {"_zip", "_period", "month_date_yyyymm"}
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].notna().any()
    ]
    with col_metric:
        default_m = numeric_cols.index("median_listing_price") if "median_listing_price" in numeric_cols else 0
        metric = st.selectbox("Metric", options=numeric_cols, index=default_m)

    min_p, max_p = df["_period"].min(), df["_period"].max()
    with col_range:
        start, end = st.slider(
            "Date range",
            min_value=min_p.to_pydatetime(),
            max_value=max_p.to_pydatetime(),
            value=(min_p.to_pydatetime(), max_p.to_pydatetime()),
            format="YYYY-MM",
        )

    zip_df = (
        df[df["_zip"] == selected_zip][["_period", metric]]
        .dropna()
        .sort_values("_period")
        .copy()
    )
    zip_df = zip_df[
        (zip_df["_period"] >= pd.Timestamp(start))
        & (zip_df["_period"] <= pd.Timestamp(end))
    ].copy()

    if zip_df.empty:
        st.info("No data for this ZIP / date range.")
    else:
        # ── YoY comparison ────────────────────────────────────────────────
        st.subheader("Year-over-Year comparison")

        yoy = zip_df.copy()
        yoy["prev_year"] = yoy[metric].shift(12)
        yoy["yoy_delta"] = yoy[metric] - yoy["prev_year"]
        yoy["yoy_pct"] = np.where(
            yoy["prev_year"].notna() & (yoy["prev_year"] != 0),
            yoy["yoy_delta"] / yoy["prev_year"],
            np.nan,
        )

        if yoy["prev_year"].notna().any():
            tbl = yoy.tail(24).copy()
            tbl["Period"] = tbl["_period"].dt.strftime("%Y-%m")
            tbl = tbl.rename(columns={metric: "Current", "prev_year": "Prev Year", "yoy_delta": "YoY Δ", "yoy_pct": "YoY %"})
            st.dataframe(
                tbl[["Period", "Current", "Prev Year", "YoY Δ", "YoY %"]]
                .style.format({
                    "Current": "{:,.0f}",
                    "Prev Year": "{:,.0f}",
                    "YoY Δ": "{:,.0f}",
                    "YoY %": lambda x: f"{x:.1%}" if pd.notna(x) else "",
                }),
                use_container_width=True,
            )

            plot = yoy.dropna(subset=["prev_year"]).copy()
            plot["period_s"] = plot["_period"].dt.strftime("%Y-%m")

            cur_line = alt.Chart(plot).mark_line(point=True).encode(
                x=alt.X("_period:T", title="Month"),
                y=alt.Y(f"{metric}:Q", title=metric.replace("_", " ").title()),
                color=alt.value("#1f77b4"),
                tooltip=["period_s:N", alt.Tooltip(f"{metric}:Q", format=",.0f", title="Current")],
            )
            prev_line = alt.Chart(plot).mark_line(point=True, strokeDash=[4, 2]).encode(
                x=alt.X("_period:T"),
                y=alt.Y("prev_year:Q"),
                color=alt.value("#aec7e8"),
                tooltip=["period_s:N", alt.Tooltip("prev_year:Q", format=",.0f", title="Prev Year")],
            )
            st.altair_chart(
                (cur_line + prev_line).properties(title=f"ZIP {selected_zip}").interactive(),
                use_container_width=True,
            )
        else:
            st.info("Fewer than 13 months of data — YoY not available.")

        # ── Seasonality by year ───────────────────────────────────────────
        st.subheader("Seasonal pattern by year")

        seas = zip_df.copy()
        seas["year"] = seas["_period"].dt.year.astype(str)
        seas["month_num"] = seas["_period"].dt.month
        seas["month_label"] = seas["_period"].dt.strftime("%b")

        seas_chart = (
            alt.Chart(seas)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "month_num:O",
                    title="Month",
                    axis=alt.Axis(
                        values=list(range(1, 13)),
                        labelExpr='["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][datum.value-1]',
                    ),
                ),
                y=alt.Y(f"{metric}:Q", title=metric.replace("_", " ").title()),
                color=alt.Color("year:N", title="Year"),
                detail="year:N",
                tooltip=[
                    alt.Tooltip("year:N", title="Year"),
                    alt.Tooltip("month_label:N", title="Month"),
                    alt.Tooltip(f"{metric}:Q", format=",.0f"),
                ],
            )
            .interactive()
        )
        st.altair_chart(seas_chart, use_container_width=True)

        # ── Prophet forecast ──────────────────────────────────────────────
        st.subheader("12-month forecast (Prophet)")

        series = zip_df.rename(columns={"_period": "ds", metric: "y"}).copy()
        series["ds"] = pd.to_datetime(series["ds"]).dt.tz_localize(None)

        if len(series) < 12:
            st.warning("Need at least 12 months of history to forecast.")
        else:
            try:
                with st.spinner("Training Prophet model …"):
                    fcst = _run_prophet(series, periods=12)

                cutoff = series["ds"].max()
                hist_data = series.rename(columns={"ds": "date", "y": "value"})
                fc_all = fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": "date"})
                fc_future = fc_all[fc_all["date"] > cutoff].copy()

                hist_line = alt.Chart(hist_data).mark_line(point=True).encode(
                    x=alt.X("date:T", title="Month"),
                    y=alt.Y("value:Q", title=metric.replace("_", " ").title()),
                    color=alt.value("#1f77b4"),
                    tooltip=[alt.Tooltip("date:T"), alt.Tooltip("value:Q", format=",.0f", title="Actual")],
                )
                fc_line = alt.Chart(fc_future).mark_line(point=True, strokeDash=[6, 3]).encode(
                    x="date:T",
                    y="yhat:Q",
                    color=alt.value("#ff7f0e"),
                    tooltip=[alt.Tooltip("date:T"), alt.Tooltip("yhat:Q", format=",.0f", title="Forecast")],
                )
                band = alt.Chart(fc_future).mark_area(opacity=0.2).encode(
                    x="date:T",
                    y=alt.Y("yhat_lower:Q", title=""),
                    y2="yhat_upper:Q",
                    color=alt.value("#ff7f0e"),
                )
                st.altair_chart(
                    alt.layer(hist_line, fc_line, band).resolve_scale(y="shared").interactive(),
                    use_container_width=True,
                )

                fc_tbl = fc_future.copy()
                fc_tbl["Month"] = fc_tbl["date"].dt.strftime("%Y-%m")
                st.dataframe(
                    fc_tbl[["Month", "yhat", "yhat_lower", "yhat_upper"]]
                    .rename(columns={"yhat": "Forecast", "yhat_lower": "Low (90%)", "yhat_upper": "High (90%)"})
                    .style.format({"Forecast": ",.0f", "Low (90%)": ",.0f", "High (90%)": ",.0f"}),
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Prophet failed: {e}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Market Mining
# ════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Market Mining")
    scores = load_market_scores()

    if scores is None:
        _pipeline_hint("mine")
        st.stop()

    scores = scores.copy()
    col_n, col_state = st.columns([1, 3])

    with col_n:
        top_n = st.slider("Show top N markets", 10, 100, 25)

    with col_state:
        all_names = scores["zip_name"].dropna().unique().tolist()
        states = sorted({n.split(", ")[-1].strip() for n in all_names if ", " in n})
        state_filter = st.multiselect("Filter by state", states, placeholder="All states")

    filtered = scores.dropna(subset=["heat_score"]).copy()
    if state_filter:
        filtered = filtered[filtered["zip_name"].str.contains("|".join(state_filter), na=False)]

    top = filtered.nlargest(top_n, "heat_score").copy()
    top["label"] = top["_zip"] + "  " + top["zip_name"].fillna("")

    # ── Heat score bar chart ──────────────────────────────────────────────
    st.subheader(f"Top {top_n} hottest markets")

    bar = (
        alt.Chart(top)
        .mark_bar()
        .encode(
            x=alt.X("heat_score:Q", title="Heat Score (0–100)"),
            y=alt.Y("label:N", sort="-x", title=None),
            color=alt.Color(
                "heat_score:Q",
                scale=alt.Scale(scheme="reds"),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("_zip:N", title="ZIP"),
                alt.Tooltip("zip_name:N", title="Market"),
                alt.Tooltip("heat_score:Q", format=".1f", title="Heat Score"),
                alt.Tooltip("median_listing_price:Q", format=",.0f", title="Median Price ($)"),
                alt.Tooltip("median_listing_price_yy:Q", format=".1%", title="Price YoY"),
                alt.Tooltip("pending_ratio:Q", format=".2f", title="Pending Ratio"),
                alt.Tooltip("median_days_on_market:Q", format=".0f", title="Days on Market"),
            ],
        )
        .properties(height=max(300, top_n * 22))
    )
    st.altair_chart(bar, use_container_width=True)

    # ── Scatter: price vs pending ratio ──────────────────────────────────
    st.subheader("Market scatter — Listing price vs. Pending ratio")

    scatter_df = filtered.dropna(subset=["median_listing_price", "pending_ratio", "heat_score"]).copy()
    scatter_df["median_listing_price_k"] = scatter_df["median_listing_price"] / 1000

    scatter = (
        alt.Chart(scatter_df.sample(min(2000, len(scatter_df)), random_state=42))
        .mark_circle(size=60, opacity=0.7)
        .encode(
            x=alt.X("median_listing_price_k:Q", title="Median Listing Price ($k)"),
            y=alt.Y("pending_ratio:Q", title="Pending Ratio"),
            color=alt.Color(
                "heat_score:Q",
                scale=alt.Scale(scheme="plasma"),
                title="Heat Score",
            ),
            tooltip=[
                alt.Tooltip("_zip:N", title="ZIP"),
                alt.Tooltip("zip_name:N", title="Market"),
                alt.Tooltip("heat_score:Q", format=".1f", title="Heat"),
                alt.Tooltip("median_listing_price_k:Q", format=",.0f", title="Price ($k)"),
                alt.Tooltip("pending_ratio:Q", format=".2f", title="Pending Ratio"),
                alt.Tooltip("median_days_on_market:Q", format=".0f", title="Days on Market"),
            ],
        )
        .interactive()
    )
    st.altair_chart(scatter, use_container_width=True)

    # ── Cluster breakdown ─────────────────────────────────────────────────
    if "cluster_label" in scores.columns:
        st.subheader("Market clusters")

        cluster_summary = (
            scores.dropna(subset=["cluster_label"])
            .groupby("cluster_label")
            .agg(
                count=("_zip", "count"),
                avg_heat=("heat_score", "mean"),
                avg_price=("median_listing_price", "mean"),
                avg_price_yy=("median_listing_price_yy", "mean"),
                avg_pending_ratio=("pending_ratio", "mean"),
                avg_days_on_market=("median_days_on_market", "mean"),
            )
            .reset_index()
            .rename(columns={
                "cluster_label": "Cluster",
                "count": "ZIP Count",
                "avg_heat": "Avg Heat Score",
                "avg_price": "Avg Price ($)",
                "avg_price_yy": "Avg Price YoY",
                "avg_pending_ratio": "Avg Pending Ratio",
                "avg_days_on_market": "Avg Days on Market",
            })
        )
        order = ["Hot", "Rising", "Balanced", "Cooling", "Soft"]
        cluster_summary["_order"] = cluster_summary["Cluster"].map({v: i for i, v in enumerate(order)})
        cluster_summary = cluster_summary.sort_values("_order").drop(columns="_order")

        st.dataframe(
            cluster_summary.style.format({
                "Avg Heat Score": "{:.1f}",
                "Avg Price ($)": "{:,.0f}",
                "Avg Price YoY": "{:.1%}",
                "Avg Pending Ratio": "{:.3f}",
                "Avg Days on Market": "{:.0f}",
            }),
            use_container_width=True,
        )

        cluster_box = (
            alt.Chart(scores.dropna(subset=["cluster_label", "heat_score"]))
            .mark_boxplot()
            .encode(
                x=alt.X("cluster_label:N", sort=order, title="Cluster"),
                y=alt.Y("heat_score:Q", title="Heat Score"),
                color=alt.Color(
                    "cluster_label:N",
                    sort=order,
                    scale=alt.Scale(
                        domain=order,
                        range=["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"],
                    ),
                    legend=None,
                ),
            )
        )
        st.altair_chart(cluster_box, use_container_width=True)

    # ── Full scores table ─────────────────────────────────────────────────
    with st.expander("Full market scores table"):
        display_cols = [c for c in [
            "_zip", "zip_name", "heat_score", "cluster_label",
            "median_listing_price", "median_listing_price_yy",
            "median_days_on_market", "pending_ratio", "price_reduced_share",
        ] if c in scores.columns]
        st.dataframe(
            filtered.nlargest(500, "heat_score")[display_cols]
            .style.format({
                "heat_score": "{:.1f}",
                "median_listing_price": "{:,.0f}",
                "median_listing_price_yy": "{:.1%}",
                "median_days_on_market": "{:.0f}",
                "pending_ratio": "{:.3f}",
                "price_reduced_share": "{:.3f}",
            }),
            use_container_width=True,
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Batch Forecasts
# ════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("Batch Forecasts")
    forecasts = load_forecasts()
    scores_for_labels = load_market_scores()

    if forecasts is None:
        _pipeline_hint("forecast")
        st.stop()

    forecast_zips = sorted(forecasts["_zip"].unique().tolist())

    if scores_for_labels is not None:
        zip_name_map = scores_for_labels.set_index("_zip")["zip_name"].to_dict()
    else:
        zip_name_map = {}

    def _zip_label(z: str) -> str:
        name = zip_name_map.get(z, "")
        return f"{z}  {name}" if name else z

    zip_options = [_zip_label(z) for z in forecast_zips]

    st.markdown("Select markets to compare. Each line shows historical fit + 12-month forecast.")

    n_compare = st.slider("Compare top N markets", 2, min(20, len(forecast_zips)), 5)

    if scores_for_labels is not None:
        default_zips = (
            scores_for_labels[scores_for_labels["_zip"].isin(forecast_zips)]
            .nlargest(n_compare, "heat_score")["_zip"]
            .tolist()
        )
    else:
        default_zips = forecast_zips[:n_compare]

    default_labels = [_zip_label(z) for z in default_zips if _zip_label(z) in zip_options]
    selected_labels = st.multiselect("Markets", zip_options, default=default_labels)
    selected_zips = [lbl.split("  ")[0] for lbl in selected_labels]

    if not selected_zips:
        st.info("Select at least one market above.")
        st.stop()

    fc_sel = forecasts[forecasts["_zip"].isin(selected_zips)].copy()
    fc_sel["market"] = fc_sel["_zip"].map(lambda z: _zip_label(z))
    fc_sel["type"] = fc_sel["is_forecast"].map({True: "Forecast", False: "Historical"})

    # ── Overlay line chart ────────────────────────────────────────────────
    st.subheader("Forecast comparison")

    hist_layer = (
        alt.Chart(fc_sel[~fc_sel["is_forecast"]])
        .mark_line()
        .encode(
            x=alt.X("ds:T", title="Month"),
            y=alt.Y("yhat:Q", title="Median Listing Price ($)"),
            color=alt.Color("market:N", title="Market"),
            tooltip=[
                alt.Tooltip("ds:T", title="Month"),
                alt.Tooltip("market:N", title="Market"),
                alt.Tooltip("yhat:Q", format=",.0f", title="Fitted"),
            ],
        )
    )

    fc_layer = (
        alt.Chart(fc_sel[fc_sel["is_forecast"]])
        .mark_line(strokeDash=[6, 3])
        .encode(
            x="ds:T",
            y="yhat:Q",
            color=alt.Color("market:N"),
            tooltip=[
                alt.Tooltip("ds:T", title="Month"),
                alt.Tooltip("market:N", title="Market"),
                alt.Tooltip("yhat:Q", format=",.0f", title="Forecast"),
                alt.Tooltip("yhat_lower:Q", format=",.0f", title="Low (90%)"),
                alt.Tooltip("yhat_upper:Q", format=",.0f", title="High (90%)"),
            ],
        )
    )

    band_layer = (
        alt.Chart(fc_sel[fc_sel["is_forecast"]])
        .mark_area(opacity=0.1)
        .encode(
            x="ds:T",
            y=alt.Y("yhat_lower:Q", title=""),
            y2="yhat_upper:Q",
            color=alt.Color("market:N"),
        )
    )

    st.altair_chart(
        alt.layer(hist_layer, fc_layer, band_layer).resolve_scale(y="shared").interactive(),
        use_container_width=True,
    )

    # ── Forecast-only table ───────────────────────────────────────────────
    st.subheader("Next 12 months")

    fc_only = fc_sel[fc_sel["is_forecast"]].copy()
    fc_only["Month"] = fc_only["ds"].dt.strftime("%Y-%m")

    pivot = fc_only.pivot_table(index="Month", columns="market", values="yhat", aggfunc="first")
    st.dataframe(
        pivot.style.format("{:,.0f}"),
        use_container_width=True,
    )

    # ── Projected price change ────────────────────────────────────────────
    st.subheader("Projected 12-month price change")

    changes = []
    for z in selected_zips:
        z_fc = fc_sel[(fc_sel["_zip"] == z) & fc_sel["is_forecast"]].sort_values("ds")
        z_hist = fc_sel[(fc_sel["_zip"] == z) & ~fc_sel["is_forecast"]].sort_values("ds")
        if z_fc.empty or z_hist.empty:
            continue
        base_price = z_hist["yhat"].iloc[-1]
        end_price = z_fc["yhat"].iloc[-1]
        changes.append({
            "Market": _zip_label(z),
            "Current Price ($)": base_price,
            "Projected Price ($)": end_price,
            "Change ($)": end_price - base_price,
            "Change (%)": (end_price - base_price) / base_price if base_price else np.nan,
        })

    if changes:
        chg_df = pd.DataFrame(changes).sort_values("Change (%)", ascending=False)
        st.dataframe(
            chg_df.style.format({
                "Current Price ($)": "{:,.0f}",
                "Projected Price ($)": "{:,.0f}",
                "Change ($)": "{:+,.0f}",
                "Change (%)": "{:+.1%}",
            }).background_gradient(subset=["Change (%)"], cmap="RdYlGn"),
            use_container_width=True,
        )
