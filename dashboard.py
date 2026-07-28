"""
Static HTML dashboard generator.

Reads the parquet/json outputs of pipeline.py (market_scores, county_scores,
forecasts, fetch_meta, mine_meta) and renders a single self-contained
data/dashboard.html — no server, no CDN dependency, works offline.

Charts are hand-rolled inline SVG (not a plotting library) so the output file
has zero external dependencies. Color roles follow the project's data-viz
palette: fixed categorical hue order, one sequential hue for magnitude,
never a dual axis. See the `dataviz` skill for the full rationale.
"""
import html
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ─── Palette (validated categorical order — see dataviz skill references/palette.md) ─
CATEGORICAL = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark":  ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
SEQUENTIAL_BLUE = "#2a78d6"

CLUSTER_ORDER = ["Hot", "Rising", "Balanced", "Cooling", "Soft"]


# ─── Small SVG helpers ───────────────────────────────────────────────────────

def _esc(s) -> str:
    return html.escape(str(s), quote=True)


def _scale(value, d0, d1, r0, r1):
    if d1 == d0:
        return r0
    return r0 + (value - d0) / (d1 - d0) * (r1 - r0)


def _fmt_money(v) -> str:
    if pd.isna(v):
        return "—"
    return f"${v:,.0f}"


def _fmt_pct(v) -> str:
    if pd.isna(v):
        return "—"
    return f"{v:+.1%}"


def _fmt_compact(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# ─── Stat tiles ───────────────────────────────────────────────────────────────

def stat_tile(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="tile-sub">{_esc(sub)}</div>' if sub else ""
    return f"""
    <div class="tile">
      <div class="tile-label">{_esc(label)}</div>
      <div class="tile-value">{_esc(value)}</div>
      {sub_html}
    </div>"""


# ─── Horizontal bar chart (heat score ranking) ───────────────────────────────

def hbar_chart(
    rows: list[dict],
    *,
    value_key: str,
    label_key: str,
    tooltip_fn,
    color: str,
    domain_max: float = 100.0,
    row_h: int = 26,
    bar_h: int = 16,
    chart_id: str = "hbar",
) -> str:
    n = len(rows)
    left_pad, right_pad, top_pad = 8, 56, 8
    label_w = 220
    plot_w = 560
    width = left_pad + label_w + plot_w + right_pad
    height = top_pad * 2 + n * row_h

    bars = []
    for i, row in enumerate(rows):
        y = top_pad + i * row_h + (row_h - bar_h) / 2
        val = row[value_key]
        bar_w = max(2, _scale(val, 0, domain_max, 0, plot_w))
        x0 = left_pad + label_w
        label = _esc(row[label_key])
        tip = _esc(tooltip_fn(row))
        show_value_label = i < 5
        value_label = (
            f'<text class="bar-value" x="{x0 + bar_w + 6:.1f}" y="{y + bar_h / 2:.1f}" '
            f'dominant-baseline="middle">{val:.1f}</text>'
            if show_value_label else ""
        )
        bars.append(f"""
      <text class="bar-label" x="{left_pad + label_w - 8}" y="{y + bar_h / 2:.1f}"
            text-anchor="end" dominant-baseline="middle">{label}</text>
      <rect class="hit bar" x="{x0:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h}"
            rx="4" fill="{color}" data-tip="{tip}"></rect>
      {value_label}""")

    return f"""
    <div class="chart-scroll">
    <svg class="chart" viewBox="0 0 {width} {height}" width="100%"
         preserveAspectRatio="xMinYMin meet" id="{chart_id}">
      {''.join(bars)}
    </svg>
    </div>"""


# ─── Cluster summary bars (fixed categorical color per cluster) ─────────────

def cluster_bar_chart(cluster_df: pd.DataFrame, colors: list[str]) -> str:
    rows = []
    for i, label in enumerate(CLUSTER_ORDER):
        sub = cluster_df[cluster_df["cluster_label"] == label]
        if sub.empty:
            continue
        rows.append({
            "label": label,
            "count": int(sub["_zip"].count()) if "_zip" in sub.columns else int(len(sub)),
            "avg_heat": float(sub["heat_score"].mean()),
            "color": colors[i % len(colors)],
        })

    if not rows:
        return '<p class="empty">No cluster data.</p>'

    row_h, bar_h = 34, 20
    left_pad, right_pad, top_pad = 8, 70, 8
    label_w = 100
    plot_w = 480
    width = left_pad + label_w + plot_w + right_pad
    height = top_pad * 2 + len(rows) * row_h

    bars = []
    for i, r in enumerate(rows):
        y = top_pad + i * row_h + (row_h - bar_h) / 2
        bar_w = max(2, _scale(r["avg_heat"], 0, 100, 0, plot_w))
        x0 = left_pad + label_w
        tip = _esc(f"{r['label']}: {r['count']} ZIPs, avg heat {r['avg_heat']:.1f}")
        bars.append(f"""
      <circle cx="{left_pad + 6}" cy="{y + bar_h / 2:.1f}" r="5" fill="{r['color']}"></circle>
      <text class="bar-label" x="{left_pad + 18}" y="{y + bar_h / 2:.1f}"
            dominant-baseline="middle">{_esc(r['label'])}</text>
      <rect class="hit bar" x="{x0:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h}"
            rx="4" fill="{r['color']}" data-tip="{tip}"></rect>
      <text class="bar-value" x="{x0 + bar_w + 6:.1f}" y="{y + bar_h / 2:.1f}"
            dominant-baseline="middle">{r['avg_heat']:.1f} · {r['count']} ZIPs</text>""")

    return f"""
    <div class="chart-scroll">
    <svg class="chart" viewBox="0 0 {width} {height}" width="100%"
         preserveAspectRatio="xMinYMin meet">
      {''.join(bars)}
    </svg>
    </div>"""


# ─── Multi-series line chart (forecast comparison) ───────────────────────────

def forecast_line_chart(fc: pd.DataFrame, zip_labels: dict, colors: list[str]) -> str:
    zips = fc["_zip"].unique().tolist()[: len(colors)]
    if not zips:
        return '<p class="empty">No forecast data.</p>'

    width, height = 900, 380
    pad_l, pad_r, pad_t, pad_b = 56, 170, 16, 32
    plot_l, plot_r = pad_l, width - pad_r
    plot_t, plot_b = pad_t, height - pad_b

    all_pts = fc[fc["_zip"].isin(zips)]
    x_min, x_max = all_pts["ds"].min(), all_pts["ds"].max()
    y_min, y_max = all_pts["yhat_lower"].min(), all_pts["yhat_upper"].max()
    y_pad = (y_max - y_min) * 0.08 or 1
    y_min, y_max = y_min - y_pad, y_max + y_pad

    def xp(d):
        return _scale(d.value, x_min.value, x_max.value, plot_l, plot_r)

    def yp(v):
        return _scale(v, y_min, y_max, plot_b, plot_t)

    # y gridlines / ticks (4 clean steps)
    grid = []
    for i in range(5):
        gv = y_min + (y_max - y_min) * i / 4
        gy = yp(gv)
        grid.append(
            f'<line class="grid" x1="{plot_l}" x2="{plot_r}" y1="{gy:.1f}" y2="{gy:.1f}"></line>'
            f'<text class="axis-label" x="{plot_l - 8}" y="{gy:.1f}" text-anchor="end" '
            f'dominant-baseline="middle">${gv / 1000:,.0f}k</text>'
        )

    series_svg = []
    legend_items = []
    end_labels = []  # (y_pixel, svg_string) — compacted after the fact

    for i, z in enumerate(zips):
        color = colors[i % len(colors)]
        s = fc[fc["_zip"] == z].sort_values("ds")
        hist = s[~s["is_forecast"]]
        forecast = s[s["is_forecast"]]

        def path_d(sub):
            pts = [f"{xp(row.ds):.1f},{yp(row.yhat):.1f}" for row in sub.itertuples()]
            return "M" + " L".join(pts) if pts else ""

        if len(hist):
            series_svg.append(f'<path class="line" d="{path_d(hist)}" stroke="{color}" fill="none"></path>')
        if len(forecast):
            # connect last historical point to keep the dashed segment contiguous
            bridge = pd.concat([hist.tail(1), forecast]) if len(hist) else forecast
            series_svg.append(
                f'<path class="line dashed" d="{path_d(bridge)}" stroke="{color}" fill="none"></path>'
            )

        # sparse hover markers: monthly points already sparse enough to mark all
        for row in s.itertuples():
            tip = _esc(
                f"{zip_labels.get(z, z)} · {row.ds:%Y-%m}: ${row.yhat:,.0f}"
                + (" (forecast)" if row.is_forecast else "")
            )
            series_svg.append(
                f'<circle class="hit dot" cx="{xp(row.ds):.1f}" cy="{yp(row.yhat):.1f}" r="4" '
                f'fill="{color}" data-tip="{tip}"></circle>'
            )

        last = s.iloc[-1]
        end_labels.append([yp(last["yhat"]), zip_labels.get(z, z), color])
        legend_items.append(
            f'<span class="legend-item"><span class="legend-swatch" '
            f'style="background:{color}"></span>{_esc(zip_labels.get(z, z))}</span>'
        )

    # compact overlapping end labels (simple greedy vertical spread)
    end_labels.sort(key=lambda t: t[0])
    min_gap = 14
    for i in range(1, len(end_labels)):
        if end_labels[i][0] - end_labels[i - 1][0] < min_gap:
            end_labels[i][0] = end_labels[i - 1][0] + min_gap

    label_svg = [
        f'<text class="end-label" x="{plot_r + 10}" y="{y:.1f}" dominant-baseline="middle" '
        f'fill="{color}">{_esc(name)}</text>'
        for y, name, color in end_labels
    ]

    return f"""
    <div class="chart-scroll">
    <svg class="chart line-chart" viewBox="0 0 {width} {height}" width="100%"
         preserveAspectRatio="xMinYMin meet">
      {''.join(grid)}
      <line class="baseline" x1="{plot_l}" x2="{plot_r}" y1="{plot_b}" y2="{plot_b}"></line>
      {''.join(series_svg)}
      {''.join(label_svg)}
    </svg>
    </div>
    <div class="legend">{''.join(legend_items)}</div>"""


# ─── Page assembly ────────────────────────────────────────────────────────────

CSS = """
:root {
  color-scheme: light;
  --surface-1:      #fcfcfb;
  --page-plane:     #f9f9f7;
  --text-primary:   #0b0b0b;
  --text-secondary: #52514e;
  --text-muted:     #898781;
  --gridline:       #e1e0d9;
  --baseline:       #c3c2b7;
  --border:         rgba(11,11,11,0.10);
  --tile-bg:        #ffffff;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface-1:      #1a1a19;
    --page-plane:     #0d0d0d;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted:     #898781;
    --gridline:       #2c2c2a;
    --baseline:       #383835;
    --border:         rgba(255,255,255,0.10);
    --tile-bg:        #1f1f1e;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-1:      #1a1a19;
  --page-plane:     #0d0d0d;
  --text-primary:   #ffffff;
  --text-secondary: #c3c2b7;
  --text-muted:     #898781;
  --gridline:       #2c2c2a;
  --baseline:       #383835;
  --border:         rgba(255,255,255,0.10);
  --tile-bg:        #1f1f1e;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  background: var(--page-plane);
  color: var(--text-primary);
}
.wrap { max-width: 1100px; margin: 0 auto; padding: 24px 20px 64px; }
header { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; flex-wrap: wrap; margin-bottom: 20px; }
h1 { font-size: 1.5rem; margin: 0 0 4px; }
.subtitle { color: var(--text-secondary); font-size: 0.9rem; }
.theme-toggle {
  border: 1px solid var(--border); background: var(--tile-bg); color: var(--text-primary);
  border-radius: 8px; padding: 6px 12px; font-size: 0.85rem; cursor: pointer;
}
.tiles { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 32px; }
.tile {
  background: var(--tile-bg); border: 1px solid var(--border); border-radius: 10px;
  padding: 14px 18px; min-width: 160px; flex: 1 1 160px;
}
.tile-label { font-size: 0.78rem; color: var(--text-secondary); margin-bottom: 4px; }
.tile-value { font-size: 1.6rem; font-weight: 600; }
.tile-sub { font-size: 0.78rem; color: var(--text-muted); margin-top: 2px; }
section { margin-bottom: 40px; }
h2 { font-size: 1.1rem; margin: 0 0 4px; }
.section-sub { color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 14px; }
.chart-scroll { overflow-x: auto; background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 8px; }
.chart { display: block; }
.bar-label { font-size: 11px; fill: var(--text-secondary); }
.bar-value { font-size: 11px; fill: var(--text-primary); font-variant-numeric: tabular-nums; }
.axis-label { font-size: 10px; fill: var(--text-muted); font-variant-numeric: tabular-nums; }
.end-label { font-size: 11px; font-weight: 600; }
.grid { stroke: var(--gridline); stroke-width: 1; }
.baseline { stroke: var(--baseline); stroke-width: 1; }
.line { stroke-width: 2; }
.line.dashed { stroke-dasharray: 6 3; }
.hit { cursor: pointer; }
.hit.bar { transition: opacity .1s; }
.hit.bar:hover { opacity: 0.85; }
.hit.dot:hover { r: 6; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 10px; font-size: 0.82rem; color: var(--text-secondary); }
.legend-item { display: inline-flex; align-items: center; gap: 6px; }
.legend-swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
.cluster-note { font-size: 0.78rem; color: var(--text-muted); margin-top: 8px; }
table.data-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
table.data-table th, table.data-table td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border); }
table.data-table th { color: var(--text-secondary); font-weight: 600; position: sticky; top: 0; background: var(--surface-1); }
table.data-table td { font-variant-numeric: tabular-nums; }
.table-scroll { max-height: 420px; overflow-y: auto; border: 1px solid var(--border); border-radius: 10px; background: var(--surface-1); }
.filter-input {
  width: 100%; max-width: 320px; padding: 8px 10px; margin-bottom: 10px;
  border: 1px solid var(--border); border-radius: 8px; background: var(--tile-bg); color: var(--text-primary);
}
#tooltip {
  position: fixed; pointer-events: none; background: var(--tile-bg); color: var(--text-primary);
  border: 1px solid var(--border); border-radius: 6px; padding: 6px 10px; font-size: 0.8rem;
  box-shadow: 0 4px 12px rgba(0,0,0,0.15); display: none; z-index: 100; max-width: 260px;
}
footer { color: var(--text-muted); font-size: 0.78rem; margin-top: 40px; }
"""

JS = """
const tooltip = document.getElementById('tooltip');
document.querySelectorAll('.hit').forEach(el => {
  el.addEventListener('mouseenter', () => {
    tooltip.textContent = el.dataset.tip;
    tooltip.style.display = 'block';
  });
  el.addEventListener('mousemove', (e) => {
    tooltip.style.left = (e.clientX + 14) + 'px';
    tooltip.style.top = (e.clientY + 14) + 'px';
  });
  el.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });
});

const toggle = document.getElementById('theme-toggle');
if (toggle) {
  const stored = localStorage.getItem('re-dashboard-theme');
  if (stored) document.documentElement.setAttribute('data-theme', stored);
  toggle.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme')
      || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('re-dashboard-theme', next);
  });
}

document.querySelectorAll('.filter-input').forEach(input => {
  const tableId = input.dataset.table;
  const table = document.getElementById(tableId);
  if (!table) return;
  input.addEventListener('input', () => {
    const q = input.value.toLowerCase();
    table.querySelectorAll('tbody tr').forEach(tr => {
      tr.style.display = tr.textContent.toLowerCase().includes(q) ? '' : 'none';
    });
  });
});
"""


def data_table(df: pd.DataFrame, columns: list[tuple[str, str]], table_id: str, filter_placeholder: str) -> str:
    """columns: list of (df_col, header_label)."""
    thead = "".join(f"<th>{_esc(h)}</th>" for _, h in columns)
    rows = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{_esc(row[c]) if pd.notna(row[c]) else '—'}</td>" for c, _ in columns)
        rows.append(f"<tr>{cells}</tr>")
    return f"""
    <input class="filter-input" data-table="{table_id}" placeholder="{_esc(filter_placeholder)}">
    <div class="table-scroll">
      <table class="data-table" id="{table_id}">
        <thead><tr>{thead}</tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>"""


def build_dashboard(data_dir: Path, out_path: Path) -> None:
    scores_path = data_dir / "market_scores.parquet"
    if not scores_path.exists():
        raise SystemExit("market_scores.parquet not found — run 'mine' first.")
    scores = pd.read_parquet(scores_path)

    county_scores = None
    county_path = data_dir / "county_scores.parquet"
    if county_path.exists():
        county_scores = pd.read_parquet(county_path)

    forecasts = None
    fc_path = data_dir / "forecasts.parquet"
    if fc_path.exists():
        forecasts = pd.read_parquet(fc_path)
        forecasts["ds"] = pd.to_datetime(forecasts["ds"])

    mine_meta = {}
    mine_meta_path = data_dir / "mine_meta.json"
    if mine_meta_path.exists():
        mine_meta = json.loads(mine_meta_path.read_text())

    fetch_meta = {}
    fetch_meta_path = data_dir / "fetch_meta.json"
    if fetch_meta_path.exists():
        fetch_meta = json.loads(fetch_meta_path.read_text())

    scored = scores.dropna(subset=["heat_score"]).copy()
    scored["state"] = scored["zip_name"].fillna("").apply(
        lambda s: s.split(", ")[-1].strip() if ", " in s else None
    )
    hottest_state = "—"
    if scored["state"].notna().any():
        by_state = scored.dropna(subset=["state"]).groupby("state")["heat_score"].mean()
        if len(by_state):
            hottest_state = by_state.idxmax()

    # ── Stat tiles ──────────────────────────────────────────────────────────
    tiles = [
        stat_tile("ZIPs scored", f"{len(scored):,}"),
        stat_tile("Avg heat score", f"{scored['heat_score'].mean():.1f}", "0–100 scale"),
        stat_tile("Hottest state (avg)", hottest_state),
        stat_tile(
            "Markets forecasted",
            f"{forecasts['_zip'].nunique():,}" if forecasts is not None else "—",
            "12-month Prophet horizon" if forecasts is not None else "run `pipeline.py forecast`",
        ),
    ]

    # ── Top ZIPs bar chart ─────────────────────────────────────────────────
    top25 = scored.nlargest(25, "heat_score").to_dict("records")

    def zip_tip(r):
        return (
            f"{r['_zip']} {r.get('zip_name') or ''} · heat {r['heat_score']:.1f} · "
            f"{_fmt_money(r.get('median_listing_price'))} · "
            f"YoY {_fmt_pct(r.get('median_listing_price_yy'))}"
        )

    for r in top25:
        r["label"] = f"{r['_zip']}  {r.get('zip_name') or ''}"
    top_zip_chart = hbar_chart(
        top25, value_key="heat_score", label_key="label",
        tooltip_fn=zip_tip, color=SEQUENTIAL_BLUE, chart_id="top-zips",
    )

    # ── Cluster breakdown ──────────────────────────────────────────────────
    cluster_section = ""
    if "cluster_label" in scores.columns and scores["cluster_label"].notna().any():
        cluster_section = cluster_bar_chart(scored, CATEGORICAL["light"])

    # ── Forecast comparison ────────────────────────────────────────────────
    forecast_section = ""
    if forecasts is not None:
        top5 = scored.nlargest(5, "heat_score")["_zip"].tolist()
        top5 = [z for z in top5 if z in set(forecasts["_zip"])]
        zip_labels = scored.set_index("_zip")["zip_name"].to_dict()
        forecast_section = forecast_line_chart(
            forecasts[forecasts["_zip"].isin(top5)], zip_labels, CATEGORICAL["light"]
        )

    # ── County section ─────────────────────────────────────────────────────
    county_section = ""
    if county_scores is not None:
        c_scored = county_scores.dropna(subset=["heat_score"]).nlargest(15, "heat_score").to_dict("records")
        for r in c_scored:
            r["label"] = r.get("county_name") or r.get("county_fips")

        def county_tip(r):
            return (
                f"{r.get('county_name') or r.get('county_fips')} · heat {r['heat_score']:.1f} · "
                f"{_fmt_money(r.get('median_listing_price'))} · YoY {_fmt_pct(r.get('median_listing_price_yy'))}"
            )

        county_section = hbar_chart(
            c_scored, value_key="heat_score", label_key="label",
            tooltip_fn=county_tip, color=CATEGORICAL["light"][2], chart_id="top-counties",
        )

    # ── Full ZIP table ─────────────────────────────────────────────────────
    table_cols = [
        ("_zip", "ZIP"), ("zip_name", "Market"), ("heat_score", "Heat"),
        ("cluster_label", "Cluster"), ("median_listing_price", "Median Price"),
        ("median_listing_price_yy", "Price YoY"), ("median_days_on_market", "Days on Market"),
    ]
    table_df = scored.nlargest(300, "heat_score").copy()
    table_df["heat_score"] = table_df["heat_score"].round(1)
    if "median_listing_price" in table_df:
        table_df["median_listing_price"] = table_df["median_listing_price"].map(_fmt_money)
    if "median_listing_price_yy" in table_df:
        table_df["median_listing_price_yy"] = table_df["median_listing_price_yy"].map(_fmt_pct)
    zip_table_html = data_table(table_df, table_cols, "zip-table", "Filter by ZIP, market, or cluster…")

    # ── Freshness strings ───────────────────────────────────────────────────
    data_through = mine_meta.get("data_through")
    if not data_through and forecasts is not None and (~forecasts["is_forecast"]).any():
        data_through = forecasts.loc[~forecasts["is_forecast"], "ds"].max().strftime("%Y-%m-%d")
    data_through = data_through or "—"
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    freshness_bits = [f"data through {data_through}"]
    if fetch_meta:
        newest = max(fetch_meta.values())
        freshness_bits.append(f"sources last fetched {datetime.fromisoformat(newest):%Y-%m-%d}")

    county_html = (
        f"""
    <section>
      <h2>Top counties by heat score</h2>
      <p class="section-sub">County-level rollup from RDC_Inventory_Core_Metrics_County_History.</p>
      {county_section}
    </section>""" if county_section else ""
    )

    forecast_html = (
        f"""
    <section>
      <h2>Forecast comparison — top 5 hottest markets</h2>
      <p class="section-sub">Solid = historical fit, dashed = 12-month Prophet forecast.</p>
      {forecast_section}
    </section>""" if forecast_section else """
    <section>
      <h2>Forecast comparison</h2>
      <p class="section-sub">Run <code>poetry run python pipeline.py forecast</code> to populate this section.</p>
    </section>"""
    )

    cluster_html = (
        f"""
    <section>
      <h2>Market clusters</h2>
      <p class="section-sub">Average heat score and ZIP count per cluster.</p>
      {cluster_section}
    </section>""" if cluster_section else ""
    )

    body = f"""
<div class="wrap">
  <header>
    <div>
      <h1>Real Estate Market Dashboard</h1>
      <div class="subtitle">{_esc(' · '.join(freshness_bits))} · generated {generated_at}</div>
    </div>
    <button class="theme-toggle" id="theme-toggle">Toggle theme</button>
  </header>

  <div class="tiles">{''.join(tiles)}</div>

  <section>
    <h2>Top 25 hottest ZIPs</h2>
    <p class="section-sub">Weighted heat score (0–100) from price growth, days on market, pending ratio, price reductions, and supply.</p>
    {top_zip_chart}
  </section>

  {cluster_html}
  {forecast_html}
  {county_html}

  <section>
    <h2>All scored ZIPs</h2>
    <p class="section-sub">Top 300 by heat score. Filter below.</p>
    {zip_table_html}
  </section>

  <footer>Generated by <code>pipeline.py report</code> from local parquet outputs. No external network calls at view time.</footer>
</div>
<div id="tooltip"></div>
"""

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Real Estate Market Dashboard</title>
<style>{CSS}</style>
</head>
<body>
{body}
<script>{JS}</script>
</body>
</html>
"""
    out_path.write_text(page, encoding="utf-8")
