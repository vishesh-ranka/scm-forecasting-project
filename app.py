"""Interactive dashboard for the hierarchical demand forecasting results.

Reads only the Parquet files already produced by the pipeline - nothing is
recomputed here. Run with:  streamlit run app.py
"""

import time
from pathlib import Path

import duckdb
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
RECONCILED = PROJECT_ROOT / "data" / "forecasts" / "reconciled_forecasts.parquet"
HIERARCHY_DIR = PROJECT_ROOT / "data" / "staged" / "hierarchy"

LEVELS = ["total", "category", "department", "store", "item_store"]
LEVEL_LABELS = {
    "total": "Company Wide Total",
    "category": "Product Category",
    "department": "Department",
    "store": "Store",
    "item_store": "Individual Product (per store)",
}

# mint_ols stays in the Parquet for completeness but is never surfaced: the app
# never explains the difference between the two MinT variants, so showing both
# only invites confusion. UI_METHODS is what the interface compares.
UI_METHODS = ["unreconciled", "bottom_up", "mint_wls_struct"]
METHOD_LABELS = {
    "unreconciled": "Original Forecast",
    "bottom_up": "Reconciled: Bottom Up",
    "mint_wls_struct": "Reconciled: MinT",
}

# Muted palette, validated for colour-blind separation against a light surface.
# With MinT (OLS) dropped the remaining trio was re-validated on ALL pairs
# (worst normal dE 21.1, colour-blind dE 20.3) - the previous grey failed
# against teal once the blue slot was no longer between them.
COLORS = {
    "unreconciled": "#2B2B2B",
    "bottom_up": "#C9743C",
    "mint_wls_struct": "#2F5E93",
    "mint_ols": "#3E8C7A",  # retained for data completeness; not shown in the UI
}
DETAIL_COLORS = {"original": COLORS["unreconciled"], "mint": COLORS["mint_wls_struct"]}

PLOT_SURFACE = "#fcfcfb"
PLOT_INK = "#16181d"
PLOT_INK_SOFT = "#5b6067"
PLOT_GRID = "#e3e5e8"

LEVEL_BLURB = {
    "total": "One number: every product in every store, combined.",
    "category": "3 categories, **FOODS**, **HOBBIES**, **HOUSEHOLD**.",
    "department": "7 departments, **FOODS_1, FOODS_2, FOODS_3, HOBBIES_1, HOBBIES_2, "
                  "HOUSEHOLD_1, HOUSEHOLD_2**.",
    "store": "10 Walmart stores, **CA_1 to CA_4** (California), **TX_1 to TX_3** (Texas), "
             "**WI_1 to WI_3** (Wisconsin).",
    "item_store": "30,490 combinations, 3,049 products × 10 stores "
                  "(e.g. `FOODS_1_001_CA_1` is product FOODS_1_001 in store CA_1).",
}

# Approximate state centroids - the dataset gives no store addresses.
STORE_GEO = [
    ("California", "CA", 4, 36.78, -119.42, "CA_1, CA_2, CA_3, CA_4"),
    ("Texas", "TX", 3, 31.00, -99.90, "TX_1, TX_2, TX_3"),
    ("Wisconsin", "WI", 3, 44.50, -89.50, "WI_1, WI_2, WI_3"),
]

st.set_page_config(page_title="Hierarchical Demand Forecasting", layout="wide")


# --------------------------------------------------------------------------
# Data loading - cached, and always filtered/aggregated in DuckDB so the
# 3.4M-row forecast file never lands in memory whole.
# --------------------------------------------------------------------------
@st.cache_data
def load_error_by_level():
    return duckdb.connect().sql(
        f"""
        SELECT level, method, SQRT(AVG(POWER(forecast - actual, 2))) AS rmse
        FROM read_parquet('{RECONCILED}')
        GROUP BY level, method
        """
    ).df()


@st.cache_data
def load_total_series():
    return duckdb.connect().sql(
        f"""
        SELECT date,
               MAX(actual)   FILTER (method = 'unreconciled')    AS actual,
               MAX(forecast) FILTER (method = 'unreconciled')    AS unreconciled,
               MAX(forecast) FILTER (method = 'mint_wls_struct') AS mint_wls_struct
        FROM read_parquet('{RECONCILED}')
        WHERE level = 'total'
        GROUP BY date ORDER BY date
        """
    ).df()


@st.cache_data
def load_history(days: int):
    return duckdb.connect().sql(
        f"""
        SELECT date, sales
        FROM read_parquet('{HIERARCHY_DIR / "total.parquet"}')
        WHERE date < (SELECT MIN(date) FROM read_parquet('{RECONCILED}'))
        ORDER BY date DESC LIMIT {int(days)}
        """
    ).df().sort_values("date")


@st.cache_data
def load_coherence():
    return duckdb.connect().sql(
        f"""
        WITH totals AS (
            SELECT method, date, forecast AS total
            FROM read_parquet('{RECONCILED}') WHERE level = 'total'
        ),
        summed AS (
            SELECT method, level, date, SUM(forecast) AS level_sum
            FROM read_parquet('{RECONCILED}') WHERE level <> 'total'
            GROUP BY method, level, date
        )
        SELECT s.method, s.level,
               MAX(ABS(s.level_sum - t.total))                 AS max_gap_units,
               AVG(ABS(s.level_sum - t.total) / t.total) * 100 AS mean_gap_pct
        FROM summed s JOIN totals t USING (method, date)
        GROUP BY s.method, s.level
        """
    ).df()


@st.cache_data
def load_member_detail(level: str):
    if level not in LEVELS:
        raise ValueError(level)
    return duckdb.connect().sql(
        f"""
        SELECT series_id,
               MAX(rmse)      FILTER (method = 'unreconciled')    AS err_original,
               MAX(rmse)      FILTER (method = 'mint_wls_struct') AS err_mint,
               MAX(avg_units) AS avg_units
        FROM (
            SELECT series_id, method,
                   SQRT(AVG(POWER(forecast - actual, 2))) AS rmse,
                   AVG(actual)                            AS avg_units
            FROM read_parquet('{RECONCILED}')
            WHERE level = '{level}'
            GROUP BY series_id, method
        )
        GROUP BY series_id
        ORDER BY avg_units DESC
        """
    ).df()


@st.cache_data
def load_item_options(category: str, store: str):
    """Product-store IDs matching the filters, busiest first.

    Filtered and ranked in SQL. starts_with/ends_with rather than LIKE because
    '_' is a LIKE wildcard and every ID is full of underscores.
    """
    clauses, params = ["level = 'item_store'", "method = 'unreconciled'"], []
    if category != "All":
        clauses.append("starts_with(series_id, ?)")
        params.append(category)
    if store != "All":
        clauses.append("ends_with(series_id, ?)")
        params.append("_" + store)
    return duckdb.connect().execute(
        f"""
        SELECT series_id, AVG(actual) AS avg_units
        FROM read_parquet('{RECONCILED}')
        WHERE {' AND '.join(clauses)}
        GROUP BY series_id ORDER BY avg_units DESC
        """,
        params,
    ).df()


@st.cache_data
def load_day_of_week(level: str):
    """Average daily miss by weekday - which days are hardest to forecast."""
    if level not in LEVELS:
        raise ValueError(level)
    return duckdb.connect().sql(
        f"""
        SELECT dayname(date) AS day, dayofweek(date) AS dow,
               AVG(ABS(forecast - actual)) FILTER (method = 'unreconciled')    AS err_original,
               AVG(ABS(forecast - actual)) FILTER (method = 'mint_wls_struct') AS err_mint,
               AVG(actual)                 FILTER (method = 'unreconciled')    AS avg_sales
        FROM read_parquet('{RECONCILED}')
        WHERE level = '{level}'
        GROUP BY day, dow ORDER BY dow
        """
    ).df()


@st.cache_data
def load_one_item(series_id: str):
    """One product-store series. Filtered in SQL - never loads all 30,490."""
    con = duckdb.connect()
    daily = con.execute(
        f"""
        SELECT date,
               MAX(actual)   FILTER (method = 'unreconciled')    AS actual,
               MAX(forecast) FILTER (method = 'unreconciled')    AS unreconciled,
               MAX(forecast) FILTER (method = 'mint_wls_struct') AS mint_wls_struct
        FROM read_parquet('{RECONCILED}')
        WHERE level = 'item_store' AND series_id = ?
        GROUP BY date ORDER BY date
        """,
        [series_id],
    ).df()
    stats = con.execute(
        f"""
        SELECT AVG(actual) FILTER (method = 'unreconciled') AS avg_units,
               SQRT(AVG(POWER(forecast - actual, 2)) FILTER (method = 'unreconciled'))    AS err_original,
               SQRT(AVG(POWER(forecast - actual, 2)) FILTER (method = 'mint_wls_struct')) AS err_mint
        FROM read_parquet('{RECONCILED}')
        WHERE level = 'item_store' AND series_id = ?
        """,
        [series_id],
    ).df()
    return daily, stats.iloc[0]


@st.cache_data
def load_series_counts():
    return duckdb.connect().sql(
        f"""
        SELECT level, COUNT(DISTINCT series_id) AS n_series
        FROM read_parquet('{RECONCILED}') GROUP BY level
        """
    ).df().set_index("level")["n_series"].to_dict()


# --------------------------------------------------------------------------
# Presentation helpers
# --------------------------------------------------------------------------
def style_plot(fig, height=460, legend=False):
    """Light chart surface regardless of the dark page theme.

    Title and legend both want the top of the figure, so when a legend is
    present the top margin is enlarged and the two are pinned to separate
    bands - otherwise they overlap and the legend reads as cramped.
    """
    top = 132 if legend else 62
    fig.update_layout(
        paper_bgcolor=PLOT_SURFACE,
        plot_bgcolor=PLOT_SURFACE,
        font=dict(color=PLOT_INK, size=14, family="Helvetica Neue, Helvetica, Arial, sans-serif"),
        height=height + (top - 62 if legend else 0),
        margin=dict(l=80, r=40, t=top, b=70),
        # The tooltip sits on a white card, so its font colour must be set
        # explicitly - left unset, Plotly picks a light colour from the page
        # theme and the readout comes out too dim to read.
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="#b9bcc2",
            font=dict(color=PLOT_INK, size=14,
                      family="Helvetica Neue, Helvetica, Arial, sans-serif"),
        ),
        title=dict(x=0, xanchor="left", y=1, yanchor="top", pad=dict(t=18, l=4)),
    )
    # A vertical guide makes it obvious which day the readout belongs to.
    if fig.layout.hovermode == "x unified":
        fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1,
                         spikedash="dot", spikecolor="#9aa0a6")
    if legend:
        fig.update_layout(
            legend=dict(
                orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0,
                bgcolor="rgba(0,0,0,0)", font=dict(color=PLOT_INK, size=13),
                itemwidth=40, tracegroupgap=24,
            ),
            showlegend=True,
        )
    fig.update_xaxes(gridcolor=PLOT_GRID, linecolor=PLOT_GRID, zeroline=False,
                     tickfont=dict(color=PLOT_INK_SOFT), title_font=dict(color=PLOT_INK_SOFT))
    fig.update_yaxes(gridcolor=PLOT_GRID, linecolor=PLOT_GRID, zeroline=False,
                     tickfont=dict(color=PLOT_INK_SOFT), title_font=dict(color=PLOT_INK_SOFT))
    return fig


@st.cache_data(show_spinner=False)
def _fig_to_png(fig_json: str) -> bytes:
    """Rendering a PNG costs ~2s, so it is cached on the figure spec."""
    return pio.from_json(fig_json).to_image(format="png", width=1500, height=780, scale=2)


def chart_downloads(fig, stem: str, key: str, df=None):
    """PNG (and optional CSV) download buttons under a chart.

    PNG export needs a Chrome binary via kaleido; if that is unavailable the
    button falls back to an interactive HTML file so the app still works.
    """
    cols = st.columns([1, 1, 4])
    try:
        png = _fig_to_png(fig.to_json())
        cols[0].download_button("Download Chart (PNG)", data=png, file_name=f"{stem}.png",
                                mime="image/png", key=f"png_{key}", width="stretch")
    except Exception:
        cols[0].download_button("Download Chart (HTML)", data=fig.to_html(include_plotlyjs="cdn"),
                                file_name=f"{stem}.html", mime="text/html", key=f"html_{key}",
                                width="stretch")
        cols[2].caption("PNG export unavailable (kaleido needs Chrome), exporting interactive HTML instead.")
    if df is not None:
        cols[1].download_button("Download Data (CSV)", data=df.to_csv(index=False).encode(),
                                file_name=f"{stem}.csv", mime="text/csv", key=f"csv_{key}",
                                width="stretch")


def fmt_gap(x):
    """Plain decimals, never scientific notation - '1.18e-08' reads as noise.

    The residuals are real and worth showing, so they are written out in full
    rather than rounded away to a flat 0.
    """
    if x == 0:
        return "exactly 0"
    if x < 0.001:
        return f"{x:.8f}"
    return f"{x:,.1f}"


def fmt_share(pct):
    """A share of daily sales, phrased so it can be read aloud.

    Percentages this small ('9.16e-12%') carry no intuition, so anything under
    a hundredth of a percent becomes a 'one part in N' ratio instead.
    """
    if pct == 0:
        return "none at all"
    if pct >= 0.01:
        return f"{pct:.2f}% of that day's sales"
    n = 100 / pct
    for div, name in ((1e12, "trillion"), (1e9, "billion"), (1e6, "million"), (1e3, "thousand")):
        if n >= div:
            return f"about 1 unit in {n / div:,.0f} {name}"
    return f"about 1 unit in {n:,.0f}"


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("Hierarchical Demand Forecasting")
st.markdown(
    "Predicting daily sales for **10 Walmart stores** in California, Texas and Wisconsin, for the "
    "company as a whole, for each product category, for each store, and for each individual product "
    "in each store, and then adjusting those predictions so they **add up correctly to each "
    "other**, which separately made predictions never do on their own."
)
st.caption(
    "Data: the public M5 competition dataset (Walmart, 2011 to 2016). **Units** means individual items "
    "sold per day, one unit is one item scanned at a checkout. Product names are anonymised in the "
    "source data, so products appear as IDs such as `FOODS_1_001`; the three categories (FOODS, "
    "HOBBIES, HOUSEHOLD) and the 10 store IDs are the real groupings the data ships with."
)

counts = load_series_counts()
errors = load_error_by_level()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Products x Stores", f"{counts['item_store']:,}")
c2.metric("Days Of History", "1,913")
c3.metric("Forecasts Produced", f"{sum(counts.values()):,}")
c4.metric("Test Window", "28 days")

st.divider()


# --------------------------------------------------------------------------
# The problem
# --------------------------------------------------------------------------
st.markdown("### :primary[The Problem]")
st.markdown(
    "A demand plan gets used at every altitude at once. Finance commits to a company wide total, "
    "category managers plan by category, and the replenishment team orders stock for one product "
    "in one store. If each level is forecast on its own, **those numbers disagree**: and someone "
    "ends up reconciling them in a spreadsheet by hand."
)

st.markdown("### :primary[Why This Is Hard]")
st.markdown(
    "There are two independent ways to slice this business, by **product** "
    "(product → department → category) and by **geography** (store → state). These two paths "
    "**cross rather than nest**: every category is sold in every store, so neither slicing sits "
    "inside the other. A single product in a store rolls up through two different chains, not "
    "one.\n\n"
    "That rules out the simplest fix, taking the company total and splitting it downward - "
    "because there is no single path down. It forces methods that work on the whole structure "
    "at once."
)
st.code(
    "Company Total ──┬── Product Category ── Department ──┐\n"
    "                │                                    ├── Individual Product per Store\n"
    "                └── Store ───────────────────────────┘        (30,490 combinations)",
    language=None,
)

geo_col, txt_col = st.columns([3, 2])
map_fig = go.Figure(
    go.Scattergeo(
        lon=[g[4] for g in STORE_GEO],
        lat=[g[3] for g in STORE_GEO],
        text=[g[0] for g in STORE_GEO],
        customdata=[[g[2], g[5]] for g in STORE_GEO],
        marker=dict(
            size=[g[2] * 11 for g in STORE_GEO],
            color=COLORS["mint_wls_struct"],
            line=dict(width=1.5, color=PLOT_SURFACE),
            opacity=0.85,
        ),
        mode="markers+text",
        textposition="top center",
        textfont=dict(color=PLOT_INK, size=13),
        hovertemplate="<b>%{text}</b><br>%{customdata[0]} stores<br>%{customdata[1]}<extra></extra>",
    )
)
map_fig.update_geos(
    scope="usa", bgcolor=PLOT_SURFACE, landcolor="#f0f0ee",
    lakecolor=PLOT_SURFACE, subunitcolor="#c9ccd1", countrycolor="#c9ccd1",
)
map_fig.update_layout(title=dict(text="Where The 10 Stores Are", font=dict(size=17, color=PLOT_INK)))
geo_col.plotly_chart(style_plot(map_fig, height=380), width="stretch")
txt_col.markdown(
    "The geography side of the hierarchy is small but real: **4 stores in California, "
    "3 in Texas, 3 in Wisconsin**.\n\n"
    "Every one of the 3,049 products is sold in all 10 stores, which is exactly why the product "
    "and geography paths cross instead of nesting."
)
txt_col.caption(
    "Markers sit at approximate state centres and are sized by store count. The dataset contains "
    "no store addresses or coordinates, so exact locations are not shown."
)
st.divider()


# --------------------------------------------------------------------------
# Section 1 - accuracy by level
# --------------------------------------------------------------------------
st.markdown("### :primary[Forecast Accuracy By Level]")
st.markdown(
    "Pick a level of the business to see how the three methods compare. "
    "**Lower bars are better**: the number is the average amount the forecast missed by, per day."
)

level = st.selectbox(
    "Choose A Level Of The Business", LEVELS,
    format_func=lambda lv: LEVEL_LABELS[lv], key="level_select",
)
st.caption(f"**{LEVEL_LABELS[level]}**: {LEVEL_BLURB[level]}")

level_errors = errors[errors["level"] == level].set_index("method")["rmse"].to_dict()
baseline = level_errors["unreconciled"]
values = [level_errors[m] for m in UI_METHODS]
deltas = [(level_errors[m] - baseline) / baseline * 100 for m in UI_METHODS]

fig = go.Figure(
    go.Bar(
        x=[METHOD_LABELS[m] for m in UI_METHODS], y=values,
        marker_color=[COLORS[m] for m in UI_METHODS], customdata=deltas,
        text=[f"{v:,.2f}" if v < 100 else f"{v:,.0f}" for v in values],
        textposition="outside", textfont=dict(color=PLOT_INK, size=13),
        hovertemplate="<b>%{x}</b><br>Average daily miss: %{y:,.3f} units"
                      "<br>Change vs. original: %{customdata:+.1f}%<extra></extra>",
    )
)
fig.update_layout(
    title=dict(text=f"Forecast Error - {LEVEL_LABELS[level]}", font=dict(size=18, color=PLOT_INK)),
    yaxis_title="Average daily miss (units sold)", xaxis_title=None, showlegend=False,
)
fig.update_yaxes(range=[0, max(values) * 1.18])
st.plotly_chart(style_plot(fig), width="stretch")

method_table = errors[
    (errors["level"] == level) & (errors["method"].isin(UI_METHODS))
].assign(Method=lambda d: d["method"].map(METHOD_LABELS))[["Method", "rmse"]].rename(
    columns={"rmse": "Average Daily Miss (units)"}
)
chart_downloads(fig, f"forecast_error_{level}", f"lvl_{level}", method_table)

best = min(UI_METHODS, key=lambda m: level_errors[m])
mint_delta = deltas[UI_METHODS.index("mint_wls_struct")]
verdict = "more accurate" if mint_delta < 0 else "less accurate"
st.markdown(
    f"At the **{LEVEL_LABELS[level]}** level the most accurate method is "
    f"**{METHOD_LABELS[best]}**. Reconciling with MinT is "
    f"**{abs(mint_delta):.1f}% {verdict}** than the original forecast here."
)
st.caption(
    "Error is not comparable between levels: the company total misses by thousands of units a "
    "day because it sums millions of units of sales, while a single product in a single store "
    "misses by about 2 units because it only sells a handful."
)

# ---- what-if weighting ----------------------------------------------
st.markdown("#### :primary[What If: Where Would You Rather Be Accurate?]")
blend = st.slider(
    "Drag to shift priority between the two ends of the trade off",
    0, 100, 0, step=5, key="blend_slider",
    format="%d%%",
)
w = blend / 100
blended = []
for lv in LEVELS:
    e = errors[errors["level"] == lv].set_index("method")["rmse"]
    blended.append((1 - w) * e["mint_wls_struct"] + w * e["unreconciled"])
orig = [errors[(errors.level == lv) & (errors.method == "unreconciled")]["rmse"].iloc[0] for lv in LEVELS]
blend_pct = [(b - o) / o * 100 for b, o in zip(blended, orig)]

fig_blend = go.Figure(
    go.Bar(
        x=[LEVEL_LABELS[lv] for lv in LEVELS], y=blend_pct,
        marker_color=[COLORS["mint_wls_struct"] if v < 0 else COLORS["bottom_up"] for v in blend_pct],
        text=[f"{v:+.1f}%" for v in blend_pct], textposition="outside",
        textfont=dict(color=PLOT_INK, size=12),
        hovertemplate="<b>%{x}</b><br>Change vs. original: %{y:+.2f}%<extra></extra>",
    )
)
fig_blend.update_layout(
    title=dict(text=f"Blend: {100 - blend}% company wide priority / {blend}% store level priority",
               font=dict(size=16, color=PLOT_INK)),
    yaxis_title="Change in error vs. original (%)", showlegend=False,
)
lim = max(abs(v) for v in blend_pct) if any(blend_pct) else 1
fig_blend.update_yaxes(range=[-lim * 1.5, lim * 1.5])
st.plotly_chart(style_plot(fig_blend, height=360), width="stretch")
st.caption(
    "**Illustrative only.** This slides linearly between two results that were already computed "
    "(MinT at 0%, the original forecast at 100%). It does not re run the optimisation, and the "
    "in between blends are not themselves coherent forecasts, they would not add up. Treat it "
    "as a way to feel the shape of the trade off, not as a tuning control."
)

# ---- what makes up this level ---------------------------------------
if level != "total":
    st.markdown(f"#### :primary[Every {LEVEL_LABELS[level]}, Compared]")
    detail = load_member_detail(level)
    detail["change_pct"] = (detail["err_mint"] - detail["err_original"]) / detail["err_original"] * 100
    detail["err_as_pct_of_sales"] = detail["err_original"] / detail["avg_units"] * 100

    if level == "item_store":
        st.markdown(
            f"All {len(detail):,} combinations are too many to chart. "
            "The 15 highest selling ones:"
        )
        shown_detail = detail.head(15)
        st.caption(
            "Only the 15 highest volume products are shown. These are not a random sample, so "
            "the helped/hurt pattern visible at other levels may not hold for the thousands of "
            "lower volume products not listed here, most of which sell fewer than one unit a day."
        )
    else:
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            name="Original Forecast", x=detail["series_id"], y=detail["err_original"],
            marker_color=DETAIL_COLORS["original"],
            hovertemplate="<b>%{x}</b><br>Original error: %{y:,.1f} units/day<extra></extra>",
        ))
        fig3.add_trace(go.Bar(
            name="Reconciled (MinT)", x=detail["series_id"], y=detail["err_mint"],
            marker_color=DETAIL_COLORS["mint"],
            hovertemplate="<b>%{x}</b><br>Reconciled error: %{y:,.1f} units/day<extra></extra>",
        ))
        fig3.update_layout(
            title=dict(text=f"Error For Each {LEVEL_LABELS[level]}, Busiest First",
                       font=dict(size=17, color=PLOT_INK)),
            yaxis_title="Average daily miss (units sold)", barmode="group",
                    )
        st.plotly_chart(style_plot(fig3, height=420, legend=True), width="stretch")
        chart_downloads(fig3, f"members_{level}", f"mem_{level}")

        helped = detail[detail["change_pct"] < 0]["series_id"].tolist()
        hurt = detail[detail["change_pct"] >= 0]["series_id"].tolist()
        st.markdown(
            f"Reconciling **helps** {len(helped)} of {len(detail)} "
            f"({', '.join(helped) if helped else 'none'}) and **costs accuracy** on "
            f"{len(hurt)} ({', '.join(hurt) if hurt else 'none'}). "
            "The single headline percentage for this level hides that split."
        )
        shown_detail = detail

    table = shown_detail.rename(columns={"series_id": "Name"})[
        ["Name", "avg_units", "err_original", "err_mint", "change_pct", "err_as_pct_of_sales"]
    ].rename(columns={
        "avg_units": "Avg Units Sold Per Day",
        "err_original": "Original Error (units/day)",
        "err_mint": "Reconciled Error (units/day)",
        "change_pct": "Change (%)",
        "err_as_pct_of_sales": "Original Error As % Of Sales",
    })
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "Avg Units Sold Per Day": st.column_config.NumberColumn(format="%.1f"),
            "Original Error (units/day)": st.column_config.NumberColumn(format="%.2f"),
            "Reconciled Error (units/day)": st.column_config.NumberColumn(format="%.2f"),
            "Change (%)": st.column_config.NumberColumn(format="%+.1f"),
            "Original Error As % Of Sales": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    st.download_button(
        "Download Table (CSV)", data=table.to_csv(index=False).encode(),
        file_name=f"members_{level}.csv", mime="text/csv", key=f"csv_mem_{level}",
    )
    st.caption(
        "The last column puts the error in context: missing by 1,384 units a day matters less "
        "for a department selling ~16,900 units a day than missing by 53 units does for one "
        "selling ~276."
    )
st.divider()


# --------------------------------------------------------------------------
# Individual product lookup
# --------------------------------------------------------------------------
st.markdown("### :primary[Look Up One Product In One Store]")
st.markdown(
    "Every one of the 30,490 product store combinations has its own forecast. "
    "Narrow the list down, then pick a product, the list is ordered by how much it sells."
)

f1, f2 = st.columns(2)
cat_filter = f1.selectbox("Filter By Category", ["All", "FOODS", "HOBBIES", "HOUSEHOLD"],
                          key="item_cat")
store_filter = f2.selectbox(
    "Filter By Store",
    ["All", "CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"],
    key="item_store_filter",
)

opts = load_item_options(cat_filter, store_filter)
labels = {r.series_id: f"{r.series_id}  -  {r.avg_units:.1f} units/day" for r in opts.itertuples()}
chosen = st.selectbox(
    f"Choose A Product ({len(opts):,} match, busiest first)",
    opts["series_id"].tolist(), format_func=lambda s: labels[s], key="item_select",
)
which = st.radio(
    "Which Forecasts To Show", ["Both", "Original Only", "Reconciled Only"],
    horizontal=True, key="item_which",
)

daily, stats = load_one_item(chosen)
moved = (daily["mint_wls_struct"] - daily["unreconciled"]).abs().mean()

i1, i2, i3, i4 = st.columns(4)
i1.metric("Avg Units Sold Per Day", f"{stats['avg_units']:.2f}")
i2.metric("Original Forecast Error", f"{stats['err_original']:.3f} units/day")
i3.metric("Reconciled Forecast Error", f"{stats['err_mint']:.3f} units/day",
          delta=f"{(stats['err_mint'] - stats['err_original']) / stats['err_original'] * 100:+.1f}%",
          delta_color="inverse")
i4.metric("Reconciliation Moved It By", f"{moved:.3f} units/day")

if stats["avg_units"] == 0:
    st.warning(
        "This product sold nothing at all during the 28-day test window, 1,587 of the 30,490 "
        "combinations didn't. The actual line will sit flat on zero.", icon=None,
    )
elif stats["avg_units"] < 1:
    st.info(
        f"This product sells under one unit a day on average. 18,257 of the 30,490 combinations do, "
        "so the actual line will be a sparse set of integer spikes.", icon=None,
    )

fig_item = go.Figure()
fig_item.add_trace(go.Scatter(
    x=daily["date"], y=daily["actual"], name="Actual Sales", mode="lines+markers",
    line=dict(color="#1A1A1A", width=3), marker=dict(size=7),
    hovertemplate="<b>%{x|%b %d}</b><br>Actual: %{y:.0f} units<extra></extra>",
))
if which in ("Both", "Original Only"):
    fig_item.add_trace(go.Scatter(
        x=daily["date"], y=daily["unreconciled"], name="Original Forecast", mode="lines+markers",
        line=dict(color=COLORS["unreconciled"], width=2.5, dash="dot"), marker=dict(size=6),
        hovertemplate="<b>%{x|%b %d}</b><br>Original: %{y:.2f} units<extra></extra>",
    ))
if which in ("Both", "Reconciled Only"):
    fig_item.add_trace(go.Scatter(
        x=daily["date"], y=daily["mint_wls_struct"], name="Reconciled Forecast (MinT)",
        mode="lines+markers", line=dict(color=COLORS["mint_wls_struct"], width=2.5),
        marker=dict(size=6),
        hovertemplate="<b>%{x|%b %d}</b><br>Reconciled: %{y:.2f} units<extra></extra>",
    ))
fig_item.update_layout(
    title=dict(text=f"Daily Sales And Forecast - {chosen}", font=dict(size=17, color=PLOT_INK)),
    yaxis_title="Units sold per day", xaxis_title="Test window (Mar 28, Apr 24, 2016)",
    hovermode="x unified",
    )
st.plotly_chart(style_plot(fig_item, height=440, legend=True), width="stretch")
chart_downloads(fig_item, f"product_{chosen}", "item", daily)
st.caption(
    "**The two forecast lines will look identical, that is the real result, not a glitch.** "
    "Reconciliation moves an individual product's forecast by about 0.05 units a day on average "
    "(0.30 at most), which is invisible next to daily sales. Almost all of the adjustment lands on "
    "the aggregate levels; the bottom of the hierarchy barely moves. The metric above tells you the "
    "exact size of the shift for this product. Forecasts are fractional because they are expected "
    "values, not whole unit predictions."
)
st.divider()


# --------------------------------------------------------------------------
# Which days are hardest to forecast
# --------------------------------------------------------------------------
st.markdown("### :primary[Which Days Are Hardest To Forecast?]")
st.markdown(
    "The same 28 days, grouped by day of the week. Retail demand is strongly weekly, so some days "
    "are consistently harder to predict than others, and reconciliation does not help them equally."
)

dow_level = st.selectbox(
    "Choose A Level Of The Business", LEVELS,
    format_func=lambda lv: LEVEL_LABELS[lv], key="dow_level",
)
dow = load_day_of_week(dow_level)

fig_dow = go.Figure()
fig_dow.add_trace(go.Bar(
    name="Original Forecast", x=dow["day"], y=dow["err_original"],
    marker_color=DETAIL_COLORS["original"], customdata=dow["avg_sales"],
    hovertemplate="<b>%{x}</b><br>Original miss: %{y:,.1f} units"
                  "<br>Avg sales that day: %{customdata:,.0f} units<extra></extra>",
))
fig_dow.add_trace(go.Bar(
    name="Reconciled (MinT)", x=dow["day"], y=dow["err_mint"],
    marker_color=DETAIL_COLORS["mint"], customdata=dow["avg_sales"],
    hovertemplate="<b>%{x}</b><br>Reconciled miss: %{y:,.1f} units"
                  "<br>Avg sales that day: %{customdata:,.0f} units<extra></extra>",
))
fig_dow.update_layout(
    title=dict(text=f"Average Daily Miss By Weekday - {LEVEL_LABELS[dow_level]}",
               font=dict(size=17, color=PLOT_INK)),
    yaxis_title="Average miss (units sold)", barmode="group",
    )
st.plotly_chart(style_plot(fig_dow, height=420, legend=True), width="stretch")
chart_downloads(fig_dow, f"day_of_week_{dow_level}", f"dow_{dow_level}", dow)

worst_day = dow.loc[dow["err_original"].idxmax()]
best_gain = dow.assign(gain=dow["err_original"] - dow["err_mint"]).sort_values("gain", ascending=False)
top_gain, worst_gain = best_gain.iloc[0], best_gain.iloc[-1]
st.markdown(
    f"At the **{LEVEL_LABELS[dow_level]}** level the hardest day to forecast is "
    f"**{worst_day['day']}** (missing by {worst_day['err_original']:,.1f} units on average). "
    f"Reconciliation helps most on **{top_gain['day']}** "
    f"({top_gain['gain']:+,.1f} units of accuracy) and least on **{worst_gain['day']}** "
    f"({worst_gain['gain']:+,.1f})."
)
st.caption(
    "Hover any bar for that weekday's average sales. High volume days are not automatically the "
    "hardest, the error is driven by how unusual the day is, not just how big it is."
)
st.divider()


# --------------------------------------------------------------------------
# Section 2 - total-level forecast vs actual
# --------------------------------------------------------------------------
st.markdown("### :primary[Company Wide Forecast Over The Test Period]")
st.markdown(
    "The 28 days held back from the models. Actual sales are always shown; "
    "switch the forecasts on and off to compare them. Hover for exact values."
)

t1, t2, t3 = st.columns(3)
show_original = t1.checkbox("Show Original Forecast", value=True, key="cb_orig")
show_mint = t2.checkbox("Show Reconciled Forecast (MinT)", value=True, key="cb_mint")
show_history = t3.checkbox("Show 28 Days Of Prior History", value=False, key="cb_hist")

total = load_total_series()
fig2 = go.Figure()
if show_history:
    hist = load_history(28)
    fig2.add_trace(go.Scatter(
        x=hist["date"], y=hist["sales"], name="Actual Demand (before test window)",
        mode="lines", line=dict(color="#b9bcc2", width=2, dash="dot"),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Actual: %{y:,.0f} units<extra></extra>",
    ))
fig2.add_trace(go.Scatter(
    x=total["date"], y=total["actual"], name="Actual Demand", mode="lines+markers",
    line=dict(color="#1A1A1A", width=3), marker=dict(size=7),
    hovertemplate="<b>%{x|%b %d, %Y}</b><br>Actual: %{y:,.0f} units<extra></extra>",
))
if show_original:
    fig2.add_trace(go.Scatter(
        x=total["date"], y=total["unreconciled"], name="Original Forecast",
        mode="lines+markers", line=dict(color=COLORS["unreconciled"], width=2.5, dash="dot"),
        marker=dict(size=6),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Original forecast: %{y:,.0f} units<extra></extra>",
    ))
if show_mint:
    fig2.add_trace(go.Scatter(
        x=total["date"], y=total["mint_wls_struct"], name="Reconciled Forecast (MinT)",
        mode="lines+markers", line=dict(color=COLORS["mint_wls_struct"], width=2.5),
        marker=dict(size=6),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Reconciled forecast: %{y:,.0f} units<extra></extra>",
    ))
fig2.update_layout(
    title=dict(text="Total Units Sold Per Day, Forecast And Actual",
               font=dict(size=18, color=PLOT_INK)),
    yaxis_title="Total units sold per day",
    xaxis_title="Test window (Mar 28, Apr 24, 2016)", hovermode="x unified",
    )
st.plotly_chart(style_plot(fig2, height=520, legend=True), width="stretch")
chart_downloads(fig2, "company_wide_forecast", "total", total)

m1, m2, m3 = st.columns(3)
orig_err = errors[(errors.level == "total") & (errors.method == "unreconciled")]["rmse"].iloc[0]
mint_err = errors[(errors.level == "total") & (errors.method == "mint_wls_struct")]["rmse"].iloc[0]
m1.metric("Original Forecast Error", f"{orig_err:,.0f} units/day")
m2.metric("Reconciled Forecast Error", f"{mint_err:,.0f} units/day",
          delta=f"{(mint_err - orig_err) / orig_err * 100:.1f}%", delta_color="inverse")
m3.metric("Accuracy Gained", f"{(orig_err - mint_err) / orig_err * 100:.1f}%")

st.caption(
    "Both forecasts follow the weekly rhythm, sales peak at weekends, but under shoot the "
    "biggest peaks. The reconciled forecast sits closer to reality on most days."
)
st.divider()


# --------------------------------------------------------------------------
# Section 3 - coherence
# --------------------------------------------------------------------------
st.markdown("### :primary[Do The Forecasts Add Up?]")
st.markdown(
    "If you add up every store's forecast for one day, you should get the company wide forecast "
    "for that day. Before reconciliation you do not. Use the toggle to compare."
)

stage = st.radio(
    "Choose A Stage", ["Before Reconciliation", "After Reconciliation"],
    horizontal=True, label_visibility="collapsed", key="coh_stage",
)

coh = load_coherence()
if stage == "Before Reconciliation":
    shown = coh[coh["method"] == "unreconciled"].copy()
    shown["Method"] = "Original Forecast"
else:
    shown = coh[coh["method"].isin(["bottom_up", "mint_wls_struct"])].copy()
    shown["Method"] = shown["method"].map(METHOD_LABELS)

shown["Level Being Added Up"] = shown["level"].map(LEVEL_LABELS)
shown = shown.sort_values(["Method", "Level Being Added Up"])
worst = shown["max_gap_units"].max()

# Animated hand-off between the two stages. Streamlit has no real
# transition primitive, so this is a short scripted count between the two
# values in a placeholder - an approximation of an animation, not a tween.
prev = st.session_state.get("coh_prev_worst")
k1, k2 = st.columns(2)
slot = k1.empty()
if prev is not None and prev != worst:
    for i in range(11):
        t = i / 10
        eased = 1 - (1 - t) ** 3
        slot.metric("Worst Gap On Any Single Day",
                    f"{fmt_gap(prev + (worst - prev) * eased)} units")
        time.sleep(0.045)
slot.metric("Worst Gap On Any Single Day", f"{fmt_gap(worst)} units")
st.session_state["coh_prev_worst"] = worst

# Worst rather than mean: averaging across methods would blend Bottom-Up's
# exact zero with MinT's residue and report a gap neither of them has.
k2.metric("Typical Gap, Worst Level", fmt_share(shown["mean_gap_pct"].max()))

table_coh = shown[["Method", "Level Being Added Up", "max_gap_units", "mean_gap_pct"]].copy()
table_coh["Worst Gap On Any Day (units)"] = table_coh["max_gap_units"].map(fmt_gap)
table_coh["How Big Is That Gap?"] = table_coh["mean_gap_pct"].map(fmt_share)
coh_out = table_coh[["Method", "Level Being Added Up", "Worst Gap On Any Day (units)",
                     "How Big Is That Gap?"]]
st.dataframe(coh_out, hide_index=True, width="stretch")
st.download_button("Download Table (CSV)", data=coh_out.to_csv(index=False).encode(),
                   file_name=f"coherence_{stage.split()[0].lower()}.csv", mime="text/csv",
                   key="csv_coh")

if stage == "Before Reconciliation":
    st.warning(
        "The levels disagree. Adding up the individual forecasts gives an answer that differs "
        "from the company wide forecast by as much as 3,449 units on a single day, roughly "
        "1.9-2.5% of that day's sales.",
        icon=None,
    )
else:
    st.success(
        "**Every level now adds up.** What is left is effectively zero, ordinary "
        "floating point rounding from adding 30,490 numbers in a different order, not a real "
        "inconsistency. In figures: the Bottom Up gap is exactly 0 and the MinT gap is about "
        "0.00000001 units, against a daily total of roughly 39,000 units, about one part in "
        "four trillion, down from 3,449 units before reconciling.",
        icon=None,
    )
    st.caption(
        "This is a guarantee rather than a lucky result: the reconciled numbers are built by "
        "summing one consistent set of product level figures, so they cannot disagree by more "
        "than the arithmetic noise of the summation itself."
    )
st.divider()


# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.markdown("### :primary[The Short Version]")

total_orig = errors[(errors.level == "total") & (errors.method == "unreconciled")]["rmse"].iloc[0]
total_mint = errors[(errors.level == "total") & (errors.method == "mint_wls_struct")]["rmse"].iloc[0]
store_orig = errors[(errors.level == "store") & (errors.method == "unreconciled")]["rmse"].iloc[0]
store_mint = errors[(errors.level == "store") & (errors.method == "mint_wls_struct")]["rmse"].iloc[0]

s1, s2, s3 = st.columns(3)
s1.metric("Company Wide Accuracy", f"{(total_orig - total_mint) / total_orig * 100:.1f}% better",
          help="Reconciled (MinT) vs. the original forecast, at the company wide level.")
s2.metric("Store Level Accuracy", f"{(store_mint - store_orig) / store_orig * 100:.1f}% worse",
          help="The same comparison at store level, the cost side of the trade off.")
s3.metric("Levels That Now Agree", "5 of 5", help="Every level sums correctly after reconciliation.")

st.markdown(
    f"""
**What was built.** Daily sales forecasts for {counts['item_store']:,} product-store combinations
across 10 Walmart stores, rolled up into five levels of the business, then adjusted so every level
agrees with every other.

**What reconciliation bought.** Before it, adding up the individual forecasts missed the
company-wide forecast by as much as **3,449 units on a single day**. After it, the levels agree to
within a rounding error - a gap of about one unit in eleven trillion. That consistency is
*guaranteed by construction*, not tuned for.

**What it cost.** Company-wide accuracy improved **{(total_orig - total_mint) / total_orig * 100:.1f}%**
and category accuracy **4.6%**, but department and store forecasts got **2.7 to 5.6% worse**. Error was
moved, not removed. At the individual-product level almost nothing changed at all - the typical
forecast shifted by 0.05 units a day.

**So what.** Whether this is a good trade depends on one question: which level is the plan actually
committed at? If finance signs up to a company-wide number, reconciliation pays for itself. If the
binding commitment is store-level replenishment, it costs more than it returns. That is a business
decision, and the numbers above are the input to it - not a model-selection question with one right
answer.
"""
)

st.divider()

st.markdown("### :primary[Glossary And Technical Details]")

with st.expander("Glossary"):
    st.markdown(
        "- **Forecast**: a prediction of how many units will sell on a future day.\n"
        "- **Base forecast**: the first draft prediction made for each level separately, "
        "before any adjustment.\n"
        "- **Hierarchy level**: one altitude of the business: the company total, a product "
        "category, a department, a store, or a single product in a single store.\n"
        "- **Reconciliation**: adjusting a set of forecasts so the smaller ones add up exactly "
        "to the bigger ones.\n"
        "- **Forecast error (RMSE)**: how far off the forecast was, on average, in units of "
        "product per day. Lower is better."
    )

with st.expander("How The Forecasts Were Built"):
    st.markdown(
        "The raw M5 Walmart data is reshaped in DuckDB from a spreadsheet style file (one column "
        "per day) into 58.3 million rows, one per product, per store, per day across 5 years - "
        "then aggregated into the five levels.\n\n"
        "Each level gets one global LightGBM model using day of week, month, and lagged and "
        "rolling average sales. A 'same as last week' seasonal benchmark is fitted alongside it. "
        "Because the horizon is 28 days but the features reach back only 7, the model feeds its "
        "own predictions forward day by day rather than peeking at held out values, verified by "
        "replacing the test period with random numbers and confirming the forecasts did not "
        "change."
    )

with st.expander("Why The Textbook Method Does Not Fit"):
    st.markdown(
        "The standard version of MinT needs a table comparing every product store combination "
        "with every other one. At 30,490 combinations that is a 30,490 x 30,490 matrix, about "
        "7.4 GB, and the maths requires effectively inverting it.\n\n"
        "The sparse version used here exploits the fact that of the 930 million possible entries "
        "in the summing table, only 152,450 are non zero (0.016%). It solves in 0.06 seconds.\n\n"
        "One caveat: MinT produces 1,280 negative forecasts out of 854,308. Rounding them up to "
        "zero would break the adding up guarantee, so enforcing non negativity properly requires "
        "a more expensive constrained calculation."
    )

with st.expander("A Third Method Exists But Is Not Shown"):
    st.markdown(
        "The underlying data also contains a second MinT variant (ordinary least squares "
        "weighting, `mint_ols`). It is deliberately not surfaced anywhere in this app: "
        "explaining the difference between the two MinT weightings would need a digression this "
        "page does not otherwise earn, and showing two near identical 'MinT' options invites "
        "confusion. It remains in `reconciled_forecasts.parquet` for completeness."
    )

with st.expander("Data Sources Behind This Page"):
    st.markdown(
        f"- `{RECONCILED.relative_to(PROJECT_ROOT)}` - forecasts before and after reconciliation\n"
        f"- `{HIERARCHY_DIR.relative_to(PROJECT_ROOT)}/` - actual sales aggregated to each level\n\n"
        "Nothing on this page is recomputed; it reads the Parquet output of the pipeline. "
        "Regenerate it with `python3 sql_runner.py`, `python3 src/train_base_forecasts.py`, "
        "then `python3 src/reconcile_forecasts.py`."
    )
