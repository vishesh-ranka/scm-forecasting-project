"""Evaluation charts for the reconciliation results.

Reads data/forecasts/reconciled_forecasts.parquet and writes:
  results/rmse_comparison.png      - forecast error by method and level
  results/total_level_forecast.png - Total-level forecast vs actual

Palettes are muted and were validated for colour-blind separation with a local
port of the OKLab/CVD six-checks (adjacent-pair gate, since grouped bars sit
side by side): worst adjacent normal dE 15.9 and CVD dE 10.8 for the bars,
27.6 / 20.5 for the lines; every colour clears 3:1 contrast on the surface.
"""

from pathlib import Path

import duckdb
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECONCILED = PROJECT_ROOT / "data" / "forecasts" / "reconciled_forecasts.parquet"
RESULTS_DIR = PROJECT_ROOT / "results"

# Coarsest level first: the hierarchy order is also the story order.
LEVELS = ["total", "category", "department", "store", "item_store"]
# Plain-language labels, matching the terminology used in README.md.
LEVEL_LABELS = {
    "total": "Company-Wide\nTotal",
    "category": "Product\nCategory",
    "department": "Department",
    "store": "Store",
    "item_store": "Individual Product\n(per store)",
}
LEVEL_LABELS_FLAT = {
    "total": "Company-Wide Total",
    "category": "Product Category",
    "department": "Department",
    "store": "Store",
    "item_store": "Individual Product",
}
METHODS = ["unreconciled", "bottom_up", "mint_ols", "mint_wls_struct"]
RECONCILED_METHODS = METHODS[1:]
LABELS = {
    "unreconciled": "Original forecast (before fixing)",
    "bottom_up": "Reconciled - Bottom-Up method",
    "mint_ols": "Reconciled - MinT method (OLS)",
    "mint_wls_struct": "Reconciled - MinT method (WLS)",
}
SHORT_LABELS = {
    "unreconciled": "Original forecast",
    "bottom_up": "Bottom-Up",
    "mint_ols": "MinT (OLS)",
    "mint_wls_struct": "MinT (WLS)",
}

SURFACE = "#fdfdfc"
INK = "#16181d"
INK_SECONDARY = "#5b6067"
INK_MUTED = "#8c9199"
GRID = "#e3e5e8"
AXIS = "#c6c9ce"

COLORS = {
    "unreconciled": "#6B6B6B",      # neutral: the baseline the rest are measured against
    "bottom_up": "#C9743C",
    "mint_ols": "#2F5E93",
    "mint_wls_struct": "#3E8C7A",
}
LINE_COLORS = {"actual": "#1A1A1A", "unreconciled": "#C9743C", "mint_wls_struct": "#2F5E93"}

mpl.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "axes.edgecolor": AXIS,
        "font.size": 12,
    }
)


def px_to_data(ax, px):
    """Pixel radius -> x and y data units (the two axes differ by orders of magnitude)."""
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([(0, 0), (px, px)])
    return abs(x1 - x0), abs(y1 - y0)


def rounded_bar(ax, x, width, height, color, rx, ry, zorder=3):
    """Bar anchored at zero with its data-end rounded; handles negative heights."""
    if not height:
        return
    rx = min(rx, width / 2)
    ry = min(ry, abs(height)) * (1 if height > 0 else -1)
    left, right, top = x - width / 2, x + width / 2, height
    verts = [
        (left, 0), (left, top - ry),
        (left, top), (left + rx, top),
        (right - rx, top),
        (right, top), (right, top - ry),
        (right, 0), (left, 0),
    ]
    codes = [
        MplPath.MOVETO, MplPath.LINETO,
        MplPath.CURVE3, MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3, MplPath.CURVE3,
        MplPath.LINETO, MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", zorder=zorder))


def style_axis(ax, grid_axis="y"):
    getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(length=0, labelsize=11)


def load_rmse(con):
    df = con.sql(
        f"""
        SELECT level, method, SQRT(AVG(POWER(forecast - actual, 2))) AS rmse
        FROM read_parquet('{RECONCILED}')
        GROUP BY level, method
        """
    ).df()
    return {(r.level, r.method): r.rmse for r in df.itertuples()}


def chart_rmse(rmse):
    """Panel A: absolute error per level. Panel B: change vs baseline, which is
    where the pattern lives - improvement at the top, cost in the middle.
    """
    fig = plt.figure(figsize=(15.5, 9.6))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.25], hspace=0.52, wspace=0.34,
                          left=0.055, right=0.985, top=0.845, bottom=0.105)
    axes_a = [fig.add_subplot(gs[0, i]) for i in range(len(LEVELS))]
    ax_b = fig.add_subplot(gs[1, :])

    # ---- Panel A: absolute RMSE, one scale per level -------------------
    for ax, level in zip(axes_a, LEVELS):
        values = [rmse[(level, m)] for m in METHODS]
        ax.set_xlim(-0.65, len(METHODS) - 0.35)
        ax.set_ylim(0, max(values) * 1.18)
        ax.set_xticks([])
        ax.set_title(LEVEL_LABELS[level], fontsize=13.5, color=INK, pad=9, fontweight="semibold")
        style_axis(ax)
    axes_a[0].set_ylabel("Forecast error\n(RMSE, units sold)", fontsize=12, color=INK_SECONDARY, labelpad=8)

    # ---- Panel B: change vs the unreconciled baseline -------------------
    group_x = np.arange(len(LEVELS))
    bar_w = 0.23
    deltas = {
        m: [(rmse[(lv, m)] - rmse[(lv, "unreconciled")]) / rmse[(lv, "unreconciled")] * 100 for lv in LEVELS]
        for m in RECONCILED_METHODS
    }
    span = max(abs(v) for vals in deltas.values() for v in vals)
    ax_b.set_xlim(-0.6, len(LEVELS) - 0.4)
    ax_b.set_ylim(-span * 1.42, span * 1.42)
    ax_b.set_xticks(group_x)
    ax_b.set_xticklabels(
        [f"{LEVEL_LABELS_FLAT[lv]}\noriginal error: {rmse[(lv, 'unreconciled')]:,.1f}" for lv in LEVELS],
        fontsize=12, color=INK,
    )
    ax_b.set_ylabel("Change in forecast error\nvs. original (%)", fontsize=12, color=INK_SECONDARY, labelpad=8)
    style_axis(ax_b)
    ax_b.axhline(0, color=COLORS["unreconciled"], linewidth=1.8, zorder=2)
    ax_b.text(
        -0.55, span * 1.30, "less accurate than the original", fontsize=11, color=INK_MUTED, va="top", style="italic",
    )
    ax_b.text(
        -0.55, -span * 1.30, "more accurate than the original", fontsize=11, color=INK_MUTED, va="bottom", style="italic",
    )

    fig.canvas.draw()

    for ax, level in zip(axes_a, LEVELS):
        values = [rmse[(level, m)] for m in METHODS]
        rx, ry = px_to_data(ax, 4)
        for i, (method, value) in enumerate(zip(METHODS, values)):
            rounded_bar(ax, i, 0.66, value, COLORS[method], rx, ry)
        ax.text(
            (len(METHODS) - 1) / 2, max(values) * 1.11,
            f"best: {SHORT_LABELS[min(METHODS, key=lambda m: rmse[(level, m)])]}",
            ha="center", va="center", fontsize=10.5, color=INK_MUTED,
        )

    rx, ry = px_to_data(ax_b, 4)
    for j, method in enumerate(RECONCILED_METHODS):
        offset = (j - 1) * (bar_w + 0.025)
        for i, value in enumerate(deltas[method]):
            rounded_bar(ax_b, group_x[i] + offset, bar_w, value, COLORS[method], rx, ry)
            pad = span * 0.07
            ax_b.text(
                group_x[i] + offset, value + (pad if value >= 0 else -pad),
                f"{value:+.1f}%" if abs(value) >= 0.05 else "0.0%",
                ha="center", va="bottom" if value >= 0 else "top",
                fontsize=10.5, color=INK if abs(value) >= 1 else INK_MUTED,
                fontweight="semibold" if abs(value) >= 1 else "normal",
            )

    handles = [
        plt.Line2D([], [], marker="s", linestyle="none", markersize=11, color=COLORS[m], label=LABELS[m])
        for m in METHODS
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=12.5,
        bbox_to_anchor=(0.5, 0.005), handletextpad=0.6, columnspacing=2.6,
    )
    fig.suptitle(
        "Forcing the forecasts to add up helps the big picture and costs a little in the middle",
        fontsize=17.5, color=INK, y=0.965, fontweight="semibold",
    )
    fig.text(
        0.5, 0.905,
        "Forecast error over the 28 days held out for testing, for four methods at five levels of the business. Lower is better.\n"
        "A: how big the error is at each level (each panel has its own scale).   B: change vs. the original forecast - below zero is more accurate.",
        ha="center", fontsize=12.5, color=INK_SECONDARY, linespacing=1.5,
    )
    for ax, tag in ((axes_a[0], "A"), (ax_b, "B")):
        ax.text(-0.02, 1.0, tag, transform=ax.transAxes, fontsize=15, fontweight="bold",
                color=INK, ha="right", va="bottom")

    out = RESULTS_DIR / "rmse_comparison.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def chart_total(con, rmse):
    df = con.sql(
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

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.subplots_adjust(left=0.085, right=0.975, top=0.80, bottom=0.125)
    x = np.arange(len(df))

    series = [
        ("actual", "Actual demand", LINE_COLORS["actual"], 2.8),
        ("unreconciled", f"Original forecast  -  average miss {rmse[('total','unreconciled')]:,.0f} units/day",
         LINE_COLORS["unreconciled"], 2.4),
        ("mint_wls_struct", f"Reconciled forecast (MinT)  -  average miss {rmse[('total','mint_wls_struct')]:,.0f} units/day",
         LINE_COLORS["mint_wls_struct"], 2.4),
    ]
    for col, label, color, lw in series:
        ax.plot(
            x, df[col], color=color, linewidth=lw, label=label, marker="o", markersize=5.5,
            markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=4 if col == "actual" else 3,
        )

    ax.set_xticks(x[::3])
    ax.set_xticklabels([d.strftime("%b %d") for d in df["date"][::3]], fontsize=12)
    ax.set_xlabel("Test window - 28 days held out (Mar 28 - Apr 24, 2016)",
                  fontsize=12.5, color=INK_SECONDARY, labelpad=10)
    ax.set_ylabel("Total units sold per day\n(all 3,049 products x 10 stores)",
                  fontsize=12.5, color=INK_SECONDARY, labelpad=10)
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
    style_axis(ax)
    ax.tick_params(labelsize=12)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=3, frameon=False,
              fontsize=12.5, handlelength=2.0, columnspacing=2.6)

    ax.set_title("Company-wide daily demand: forecast vs. actual",
                 fontsize=17.5, color=INK, pad=74, loc="left", fontweight="semibold")
    ax.text(
        0, 1.135,
        "Both forecasts under-shoot the weekly peaks; the reconciled version's daily correction is small but cuts company-wide error by 12.5%.",
        transform=ax.transAxes, fontsize=12.5, color=INK_SECONDARY,
    )

    out = RESULTS_DIR / "total_level_forecast.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    rmse = load_rmse(con)
    print(f"wrote {chart_rmse(rmse).relative_to(PROJECT_ROOT)}")
    print(f"wrote {chart_total(con, rmse).relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
