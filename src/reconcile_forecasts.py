"""Hierarchical reconciliation of the LightGBM base forecasts.

Builds the crossed summing matrix S over the five staged hierarchy levels,
reconciles the 28-day LightGBM forecasts with bottom-up and MinT (trace
minimization), verifies coherence, and compares RMSE before vs after.

Uses the sparse reconcilers from hierarchicalforecast: with 30,490 bottom
series the dense estimators would need a 30,490 x 30,490 matrix (~7.4 GB) and
an O(n^3) inversion, which is why the covariance-based MinT variants
(mint_shrink / mint_cov) are not reachable at this scale. See the note printed
at the end of the run.
"""

import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from hierarchicalforecast.methods import BottomUpSparse, MinTraceSparse
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HIERARCHY_DIR = PROJECT_ROOT / "data" / "staged" / "hierarchy"
BASE_FORECASTS = PROJECT_ROOT / "data" / "forecasts" / "base_forecasts.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "forecasts" / "reconciled_forecasts.parquet"

BASE_MODEL = "lightgbm"
HORIZON = 28

# Aggregate levels, coarsest first. The bottom level (item_store) is appended
# after these, because the sparse reconcilers require the bottom series to
# occupy the final n_bottom rows of S (BottomUpSparse builds its P as
# eye(n_bottom, n_hiers, n_hiers - n_bottom)).
AGG_LEVELS = ["total", "category", "department", "store"]
BOTTOM_LEVEL = "item_store"


def load_bottom_keys(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """One row per bottom series, carrying its parent keys at every level."""
    bottom = con.sql(
        f"""
        SELECT DISTINCT item_id, store_id, cat_id, dept_id
        FROM read_parquet('{HIERARCHY_DIR / "item_store.parquet"}')
        ORDER BY item_id, store_id
        """
    ).df()
    bottom["series_id"] = bottom["item_id"] + "_" + bottom["store_id"]
    return bottom


def build_summing_matrix(bottom: pd.DataFrame):
    """Sparse S of shape (n_hiers, n_bottom) plus the aligned series index.

    Each bottom series contributes to exactly four aggregates - Total, its
    category, its department, its store - and to itself, so S holds
    5 * n_bottom nonzeros. The two paths are crossed rather than nested:
    category and store are both sums over bottom series, and neither is an
    ancestor of the other, so S is the stacked union of the product path and
    the geography path rather than a single tree.
    """
    n_bottom = len(bottom)
    cats = sorted(bottom["cat_id"].unique())
    depts = sorted(bottom["dept_id"].unique())
    stores = sorted(bottom["store_id"].unique())

    # Row layout: Total, categories, departments, stores, then the bottom block.
    offsets = {}
    cursor = 0
    for name, members in [
        ("total", ["total"]),
        ("category", cats),
        ("department", depts),
        ("store", stores),
    ]:
        offsets[name] = (cursor, members)
        cursor += len(members)
    n_agg = cursor

    col = np.arange(n_bottom)
    agg_rows = [
        np.zeros(n_bottom, dtype=np.int64),  # Total: every bottom series
        offsets["category"][0] + bottom["cat_id"].map({c: i for i, c in enumerate(cats)}).to_numpy(),
        offsets["department"][0] + bottom["dept_id"].map({d: i for i, d in enumerate(depts)}).to_numpy(),
        offsets["store"][0] + bottom["store_id"].map({s: i for i, s in enumerate(stores)}).to_numpy(),
        n_agg + col,  # identity block
    ]
    rows = np.concatenate(agg_rows)
    cols = np.tile(col, len(agg_rows))
    S = sparse.csr_matrix(
        (np.ones(rows.size, dtype=np.float64), (rows, cols)),
        shape=(n_agg + n_bottom, n_bottom),
    )

    index = pd.DataFrame(
        {
            "level": np.concatenate(
                [np.repeat(name, len(members)) for name, (_, members) in offsets.items()]
                + [np.repeat(BOTTOM_LEVEL, n_bottom)]
            ),
            "series_id": np.concatenate(
                [np.asarray(members, dtype=object) for _, (_, members) in offsets.items()]
                + [bottom["series_id"].to_numpy()]
            ),
        }
    )
    index["row_idx"] = np.arange(len(index))
    tags = {name: index.loc[index["level"] == name, "series_id"].to_numpy() for name in index["level"].unique()}
    return S, index, tags, n_agg


def load_base_forecasts(con: duckdb.DuckDBPyConnection, index: pd.DataFrame):
    """Base forecasts and actuals as (n_hiers, HORIZON) matrices, S-aligned."""
    con.register("series_index", index)
    rows = con.sql(
        f"""
        SELECT f.forecast, f.actual
        FROM read_parquet('{BASE_FORECASTS}') f
        JOIN series_index i USING (level, series_id)
        WHERE f.model = '{BASE_MODEL}'
        ORDER BY i.row_idx, f.date
        """
    ).fetchnumpy()

    expected = len(index) * HORIZON
    if len(rows["forecast"]) != expected:
        raise ValueError(f"expected {expected} base forecast rows, got {len(rows['forecast'])}")

    y_hat = np.asarray(rows["forecast"], dtype=np.float64).reshape(len(index), HORIZON)
    y_true = np.asarray(rows["actual"], dtype=np.float64).reshape(len(index), HORIZON)
    dates = con.sql(
        f"SELECT DISTINCT date FROM read_parquet('{BASE_FORECASTS}') ORDER BY date"
    ).fetchnumpy()["date"]
    return y_hat, y_true, dates


def coherence_gap(matrix, index, n_agg):
    """Mean absolute daily gap, in %, between each level's sum and Total."""
    total = matrix[0]
    gaps = {}
    for level in AGG_LEVELS[1:] + [BOTTOM_LEVEL]:
        mask = (index["level"] == level).to_numpy()
        gaps[level] = float(np.mean(np.abs(matrix[mask].sum(axis=0) - total) / np.abs(total)) * 100)
    return gaps


def rmse_by_level(pred, y_true, index):
    out = {}
    for level in AGG_LEVELS + [BOTTOM_LEVEL]:
        mask = (index["level"] == level).to_numpy()
        out[level] = float(np.sqrt(np.mean((pred[mask] - y_true[mask]) ** 2)))
    return out


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    bottom = load_bottom_keys(con)
    S, index, tags, n_agg = build_summing_matrix(bottom)
    y_hat, y_true, dates = load_base_forecasts(con, index)
    n_hiers, n_bottom = S.shape
    print(f"S: {n_hiers:,} x {n_bottom:,}  ({S.nnz:,} nonzeros, {S.nnz / (n_hiers * n_bottom) * 100:.4f}% dense)")
    print(f"aggregate series: {n_agg}   bottom series: {n_bottom:,}\n")

    # Bottom-up reproduces the base bottom-level forecasts and sums them up.
    # MinT solves min tr(SG W G'S') subject to coherence; the sparse variant
    # solves the normal equations iteratively instead of inverting S'W^-1 S.
    reconcilers = {
        "bottom_up": BottomUpSparse(),
        "mint_ols": MinTraceSparse(method="ols"),
        "mint_wls_struct": MinTraceSparse(method="wls_struct"),
    }

    reconciled = {"unreconciled": y_hat}
    for name, rec in reconcilers.items():
        started = time.perf_counter()
        reconciled[name] = rec.fit_predict(S=S, y_hat=y_hat, tags=tags)["mean"]
        print(f"  {name:<16} {time.perf_counter() - started:6.2f}s")

    # --- coherence -------------------------------------------------------
    # Every reconciled vector is S @ b for some bottom vector b, so coherence
    # should hold to floating point regardless of the iterative solver's
    # tolerance. Checked directly against S rather than assumed.
    print(f"\n{'method':<16}{'max |S@bottom - y|':>22}{'   per-level gap vs Total (mean abs %)'}")
    print("-" * 96)
    for name, mat in reconciled.items():
        residual = float(np.abs(S @ mat[n_agg:] - mat).max())
        gaps = coherence_gap(mat, index, n_agg)
        gap_str = "  ".join(f"{lvl[:4]}={g:6.3f}%" for lvl, g in gaps.items())
        print(f"{name:<16}{residual:>22.3e}   {gap_str}")

    # --- accuracy --------------------------------------------------------
    scores = {name: rmse_by_level(mat, y_true, index) for name, mat in reconciled.items()}
    names = list(reconciled)
    print(f"\n{'level':<12}" + "".join(f"{n:>18}" for n in names))
    print("-" * (12 + 18 * len(names)))
    for level in AGG_LEVELS + [BOTTOM_LEVEL]:
        line = f"{level:<12}"
        best = min(scores[n][level] for n in names)
        for n in names:
            v = scores[n][level]
            line += f"{v:>17,.3f}{'*' if v == best else ' '}"
        print(line)
    print("* best per level. RMSE compares methods within a level, not across levels.")

    negatives = {n: int((mat < 0).sum()) for n, mat in reconciled.items()}
    print(f"\nnegative forecasts: {negatives}")

    # --- persist ---------------------------------------------------------
    frames = []
    for name, mat in reconciled.items():
        frames.append(
            pd.DataFrame(
                {
                    "level": np.repeat(index["level"].to_numpy(), HORIZON),
                    "series_id": np.repeat(index["series_id"].to_numpy(), HORIZON),
                    "date": np.tile(dates, n_hiers),
                    "base_model": BASE_MODEL,
                    "method": name,
                    "forecast": mat.ravel(),
                    "actual": y_true.ravel(),
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    con.register("reconciled", out)
    con.sql(
        "SELECT level, series_id, CAST(date AS DATE) AS date, base_model, method, forecast, actual "
        "FROM reconciled"
    ).write_parquet(str(OUTPUT_PATH))
    print(f"\nwrote {len(out):,} rows to {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")

    dense_gb = n_bottom**2 * 8 / 1e9
    print(
        f"\nNote: mint_shrink / mint_cov would need a dense {n_bottom:,} x {n_bottom:,} "
        f"error covariance (~{dense_gb:.1f} GB) plus in-sample residuals, so they are "
        f"out of reach here; wls_struct is the scalable MinT weighting."
    )


if __name__ == "__main__":
    main()
