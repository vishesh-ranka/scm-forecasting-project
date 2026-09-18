"""Baseline forecasts for every level of the M5 aggregation hierarchy.

Holds out the last 28 days of each level, fits a seasonal-naive baseline and a
per-level LightGBM model, writes every forecast to
data/forecasts/base_forecasts.parquet, and prints a naive-vs-LightGBM RMSE table.
"""

import glob
import os
import sys
import sysconfig
import time
from pathlib import Path


def _bootstrap_openmp() -> None:
    """LightGBM's macOS wheel needs libomp, which isn't installed system-wide
    here and can't be linked in without Xcode CLT. scikit-learn ships a copy,
    so point dyld at it and re-exec once. No-op once libomp is installed
    properly (e.g. `brew install libomp`), and on non-macOS platforms.
    """
    if sys.platform != "darwin" or os.environ.get("_LIBOMP_BOOTSTRAP"):
        return
    try:
        import lightgbm  # noqa: F401

        return
    except OSError:
        pass

    roots = {sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]}
    found = [p for r in roots for p in glob.glob(str(Path(r) / "*" / ".dylibs" / "libomp.dylib"))]
    if not found:
        raise RuntimeError("libomp not found - install it with `brew install libomp`")

    env = dict(os.environ, _LIBOMP_BOOTSTRAP="1")
    env["DYLD_LIBRARY_PATH"] = os.pathsep.join(
        p for p in (str(Path(found[0]).parent), env.get("DYLD_LIBRARY_PATH", "")) if p
    )
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


_bootstrap_openmp()

import duckdb  # noqa: E402
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HIERARCHY_DIR = PROJECT_ROOT / "data" / "staged" / "hierarchy"
OUTPUT_PATH = PROJECT_ROOT / "data" / "forecasts" / "base_forecasts.parquet"

HORIZON = 28  # M5's actual forecast horizon
SEASON = 7  # weekly seasonality
MAX_LAG = 28  # longest lag/window, so the first usable target day is day 28

# Level name -> key columns that identify a series within that level.
LEVELS = {
    "total": [],
    "category": ["cat_id"],
    "department": ["dept_id"],
    "store": ["store_id"],
    "item_store": ["item_id", "store_id"],
}

FEATURES = ["dayofweek", "month", "lag_7", "lag_28", "rolling_mean_7", "rolling_mean_28"]

LGB_PARAMS = {
    "objective": "regression",  # L2, matching the RMSE we report
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "force_col_wise": True,  # 6 features over many rows
    "num_threads": os.cpu_count(),
    "verbose": -1,
    "seed": 0,
}
NUM_ROUNDS = 200


def load_level(con: duckdb.DuckDBPyConnection, path: Path, keys: list[str]):
    """Load a level as a dense (n_series, n_days) matrix of sales.

    Fetches the series list and the value column separately so that 58M string
    keys never have to be materialised for the item_store level.
    """
    dates = con.sql(f"SELECT DISTINCT date FROM read_parquet('{path}') ORDER BY date").fetchnumpy()["date"]

    if keys:
        key_sql = ", ".join(keys)
        ids = con.sql(f"SELECT DISTINCT {key_sql} FROM read_parquet('{path}') ORDER BY {key_sql}").df()
        series_ids = ids[keys[0]].astype(str)
        for k in keys[1:]:
            series_ids = series_ids + "_" + ids[k].astype(str)
        series_ids = series_ids.to_numpy()
        order_sql = f"{key_sql}, date"
    else:
        series_ids = np.array(["total"])
        order_sql = "date"

    sales = con.sql(f"SELECT sales FROM read_parquet('{path}') ORDER BY {order_sql}").fetchnumpy()["sales"]
    expected = len(series_ids) * len(dates)
    if len(sales) != expected:
        raise ValueError(f"{path.name}: expected {expected} rows (dense panel), got {len(sales)}")

    matrix = np.asarray(sales, dtype=np.float32).reshape(len(series_ids), len(dates))
    return series_ids, dates, matrix


def build_features(matrix, cumsum, dates, t0, t1):
    """Feature matrix for target days [t0, t1) across all series.

    Rows are series-major: series 0's days, then series 1's, etc. Every feature
    is strictly lagged - rolling_mean_7 at day t covers days t-7..t-1 - so no
    row ever sees its own target.
    """
    n_series = matrix.shape[0]
    n_days = t1 - t0
    dow = pd.DatetimeIndex(dates[t0:t1]).dayofweek.to_numpy()
    month = pd.DatetimeIndex(dates[t0:t1]).month.to_numpy()

    # Filled column by column to avoid holding six full-size temporaries at once.
    X = np.empty((n_series * n_days, len(FEATURES)), dtype=np.float32)
    X[:, 0] = np.tile(dow, n_series)
    X[:, 1] = np.tile(month, n_series)
    X[:, 2] = matrix[:, t0 - 7 : t1 - 7].ravel()
    X[:, 3] = matrix[:, t0 - 28 : t1 - 28].ravel()
    X[:, 4] = ((cumsum[:, t0:t1] - cumsum[:, t0 - 7 : t1 - 7]) / 7).ravel()
    X[:, 5] = ((cumsum[:, t0:t1] - cumsum[:, t0 - 28 : t1 - 28]) / 28).ravel()
    return X


def seasonal_naive(train):
    """Repeat the last observed week across the horizon.

    Each forecast day equals the same weekday 7 days earlier; applied
    recursively over 28 days that is exactly the last training week tiled 4x.
    Taking it from actuals instead would read test data the model can't see.
    """
    reps = -(-HORIZON // SEASON)
    return np.tile(train[:, -SEASON:], (1, reps))[:, :HORIZON]


def lightgbm_forecast(matrix, cumsum, dates, n_train):
    """Train one global LightGBM per level, then forecast 28 days recursively.

    lag_7 and the rolling means reach less than 28 days back, so beyond day 7 of
    the horizon they would need values that don't exist yet at forecast time.
    Predictions are fed back in step by step rather than using held-out actuals.
    """
    X = build_features(matrix, cumsum, dates, MAX_LAG, n_train)
    y = matrix[:, MAX_LAG:n_train].ravel()

    model = lgb.train(LGB_PARAMS, lgb.Dataset(X, y, feature_name=FEATURES), num_boost_round=NUM_ROUNDS)
    n_train_rows = X.shape[0]
    del X, y

    # Rolling buffer: 28 days of real history followed by the 28 predicted days.
    n_series = matrix.shape[0]
    buf = np.zeros((n_series, MAX_LAG + HORIZON), dtype=np.float32)
    buf[:, :MAX_LAG] = matrix[:, n_train - MAX_LAG : n_train]
    test_dates = pd.DatetimeIndex(dates[n_train : n_train + HORIZON])

    Xp = np.empty((n_series, len(FEATURES)), dtype=np.float32)
    for k in range(HORIZON):
        t = MAX_LAG + k
        window = buf[:, t - MAX_LAG : t]
        Xp[:, 0] = test_dates.dayofweek[k]
        Xp[:, 1] = test_dates.month[k]
        Xp[:, 2] = buf[:, t - 7]
        Xp[:, 3] = buf[:, t - 28]
        Xp[:, 4] = window[:, -7:].mean(axis=1)
        Xp[:, 5] = window.mean(axis=1)
        buf[:, t] = np.maximum(model.predict(Xp), 0)  # unit sales can't be negative

    return buf[:, MAX_LAG:], n_train_rows


def rmse(pred, actual):
    return float(np.sqrt(np.mean((pred.astype(np.float64) - actual.astype(np.float64)) ** 2)))


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    frames = []
    summary = []

    for level, keys in LEVELS.items():
        started = time.perf_counter()
        series_ids, dates, matrix = load_level(con, HIERARCHY_DIR / f"{level}.parquet", keys)

        # Last 28 days are the test set; everything before is training data.
        n_train = matrix.shape[1] - HORIZON
        train, actual = matrix[:, :n_train], matrix[:, n_train:]
        test_dates = dates[n_train:]

        # float64 cumsum: a float32 running total loses precision at the top level.
        cumsum = np.zeros((matrix.shape[0], matrix.shape[1] + 1), dtype=np.float64)
        np.cumsum(matrix, axis=1, out=cumsum[:, 1:])

        preds = {
            "seasonal_naive": seasonal_naive(train),
            "lightgbm": None,
        }
        preds["lightgbm"], n_train_rows = lightgbm_forecast(matrix, cumsum, dates, n_train)
        del cumsum

        for model_name, pred in preds.items():
            frames.append(
                pd.DataFrame(
                    {
                        "level": level,
                        "series_id": np.repeat(series_ids, HORIZON),
                        "date": np.tile(test_dates, len(series_ids)),
                        "model": model_name,
                        "forecast": pred.ravel().astype(np.float64),
                        "actual": actual.ravel().astype(np.float64),
                    }
                )
            )

        summary.append(
            {
                "level": level,
                "series": len(series_ids),
                "train_rows": n_train_rows,
                "naive_rmse": rmse(preds["seasonal_naive"], actual),
                "lgbm_rmse": rmse(preds["lightgbm"], actual),
                "secs": time.perf_counter() - started,
            }
        )
        print(f"  {level:<12} done in {summary[-1]['secs']:6.1f}s", flush=True)

    out = pd.concat(frames, ignore_index=True)
    # Written through DuckDB so date lands as DATE, matching the hierarchy files.
    con.register("forecasts", out)
    con.sql(
        "SELECT level, series_id, CAST(date AS DATE) AS date, model, forecast, actual FROM forecasts"
    ).write_parquet(str(OUTPUT_PATH))
    print(f"\nwrote {len(out):,} rows to {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")

    print(f"\n{'level':<12}{'series':>8}{'train rows':>14}{'naive RMSE':>14}{'LGBM RMSE':>14}{'improvement':>13}")
    print("-" * 75)
    for r in summary:
        delta = (r["naive_rmse"] - r["lgbm_rmse"]) / r["naive_rmse"] * 100
        print(
            f"{r['level']:<12}{r['series']:>8,}{r['train_rows']:>14,}"
            f"{r['naive_rmse']:>14,.3f}{r['lgbm_rmse']:>14,.3f}{delta:>12.1f}%"
        )
    print("\nRMSE is comparable between models within a level, not across levels.")


if __name__ == "__main__":
    main()
