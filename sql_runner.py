"""Runs the staging SQL scripts against DuckDB and writes the results to
data/staged/ as Parquet files.

01_staging.sql is a single SELECT, so this script writes its one output.
02_hierarchy.sql writes its own five outputs via COPY, so this script just
executes it.
"""

import os
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent
SQL_DIR = PROJECT_ROOT / "sql"
STAGED_DIR = PROJECT_ROOT / "data" / "staged"
HIERARCHY_DIR = STAGED_DIR / "hierarchy"

STAGED_SALES_PATH = STAGED_DIR / "sales_long.parquet"

HIERARCHY_LEVELS = [
    "total",
    "category",
    "department",
    "store",
    "item_store",
]


def stage_sales(con: duckdb.DuckDBPyConnection) -> None:
    query = (SQL_DIR / "01_staging.sql").read_text()
    con.sql(query).write_parquet(str(STAGED_SALES_PATH))

    row_count = con.sql(
        f"SELECT COUNT(*) FROM read_parquet('{STAGED_SALES_PATH}')"
    ).fetchone()[0]
    print(f"staged sales_long.parquet: {row_count:,} rows")


def build_hierarchy(con: duckdb.DuckDBPyConnection) -> None:
    query = (SQL_DIR / "02_hierarchy.sql").read_text()
    con.execute(query)

    for level in HIERARCHY_LEVELS:
        path = HIERARCHY_DIR / f"{level}.parquet"
        row_count = con.sql(f"SELECT COUNT(*) FROM read_parquet('{path}')").fetchone()[0]
        print(f"hierarchy/{level}.parquet: {row_count:,} rows")


def main() -> None:
    HIERARCHY_DIR.mkdir(parents=True, exist_ok=True)

    # The SQL files reference the CSVs and the staged Parquet with relative
    # paths, so run from the project root regardless of the caller's
    # current directory.
    os.chdir(PROJECT_ROOT)

    con = duckdb.connect()
    stage_sales(con)
    build_hierarchy(con)


if __name__ == "__main__":
    main()
