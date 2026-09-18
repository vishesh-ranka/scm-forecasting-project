-- 02_hierarchy.sql
--
-- Builds the aggregation hierarchy from data/staged/sales_long.parquet.
--
-- sales_long.parquet only carries (item_id, store_id, date, sales) because
-- the unpivot in 01_staging.sql dropped the descriptive columns. This script
-- joins dept_id / cat_id / state_id back in from the source CSV, then rolls
-- the base series up to each level and writes one Parquet file per level.
--
-- Every level is derived from the same sales_base view, so the levels are
-- coherent by construction: each one sums to the same daily total.
--
-- The SUMs are cast back to BIGINT: SUM() over a BIGINT returns an int128,
-- which Parquet has no type for and would silently store as DOUBLE, leaving
-- the aggregate levels with a different sales type than the base level.
--
-- Assumes data/staged/hierarchy/ already exists (sql_runner.py creates it).

-- The descriptive columns, keyed by the same grain as the staged sales.
-- (item_id, store_id) is a unique key in the source CSV - 30,490 rows,
-- 30,490 distinct pairs - so this join is 1:1 and cannot fan out rows.
CREATE OR REPLACE TEMP VIEW item_store_dim AS
SELECT item_id, store_id, dept_id, cat_id, state_id
FROM read_csv_auto('sales_train_validation.csv');

-- The base fact table, re-enriched with the descriptive columns. Left as a
-- view rather than a materialized table: each roll-up below re-reads the
-- Parquet file, which is cheaper than holding 58M enriched rows in memory.
CREATE OR REPLACE TEMP VIEW sales_base AS
SELECT
    s.date,
    d.cat_id,
    d.dept_id,
    s.item_id,
    d.state_id,
    s.store_id,
    s.sales
FROM read_parquet('data/staged/sales_long.parquet') s
JOIN item_store_dim d USING (item_id, store_id);

-- Level: Total - one series. The top of the hierarchy and the reconciliation
-- constraint every other level must sum to. 1,913 rows (one per day).
COPY (
    SELECT date, CAST(SUM(sales) AS BIGINT) AS sales
    FROM sales_base
    GROUP BY date
    ORDER BY date
) TO 'data/staged/hierarchy/total.parquet' (FORMAT PARQUET);

-- Level: Category - 3 series (FOODS, HOBBIES, HOUSEHOLD).
-- First split of the product dimension. 3 x 1,913 = 5,739 rows.
COPY (
    SELECT date, cat_id, CAST(SUM(sales) AS BIGINT) AS sales
    FROM sales_base
    GROUP BY date, cat_id
    ORDER BY cat_id, date
) TO 'data/staged/hierarchy/category.parquet' (FORMAT PARQUET);

-- Level: Department - 7 series. Departments nest strictly inside categories
-- (FOODS_1..3 -> FOODS, HOBBIES_1..2 -> HOBBIES, HOUSEHOLD_1..2 ->
-- HOUSEHOLD), so cat_id is carried alongside dept_id: it makes each row's
-- parent explicit and lets the summing matrix be built without re-joining.
-- 7 x 1,913 = 13,391 rows.
COPY (
    SELECT date, cat_id, dept_id, CAST(SUM(sales) AS BIGINT) AS sales
    FROM sales_base
    GROUP BY date, cat_id, dept_id
    ORDER BY dept_id, date
) TO 'data/staged/hierarchy/department.parquet' (FORMAT PARQUET);

-- Level: Store - 10 series. This is the geographic dimension, independent of
-- the product dimension above. Stores nest strictly inside states (CA_1..4 ->
-- CA, TX_1..3 -> TX, WI_1..3 -> WI), so state_id is carried for the same
-- reason cat_id is carried above: it makes the state level a pure roll-up of
-- this file. 10 x 1,913 = 19,130 rows.
COPY (
    SELECT date, state_id, store_id, CAST(SUM(sales) AS BIGINT) AS sales
    FROM sales_base
    GROUP BY date, state_id, store_id
    ORDER BY store_id, date
) TO 'data/staged/hierarchy/store.parquet' (FORMAT PARQUET);

-- Level: Item-store - 30,490 series. The bottom of the hierarchy and the only
-- level that is observed rather than aggregated; every level above is a sum of
-- these. No GROUP BY: (item_id, store_id, date) is already unique in the
-- staged file, so this is a pass-through that adds the descriptive columns.
-- 30,490 x 1,913 = 58,327,370 rows.
COPY (
    SELECT date, cat_id, dept_id, item_id, state_id, store_id, sales
    FROM sales_base
) TO 'data/staged/hierarchy/item_store.parquet' (FORMAT PARQUET);
