-- 01_staging.sql
--
-- Reshapes sales_train_validation.csv from wide format (one column per day,
-- d_1 .. d_1913) into long format (one row per item_id, store_id, date,
-- sales), with the real calendar date attached via calendar.csv.
--
-- Source shape:  30,490 rows x 1,919 columns
--                (6 id columns + 1,913 day columns)
-- Target shape:  ~58.3M rows x 4 columns
--                (item_id, store_id, date, sales)

WITH wide_sales AS (
    -- Read the raw CSV. read_csv_auto lets DuckDB query the file in place
    -- (infers types/columns from the header) without a separate load step.
    SELECT *
    FROM read_csv_auto('sales_train_validation.csv')
),

unpivoted AS (
    -- UNPIVOT turns the 1,913 day columns into two columns: `d` (the
    -- original column name, e.g. "d_1", "d_2", ...) and `sales` (the value
    -- that was in that column). Every one of the 30,490 source rows becomes
    -- 1,913 output rows, one per day.
    --
    -- ON COLUMNS(* EXCLUDE (...)) tells DuckDB which columns to unpivot: all
    -- of them EXCEPT the six identifier columns listed. Those six are left
    -- alone and repeated across every generated row, so each long row still
    -- knows which item/store/dept/cat/state it came from.
    UNPIVOT wide_sales
    ON COLUMNS(* EXCLUDE (id, item_id, dept_id, cat_id, store_id, state_id))
    INTO
        NAME d
        VALUE sales
),

calendar AS (
    -- calendar.csv has one row per real day and a `d` column holding the
    -- same "d_1", "d_2", ... keys that were column names in the sales file.
    -- We only need `d` and `date` here to resolve day-key -> calendar date.
    SELECT d, date
    FROM read_csv_auto('calendar.csv')
)

-- Join the unpivoted sales onto the calendar using the shared `d` key
-- (e.g. "d_1" in both sides) so each row gets a real date instead of a
-- synthetic day label. This is the bridge that lets the long table be used
-- for time-series work (resampling, joining sell_prices via date, etc).
SELECT
    u.item_id,
    u.store_id,
    c.date,
    u.sales
FROM unpivoted u
JOIN calendar c
    ON u.d = c.d
ORDER BY u.item_id, u.store_id, c.date;
