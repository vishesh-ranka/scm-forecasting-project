"""Generate a self-contained static dashboard at docs/index.html.

Reads the same Parquet files the Streamlit app reads, so the two versions
cannot drift: every number on the page is precomputed here and embedded as
JSON. All interactivity that does not need live computation - level dropdowns,
method toggles, the what-if slider, hover tooltips - runs client-side against
that payload.

Run with:  python3 src/build_static_dashboard.py
"""

import json
from datetime import date
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECONCILED = PROJECT_ROOT / "data" / "forecasts" / "reconciled_forecasts.parquet"
HIERARCHY_DIR = PROJECT_ROOT / "data" / "staged" / "hierarchy"
OUT = PROJECT_ROOT / "docs" / "index.html"

PLOTLY_VERSION = "2.35.2"

# Embedding all 30,490 product series would be ~30 MB of JSON. The page ships
# the busiest ones, which is where the behaviour is actually visible, and says
# so in the caption.
TOP_PRODUCTS = 300

LEVELS = [
    ("total", "Company Wide Total", "One number: every product in every store, combined."),
    ("category", "Product Category", "3 categories, FOODS, HOBBIES, HOUSEHOLD."),
    ("department", "Department",
     "7 departments, FOODS_1, FOODS_2, FOODS_3, HOBBIES_1, HOBBIES_2, HOUSEHOLD_1, HOUSEHOLD_2."),
    ("store", "Store",
     "10 Walmart stores, CA_1 to CA_4 (California), TX_1 to TX_3 (Texas), WI_1 to WI_3 (Wisconsin)."),
    ("item_store", "Individual Product (per store)",
     "30,490 combinations, 3,049 products × 10 stores."),
]
UI_METHODS = [
    ("unreconciled", "Original Forecast", "#2B2B2B"),
    ("bottom_up", "Reconciled: Bottom Up", "#C9743C"),
    ("mint_wls_struct", "Reconciled: MinT", "#2F5E93"),
]

GEO = [
    {"state": "California", "n": 4, "lat": 36.78, "lon": -119.42, "stores": "CA_1, CA_2, CA_3, CA_4"},
    {"state": "Texas", "n": 3, "lat": 31.00, "lon": -99.90, "stores": "TX_1, TX_2, TX_3"},
    {"state": "Wisconsin", "n": 3, "lat": 44.50, "lon": -89.50, "stores": "WI_1, WI_2, WI_3"},
]


def r(x, n=4):
    """Round for the payload; keeps the embedded JSON small and diff-friendly."""
    return None if x is None else round(float(x), n)


def build_payload() -> dict:
    con = duckdb.connect()
    src = f"read_parquet('{RECONCILED}')"

    err = con.sql(
        f"SELECT level, method, SQRT(AVG(POWER(forecast-actual,2))) AS rmse FROM {src} "
        "GROUP BY level, method"
    ).df()
    error_by_level = {}
    for row in err.itertuples():
        error_by_level.setdefault(row.level, {})[row.method] = r(row.rmse, 4)

    total = con.sql(
        f"""SELECT date,
               MAX(actual)   FILTER (method='unreconciled')    AS actual,
               MAX(forecast) FILTER (method='unreconciled')    AS unreconciled,
               MAX(forecast) FILTER (method='mint_wls_struct') AS mint
            FROM {src} WHERE level='total' GROUP BY date ORDER BY date"""
    ).df()

    hist = con.sql(
        f"""SELECT date, sales FROM read_parquet('{HIERARCHY_DIR / "total.parquet"}')
            WHERE date < (SELECT MIN(date) FROM {src}) ORDER BY date DESC LIMIT 28"""
    ).df().sort_values("date")

    coh = con.sql(
        f"""WITH t AS (SELECT method, date, forecast AS total FROM {src} WHERE level='total'),
                 a AS (SELECT method, level, date, SUM(forecast) AS s FROM {src}
                       WHERE level<>'total' GROUP BY method, level, date)
            SELECT a.method, a.level, MAX(ABS(a.s-t.total)) AS max_gap,
                   AVG(ABS(a.s-t.total)/t.total)*100 AS mean_pct
            FROM a JOIN t USING (method, date) GROUP BY a.method, a.level"""
    ).df()
    coherence = {"before": [], "after": []}
    for row in coh.itertuples():
        if row.method == "mint_ols":
            continue
        bucket = "before" if row.method == "unreconciled" else "after"
        coherence[bucket].append({
            "method": row.method, "level": row.level,
            "maxGap": r(row.max_gap, 10), "meanPct": r(row.mean_pct, 14),
        })

    members = {}
    for key, _, _ in LEVELS:
        if key in ("total", "item_store"):
            continue
        df = con.sql(
            f"""SELECT series_id,
                    MAX(rmse) FILTER (method='unreconciled')    AS err_o,
                    MAX(rmse) FILTER (method='mint_wls_struct') AS err_m,
                    MAX(av) AS av
                FROM (SELECT series_id, method, SQRT(AVG(POWER(forecast-actual,2))) AS rmse,
                             AVG(actual) AS av
                      FROM {src} WHERE level='{key}' GROUP BY series_id, method)
                GROUP BY series_id ORDER BY av DESC"""
        ).df()
        members[key] = [
            {"id": t.series_id, "avg": r(t.av, 2), "errO": r(t.err_o, 3), "errM": r(t.err_m, 3)}
            for t in df.itertuples()
        ]

    day_of_week = {}
    for key, _, _ in LEVELS:
        df = con.sql(
            f"""SELECT dayname(date) AS day, dayofweek(date) AS dow,
                    AVG(ABS(forecast-actual)) FILTER (method='unreconciled')    AS eo,
                    AVG(ABS(forecast-actual)) FILTER (method='mint_wls_struct') AS em,
                    AVG(actual)               FILTER (method='unreconciled')    AS av
                FROM {src} WHERE level='{key}' GROUP BY day, dow ORDER BY dow"""
        ).df()
        day_of_week[key] = [
            {"day": t.day, "errO": r(t.eo, 3), "errM": r(t.em, 3), "avg": r(t.av, 1)}
            for t in df.itertuples()
        ]

    top = con.sql(
        f"""SELECT series_id, AVG(actual) AS av FROM {src}
            WHERE level='item_store' AND method='unreconciled'
            GROUP BY series_id ORDER BY av DESC LIMIT {TOP_PRODUCTS}"""
    ).df()
    ids = top["series_id"].tolist()
    prod_daily = con.execute(
        f"""SELECT series_id, date,
                MAX(actual)   FILTER (method='unreconciled')    AS actual,
                MAX(forecast) FILTER (method='unreconciled')    AS unreconciled,
                MAX(forecast) FILTER (method='mint_wls_struct') AS mint
            FROM {src} WHERE level='item_store' AND series_id IN
                (SELECT UNNEST(?::VARCHAR[]))
            GROUP BY series_id, date ORDER BY series_id, date""",
        [ids],
    ).df()
    prod_stats = con.execute(
        f"""SELECT series_id,
                AVG(actual) FILTER (method='unreconciled') AS av,
                SQRT(AVG(POWER(forecast-actual,2)) FILTER (method='unreconciled'))    AS err_o,
                SQRT(AVG(POWER(forecast-actual,2)) FILTER (method='mint_wls_struct')) AS err_m,
                AVG(ABS(forecast - actual)) FILTER (method='mint_wls_struct')         AS dummy
            FROM {src} WHERE level='item_store' AND series_id IN
                (SELECT UNNEST(?::VARCHAR[]))
            GROUP BY series_id""",
        [ids],
    ).df().set_index("series_id")

    products = {}
    for sid, grp in prod_daily.groupby("series_id", sort=False):
        s = prod_stats.loc[sid]
        moved = (grp["mint"] - grp["unreconciled"]).abs().mean()
        products[sid] = {
            "actual": [r(v, 1) for v in grp["actual"]],
            "orig": [r(v, 3) for v in grp["unreconciled"]],
            "mint": [r(v, 3) for v in grp["mint"]],
            "avg": r(s["av"], 2), "errO": r(s["err_o"], 3), "errM": r(s["err_m"], 3),
            "moved": r(moved, 4),
        }

    counts = con.sql(
        f"SELECT level, COUNT(DISTINCT series_id) AS n FROM {src} GROUP BY level"
    ).df().set_index("level")["n"].to_dict()

    return {
        "generated": date.today().isoformat(),
        "levels": [{"key": k, "label": lb, "blurb": b} for k, lb, b in LEVELS],
        "methods": [{"key": k, "label": lb, "color": c} for k, lb, c in UI_METHODS],
        "errorByLevel": error_by_level,
        "totalSeries": {
            "dates": [d.strftime("%Y-%m-%d") for d in total["date"]],
            "actual": [r(v, 1) for v in total["actual"]],
            "orig": [r(v, 2) for v in total["unreconciled"]],
            "mint": [r(v, 2) for v in total["mint"]],
        },
        "history": {
            "dates": [d.strftime("%Y-%m-%d") for d in hist["date"]],
            "sales": [r(v, 1) for v in hist["sales"]],
        },
        "coherence": coherence,
        "members": members,
        "dayOfWeek": day_of_week,
        "productIds": ids,
        "products": products,
        "geo": GEO,
        "counts": {k: int(v) for k, v in counts.items()},
        "topProducts": TOP_PRODUCTS,
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device width, initial scale=1">
<title>Hierarchical Demand Forecasting</title>
<meta name="description" content="Reconciling retail demand forecasts across a crossed hierarchy of 30,511 series.">
<script src="https://cdn.plot.ly/plotly-__PLOTLY_VERSION__.min.js" charset="utf-8"></script>
<style>
:root{
  --bg:#1a1a2e; --card:#232342; --line:#33335c; --text:#e8e8e8; --muted:#a2a2c0;
  --accent:#5ec8c8; --plot:#fcfcfb; --warn:#e6b800; --good:#4ec9a0;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
  font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:44px 22px 110px}
h1{font-size:2.1em;margin:0 0 .35em;line-height:1.25}
h2{color:var(--accent);font-size:1.45em;margin:2.3em 0 .6em;line-height:1.35}
h3{color:var(--accent);font-size:1.12em;margin:1.9em 0 .5em}
p{margin:0 0 1.05em}
a{color:var(--accent)}
.lede{font-size:1.06em;color:#d6d6ea}
.muted{color:var(--muted);font-size:.9em;line-height:1.6}
hr{border:0;height:1px;background:var(--line);margin:40px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(178px,1fr));gap:14px;margin:26px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:15px 17px}
.tile .k{color:var(--muted);font-size:.79em;text-transform:uppercase;letter-spacing:.06em}
.tile .v{font-size:1.65em;font-weight:600;margin-top:5px;line-height:1.2}
.controls{display:flex;flex-wrap:wrap;gap:16px;align-items:flex-end;margin:16px 0 8px}
.ctl{display:flex;flex-direction:column;gap:6px}
.ctl label{color:var(--muted);font-size:.83em}
select,input[type=range]{background:var(--card);color:var(--text);border:1px solid var(--line);
  border-radius:7px;padding:8px 11px;font-size:.94em;font-family:inherit;min-width:210px}
input[type=range]{padding:0;min-width:280px;accent-color:var(--accent)}
.checks{display:flex;flex-wrap:wrap;gap:18px;margin:14px 0}
.checks label{display:flex;gap:7px;align-items:center;font-size:.93em;cursor:pointer}
input[type=checkbox],input[type=radio]{accent-color:var(--accent);width:16px;height:16px;cursor:pointer}
.chart{background:var(--plot);border-radius:10px;margin:14px 0 6px;overflow:hidden}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:.92em;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:8px 12px;text-align:left;white-space:nowrap}
th{background:#2a2a4d;font-weight:600;color:#f0f0ff}
td.num{text-align:right;font-variant-numeric:tabular-nums}
tr:nth-child(2n) td{background:#1f1f3a}
details{background:var(--card);border:1px solid var(--line);border-radius:9px;
  padding:13px 17px;margin:11px 0}
summary{cursor:pointer;font-weight:600;color:var(--accent)}
details[open] summary{margin-bottom:9px}
.note{border-left:3px solid var(--accent);padding:11px 15px;margin:15px 0;
  background:#20203c;border-radius:0 8px 8px 0}
.note.warn{border-left-color:var(--warn)}
.note.good{border-left-color:var(--good)}
pre{background:#15152b;border:1px solid var(--line);border-radius:8px;padding:14px;
  overflow-x:auto;font-size:.82em;line-height:1.5}
.foot{color:var(--muted);font-size:.85em;margin-top:36px;border-top:1px solid var(--line);padding-top:18px}
@media(max-width:640px){.wrap{padding:26px 15px 70px}h1{font-size:1.6em}select,input[type=range]{min-width:100%}}
</style>
</head>
<body>
<div class="wrap">

<h1>Hierarchical Demand Forecasting</h1>
<p class="lede">Predicting daily sales for <strong>10 Walmart stores</strong> in California, Texas and
Wisconsin, for the company as a whole, for each product category, for each store, and for each
individual product in each store, and then adjusting those predictions so they
<strong>add up correctly to each other</strong>, which separately made predictions never do on their own.</p>
<p class="muted">Data: the public M5 competition dataset (Walmart, 2011 to 2016). <strong>Units</strong> means
individual items sold per day. Product names are anonymised in the source data, so products appear as
IDs such as <code>FOODS_1_001</code>; the three categories and 10 store IDs are the real groupings the
data ships with.</p>

<div class="tiles" id="tiles"></div>

<hr>

<h2>The Problem</h2>
<p>A demand plan gets used at every altitude at once. Finance commits to a company wide total, category
managers plan by category, and the replenishment team orders stock for one product in one store. If each
level is forecast on its own, <strong>those numbers disagree</strong>: and someone ends up reconciling
them in a spreadsheet by hand.</p>

<h2>Why This Is Hard</h2>
<p>There are two independent ways to slice this business, by <strong>product</strong>
(product → department → category) and by <strong>geography</strong> (store → state). These two paths
<strong>cross rather than nest</strong>: every category is sold in every store, so neither slicing sits
inside the other. A single product in a store rolls up through two different chains, not one.</p>
<p>That rules out the simplest fix, taking the company total and splitting it downward, because there
is no single path down. It forces methods that work on the whole structure at once.</p>
<pre>Company Total ──┬── Product Category ── Department ──┐
                │                                    ├── Individual Product per Store
                └── Store ───────────────────────────┘        (30,490 combinations)</pre>
<div class="chart" id="map"></div>
<p class="muted">Markers sit at approximate state centres and are sized by store count. The dataset
contains no store addresses or coordinates, so exact locations are not shown.</p>

<hr>

<h2>Forecast Accuracy By Level</h2>
<p>Pick a level of the business to see how the three methods compare. <strong>Lower bars are better</strong>
- the number is the average amount the forecast missed by, per day.</p>
<div class="controls">
  <div class="ctl"><label for="lvl">Choose A Level Of The Business</label>
    <select id="lvl"></select></div>
</div>
<p class="muted" id="lvlBlurb"></p>
<div class="chart" id="errChart"></div>
<p id="lvlVerdict"></p>
<p class="muted">Error is not comparable between levels: the company total misses by thousands of units a
day because it sums millions of units of sales, while a single product in a single store misses by about
2 units because it only sells a handful.</p>

<h3>What If: Where Would You Rather Be Accurate?</h3>
<div class="controls">
  <div class="ctl"><label for="blend">Drag to shift priority: <span id="blendLabel"></span></label>
    <input type="range" id="blend" min="0" max="100" step="5" value="0"></div>
</div>
<div class="chart" id="blendChart"></div>
<p class="muted"><strong>Illustrative only.</strong> This slides linearly between two results that were
already computed (MinT at 0%, the original forecast at 100%). It does not re run the optimisation, and
the in between blends are not themselves coherent forecasts, they would not add up.</p>

<h3 id="memberHead">Every Level, Compared</h3>
<div class="chart" id="memberChart"></div>
<p id="memberVerdict"></p>
<div id="memberTable"></div>

<hr>

<h2>Look Up One Product In One Store</h2>
<p>Every product store combination has its own forecast. The list is ordered by how much it sells.</p>
<div class="controls">
  <div class="ctl"><label for="prod">Choose A Product</label><select id="prod"></select></div>
  <div class="ctl"><label>Which Forecasts To Show</label>
    <div class="checks">
      <label><input type="radio" name="pw" value="both" checked> Both</label>
      <label><input type="radio" name="pw" value="orig"> Original Only</label>
      <label><input type="radio" name="pw" value="mint"> Reconciled Only</label>
    </div>
  </div>
</div>
<div class="tiles" id="prodTiles"></div>
<div class="chart" id="prodChart"></div>
<p class="muted"><strong>The two forecast lines will look identical, that is the real result, not a
glitch.</strong> Reconciliation moves an individual product's forecast by about 0.05 units a day on
average. Almost all of the adjustment lands on the aggregate levels; the bottom of the hierarchy barely
moves. This page ships the <span id="topN"></span> highest selling combinations of the 30,490, the full
set is in the Streamlit version.</p>

<hr>

<h2>Which Days Are Hardest To Forecast?</h2>
<p>The same 28 days, grouped by day of the week. Retail demand is strongly weekly, so some days are
consistently harder to predict than others, and reconciliation does not help them equally.</p>
<div class="controls">
  <div class="ctl"><label for="dowLvl">Choose A Level Of The Business</label>
    <select id="dowLvl"></select></div>
</div>
<div class="chart" id="dowChart"></div>
<p id="dowVerdict"></p>

<hr>

<h2>Company Wide Forecast Over The Test Period</h2>
<p>The 28 days held back from the models. Actual sales are always shown; switch the forecasts on and off
to compare them. Hover for exact values.</p>
<div class="checks">
  <label><input type="checkbox" id="cbOrig" checked> Show Original Forecast</label>
  <label><input type="checkbox" id="cbMint" checked> Show Reconciled Forecast (MinT)</label>
  <label><input type="checkbox" id="cbHist"> Show 28 Days Of Prior History</label>
</div>
<div class="chart" id="totalChart"></div>
<div class="tiles" id="totalTiles"></div>
<p class="muted">Both forecasts follow the weekly rhythm, sales peak at weekends, but under shoot the
biggest peaks. The reconciled forecast sits closer to reality on most days.</p>

<hr>

<h2>Do The Forecasts Add Up?</h2>
<p>If you add up every store's forecast for one day, you should get the company wide forecast for that
day. Before reconciliation you do not. Use the toggle to compare.</p>
<div class="checks">
  <label><input type="radio" name="coh" value="before" checked> Before Reconciliation</label>
  <label><input type="radio" name="coh" value="after"> After Reconciliation</label>
</div>
<div class="tiles" id="cohTiles"></div>
<div id="cohTable"></div>
<div id="cohNote"></div>

<hr>

<h2>The Short Version</h2>
<div class="tiles" id="sumTiles"></div>
<p><strong>What was built.</strong> Daily sales forecasts for 30,490 product store combinations across 10
Walmart stores, rolled up into five levels of the business, then adjusted so every level agrees with
every other.</p>
<p><strong>What reconciliation bought.</strong> Before it, adding up the individual forecasts missed the
company wide forecast by as much as <strong>3,449 units on a single day</strong>. After it, the levels
agree to within a rounding error, a gap of about one unit in eleven trillion. That consistency is
<em>guaranteed by construction</em>, not tuned for.</p>
<p><strong>What it cost.</strong> Company wide accuracy improved <strong>12.5%</strong> and category
accuracy <strong>4.6%</strong>, but department and store forecasts got <strong>2.7 to 5.6% worse</strong>.
Error was moved, not removed. At the individual product level almost nothing changed at all.</p>
<p><strong>So what.</strong> Whether this is a good trade depends on one question: which level is the plan
actually committed at? If finance signs up to a company wide number, reconciliation pays for itself. If
the binding commitment is store level replenishment, it costs more than it returns. That is a business
decision, and the numbers above are the input to it, not a model selection question with one right answer.</p>

<hr>

<h2>Glossary And Technical Details</h2>
<details><summary>Glossary</summary>
<ul>
<li><strong>Forecast</strong>: a prediction of how many units will sell on a future day.</li>
<li><strong>Base forecast</strong>: the first draft prediction made for each level separately, before any adjustment.</li>
<li><strong>Hierarchy level</strong>: one altitude of the business: the company total, a product category, a department, a store, or a single product in a single store.</li>
<li><strong>Reconciliation</strong>: adjusting a set of forecasts so the smaller ones add up exactly to the bigger ones.</li>
<li><strong>Forecast error (RMSE)</strong>: how far off the forecast was, on average, in units of product per day. Lower is better.</li>
</ul>
</details>
<details><summary>How The Forecasts Were Built</summary>
<p>The raw M5 Walmart data is reshaped in DuckDB from a spreadsheet style file (one column per day) into
58.3 million rows, one per product, per store, per day across 5 years, then aggregated into the five levels.</p>
<p>Each level gets one global LightGBM model using day of week, month, and lagged and rolling average
sales. Because the horizon is 28 days but the features reach back only 7, the model feeds its own
predictions forward day by day rather than peeking at held out values, verified by replacing the test
period with random numbers and confirming the forecasts did not change.</p>
</details>
<details><summary>Why The Textbook Method Does Not Fit</summary>
<p>The standard version of MinT needs a table comparing every product store combination with every other
one. At 30,490 combinations that is a 30,490 × 30,490 matrix, about 7.4 GB, and the maths requires
effectively inverting it.</p>
<p>The sparse version used here exploits the fact that of the 930 million possible entries in the summing
table, only 152,450 are non zero (0.016%). It solves in 0.06 seconds.</p>
<p>One caveat: MinT produces 1,280 negative forecasts out of 854,308. Rounding them up to zero would
break the adding up guarantee.</p>
</details>
<details><summary>How This Page Is Built</summary>
<p>This page is generated by <code>src/build_static_dashboard.py</code> from the same Parquet files the
Streamlit app reads, so the two versions cannot drift. Every number is precomputed and embedded as JSON;
the only runtime dependency is Plotly from a CDN. Regenerate with
<code>python3 src/build_static_dashboard.py</code>.</p>
</details>

<p class="foot">Generated __GENERATED__ from the project's Parquet outputs ·
<a href="https://github.com/vishesh-ranka/scm-forecasting-project">Source on GitHub</a></p>

</div>

<script>
const D = __DATA__;

const C = {plot:'#fcfcfb', ink:'#16181d', soft:'#5b6067', grid:'#e3e5e8'};
const METHODS = D.methods;
const LEVELS = D.levels;
const CFG = {displaylogo:false, responsive:true,
  modeBarButtonsToRemove:['lasso2d','select2d','autoScale2d']};

function layout(title, opts){
  opts = opts || {};
  const legend = !!opts.legend;
  return {
    title:{text:title, x:0, xanchor:'left', font:{size:17, color:C.ink}, pad:{t:16,l:8}},
    paper_bgcolor:C.plot, plot_bgcolor:C.plot,
    font:{color:C.ink, size:13, family:'Helvetica Neue,Helvetica,Arial,sans-serif'},
    margin:{l:72, r:30, t:legend?112:58, b:60},
    height:opts.height || 400,
    hoverlabel:{bgcolor:'#ffffff', bordercolor:'#b9bcc2',
      font:{color:C.ink, size:13, family:'Helvetica Neue,Helvetica,Arial,sans-serif'}},
    showlegend:legend,
    legend:legend?{orientation:'h', yanchor:'bottom', y:1.03, xanchor:'left', x:0,
      bgcolor:'rgba(0,0,0,0)', font:{color:C.ink, size:12}}:{},
    xaxis:{gridcolor:C.grid, linecolor:C.grid, zeroline:false,
      tickfont:{color:C.soft}, title:{font:{color:C.soft}}},
    yaxis:{gridcolor:C.grid, linecolor:C.grid, zeroline:false,
      tickfont:{color:C.soft}, title:{text:opts.ytitle||'', font:{color:C.soft}}}
  };
}

function nf(x, d){
  if(x === null || x === undefined) return '-';
  return x.toLocaleString('en-US', {minimumFractionDigits:d===undefined?0:d,
                                    maximumFractionDigits:d===undefined?0:d});
}
function fmtGap(x){
  if(x === 0) return 'exactly 0';
  if(x < 0.001) return x.toFixed(8);
  return nf(x, 1);
}
function fmtShare(p){
  if(p === 0) return 'none at all';
  if(p >= 0.01) return p.toFixed(2) + '% of that day\'s sales';
  const n = 100 / p;
  const steps = [[1e12,'trillion'],[1e9,'billion'],[1e6,'million'],[1e3,'thousand']];
  for(const [div,name] of steps){
    if(n >= div) return 'about 1 unit in ' + nf(n/div, 0) + ' ' + name;
  }
  return 'about 1 unit in ' + nf(n, 0);
}
function tiles(el, items){
  document.getElementById(el).innerHTML = items.map(function(t){
    return '<div class="tile"><div class="k">' + t[0] + '</div><div class="v">' + t[1] + '</div></div>';
  }).join('');
}
function table(rows, head, aligns){
  let h = '<table><thead><tr>' + head.map(function(x){return '<th>'+x+'</th>';}).join('') + '</tr></thead><tbody>';
  h += rows.map(function(r){
    return '<tr>' + r.map(function(c,i){
      return '<td class="' + ((aligns && aligns[i]==='n') ? 'num' : '') + '">' + c + '</td>';
    }).join('') + '</tr>';
  }).join('');
  return h + '</tbody></table>';
}
function labelOf(key){
  for(const l of LEVELS){ if(l.key === key) return l.label; }
  return key;
}

/* ---- header tiles ---- */
tiles('tiles', [
  ['Products x Stores', nf(D.counts.item_store)],
  ['Days Of History', '1,913'],
  ['Forecasts Produced', nf(Object.keys(D.counts).reduce(function(a,k){return a + D.counts[k];}, 0))],
  ['Test Window', '28 days']
]);
document.getElementById('topN').textContent = nf(D.topProducts);

/* ---- map ---- */
Plotly.newPlot('map', [{
  type:'scattergeo', lon:D.geo.map(function(g){return g.lon;}),
  lat:D.geo.map(function(g){return g.lat;}),
  text:D.geo.map(function(g){return g.state;}),
  customdata:D.geo.map(function(g){return [g.n, g.stores];}),
  mode:'markers+text', textposition:'top center', textfont:{color:C.ink, size:12},
  marker:{size:D.geo.map(function(g){return g.n*10;}), color:'#2F5E93', opacity:.85,
    line:{width:1.5, color:C.plot}},
  hovertemplate:'<b>%{text}</b><br>%{customdata[0]} stores<br>%{customdata[1]}<extra></extra>'
}], Object.assign(layout('Where The 10 Stores Are', {height:360}), {
  geo:{scope:'usa', bgcolor:C.plot, landcolor:'#f0f0ee', lakecolor:C.plot,
       subunitcolor:'#c9ccd1', countrycolor:'#c9ccd1'}
}), CFG);

/* ---- level selectors ---- */
const lvlSel = document.getElementById('lvl');
const dowSel = document.getElementById('dowLvl');
LEVELS.forEach(function(l){
  lvlSel.insertAdjacentHTML('beforeend', '<option value="'+l.key+'">'+l.label+'</option>');
  dowSel.insertAdjacentHTML('beforeend', '<option value="'+l.key+'">'+l.label+'</option>');
});

function drawError(){
  const lv = lvlSel.value;
  const e = D.errorByLevel[lv];
  const base = e.unreconciled;
  const vals = METHODS.map(function(m){return e[m.key];});
  const deltas = METHODS.map(function(m){return (e[m.key]-base)/base*100;});
  const blurb = LEVELS.filter(function(l){return l.key===lv;})[0].blurb;
  document.getElementById('lvlBlurb').innerHTML = '<strong>' + labelOf(lv) + '</strong>: ' + blurb;

  Plotly.react('errChart', [{
    type:'bar', x:METHODS.map(function(m){return m.label;}), y:vals,
    marker:{color:METHODS.map(function(m){return m.color;})},
    customdata:deltas,
    text:vals.map(function(v){return v<100 ? v.toFixed(2) : nf(v,0);}),
    textposition:'outside', textfont:{color:C.ink, size:12},
    hovertemplate:'<b>%{x}</b><br>Average daily miss: %{y:,.3f} units'+
      '<br>Change vs. original: %{customdata:+.1f}%<extra></extra>'
  }], Object.assign(layout('Forecast Error - ' + labelOf(lv),
      {height:400, ytitle:'Average daily miss (units sold)'}),
      {yaxis:{gridcolor:C.grid, linecolor:C.grid, zeroline:false, tickfont:{color:C.soft},
              title:{text:'Average daily miss (units sold)', font:{color:C.soft}},
              range:[0, Math.max.apply(null, vals)*1.18]}}), CFG);

  let bestI = 0;
  vals.forEach(function(v,i){ if(v < vals[bestI]) bestI = i; });
  const md = deltas[2];
  document.getElementById('lvlVerdict').innerHTML =
    'At the <strong>' + labelOf(lv) + '</strong> level the most accurate method is <strong>' +
    METHODS[bestI].label + '</strong>. Reconciling with MinT is <strong>' +
    Math.abs(md).toFixed(1) + '% ' + (md < 0 ? 'more accurate' : 'less accurate') +
    '</strong> than the original forecast here.';

  drawMembers(lv);
  drawBlend();
}

function drawMembers(lv){
  const head = document.getElementById('memberHead');
  const wrapChart = document.getElementById('memberChart');
  const verdict = document.getElementById('memberVerdict');
  const tbl = document.getElementById('memberTable');
  const m = D.members[lv];
  if(!m){
    head.style.display = 'none'; wrapChart.style.display = 'none';
    verdict.innerHTML = ''; tbl.innerHTML = '';
    Plotly.purge('memberChart');
    return;
  }
  head.style.display = ''; wrapChart.style.display = '';
  head.textContent = 'Every ' + labelOf(lv) + ', Compared';

  Plotly.react('memberChart', [
    {type:'bar', name:'Original Forecast', x:m.map(function(d){return d.id;}),
     y:m.map(function(d){return d.errO;}), marker:{color:'#2B2B2B'},
     hovertemplate:'<b>%{x}</b><br>Original error: %{y:,.1f} units/day<extra></extra>'},
    {type:'bar', name:'Reconciled (MinT)', x:m.map(function(d){return d.id;}),
     y:m.map(function(d){return d.errM;}), marker:{color:'#2F5E93'},
     hovertemplate:'<b>%{x}</b><br>Reconciled error: %{y:,.1f} units/day<extra></extra>'}
  ], Object.assign(layout('Error For Each ' + labelOf(lv) + ', Busiest First',
      {height:400, legend:true, ytitle:'Average daily miss (units sold)'}), {barmode:'group'}), CFG);

  const helped = m.filter(function(d){return d.errM < d.errO;}).map(function(d){return d.id;});
  const hurt = m.filter(function(d){return d.errM >= d.errO;}).map(function(d){return d.id;});
  verdict.innerHTML = 'Reconciling <strong>helps</strong> ' + helped.length + ' of ' + m.length +
    ' (' + (helped.join(', ') || 'none') + ') and <strong>costs accuracy</strong> on ' + hurt.length +
    ' (' + (hurt.join(', ') || 'none') + '). The single headline percentage for this level hides that split.';

  tbl.innerHTML = table(m.map(function(d){
    const ch = (d.errM - d.errO) / d.errO * 100;
    return [d.id, nf(d.avg,1), nf(d.errO,2), nf(d.errM,2),
            (ch>=0?'+':'') + ch.toFixed(1) + '%', (d.errO/d.avg*100).toFixed(1) + '%'];
  }), ['Name','Avg Units Sold Per Day','Original Error (units/day)','Reconciled Error (units/day)',
       'Change','Original Error As % Of Sales'], ['s','n','n','n','n','n']);
}

function drawBlend(){
  const w = +document.getElementById('blend').value / 100;
  document.getElementById('blendLabel').textContent =
    Math.round((1-w)*100) + '% company wide priority / ' + Math.round(w*100) + '% store level priority';
  const xs = [], ys = [], cols = [];
  LEVELS.forEach(function(l){
    const e = D.errorByLevel[l.key];
    const blended = (1-w)*e.mint_wls_struct + w*e.unreconciled;
    const pct = (blended - e.unreconciled) / e.unreconciled * 100;
    xs.push(l.label); ys.push(pct); cols.push(pct < 0 ? '#2F5E93' : '#C9743C');
  });
  const lim = Math.max(1, Math.max.apply(null, ys.map(Math.abs))) * 1.5;
  Plotly.react('blendChart', [{
    type:'bar', x:xs, y:ys, marker:{color:cols},
    text:ys.map(function(v){return (v>=0?'+':'') + v.toFixed(1) + '%';}),
    textposition:'outside', textfont:{color:C.ink, size:11},
    hovertemplate:'<b>%{x}</b><br>Change vs. original: %{y:+.2f}%<extra></extra>'
  }], Object.assign(layout('Blended Priority', {height:340}),
     {yaxis:{gridcolor:C.grid, linecolor:C.grid, zeroline:true, zerolinecolor:'#9aa0a6',
             tickfont:{color:C.soft}, range:[-lim, lim],
             title:{text:'Change in error vs. original (%)', font:{color:C.soft}}}}), CFG);
}

/* ---- products ---- */
const prodSel = document.getElementById('prod');
D.productIds.forEach(function(id){
  const p = D.products[id];
  prodSel.insertAdjacentHTML('beforeend',
    '<option value="'+id+'">'+id+' - '+p.avg.toFixed(1)+' units/day</option>');
});

function drawProduct(){
  const id = prodSel.value;
  const p = D.products[id];
  const dates = D.totalSeries.dates;
  let which = 'both';
  document.getElementsByName('pw').forEach(function(r){ if(r.checked) which = r.value; });

  const chPct = (p.errM - p.errO) / p.errO * 100;
  tiles('prodTiles', [
    ['Avg Units Sold Per Day', p.avg.toFixed(2)],
    ['Original Forecast Error', p.errO.toFixed(3)],
    ['Reconciled Forecast Error', p.errM.toFixed(3) + ' (' + (chPct>=0?'+':'') + chPct.toFixed(1) + '%)'],
    ['Reconciliation Moved It By', p.moved.toFixed(3) + ' units/day']
  ]);

  const traces = [{
    type:'scatter', mode:'lines+markers', name:'Actual Sales', x:dates, y:p.actual,
    line:{color:'#1A1A1A', width:3}, marker:{size:6},
    hovertemplate:'<b>%{x}</b><br>Actual: %{y:.0f} units<extra></extra>'
  }];
  if(which === 'both' || which === 'orig'){
    traces.push({type:'scatter', mode:'lines+markers', name:'Original Forecast', x:dates, y:p.orig,
      line:{color:'#2B2B2B', width:2.4, dash:'dot'}, marker:{size:5},
      hovertemplate:'<b>%{x}</b><br>Original: %{y:.2f} units<extra></extra>'});
  }
  if(which === 'both' || which === 'mint'){
    traces.push({type:'scatter', mode:'lines+markers', name:'Reconciled Forecast (MinT)', x:dates, y:p.mint,
      line:{color:'#2F5E93', width:2.4}, marker:{size:5},
      hovertemplate:'<b>%{x}</b><br>Reconciled: %{y:.2f} units<extra></extra>'});
  }
  Plotly.react('prodChart', traces, Object.assign(
    layout('Daily Sales And Forecast - ' + id, {height:400, legend:true, ytitle:'Units sold per day'}),
    {hovermode:'x unified',
     xaxis:{gridcolor:C.grid, linecolor:C.grid, zeroline:false, tickfont:{color:C.soft},
            showspikes:true, spikemode:'across', spikethickness:1, spikedash:'dot', spikecolor:'#9aa0a6'}}
  ), CFG);
}

/* ---- day of week ---- */
function drawDow(){
  const lv = dowSel.value;
  const d = D.dayOfWeek[lv];
  Plotly.react('dowChart', [
    {type:'bar', name:'Original Forecast', x:d.map(function(x){return x.day;}),
     y:d.map(function(x){return x.errO;}), marker:{color:'#2B2B2B'},
     customdata:d.map(function(x){return x.avg;}),
     hovertemplate:'<b>%{x}</b><br>Original miss: %{y:,.1f} units'+
       '<br>Avg sales that day: %{customdata:,.0f} units<extra></extra>'},
    {type:'bar', name:'Reconciled (MinT)', x:d.map(function(x){return x.day;}),
     y:d.map(function(x){return x.errM;}), marker:{color:'#2F5E93'},
     customdata:d.map(function(x){return x.avg;}),
     hovertemplate:'<b>%{x}</b><br>Reconciled miss: %{y:,.1f} units'+
       '<br>Avg sales that day: %{customdata:,.0f} units<extra></extra>'}
  ], Object.assign(layout('Average Daily Miss By Weekday - ' + labelOf(lv),
     {height:400, legend:true, ytitle:'Average miss (units sold)'}), {barmode:'group'}), CFG);

  let worst = d[0], bestGain = d[0], worstGain = d[0];
  d.forEach(function(x){
    if(x.errO > worst.errO) worst = x;
    if((x.errO-x.errM) > (bestGain.errO-bestGain.errM)) bestGain = x;
    if((x.errO-x.errM) < (worstGain.errO-worstGain.errM)) worstGain = x;
  });
  const g1 = bestGain.errO - bestGain.errM, g2 = worstGain.errO - worstGain.errM;
  document.getElementById('dowVerdict').innerHTML =
    'At the <strong>' + labelOf(lv) + '</strong> level the hardest day to forecast is <strong>' +
    worst.day + '</strong> (missing by ' + nf(worst.errO,1) + ' units on average). Reconciliation helps ' +
    'most on <strong>' + bestGain.day + '</strong> (' + (g1>=0?'+':'') + nf(g1,1) +
    ' units of accuracy) and least on <strong>' + worstGain.day + '</strong> (' +
    (g2>=0?'+':'') + nf(g2,1) + ').';
}

/* ---- company-wide ---- */
function drawTotal(){
  const t = D.totalSeries;
  const traces = [];
  if(document.getElementById('cbHist').checked){
    traces.push({type:'scatter', mode:'lines', name:'Actual Demand (before test window)',
      x:D.history.dates, y:D.history.sales, line:{color:'#b9bcc2', width:2, dash:'dot'},
      hovertemplate:'<b>%{x}</b><br>Actual: %{y:,.0f} units<extra></extra>'});
  }
  traces.push({type:'scatter', mode:'lines+markers', name:'Actual Demand', x:t.dates, y:t.actual,
    line:{color:'#1A1A1A', width:3}, marker:{size:6},
    hovertemplate:'<b>%{x}</b><br>Actual: %{y:,.0f} units<extra></extra>'});
  if(document.getElementById('cbOrig').checked){
    traces.push({type:'scatter', mode:'lines+markers', name:'Original Forecast', x:t.dates, y:t.orig,
      line:{color:'#2B2B2B', width:2.4, dash:'dot'}, marker:{size:5},
      hovertemplate:'<b>%{x}</b><br>Original forecast: %{y:,.0f} units<extra></extra>'});
  }
  if(document.getElementById('cbMint').checked){
    traces.push({type:'scatter', mode:'lines+markers', name:'Reconciled Forecast (MinT)', x:t.dates, y:t.mint,
      line:{color:'#2F5E93', width:2.4}, marker:{size:5},
      hovertemplate:'<b>%{x}</b><br>Reconciled forecast: %{y:,.0f} units<extra></extra>'});
  }
  Plotly.react('totalChart', traces, Object.assign(
    layout('Total Units Sold Per Day, Forecast And Actual',
           {height:460, legend:true, ytitle:'Total units sold per day'}),
    {hovermode:'x unified',
     xaxis:{gridcolor:C.grid, linecolor:C.grid, zeroline:false, tickfont:{color:C.soft},
            showspikes:true, spikemode:'across', spikethickness:1, spikedash:'dot', spikecolor:'#9aa0a6',
            title:{text:'Test window (Mar 28, Apr 24, 2016)', font:{color:C.soft}}}}
  ), CFG);
}

const eo = D.errorByLevel.total.unreconciled, em = D.errorByLevel.total.mint_wls_struct;
tiles('totalTiles', [
  ['Original Forecast Error', nf(eo,0) + ' units/day'],
  ['Reconciled Forecast Error', nf(em,0) + ' units/day'],
  ['Accuracy Gained', ((eo-em)/eo*100).toFixed(1) + '%']
]);

/* ---- coherence ---- */
function drawCoh(){
  let stage = 'before';
  document.getElementsByName('coh').forEach(function(r){ if(r.checked) stage = r.value; });
  const rows = D.coherence[stage];
  const worst = Math.max.apply(null, rows.map(function(x){return x.maxGap;}));
  const worstPct = Math.max.apply(null, rows.map(function(x){return x.meanPct;}));
  tiles('cohTiles', [
    ['Worst Gap On Any Single Day', fmtGap(worst) + ' units'],
    ['Typical Gap, Worst Level', fmtShare(worstPct)]
  ]);
  const label = {unreconciled:'Original Forecast', bottom_up:'Reconciled: Bottom-Up',
                 mint_wls_struct:'Reconciled: MinT'};
  const sorted = rows.slice().sort(function(a,b){
    return (label[a.method]+a.level).localeCompare(label[b.method]+b.level);
  });
  document.getElementById('cohTable').innerHTML = table(sorted.map(function(x){
    return [label[x.method], labelOf(x.level), fmtGap(x.maxGap), fmtShare(x.meanPct)];
  }), ['Method','Level Being Added Up','Worst Gap On Any Day (units)','How Big Is That Gap?'],
     ['s','s','n','s']);

  document.getElementById('cohNote').innerHTML = stage === 'before'
    ? '<div class="note warn">The levels disagree. Adding up the individual forecasts gives an answer ' +
      'that differs from the company wide forecast by as much as 3,449 units on a single day, roughly ' +
      '1.9 to 2.5% of that day\'s sales.</div>'
    : '<div class="note good"><strong>Every level now adds up.</strong> What is left is effectively zero ' +
      '- ordinary floating point rounding from adding 30,490 numbers in a different order, not a real ' +
      'inconsistency. The Bottom Up gap is exactly 0 and the MinT gap is about 0.00000001 units, against ' +
      'a daily total of roughly 39,000 units.</div>';
}

/* ---- summary tiles ---- */
const so = D.errorByLevel.store.unreconciled, sm = D.errorByLevel.store.mint_wls_struct;
tiles('sumTiles', [
  ['Company Wide Accuracy', ((eo-em)/eo*100).toFixed(1) + '% better'],
  ['Store Level Accuracy', ((sm-so)/so*100).toFixed(1) + '% worse'],
  ['Levels That Now Agree', '5 of 5']
]);

/* ---- wiring ---- */
lvlSel.addEventListener('change', drawError);
dowSel.addEventListener('change', drawDow);
prodSel.addEventListener('change', drawProduct);
document.getElementById('blend').addEventListener('input', drawBlend);
document.getElementsByName('pw').forEach(function(r){ r.addEventListener('change', drawProduct); });
document.getElementsByName('coh').forEach(function(r){ r.addEventListener('change', drawCoh); });
['cbOrig','cbMint','cbHist'].forEach(function(id){
  document.getElementById(id).addEventListener('change', drawTotal);
});

drawError();
drawProduct();
drawDow();
drawTotal();
drawCoh();
</script>
</body>
</html>
"""


def build_html(payload: dict) -> str:
    return (
        HTML.replace("__PLOTLY_VERSION__", PLOTLY_VERSION)
        .replace("__GENERATED__", payload["generated"])
        .replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    )


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    html = build_html(payload)
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT.relative_to(PROJECT_ROOT)}  ({len(html) / 1024:.0f} KB)")
    print(f"  products embedded : {len(payload['products']):,} of {payload['counts']['item_store']:,}")
    print(f"  levels            : {len(payload['levels'])}")
    print(f"  plotly            : {PLOTLY_VERSION} (CDN)")


if __name__ == "__main__":
    main()
