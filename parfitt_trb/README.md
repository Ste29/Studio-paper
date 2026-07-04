# `parfitt_trb` — Parfitt–Collins brand-share prediction

A faithful, tested implementation of Parfitt & Collins (1968), *Use of Consumer
Panels for Brand-Share Prediction*, JMR 5(2):131–145.

```
Market Share = Trial (penetration) × Repeat (RBR) × Buying index
```

It predicts the **stabilised** share of a launched (or promoted) brand from
continuous panel data, long before the share settles in the raw sales series.

---

## Why a single Spark engine

Almost every step is a `groupBy` whose **output is tiny** (a ~100-point
penetration series, an RBR series, a few scalars) — the per-shopper dimension
always collapses. So the code is split in two:

- **`aggregation/`** — the *only* engine-specific code (`SparkAggregator`), split
  by theme: `calendar.py` (date dimension + axis descriptors), `_expr.py` (Spark
  column expressions), `aggregator.py` (the class). Every heavy join / group-by
  (trial identification, interval & period assignment, the RBR and buying
  reductions) runs **in Spark**; only the small,
  already-aggregated tables (one row per interval / period / cohort / scope) are
  collected. No transaction-level or per-card frame is ever pulled to the driver,
  so it scales to a real single-retailer panel of millions of lines.
- **everything else** (`core.py`, `model.py`, `plots.py`) — one numpy/pandas core
  that only ever sees those small tables.

```python
from parfitt_trb import TRBConfig, run_trb

cfg = TRBConfig(launch_date="2024-01-01", period_length_days=14)
res = run_trb(transactions_spark, cfg)   # transactions_spark is a Spark DataFrame
```

The hand-computed anchor tests (traceable to the paper) run through Spark and pin
the Spark column expressions; `parfitt_trb.local_spark.build_local_spark()` gives
a resource-limited local session for tests and the examples.

---

## Input schema

One row per purchase line (column names configurable in `TRBConfig`):

| column           | meaning                                    | default name     |
|------------------|--------------------------------------------|------------------|
| card             | loyalty-card / household id                | `shopper_id`     |
| date             | purchase date                              | `txn_date`       |
| brand flag       | line is the studied brand (bool)           | `is_new_product` |
| category flag    | line is in the reference category (bool)   | `is_category`    |
| measure          | quantity (volume / units / value)          | `volume`         |

The brand is part of the category (`treat_brand_as_category=True` ORs it in).

---

## What it computes

| Quantity | How | Faithful to |
|---|---|---|
| **Trial / penetration** `P(t)=ΣN/ΣF` | cumulative brand triers / category triers, weekly | appendix |
| **Ultimate penetration `K`** | fit `ΔP(t)=a(K−P(t))` (centred diff) by **discounted least squares** (Gilchrist, w=0.6) → `P(t)=K(1−e^{−at})`; optional fit-side centred smoothing of P (`penetration_smoothing_window` — the stored/plotted series stays raw) | appendix, Fig 18 |
| **RBR `R(t,s)`** | pooled brand/category volume over rolling intervals anchored to each shopper's trial, lapsed buyers kept, **furthest available interval** taken (r→∞ proxy) | §RBR, Table 1 |
| **Buying index `B`** | avg category volume of brand buyers / of all category buyers, on the analysis window; **base = triers** by default (`repeaters` optional) | Fig 17 |
| **Share (simple)** | `K × R(last) × B` | p.133, Fig 17 |
| **Share (segmented)** | `Σ Pᵢ × Rᵢ × Bᵢ` over entry cohorts + estimated future cohort | **Table 2** |
| **p.w.s.d.** | weighted RMS relative penetration-forecast error (w=0.6) | appendix, Fig 19 |
| **Promotion effect** | baseline fit before a cutoff vs realised vs re-fit after → "bought" penetration | Fig 12–15 |
| **Piecewise promo curve** | `fit_piecewise_penetration` — one `K(1−e^{−at})` segment per promo, each re-anchored on the observed penetration at its promo (change of coordinates); the composed "true" theoretical curve, also used by validation | Fig 12–15 (extended) |
| **Out-of-sample validation** | `validate_penetration` — fit up to a cutoff, project the held-out weeks (promo-aware), score p.w.s.d. of the whole curve vs the full observed series | appendix |
| **K stability diagnostic** | `penetration_stability` — K, a, observed P per estimation cutoff, to see whether K settles | appendix |

### Key modelling decisions (and where they diverge)

- **Weekly compute, monthly display.** Everything is computed on uniform 7-day
  periods (so the DLS fit sees equal spacing); `display.rollup_*` aggregate to
  `YYYY-MM` for presentation, treating months as fixed-length (a small, accepted
  distortion).
- **Share uses the *projected* `K`** (`predict_share_projected`), faithful to the
  appendix. `predict_share(rbr)` is the simple `trial × rbr × buying` multiplier
  used in the paper's worked example (34%×25%×1.00=8.5%).
- **RBR-stability is the analyst's call.** By default the share takes the
  furthest-available interval; `detect_plateau()` is a *diagnostic* only, and the
  RBR plot (pooled and, via `plot_rbr_cohorts`, per cohort) shows whether it had
  really levelled off. Once you judge it stable from some interval, set
  `TRBConfig(rbr_stable_from=t)` — the simple and segmented shares then use the
  **mean** RBR from `t` on (young cohorts with no points there fall back to their
  furthest-available rate). You can still pass an explicit
  `res.predict_share_projected(rbr_value=...)` or cap with
  `TRBConfig(max_interval=...)`.
- **Cohort share is a sum of contributions, not of RBRs.** `Σ Pᵢ Rᵢ Bᵢ`; the
  single "blended" RBR reported (`blended_rbr`) is the penetration-weighted
  average, never a sum.
- **RBR interval modes** (`rbr_interval_mode`): `"exact"` = `period_length_days`
  windows from each shopper's exact trial date; `"bucket"` = calendar weeks or
  months after the trial bucket (`rbr_bucket_unit`), so RBR(5) is the 5th
  week/month after the trial week/month.
- **Cohort RBR caveat.** Each cohort's rate is read at its furthest available
  interval; at that interval only the *earliest* members of a wide entry window
  are present, biasing the estimate. Keep entry cohorts narrow (the default
  6-week windows) or set custom `cohort_boundaries_weeks` to isolate a sub-group
  (e.g. a price-cut wave).

---

## Quick start

```python
from parfitt_trb import TRBConfig, run_trb
from parfitt_trb import plots
from parfitt_trb.local_spark import build_local_spark

spark = build_local_spark()           # local/dev; in production use the cluster session
df = spark.createDataFrame(transactions_pandas)   # or read straight from the lake

cfg = TRBConfig(launch_date="2024-01-01", period_length_days=14,  # 2-week RBR interval
                analysis_date="2024-09-30")                       # "as of" date
res = run_trb(df, cfg)

res.trial_index                 # observed penetration snapshot
res.penetration.ultimate_penetration   # projected K
res.ultimate_rbr()              # (interval, rate) furthest available
res.buying_index                # overall B (triers)
res.segmented_share()           # Σ Pᵢ Rᵢ Bᵢ   <- headline equilibrium share
res.cohort_table()              # Table 2 as a DataFrame
plots.plot_dashboard(res)       # penetration | RBR | predicted share
```

`TRBConfig` reference: see `config.py` (penetration_method, discount_weight,
penetration_denominator, rbr_interval_mode, rbr_bucket_unit, buying_index_base,
repeater_min_purchases, buying_index_window_days, cohort_boundaries_weeks,
include_prelaunch_cohort, max_interval, rbr_stable_from).

Diagnostics beyond the dashboard: `plots.plot_rbr_cohorts(res)` (per-cohort RBR
curves, to choose the cohort count), `plots.plot_dp_vs_p(pen, promo_periods=...)`
(the `ΔP=a(K−P)` difference model with a fitted line per promo segment), and
`plots.plot_penetration_piecewise(res, promo_periods)` (observed vs the composed
promo-aware curve, with the "bought" boost shaded).

### Calendar granularity (penetration / share / per-period buying index)

By default these calendar-time series are computed and labelled on **weekly**
periods. Two knobs change that (RBR has its own axis; entry cohorts stay weekly):

- `period_unit="week"` (default) / `"fortnight"` / `"month"` — origin-relative
  7-day / 14-day / calendar month buckets, derived by date arithmetic. `"week"`
  and `"fortnight"` label each period with the ISO week of its first day
  (`YYYY-Www`); `"month"` with `YYYY-MM`.
- `period_unit="iso_week"` / `"iso_fortnight"` / `"fiscal_445"` —
  **calendar-anchored** axes built from a date dimension over `[origin,
  analysis]`. `iso_week` uses ISO calendar weeks (`YYYY-Www`, handling
  `2023-W52 → 2024-W01` and 53-week years); `iso_fortnight` pairs consecutive
  ISO weeks on a fixed epoch-aligned grid (every bucket exactly 14 days; a pair
  may straddle a year boundary) and labels each pair after its **first** week
  (`YYYY-Fww`: `2023-F52` covers 2023-W52 + 2024-W01, the next pair is
  `2024-F02`); `fiscal_445` uses retail 4-4-5 periods (`YYYY-Pnn`). Because they
  live on the real calendar grid, a bucket with **no sales** (e.g. an
  out-of-stock week) keeps its slot instead of collapsing onto its neighbour.
- `share_period_unit` (optional) puts the realised-share series on a *different*
  axis from penetration (e.g. penetration on `iso_week`, share on `month`).

```python
res = run_trb(df, TRBConfig(launch_date="2024-01-01", period_unit="month"))
res.period_unit            # 'month'
res.label(3)               # '2024-03'  (period ordinal -> calendar label)
from parfitt_trb.display import label_ratio, label_cumulative
label_ratio(res.share_series, res.period_labels)        # [('2024-01', 0.06), ...]
label_cumulative(res.penetration.series, res.period_labels)
```

---

## Run it

Spark needs a working **Java** runtime (JDK 17+). Everything runs on the
resource-limited local session (`local[2]`, 1 GB), so the suite is slower than a
pure in-memory run but uses the same engine as production.

```powershell
uv run pytest tests/ -v                       # hand-computed anchors, run on Spark
uv run python -m examples.replicate_paper_figures   # -> examples/figures/*.png
uv run python -m examples.usage_template            # full analysis printout + dashboard
```

`examples/usage_template.py` is the template to copy for a real analysis.

---

## Paper-figure replication

`examples/replicate_paper_figures.py` reproduces the **single-dataset** figures
from engineered synthetic panels (`examples/synth.py`):

- Fig 1/3 & 6 — cumulative penetration (success / failure) + projection
- Fig 2/4 & 7 — repeat-buying rate (success / failure)
- Fig 5 — realised share by period
- Fig 9 — penetration by entry cohort + Table 2 reconstruction
- Fig 10 — seasonality inflating apparent (household) penetration
- Fig 12–15 — promotion vs baseline projection ("bought" penetration)
- Fig 17 — the `P × B × R` concept
- Fig 18 — early projection from 12 weeks of data

**Out of scope** (need data this single-analysis tool doesn't have): Fig 8
(predicted-vs-actual scatter over 24 brands), Fig 11 (31 case studies), Fig 16
(psychographic segments — not in the transaction schema), Fig 19 (multi-case
p.w.s.d. scatter; the p.w.s.d. *value* is still computed per analysis).
