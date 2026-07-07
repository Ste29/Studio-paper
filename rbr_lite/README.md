# `rbr_lite` — Parfitt-Collins repeat-buying rate, lite edition

Standalone distillation of the RBR half of `parfitt_trb`: the pooled
repeat-buying rate `RBR(t)` on exact-day intervals counted from each shopper's
own trial date, the optional per-cohort curves (cohort = calendar bucket of the
trial), and the stability diagnostics the analyst reads before trusting the
tail — furthest-available rate (the `r→∞` proxy), stabilised mean, plateau
detection, two plots. **Zero imports from `parfitt_trb`** — the folder can be
copied out as-is. Spark is used only to aggregate the transaction log;
everything downstream is numpy/pandas on a tiny per-interval series.

## The measure

`RBR(t) = Σ brand_qty(t) / Σ cat_qty(t)` over the brand's triers — a **ratio
of sums**, never a mean of per-shopper ratios. The conventions, all inherited
from `parfitt_trb`:

- **Intervals are repeat-buying time, not calendar time**: interval `t` covers
  days `(t−1)·P+1 … t·P` after each shopper's own trial (`P =
  period_length_days`, default 28; use 7/14/28 for weekly / fortnightly /
  monthly windows). A purchase on the trial day itself never counts as a
  repeat.
- **Trial** = the card's first brand purchase; with `launch_date` set, earlier
  brand history is ignored and the trial re-dated to the first post-launch
  brand purchase.
- **Only fully-elapsed intervals count**: a shopper contributes to interval
  `t` only once its whole window fits before the analysis date, and
  `n_eligible(t)` reports how many shoppers back each point. **Lapsed buyers
  stay in the base** — eligibility comes from elapsed time, not from
  purchasing — so an interval nobody bought in shows `rbr = 0`, and one nobody
  could reach yet shows `rbr = None`.
- The brand is treated as part of the category (a brand purchase is also a
  category purchase).
- `max_interval` caps the horizon of the **whole** analysis (pooled and cohort
  curves alike); `n_eligible` still counts the triers observable beyond the
  cap, whose early intervals are inside the sums. (This deliberately fixes a
  `parfitt_trb` undercount, since patched there too.)

## Entry cohorts

Cohorts are opt-in: pass `cohort_unit` and every trier is labelled with the
calendar bucket of its trial date — a divergence from `parfitt_trb`, which
groups triers into launch-relative week bands. Three units, pure date
arithmetic (no date dimension):

| unit | bucket | label |
|---|---|---|
| `iso_week` | ISO calendar week (Mon-aligned, 7 days) | `2023-W48` |
| `iso_fortnight` | pair of consecutive ISO weeks on a fixed epoch grid (14 days; a pair may straddle the year boundary) | `2023-F48` — named after the pair's **first** ISO week, so it is necessarily followed by `2023-F50` |
| `month` | calendar month | `2023-11` |

With no `cohort_unit` only the pooled curve is computed (`cohort_series()`
then raises).

## Usage

```python
from rbr_lite import build_rbr, plot_rbr, plot_rbr_cohorts

curve = build_rbr(spark_df, period_length_days=28,
                  launch_date="2024-01-01", cohort_unit="iso_week")
curve.to_frame()          # interval, rbr, brand_qty, cat_qty, n_eligible
curve.last_available()    # (interval, rbr) at the furthest observed interval
curve.plateau()           # first interval where the curve flattens (diagnostic)
curve.stable(6)           # mean RBR from interval 6 on (the stabilised estimate)

plot_rbr(curve, mark_plateau=True)       # pooled RBR(t)
plot_rbr_cohorts(curve)                  # per-cohort curves vs pooled (grey)
curve.cohort_series()     # {cohort label: [(interval, rbr | None), ...]}
```

### Input columns

Column names are overridable in `build_rbr` — no need to rename your dataset:

```python
curve = build_rbr(spark_df, card_col="CUSTOMER_ID", date_col="DATE_KEY",
                  brand_col="IS_BRAND", category_col="IS_CATEGORY",
                  qty_col="UNITS")
```

Defaults: `shopper_id`, `txn_date`, `is_new_product` (brand flag),
`is_category`, `volume` (the summed measure — units, value, whatever the
column holds). `synth.simulate_transactions` generates a test panel with a
planted RBR curve (and an optional per-cohort effect), recovered exactly by
`build_rbr`.

### Reading stability

The share formula ultimately wants the long-run rate, and the tail of the
curve is where the data is thinnest — check `n_eligible` before trusting it.
`last_available()` is the paper's proxy (the furthest observed rate);
`stable(from_interval)` averages the tail once the analyst judges it flat;
`plateau(tol, k)` merely suggests where that flattening starts (first interval
whose rate stays within `tol` for `k` observations). `plot_rbr_cohorts` shows
whether late triers repeat like the early ones — diverging cohort curves mean
the pooled rate mixes different behaviours.

## Tests

```powershell
uv run pytest rbr_lite -v
```

Conventions shared with the parent library (kept in sync by hand, by design):
exact-day RBR intervals (`ceil(days/P)`, 1-based, strict post-trial), ratio of
sums, eligibility from fully-elapsed windows; Monday epoch `1970-01-05` and
first-week naming for fortnights in the cohort calendar. Not ported:
calendar-bucket interval mode, launch-relative cohort bands, pre-launch
cohort handling (pre-launch brand history never counts).
