# `rbr_lite` — Parfitt-Collins repeat-buying rate, lite edition

Standalone distillation of the RBR half of `parfitt_trb`: the pooled
repeat-buying rate `RBR(t)` on exact-day **or** calendar-bucket intervals
counted from each shopper's own trial date, optional entry-cohort bands with
custom boundaries, and the stability diagnostics the analyst reads before
trusting the tail — furthest-available rate (the `r→∞` proxy), stabilised mean,
plateau detection, two plots. **Zero imports from `parfitt_trb`** — the folder
can be copied out as-is. Spark is used only to aggregate the transaction log;
everything downstream is numpy/pandas on a tiny per-interval series.

## The measure

`RBR(t) = Σ brand_qty(t) / Σ cat_qty(t)` over the brand's triers — a **ratio
of sums**, never a mean of per-shopper ratios. The conventions:

- **Intervals are repeat-buying time, not calendar time**, in one of two
  flavours:
  - *exact-day* (`interval_unit=None`, the default): interval `t` covers days
    `(t−1)·P+1 … t·P` after each shopper's own trial (`P = period_length_days`,
    default 28; use 7/14/28 for weekly / fortnightly / monthly windows).
  - *calendar-bucket* (`interval_unit="iso_week" | "iso_fortnight" | "month"`):
    the interval is the number of buckets between a purchase's bucket and the
    trial's bucket — the next bucket is interval 1, and purchases in the
    trial's **own** bucket are never repeats. `period_length_days` does not
    apply here (passing it raises). A bucket counts only once it has **fully
    elapsed** by the analysis date, so a partial current bucket is excluded —
    stricter than `parfitt_trb`, which includes the partial last bucket.
  Either way, a purchase on the trial day itself never counts as a repeat.
- **Trial** = the card's first brand purchase; with `launch_date` set, earlier
  brand history is ignored and the trial re-dated to the first post-launch
  brand purchase.
- **Only fully-elapsed intervals count**: a shopper contributes to interval
  `t` only once its whole window (or bucket) fits before the analysis date, and
  `n_eligible(t)` reports how many shoppers back each point. **Lapsed buyers
  stay in the base** — eligibility comes from elapsed time, not from
  purchasing — so an interval nobody bought in shows `rbr = 0`, and one nobody
  could reach yet shows `rbr = None`.
- The brand is treated as part of the category (a brand purchase is also a
  category purchase).

## Choosing the horizon

Two knobs bound the observation window; there is no `n_periods` argument — the
point count follows from them:

- **`analysis_date`** is the observation cutoff ("I only see data up to here";
  inclusive, default = the last transaction). It caps *all* data and sets, per
  trier, how many intervals have fully elapsed.
- **`max_interval`** caps the axis of the **whole** analysis (pooled and cohort
  curves alike). `n_eligible` still counts the triers observable beyond the
  cap, whose early intervals are inside the sums.

So `interval_unit="month", max_interval=20, analysis_date="2025-09-30"` gives
at most 20 monthly points ending at (the last month fully elapsed by)
September 2025 — the convenient "20 periods" selection, expressed as a unit +
a cap + a cutoff.

## Entry cohorts

Cohorts are opt-in: pass `cohort_boundaries`, a list of **boundaries** that cut
the triers (by trial date) into bands. Each boundary is either a calendar label
or a date, and closes its band **inclusively**:

| boundary | closes the band at |
|---|---|
| `2023-W48` (ISO week) | Sunday of that week (`2023-12-03`) |
| `2023-F48` (ISO fortnight) | the named week's Monday + 13 days |
| `2023-11` (month) | last day of the month (`2023-11-30`) |
| `2023-11-15` / `date(2023, 11, 15)` | that day itself |

Grammars may be **mixed** in one list. `N` boundaries produce `N+1` bands: the
first runs from the origin to the first boundary, the last stays open until the
analysis date. For example `cohort_boundaries=["2023-W31", "2023-W38"]` yields
three bands labelled

```
≤2023-W31   ·   2023-W32–2023-W38   ·   2023-W39+
```

Boundaries must resolve to **strictly increasing** end dates, all strictly
inside `(origin, analysis_date)` (an empty leading or trailing band raises).
Band order — not lexicographic label order — is preserved through
`cohort_series()`, `cohort_frame()` and the cohort plot. With no
`cohort_boundaries` only the pooled curve is computed (`cohort_series()` then
raises).

> This replaces `parfitt_trb`'s launch-relative week bands: instead of fixed
> "weeks since launch" groupings, you name the calendar cut points yourself.
> Passing every bucket boundary reproduces a one-band-per-bucket split if you
> ever want it.

## Usage

```python
from rbr_lite import build_rbr, plot_rbr, plot_rbr_cohorts

# exact-day intervals, three entry bands split at two ISO weeks
curve = build_rbr(spark_df, period_length_days=28, launch_date="2024-01-01",
                  cohort_boundaries=["2024-W05", "2024-W10"])

# OR: monthly buckets, 20-point horizon up to an explicit cutoff
curve = build_rbr(spark_df, interval_unit="month", max_interval=20,
                  launch_date="2024-01-01", analysis_date="2025-09-30")

curve.to_frame()          # interval, rbr, brand_qty, cat_qty, n_eligible
curve.last_available()    # (interval, rbr) at the furthest observed interval
curve.plateau()           # first interval where the curve flattens (diagnostic)
curve.stable(6)           # mean RBR from interval 6 on (the stabilised estimate)

plot_rbr(curve, mark_plateau=True)       # pooled RBR(t)
plot_rbr_cohorts(curve)                  # per-band curves vs pooled (grey)
curve.cohort_series()     # {band label: [(interval, rbr | None), ...]}
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

Conventions shared with the parent library (the calendar arithmetic is kept in
sync by hand, by design): exact-day RBR intervals (`ceil(days/P)`, 1-based,
strict post-trial), ratio of sums, eligibility from fully-elapsed windows;
Monday epoch `1970-01-05` and first-week naming for fortnights. Two deliberate
divergences from `parfitt_trb`: the calendar-bucket interval mode counts only
**fully elapsed** buckets (the partial current bucket is excluded, where
`parfitt_trb` includes it), and entry cohorts are **calendar-boundary bands**
(`cohort_boundaries`) rather than launch-relative week bands. Still not ported:
pre-launch cohort handling (pre-launch brand history never counts).
