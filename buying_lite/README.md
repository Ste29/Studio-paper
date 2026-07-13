# `buying_lite` — Parfitt-Collins buying-rate index, lite edition

Standalone distillation of the buying-index third of `parfitt_trb`: the
headline `B` that enters the share prediction (Trial × RBR × **B**), on both
membership bases (Parfitt's triers, Charan's repeaters), the growing-base
per-bucket series `B(t)` and the per-entry-cohort `B_i` diagnostic, with two
plots. **Zero imports from `parfitt_trb`** — the folder can be copied out
as-is. Spark is used only to aggregate the transaction log; everything
downstream is numpy/pandas on a tiny per-bucket series.

## The measure

`B` = (category volume per capita of the brand's buyers) / (category volume
per capita of **all** category buyers), both measured on the **same window**.
`B > 1` means the brand recruits heavier-than-average category buyers.

Following the paper ("the amount of Field A purchased in period *s* beginning
at time *t − s*"), the headline `B` is measured on a window of `window_days`
days ending at the analysis date:

- **The window narrows the volume, never the bases**: members are everyone
  seen up to the analysis date, and a member with no purchases inside the
  window weighs 0 in the per-capita average instead of dropping out.
- **`window_days` is deliberately required.** Pass `None` explicitly for the
  all-history-from-launch variant — the `parfitt_trb` / monolith default,
  which is *not* the paper's definition (it smears early heavy-buying over
  the whole life of the launch).

Two bases are always computed, no switch to choose:

- `b_triers` — Parfitt's original: every card with ≥ 1 brand purchase.
- `b_repeaters` — Charan's restatement: cards with ≥ `repeater_min_purchases`
  brand purchase **lines** (default 2; `None` when no repeaters exist yet).

The brand is treated as part of the category; pre-launch history never counts
(not as volume, not as membership; with `launch_date` set, the trial is the
first **post-launch** brand purchase).

## The series B(t) — and why it diverges from `parfitt_trb`

`B(t)` evaluates the same index on each calendar bucket with **growing
bases**: at bucket `t` the triers / repeaters / category buyers are the
members seen up to the **end of bucket `t`** — the paper's reading.
`parfitt_trb`'s per-period series instead uses a fixed panel (the whole
dataset's final membership in every bucket, which also counts the pre-trial
category volume of *future* triers in the early buckets); that fixed-panel
diagnostic was deliberately **not ported**. Membership is resolved at bucket
granularity: a card trialling mid-bucket counts as a trier for that whole
bucket, pre-trial volume included.

There is **no separate `stability()`**: unlike the penetration `K`, `B` has
no fit to converge, so watching `B(t)` settle *is* the stability check. The
last bucket usually contains the analysis date mid-bucket: the point is kept
(numerator and denominator share the same partial window) and a note is
**printed** by `to_frame()` and `plot_buying_index()` whenever that happens.

## Entry cohorts

Opt-in via `cohort_unit`: every trier is labelled with the calendar bucket of
its trial date (iso_week / iso_fortnight / month, as in `rbr_lite` — not the
launch-relative bands of `parfitt_trb`). Each cohort's `B_i` uses the **same
window** as the headline `B`, the cohort's full trier membership as base, and
the all-buyer average as denominator — the `B_i` column of the segmented
Table-2 share model.

## Usage

```python
from buying_lite import build_buying_index, plot_buying_index, plot_buying_cohorts

bi = build_buying_index(spark_df, window_days=91,        # ~ a quarter
                        unit="iso_week", launch_date="2024-01-01",
                        cohort_unit="iso_week")
bi.b_triers, bi.b_repeaters   # the headline indices on the window
bi.summary()                  # scope, n_members, cat_qty, avg_per_member, b
bi.to_frame()                 # period, label, b_triers, b_repeaters, volumes, bases
bi.cohort_frame()             # cohort, n_triers, cat_qty, b
bi.window_start               # first day whose volume enters the headline B

plot_buying_index(bi)         # B(t) both bases + 1.0 parity + windowed B marked
plot_buying_cohorts(bi)       # B_i bars per entry cohort
```

### Input columns

Column names are overridable in `build_buying_index` — no need to rename your
dataset:

```python
bi = build_buying_index(spark_df, window_days=91, card_col="CUSTOMER_ID",
                        date_col="DATE_KEY", brand_col="IS_BRAND",
                        category_col="IS_CATEGORY", qty_col="UNITS")
```

Defaults: `shopper_id`, `txn_date`, `is_new_product` (brand flag),
`is_category`, `volume` (the summed measure — units, value, whatever the
column holds). `synth.simulate_transactions` generates a test panel with a
planted index (`synth.planted_index` gives its closed form), recovered
exactly by `build_buying_index`.

## Tests

```powershell
uv run pytest buying_lite -v
```

Conventions shared with the parent library (kept in sync by hand, by design):
per-capita ratio with fixed all-time bases and zero weight for silent
members, category volume scoped on/after the launch origin, repeaters counted
on brand purchase lines; Monday epoch `1970-01-05` and first-week naming for
fortnights in the calendar. Not ported, on purpose: the fixed-panel
per-period series (replaced by the growing-base `B(t)`), the optional-window
default (`window_days` is required here), the `buying_index_base` switch
(both bases always computed), the launch-relative cohort bands and any
stability helper (`B(t)` is the stability).
