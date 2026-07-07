# `penetration_lite` — Parfitt-Collins penetration, lite edition

Standalone distillation of the penetration half of `parfitt_trb`: observed
cumulative penetration on calendar buckets, the ultimate expected penetration
`K`, the full theoretical curve `P(t) = K(1 − e^{−at})`, the **piecewise
promo-aware** composition (each post-promo segment re-anchored on the observed
penetration at its promo via a change of coordinates), p.w.s.d. validation and
the K-stability diagnostic. **Zero imports from `parfitt_trb`** — the folder
can be copied out as-is. Spark is used only to aggregate the transaction log;
everything downstream is numpy/pandas on a tiny per-bucket series.

## Calendar buckets

Three calendar-anchored units, all pure date arithmetic (no date dimension):

| unit | bucket | label |
|---|---|---|
| `iso_week` | ISO calendar week (Mon-aligned, 7 days) | `2023-W48` |
| `iso_fortnight` | pair of consecutive ISO weeks on a fixed epoch grid (14 days; a pair may straddle the year boundary) | `2023-F48` — named after the pair's **first** ISO week, so it is necessarily followed by `2023-F50` |
| `month` | calendar month | `2023-11` |

Bucket 1 is the bucket that **contains** the launch (which may fall mid-bucket).
Ordinals are gap-free by construction (they depend only on each row's date), and
labels exist for any ordinal, including projected future periods. Months
(28–31 days) are treated as equally spaced by the fit — an accepted
approximation. Every tabular output (`to_frame`, `stability`) carries a `label`
column next to the period ordinal.

## Usage

```python
from penetration_lite import (build_penetration, fit, fit_piecewise,
                              plot_penetration, stability,
                              stability_piecewise, validate)

curve = build_penetration(spark_df, unit="iso_fortnight",
                          launch_date="2024-01-01", denominator="dynamic")
fit(curve, smoothing_window=3)          # K, a; smoothing is fit-only, optional
curve.K                                 # ultimate expected penetration
curve.to_frame()                        # period, label, P_observed, P_fitted
curve.origin_iso_week                   # launch labels: '2024-W01' /
curve.origin_iso_fortnight              # '2023-F52' (the pair containing it) /
curve.origin_month                      # '2024-01'

pw = fit_piecewise(curve, promo_periods=["2024-F16"])  # composed promo-aware curve
plot_penetration(curve, piecewise=pw)   # observed vs theoretical + green boost

validate(curve, cutoff_period="2024-W34", promo_periods=["2024-F16"])  # pwsd_full / _holdout
stability(curve)                        # K, a per estimation cutoff
stability_piecewise(curve, ["2024-F16"])  # same, for the piecewise composition
```

### Input columns

Column names are overridable in `build_penetration` — no need to rename your
dataset:

```python
curve = build_penetration(spark_df, card_col="CUSTOMER_ID", date_col="DATE_KEY",
                          brand_col="IS_BRAND", category_col="IS_CATEGORY")
```

Defaults: `shopper_id`, `txn_date`, `is_new_product` (brand flag),
`is_category`. The brand is treated as part of the category.
`synth.simulate_transactions` generates a test panel with a planted `K`, `a`
and an optional promo wave.

### Promo periods and cutoffs by calendar label

`fit_piecewise`, `validate`, `stability_piecewise` and `plot_penetration`
accept promos either as period ordinals (`[16]`, as before) or as the calendar
labels the library itself prints (`["2023-F36"]`, `["2024-W16"]`,
`["2023-11"]`) — no lookup tables or period counting. The same applies to the
estimation cutoffs: `validate(cutoff_period="2024-W34")` and the `cutoffs`
argument of `stability` / `stability_piecewise`, so a training window defined
in ISO weeks picks the right fortnight/month ordinal by itself, wherever the
launch week falls in its pair. A label resolves to the bucket containing its
start day, so a weekly label on a fortnight/month axis lands in the
fortnight/month that contains the week's **Monday** (relevant for weeks
straddling a month boundary). Note that on the wider units the anchor base
`P_obs(t_promo)` may already include part of the boost when the promo falls
mid-bucket, slightly understating `K_inc`. Likewise, a weekly cutoff ending
mid-fortnight/month trains on that **whole** bucket — up to ±1 week of data
versus a window truncated mid-bucket, a limit inherent to the grid.

### K-stability of the piecewise curve

`stability_piecewise(curve, promo_periods)` refits the composed curve at every
estimation cutoff (like `stability`) and reports the composed ultimate
penetration per cutoff. Right after a promo the newest segment has too few
points to fit: `K` falls back to the most recent fitted segment's ceiling and
the `note` column flags it — those rows are fallback, not stability. Promos
after a cutoff are dropped from that fit and noted, as in `validate`.

The promo boost is highlighted by extending the pre-promo baseline for
3 buckets (`iso_week`) / 2 (`iso_fortnight` ≈ 1 month) / 1 (`month`) —
overridable via `baseline_extension`.

## Tests

```powershell
uv run pytest penetration_lite -v
```

Conventions shared with the parent library (kept in sync by hand, by design):
Monday epoch `1970-01-05`, first-week naming for fortnights, discounted
least-squares fit on the difference model `ΔP = a(K − P)` with centred
differences, w = 0.6.
