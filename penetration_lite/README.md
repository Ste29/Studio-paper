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
                              plot_penetration, stability, validate)

curve = build_penetration(spark_df, unit="iso_fortnight",
                          launch_date="2024-01-01", denominator="dynamic")
fit(curve, smoothing_window=3)          # K, a; smoothing is fit-only, optional
curve.K                                 # ultimate expected penetration
curve.to_frame()                        # period, label, P_observed, P_fitted

pw = fit_piecewise(curve, promo_periods=[16])   # composed promo-aware curve
plot_penetration(curve, piecewise=pw)   # observed vs theoretical + green boost

validate(curve, cutoff_period=20, promo_periods=[16])   # pwsd_full / _holdout
stability(curve)                        # K, a per estimation cutoff
```

Input schema (column names overridable in `build_penetration`): `shopper_id`,
`txn_date`, `is_new_product` (brand flag), `is_category`. The brand is treated
as part of the category. `synth.simulate_transactions` generates a test panel
with a planted `K`, `a` and an optional promo wave.

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
