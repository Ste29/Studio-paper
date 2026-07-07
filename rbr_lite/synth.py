"""Minimal synthetic transaction generator for tests and demos.

Plants a KNOWN repeat-buying rate on a weekly grid (use it with
``period_length_days=7``). Household i trials in week ``w = 1 + (i %
trial_weeks)`` with a brand purchase of volume 1.0; then, for every interval t
that fits inside the horizon, it buys mid-window (day ``trial + 7t - 3``) a
brand line of volume ``r_w(t)`` plus a category-only line of volume
``1 - r_w(t)``. Because the brand is part of the category, each eligible
trier's category volume at interval t is exactly 1.0 and its brand volume
exactly ``r_w(t)``, so the recovered ratio-of-sums equals the planted rate to
float precision -- per cohort, and pooled when ``cohort_effect == 0``.

Planted rate: ``r_w(t) = r_inf + (r0 - r_inf)·e^{-decay·(t-1)} -
cohort_effect·(w-1)``, clipped to [0, 1]. Analyse with
``analysis_date = launch + horizon_weeks*7`` days so trier w has exactly
``horizon_weeks - w + 1`` fully-elapsed intervals, one purchase in each (no
lapsed buyers).
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd


def planted_rate(t: int, w: int = 1, *, r0: float = 0.8, r_inf: float = 0.3,
                 decay: float = 0.5, cohort_effect: float = 0.0) -> float:
    """The rate the generator plants for trial-week w at interval t."""
    r = r_inf + (r0 - r_inf) * math.exp(-decay * (t - 1)) - cohort_effect * (w - 1)
    return min(1.0, max(0.0, r))


def simulate_transactions(*, n_households: int = 300, trial_weeks: int = 6,
                          horizon_weeks: int = 26, r0: float = 0.8,
                          r_inf: float = 0.3, decay: float = 0.5,
                          cohort_effect: float = 0.0,
                          launch: str = "2024-01-01", seed: int = 0
                          ) -> pd.DataFrame:
    """Transaction log with the planted RBR. `launch` should be a Monday so
    trial weeks map 1:1 onto iso_week cohorts. Columns match the default input
    schema: shopper_id, txn_date, is_new_product, is_category, volume."""
    rng = np.random.default_rng(seed)
    origin = date.fromisoformat(launch)
    order = rng.permutation(n_households)      # shuffle ids across trial weeks

    rows = []
    for i in range(n_households):
        h = f"h{order[i]}"
        w = 1 + (i % trial_weeks)
        trial = origin + timedelta(days=(w - 1) * 7)
        rows.append((h, trial.isoformat(), True, True, 1.0))
        t = 1
        while (w - 1) * 7 + 7 * t - 3 <= horizon_weeks * 7:
            r = planted_rate(t, w, r0=r0, r_inf=r_inf, decay=decay,
                             cohort_effect=cohort_effect)
            day = (trial + timedelta(days=7 * t - 3)).isoformat()
            rows.append((h, day, True, True, r))          # brand (⊂ category)
            rows.append((h, day, False, True, 1.0 - r))   # rest of the category
            t += 1
    return pd.DataFrame(rows, columns=["shopper_id", "txn_date",
                                       "is_new_product", "is_category", "volume"])
