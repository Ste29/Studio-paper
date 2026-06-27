"""Synthetic consumer-panel generator for the examples / figure replication.

It plants a known penetration curve P(t)=K(1-e^{-a t}) and a known repeat-buying
schedule so the model's outputs can be checked against the intended shapes.

Mechanics that matter for fidelity:
  * EVERY household buys the category on a regular occasion grid (one purchase
    every `cat_interval` weeks). By default the purchase goes to a competitor, so
    triers and non-triers are equally weighted category buyers -> buying index ≈ 1
    (heavy buyers can be added to push it above 1).
  * Brand triers are scheduled on that grid to follow the penetration curve. The
    trial is a brand purchase; afterwards each occasion goes to the brand with a
    probability that decays from `rbr_start` to a stable level (optionally
    cohort-dependent), reproducing the declining-then-flat RBR.

Because occasions are `cat_interval` weeks apart, analyse the output with
``period_length_days = cat_interval * 7`` so RBR interval k == the k-th occasion.
"""
from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Callable, Optional

import numpy as np
import pandas as pd


def simulate_panel(*, n_households: int = 4000, weeks: int = 40,
                   K: float = 0.34, a: float = 0.18,
                   rbr_start: float = 0.45, rbr_stable: float = 0.25,
                   decay: float = 0.6, cat_interval: int = 2,
                   launch: str = "2024-01-01", seed: int = 0,
                   stable_by_week: Optional[Callable[[int], float]] = None,
                   heavy_frac: float = 0.0, heavy_mult: float = 1.6,
                   entry_weeks: Optional[int] = None,
                   promo_week: Optional[int] = None, promo_K: float = 0.0,
                   promo_rbr: float = 0.06) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    origin = date.fromisoformat(launch)
    grid = list(range(1, weeks + 1, cat_interval))          # category occasion weeks
    if entry_weeks is None:
        entry_weeks = weeks

    def dstr(week: int) -> str:
        return (origin + timedelta(days=(week - 1) * 7 + 2)).isoformat()

    H = n_households
    heavy = rng.random(H) < heavy_frac

    # cumulative trier target per grid week (curve plateaus after `entry_weeks`,
    # so trials stop while observation continues -> late cohorts can still mature)
    gw = np.array(grid)
    cum = np.round(K * (1 - np.exp(-a * gw)) * H).astype(int)
    in_window = gw <= entry_weeks
    if in_window.any():                          # hold flat at the last in-window value
        cum = np.where(in_window, cum, cum[in_window].max())
    cum = np.minimum(np.maximum.accumulate(cum), H)
    if promo_week is not None and promo_K > 0:
        extra = np.where(gw >= promo_week,
                         np.round(promo_K * (1 - np.exp(-0.5 * (gw - promo_week + 1))) * H).astype(int), 0)
        cum = np.minimum(cum + extra, H)

    order = rng.permutation(H)
    trial_week = np.zeros(H, dtype=int)
    idx = 0
    for gi, w in enumerate(grid):
        while idx < cum[gi]:
            trial_week[order[idx]] = w
            idx += 1

    rows = []
    for h in range(H):
        wt = int(trial_week[h])
        vmul = heavy_mult if heavy[h] else 1.0
        if promo_week is not None and wt >= promo_week and wt > 0:
            stable = promo_rbr                                   # deal-seeker cohort
        elif stable_by_week is not None and wt > 0:
            stable = stable_by_week(wt)
        else:
            stable = rbr_stable
        for occ in grid:
            if wt == 0 or occ < wt:
                brand = False                                   # competitor
            elif occ == wt:
                brand = True                                    # trial
            else:
                k = (occ - wt) // cat_interval
                brand = rng.random() < stable + (rbr_start - stable) * math.exp(-decay * k)
            rows.append((f"h{h}", dstr(occ), bool(brand), True, vmul))

    return pd.DataFrame(rows, columns=["shopper_id", "txn_date",
                                       "is_new_product", "is_category", "volume"])
