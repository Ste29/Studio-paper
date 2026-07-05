"""Minimal synthetic transaction generator for tests and demos.

Plants a known penetration curve P(t) = K(1 - e^{-a t}) (weekly grid): every
household makes a category purchase each week from week 1, and triers make
their first brand purchase on the week that keeps the cumulative trier count on
the planted curve. An optional promo adds an extra wave of triers from
`promo_week` on. Only first purchases matter to penetration, so no repeat/RBR
mechanics are simulated.
"""
from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd


def simulate_transactions(*, n_households: int = 2000, weeks: int = 30,
                          K: float = 0.35, a: float = 0.20,
                          launch: str = "2024-01-01", seed: int = 0,
                          promo_week: Optional[int] = None,
                          promo_K: float = 0.0) -> pd.DataFrame:
    """One row per (household, week) category purchase; the trial week's row is
    flagged as a brand purchase. Columns match the default input schema:
    shopper_id, txn_date, is_new_product, is_category, volume."""
    rng = np.random.default_rng(seed)
    origin = date.fromisoformat(launch)
    H = n_households

    def dstr(week: int) -> str:
        return (origin + timedelta(days=(week - 1) * 7 + 2)).isoformat()

    gw = np.arange(1, weeks + 1)
    cum = np.round(K * (1 - np.exp(-a * gw)) * H).astype(int)
    if promo_week is not None and promo_K > 0:
        extra = np.where(gw >= promo_week,
                         np.round(promo_K * (1 - np.exp(-0.5 * (gw - promo_week + 1)))
                                  * H).astype(int), 0)
        cum = cum + extra
    cum = np.minimum(np.maximum.accumulate(cum), H)

    order = rng.permutation(H)
    trial_week = np.zeros(H, dtype=int)                 # 0 = never tries
    idx = 0
    for gi, w in enumerate(gw):
        while idx < cum[gi]:
            trial_week[order[idx]] = int(w)
            idx += 1

    rows = []
    for h in range(H):
        wt = int(trial_week[h])
        for w in range(1, weeks + 1):
            rows.append((f"h{h}", dstr(w), bool(wt and w == wt), True, 1.0))
    return pd.DataFrame(rows, columns=["shopper_id", "txn_date",
                                       "is_new_product", "is_category", "volume"])
