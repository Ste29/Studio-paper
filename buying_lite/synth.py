"""Minimal synthetic transaction generator for tests and demos.

Plants a KNOWN buying index on a weekly grid. Household types: `others` buy
1.0 of category per week, plain triers buy `heavy_trier` and repeaters
`heavy_repeater` -- but only from their own trial week on (1.0 before it, so
a staggered `trial_weeks > 1` panel exercises the growing bases). All of a
household's weekly volume sits on one line dated the Wednesday of the week;
in the trial week the line splits into a brand line of volume 0.5 (brand is
part of the category) plus a category-only line for the remainder, and a
repeater adds its second brand line the following week. Volumes are binary
fractions (0.5 / 1.0 / `heavy_*`), so with `trial_weeks == 1` the recovered
index equals :func:`planted_index` to float precision -- windowed or
all-history alike, because every weekly volume is constant from week 1.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Tuple

import numpy as np
import pandas as pd


def _counts(n_households: int, trier_share: float,
            repeater_share: float) -> Tuple[int, int, int]:
    """(n_others, n_plain_triers, n_repeaters) from the two shares."""
    n_triers = round(n_households * trier_share)
    n_rep = round(n_triers * repeater_share)
    return n_households - n_triers, n_triers - n_rep, n_rep


def planted_index(*, n_households: int = 200, trier_share: float = 0.3,
                  repeater_share: float = 0.5, heavy_trier: float = 1.5,
                  heavy_repeater: float = 2.5) -> Tuple[float, float]:
    """The steady-state (b_triers, b_repeaters) the generator plants."""
    n_o, n_tn, n_r = _counts(n_households, trier_share, repeater_share)
    avg_all = (n_o * 1.0 + n_tn * heavy_trier + n_r * heavy_repeater) \
        / n_households
    b_triers = ((n_tn * heavy_trier + n_r * heavy_repeater)
                / (n_tn + n_r)) / avg_all
    return b_triers, heavy_repeater / avg_all


def simulate_transactions(*, n_households: int = 200, trier_share: float = 0.3,
                          repeater_share: float = 0.5,
                          heavy_trier: float = 1.5,
                          heavy_repeater: float = 2.5,
                          trial_weeks: int = 1, horizon_weeks: int = 12,
                          launch: str = "2024-01-01", seed: int = 0
                          ) -> pd.DataFrame:
    """Transaction log with the planted buying index. `launch` should be a
    Monday so weeks map 1:1 onto iso_week buckets. Columns match the default
    input schema: shopper_id, txn_date, is_new_product, is_category, volume."""
    if min(heavy_trier, heavy_repeater) < 0.5:
        raise ValueError("heavy_trier / heavy_repeater must be >= 0.5 (the "
                         "trial-week brand line carves 0.5 out of the volume)")
    if not 1 <= trial_weeks < horizon_weeks:
        raise ValueError("need 1 <= trial_weeks < horizon_weeks (a repeater "
                         "buys its second brand line the week after trial)")
    n_o, n_tn, n_r = _counts(n_households, trier_share, repeater_share)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_households)      # shuffle ids across the types

    rows = []
    for i in range(n_households):
        h = f"h{order[i]}"
        is_rep, is_trier = i < n_r, i < n_r + n_tn
        heavy = heavy_repeater if is_rep else heavy_trier
        tw = 1 + (i % trial_weeks) if is_trier else None
        for w in range(1, horizon_weeks + 1):
            day = (date.fromisoformat(launch)
                   + timedelta(days=(w - 1) * 7 + 2)).isoformat()
            v = 1.0 if (tw is None or w < tw) else heavy
            if is_trier and (w == tw or (is_rep and w == tw + 1)):
                rows.append((h, day, True, True, 0.5))         # brand ⊂ category
                rows.append((h, day, False, True, v - 0.5))
            else:
                rows.append((h, day, False, True, v))
    return pd.DataFrame(rows, columns=["shopper_id", "txn_date",
                                       "is_new_product", "is_category",
                                       "volume"])
