"""Shared test helpers: build a transaction DataFrame from compact rows."""
from __future__ import annotations

from typing import List

import pandas as pd


def row(sid, d, brand, cat, vol):
    """One purchase line. units/value mirror volume; tests exercise 'volume'."""
    return dict(shopper_id=sid, txn_date=d, is_new_product=bool(brand),
                is_category=bool(cat), volume=float(vol),
                units=float(vol), value=float(vol))


def make_df(rows: List[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def approx(a, b, tol=1e-9):
    return a is not None and abs(a - b) <= tol
