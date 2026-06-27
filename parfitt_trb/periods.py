"""Calendar <-> period helpers shared by the backends and the display rollups.

The model COMPUTES on uniform weekly periods (7-day buckets counted from the
calendar origin = the launch date, or the first trial when no launch is given).
This keeps the periods equally spaced, which the discounted-least-squares
penetration fit assumes. Calendar month *labels* are attached only for display,
where months are treated as fixed-length (an accepted, documented simplification).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Union

import numpy as np
import pandas as pd

DateLike = Union[str, date, datetime, pd.Timestamp]

# A Monday, so absolute week indices align to ISO-style week starts.
_MONDAY_EPOCH = date(1970, 1, 5)


def as_date(x: DateLike) -> date:
    """Coerce a string / datetime / Timestamp to a plain ``datetime.date``."""
    if isinstance(x, str):
        return date.fromisoformat(x[:10])
    if isinstance(x, pd.Timestamp):
        return x.date()
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    raise TypeError(f"cannot interpret {x!r} as a date")


def week_period(days_since_origin: "np.ndarray | int") -> "np.ndarray | int":
    """1-based weekly period index: day 0..6 -> period 1, day 7..13 -> 2, ..."""
    return np.floor_divide(days_since_origin, 7) + 1


def week_repr_date(period: int, origin: date) -> date:
    """Representative calendar date of a weekly period (its first day)."""
    return origin + timedelta(days=(int(period) - 1) * 7)


def abs_week_index(d: date) -> int:
    """Calendar-aligned absolute week number (Monday epoch). Used by bucket-mode
    RBR so 'weeks after trial' are counted on a shared calendar grid."""
    return (d - _MONDAY_EPOCH).days // 7


def month_index(d: date) -> int:
    """Absolute month number = year*12 + (month-1); differences give month gaps."""
    return d.year * 12 + (d.month - 1)


def month_label(d: date) -> str:
    """'YYYY-MM' display label."""
    return f"{d.year:04d}-{d.month:02d}"


def iso_week_label(d: date) -> str:
    """'YYYY-Www' ISO-week display label."""
    iso = d.isocalendar()
    return f"{iso[0]:04d}-W{iso[1]:02d}"


def period_label(period: int, origin: date, unit: str) -> str:
    """Calendar label of a DERIVED period ordinal (no precomputed bucket column).

    'week'  -> the ISO-week label of the period's representative date (its first
               day); 'month' -> the 'YYYY-MM' label of the period's month.
    """
    if unit == "month":
        mi = month_index(origin) + (period - 1)        # inverse of month_index
        return f"{mi // 12:04d}-{mi % 12 + 1:02d}"
    return iso_week_label(week_repr_date(period, origin))
