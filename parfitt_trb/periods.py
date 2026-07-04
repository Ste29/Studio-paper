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


def fortnight_first_monday(d: date) -> date:
    """Monday of the FIRST ISO week of ``d``'s epoch-aligned fortnight.

    Fortnights pair consecutive ISO weeks on a fixed grid counted from the
    Monday epoch, so every bucket is exactly 14 days and a pair may straddle a
    year boundary. Scalar mirror of the Spark expression in
    aggregation.calendar.bucket_label_col('iso_fortnight') -- keep in sync."""
    fw = abs_week_index(d) // 2
    return _MONDAY_EPOCH + timedelta(days=fw * 14)


def iso_fortnight_label(d: date) -> str:
    """'YYYY-Fww' label of ``d``'s fortnight: the ISO year-week of the pair's
    FIRST week (2023-W52 + 2024-W01 -> '2023-F52')."""
    iso = fortnight_first_monday(d).isocalendar()
    return f"{iso[0]:04d}-F{iso[1]:02d}"


def period_label(period: int, origin: date, unit: str) -> str:
    """Calendar label of a DERIVED period ordinal (no precomputed bucket column).

    'week' / 'fortnight' -> the ISO-week label of the period's representative
    date (its first day); 'month' -> the 'YYYY-MM' label of the period's month.
    """
    if unit == "month":
        mi = month_index(origin) + (period - 1)        # inverse of month_index
        return f"{mi // 12:04d}-{mi % 12 + 1:02d}"
    if unit == "fortnight":
        return iso_week_label(origin + timedelta(days=(int(period) - 1) * 14))
    return iso_week_label(week_repr_date(period, origin))


# Retail 4-4-5: ISO weeks 1-4 -> period 1, 5-8 -> 2, 9-13 -> 3 (4+4+5 = 13 weeks
# per quarter), repeating; weeks 48+ fall in period 12. Single source of truth --
# the Spark expression in aggregation.calendar mirrors this.
_P445_THRESHOLDS = (4, 8, 13, 17, 21, 26, 30, 34, 39, 43, 47)


def p445_period(d: date) -> int:
    """1..12 retail 4-4-5 period of a date (scalar mirror of the Spark column)."""
    wk = d.isocalendar()[1]
    for i, thr in enumerate(_P445_THRESHOLDS, start=1):
        if wk <= thr:
            return i
    return 12


def extended_period_label(period: int, origin: date, unit: str) -> str:
    """Calendar label for ANY period ordinal >= 1 on an origin-anchored axis,
    including ordinals past the observed range (projected periods).

    The anchored ordinal maps are dense and start at the origin's bucket, so
    future ordinals continue arithmetically: iso_week buckets are exactly 7 days
    apart, and every ISO year has exactly 12 fiscal 4-4-5 periods.
    """
    period = int(period)
    if unit in ("week", "fortnight", "month"):
        return period_label(period, origin, unit)
    if unit == "iso_week":
        return iso_week_label(origin + timedelta(days=(period - 1) * 7))
    if unit == "iso_fortnight":
        return iso_fortnight_label(fortnight_first_monday(origin)
                                   + timedelta(days=(period - 1) * 14))
    if unit == "fiscal_445":
        mi = origin.isocalendar()[0] * 12 + (p445_period(origin) - 1) + (period - 1)
        return f"{mi // 12:04d}-P{mi % 12 + 1:02d}"
    return str(period)
