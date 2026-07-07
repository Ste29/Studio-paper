"""Calendar buckets for the lite penetration model.

Three calendar-anchored units, all with UNIFORM buckets whose position is pure
date arithmetic (no date dimension, no label->ordinal maps):

  ``iso_week``      ISO calendar weeks (Monday-aligned, 7 days).
  ``iso_fortnight`` pairs of consecutive ISO weeks on a fixed grid counted from
                    the Monday epoch -- every bucket is exactly 14 days and a
                    pair may straddle a year boundary (2023-W52 + 2024-W01).
  ``month``         calendar months (year*12 + month arithmetic).

Period ordinals are 1-based and gap-free by construction: the ordinal of a
transaction depends only on its own date (bucket index minus the origin's
bucket index), so a bucket with no sales keeps its slot on the axis. Ordinal 1
is the bucket that CONTAINS the origin (launch), which may fall mid-bucket.

Spark computes ONLY integer ordinals (datediff arithmetic); labels are derived
in pure Python from the ordinal, so there is no Spark-vs-Python ISO drift and
labels exist for any ordinal >= 1, including projected future periods.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Union

import pandas as pd

DateLike = Union[str, date, datetime, pd.Timestamp]

UNITS = ("iso_week", "iso_fortnight", "month")

# A Monday, so absolute week indices align to ISO week starts.
MONDAY_EPOCH = date(1970, 1, 5)


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


def _check_unit(unit: str) -> None:
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {UNITS}, got {unit!r}")


# --------------------------------------------------------------------------- #
# Absolute bucket indices (scalar) and the 1-based period ordinal.
# --------------------------------------------------------------------------- #
# Bucket length in days for the day-based (non-month) units -- the single fact
# that tells iso_week (7) from iso_fortnight (14).
_BUCKET_DAYS = {"iso_week": 7, "iso_fortnight": 14}


def bucket_index(d: date, unit: str) -> int:
    """Absolute bucket index of a date for `unit` (differences give ordinals)."""
    _check_unit(unit)
    if unit == "month":
        return d.year * 12 + (d.month - 1)
    return (d - MONDAY_EPOCH).days // _BUCKET_DAYS[unit]


def period_of(d: DateLike, origin: DateLike, unit: str) -> int:
    """1-based period ordinal of `d` on the origin-anchored axis (scalar
    mirror of :func:`period_col` -- keep the two in sync)."""
    return bucket_index(as_date(d), unit) - bucket_index(as_date(origin), unit) + 1


def period_col(F, ts_col, unit: str, origin: DateLike):
    """1-based period ordinal as a Spark Column (integers only, no labels).

    `F` is ``pyspark.sql.functions``; rows before the origin bucket map to <= 0
    (callers keep ``period >= 1``).
    """
    _check_unit(unit)
    o = as_date(origin)
    if unit == "month":
        return (F.year(ts_col) * 12 + F.month(ts_col) - 1) \
            - F.lit(bucket_index(o, "month")) + 1
    days = _BUCKET_DAYS[unit]
    epoch = F.lit(MONDAY_EPOCH.isoformat()).cast("date")
    return (F.floor(F.datediff(ts_col, epoch) / days)
            - F.lit(bucket_index(o, unit)) + 1)


# --------------------------------------------------------------------------- #
# Labels (pure Python, valid for ANY ordinal >= 1, including future periods).
# --------------------------------------------------------------------------- #
def bucket_start(period: int, origin: DateLike, unit: str) -> date:
    """First calendar day of the bucket with ordinal `period`."""
    _check_unit(unit)
    o = as_date(origin)
    p = int(period)
    if unit == "month":
        mi = bucket_index(o, "month") + (p - 1)
        return date(mi // 12, mi % 12 + 1, 1)
    days = _BUCKET_DAYS[unit]
    return MONDAY_EPOCH + timedelta(days=(bucket_index(o, unit) + p - 1) * days)


_LABEL_RE = re.compile(r"^(\d{4})-(?:([WF])(\d{2})|(\d{2}))$")


def parse_period_label(label: str) -> date:
    """First calendar day of a period label (inverse of :func:`period_label`).

    'YYYY-Www' / 'YYYY-Fww' -> Monday of that ISO week (a fortnight is named
    after its first week); 'YYYY-MM' -> first of the month. The day is meant to
    be fed to :func:`period_of`, which maps it into the bucket CONTAINING it --
    so a weekly label resolves correctly on a fortnight or month axis too.
    """
    m = _LABEL_RE.match(label.strip())
    if not m:
        raise ValueError(f"cannot parse period label {label!r}: expected "
                         "'YYYY-Www', 'YYYY-Fww' or 'YYYY-MM'")
    year = int(m.group(1))
    if m.group(4) is not None:                       # 'YYYY-MM'
        month = int(m.group(4))
        if not 1 <= month <= 12:
            raise ValueError(f"month out of range in period label {label!r}")
        return date(year, month, 1)
    week = int(m.group(3))
    try:
        return date.fromisocalendar(year, week, 1)
    except ValueError as e:
        raise ValueError(f"invalid ISO week in period label {label!r}: {e}") from e


def period_label(period: int, origin: DateLike, unit: str) -> str:
    """Calendar label of the bucket with ordinal `period`.

    ``iso_week`` -> 'YYYY-Www'; ``iso_fortnight`` -> 'YYYY-Fww' named after the
    pair's FIRST ISO week (2023-W52+2024-W01 -> '2023-F52', necessarily followed
    by '2024-F02'); ``month`` -> 'YYYY-MM'.
    """
    start = bucket_start(period, origin, unit)
    if unit == "month":
        return f"{start.year:04d}-{start.month:02d}"
    iso = start.isocalendar()
    marker = "F" if unit == "iso_fortnight" else "W"
    return f"{iso[0]:04d}-{marker}{iso[1]:02d}"
