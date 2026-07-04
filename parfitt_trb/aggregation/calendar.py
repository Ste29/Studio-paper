"""Calendar machinery for the aggregation layer.

Two concerns, both small and self-contained:

* the **date dimension** and the per-row bucket-label expressions for the
  calendar-anchored units (ISO weeks with year-boundary handling, retail 4-4-5).
  :func:`bucket_label_col` is the single source of the label formula, used both
  for the per-row period column and for the complete in-range axis maps built by
  :func:`axis_maps` (one date-dimension pass), so the two can never drift;
* the **axis descriptors** (:func:`_build_bucket_map`, :func:`_period_labels`)
  that operate on the tiny set of distinct labels in pandas and feed the display
  label rendering.

Anchored units live on the real calendar grid, so a bucket with no sales (e.g.
an out-of-stock week) keeps its slot instead of collapsing onto its neighbour.
The ``'week'`` / ``'month'`` units are plain origin-relative arithmetic and need
no map (see :mod:`parfitt_trb.periods`).
"""
from __future__ import annotations

import pandas as pd

from ..config import CALENDAR_UNITS, TRBConfig
from ..periods import _MONDAY_EPOCH, _P445_THRESHOLDS, period_label

# CALENDAR_UNITS (the calendar-anchored units; 'week'/'month' stay as
# origin-relative arithmetic in :mod:`parfitt_trb.periods`) is defined in
# :mod:`parfitt_trb.config` and re-exported here for the aggregation layer.
# _P445_THRESHOLDS lives in :mod:`parfitt_trb.periods` (single source of truth
# with the scalar mirror periods.p445_period).
__all__ = ["CALENDAR_UNITS", "bucket_label_col", "build_date_dim", "axis_maps"]


# --------------------------------------------------------------------------- #
# Date dimension + per-row bucket-label expressions (Spark).
# --------------------------------------------------------------------------- #
def _iso_year_col(F, ts):
    """ISO week-numbering year: the December tail of a year whose last days fall
    in ISO week 1 belongs to the next year, and vice-versa for the January head."""
    wk = F.weekofyear(ts)
    return (F.when((wk >= 52) & (F.month(ts) == 1), F.year(ts) - 1)
             .when((wk == 1) & (F.month(ts) == 12), F.year(ts) + 1)
             .otherwise(F.year(ts)))


def _p445_col(F, ts):
    """1..12 retail 4-4-5 period from the ISO week number."""
    wk = F.weekofyear(ts)
    col = F.when(wk <= _P445_THRESHOLDS[0], F.lit(1))
    for i, thr in enumerate(_P445_THRESHOLDS[1:], start=2):
        col = col.when(wk <= thr, F.lit(i))
    return col.otherwise(F.lit(12))


def bucket_label_col(F, ts, unit: str):
    """Per-row calendar-bucket label for an anchored ``unit`` as a Spark Column.

    ``iso_week``      -> ``'YYYY-Www'`` (ISO year/week, cross-year safe)
    ``iso_fortnight`` -> ``'YYYY-Fww'`` (pair of consecutive ISO weeks on the
                         Monday-epoch grid, named after the pair's FIRST week)
    ``fiscal_445``    -> ``'YYYY-Pnn'`` (retail 4-4-5 period within the ISO year)
    """
    # 'YYYY-Www' mirrors the scalar spec in periods.iso_week_label (keep in sync).
    if unit == "iso_week":
        return F.format_string("%04d-W%02d", _iso_year_col(F, ts), F.weekofyear(ts))
    if unit == "iso_fortnight":
        # Monday of the pair's first week; mirrors periods.fortnight_first_monday
        # (floor(days/14) == floor(abs_week/2) on the same Monday epoch).
        epoch = F.lit(_MONDAY_EPOCH.isoformat()).cast("date")
        monday = F.date_add(epoch, (F.floor(F.datediff(ts, epoch) / 14) * 14)
                            .cast("int"))
        return F.format_string("%04d-F%02d", _iso_year_col(F, monday),
                               F.weekofyear(monday))
    if unit == "fiscal_445":
        return F.format_string("%04d-P%02d", _iso_year_col(F, ts), _p445_col(F, ts))
    raise ValueError(f"{unit!r} is not a calendar-anchored unit {CALENDAR_UNITS}")


def build_date_dim(spark, start, end):
    """Daily date dimension over ``[start, end]`` (ISO strings or dates) carrying
    every anchored bucket label. One row per day -> tiny, safe to collect."""
    from pyspark.sql import functions as F
    days = spark.sql(
        f"SELECT explode(sequence(to_date('{start}'), to_date('{end}'), "
        "interval 1 day)) AS d"
    )
    out = days
    for unit in CALENDAR_UNITS:
        out = out.withColumn(unit, bucket_label_col(F, F.col("d"), unit))
    return out


def axis_maps(spark, start, end, units) -> dict:
    """Gap-free ordinal maps ``{unit: {label: 1..N}}`` for the requested
    calendar-anchored ``units``, built from a SINGLE date-dimension pass.

    Because the day range is dense, every bucket in range appears -- including
    ones with no transactions -- so the dense ordinal maps leave no calendar gaps.
    Non-anchored units (week/month) are skipped; an empty request issues no job."""
    units = [u for u in dict.fromkeys(units) if u in CALENDAR_UNITS]
    if not units:
        return {}
    # one Spark job for the whole dimension, then collapse to first-date-per-label
    # in pandas (date-range sized, tiny) for each requested unit.
    dim = build_date_dim(spark, start, end).select("d", *units).toPandas()
    return {u: _build_bucket_map(dim.groupby(u)["d"].min()) for u in units}


# --------------------------------------------------------------------------- #
# Axis descriptors (operate on the small set of distinct labels -> pandas).
# --------------------------------------------------------------------------- #
def _build_bucket_map(first_dates: pd.Series) -> dict:
    """Dense chronological ordinal map {label: 1..N} from a Series indexed by
    bucket label holding each label's first observed date."""
    return {b: i for i, b in enumerate(first_dates.sort_values().index, start=1)}


def _period_labels(periods, cfg: TRBConfig, origin, bucket_to_period: dict) -> dict:
    """Map calendar-axis period ordinals -> display labels. Anchored units invert
    the dense ordinal map (so even empty buckets get their real calendar label);
    week/month modes derive the label from the ordinal and the origin. `periods`
    is the small set of ordinals actually used."""
    if cfg.period_unit in CALENDAR_UNITS:
        inv = {p: b for b, p in bucket_to_period.items()}
        return {int(p): inv.get(int(p), str(int(p))) for p in periods}
    return {int(p): period_label(int(p), origin, cfg.period_unit) for p in periods}
