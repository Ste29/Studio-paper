"""Spark column expressions for the calendar period, the RBR interval and the
entry cohort. These mirror the calendar maths so the heavy reductions can run
entirely in Spark (no Python UDFs)."""
from __future__ import annotations

from ..cohorts import cohort_order
from ..config import TRBConfig
from ..periods import _MONDAY_EPOCH
from .calendar import CALENDAR_UNITS, bucket_label_col


def _period_col(F, ts_col, cfg: TRBConfig, origin_iso: str, origin_month: int,
                bucket_to_period: dict):
    """1-based calendar-axis period ordinal as a Spark Column. Rows before the
    origin / with an unknown bucket map to <= 0 (callers keep period >= 1)."""
    if cfg.period_unit in CALENDAR_UNITS:
        if not bucket_to_period:
            return F.lit(0)
        pairs = []
        for label, period in bucket_to_period.items():
            pairs += [F.lit(label), F.lit(int(period))]
        # The row's own calendar label is derived (single source: calendar module),
        # then mapped to its gap-free ordinal over the full in-range axis.
        return F.coalesce(
            F.create_map(*pairs)[bucket_label_col(F, ts_col, cfg.period_unit)],
            F.lit(0))
    origin = F.lit(origin_iso).cast("date")
    if cfg.period_unit == "week":
        return F.floor(F.datediff(ts_col, origin) / 7) + 1
    if cfg.period_unit == "fortnight":
        return F.floor(F.datediff(ts_col, origin) / 14) + 1
    # month: (year*12 + month - 1) - origin_month + 1
    return (F.year(ts_col) * 12 + F.month(ts_col) - 1) - F.lit(origin_month) + 1


def _abs_week_col(F, col):
    """floor(days since the Monday epoch / 7) — the calendar-aligned week index."""
    return F.floor(F.datediff(col, F.lit(_MONDAY_EPOCH.isoformat()).cast("date")) / 7)


def _month_idx_col(F, col):
    return F.year(col) * 12 + F.month(col) - 1


def _interval_col(F, ts_col, ref_col, cfg: TRBConfig):
    """1-based RBR interval index of `ts_col` relative to the trial `ref_col`."""
    if cfg.rbr_interval_mode == "exact":
        return F.ceil(F.datediff(ts_col, ref_col) / cfg.period_length_days)
    if cfg.rbr_bucket_unit == "week":
        return _abs_week_col(F, ts_col) - _abs_week_col(F, ref_col)
    return _month_idx_col(F, ts_col) - _month_idx_col(F, ref_col)


def _max_interval_col(F, trial_col, analysis_iso: str, analysis_month: int,
                      cfg: TRBConfig):
    """Highest RBR interval whose whole window has elapsed by the analysis date."""
    if cfg.rbr_interval_mode == "exact":
        adate = F.lit(analysis_iso).cast("date")
        return F.floor(F.datediff(adate, trial_col) / cfg.period_length_days)
    if cfg.rbr_bucket_unit == "week":
        adate = F.lit(analysis_iso).cast("date")
        return _abs_week_col(F, adate) - _abs_week_col(F, trial_col)
    return F.lit(analysis_month) - _month_idx_col(F, trial_col)


def _cohort_col(F, entry_week_col, boundaries, include_prelaunch: bool):
    """Entry-cohort label as a Spark Column (pure when-chain, no Python UDF).
    The label strings come from :func:`cohorts.cohort_order` so the three label
    encodings (here, ``cohort_label``, ``cohort_order``) cannot drift."""
    labels = cohort_order(boundaries, include_prelaunch)
    # cohort_order: [PRELAUNCH?, one bounded label per boundary, final '+w'].
    first_label = labels[0]                          # the <=0 / earliest bucket
    bounded = labels[1:] if include_prelaunch else labels
    col = F.when(entry_week_col <= 0, F.lit(first_label))
    for b, label in zip(boundaries, bounded):        # bounded[-1] is the '+w' tail
        col = col.when(entry_week_col <= b, F.lit(label))
    return col.otherwise(F.lit(bounded[-1]))
