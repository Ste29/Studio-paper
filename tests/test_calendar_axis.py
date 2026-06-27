"""Calendar-axis tests: the penetration / realised-share / per-period buying
index can be computed and displayed at week or month granularity, or driven by a
precomputed bucket-label column (which also handles non-daily feeds and labels
that cross a year boundary). Cohorts stay weekly (Parfitt Table 2)."""
from __future__ import annotations

import pytest

from parfitt_trb import TRBConfig, run_trb
from tests.helpers import approx, make_df, row


# --------------------------------------------------------------------------- #
# Derived month axis: penetration on calendar months (hand-computed)
# --------------------------------------------------------------------------- #
def _month_rows():
    return [
        row("b1", "2024-01-05", True, True, 1),    # brand trial, month 1 (2024-01)
        row("c1", "2024-01-07", False, True, 1),   # cat entrant month 1
        row("c2", "2024-02-06", False, True, 1),   # cat entrant month 2
        row("c3", "2024-03-07", False, True, 1),   # cat entrant month 3
    ]


def test_month_axis_dynamic_vs_static():
    base = dict(launch_date="2024-01-01", analysis_date="2024-03-31", period_unit="month")
    dyn = run_trb(make_df(_month_rows()),
                  TRBConfig(penetration_denominator="dynamic", **base),
                  project_penetration=False)
    sta = run_trb(make_df(_month_rows()),
                  TRBConfig(penetration_denominator="static", **base),
                  project_penetration=False)
    assert dyn.period_unit == "month"
    d = dict(dyn.penetration.series)
    s = dict(sta.penetration.series)
    # month1: brand b1 + cat c1 -> 1/2 ; month2: +c2 -> 1/3 ; month3: +c3 -> 1/4
    assert approx(d[1], 1 / 2) and approx(d[2], 1 / 3) and approx(d[3], 1 / 4)
    assert approx(s[1], 0.25) and approx(s[3], 0.25)
    assert approx(dyn.penetration.snapshot, 0.25)
    # display labels are calendar months
    assert dyn.label(1) == "2024-01" and dyn.label(3) == "2024-03"


def test_month_axis_share_labels():
    rows = [
        row("s1", "2024-01-05", True, True, 2), row("s1", "2024-01-20", False, True, 8),  # 2024-01 .2
        row("s1", "2024-02-05", True, True, 6), row("s1", "2024-02-20", False, True, 4),  # 2024-02 .6
    ]
    res = run_trb(make_df(rows), TRBConfig(launch_date="2024-01-01", period_unit="month"))
    sr = dict(res.share_ratio_series())
    assert approx(sr[1], 0.2) and approx(sr[2], 0.6)
    assert res.label(1) == "2024-01" and res.label(2) == "2024-02"


# --------------------------------------------------------------------------- #
# Bucket-column axis: dense chronological ordinal, cross-year, non-daily feed
# --------------------------------------------------------------------------- #
def _bucket_row(sid, d, brand, cat, vol, bucket):
    r = row(sid, d, brand, cat, vol)
    r["yw"] = bucket
    return r


def test_bucket_column_dense_ordinal_crosses_year():
    # A weekly feed labelled by ISO year-week straddling 2023 -> 2024.
    rows = [
        _bucket_row("a", "2023-12-26", True, True, 1, "2023-W52"),   # brand trial
        _bucket_row("c", "2023-12-26", False, True, 1, "2023-W52"),  # cat entrant
        _bucket_row("a", "2024-01-02", True, True, 1, "2024-W01"),
        _bucket_row("c", "2024-01-02", False, True, 1, "2024-W01"),
    ]
    res = run_trb(make_df(rows),
                  TRBConfig(launch_date="2023-12-25", bucket_column="yw",
                            analysis_date="2024-02-01"), project_penetration=False)
    assert res.period_unit == "bucket"
    # non-contiguous labels collapse to consecutive periods 1, 2
    periods = sorted(p for p, *_ in res.share_series)
    assert periods == [1, 2]
    assert res.label(1) == "2023-W52" and res.label(2) == "2024-W01"


def test_bucket_column_buying_index_on_calendar_axis():
    rows = [
        _bucket_row("r1", "2024-01-05", True, True, 10, "2024-01"),
        _bucket_row("r1", "2024-02-05", False, True, 10, "2024-02"),
        _bucket_row("n1", "2024-01-06", True, True, 5, "2024-01"),
        _bucket_row("n2", "2024-02-07", False, True, 5, "2024-02"),
    ]
    res = run_trb(make_df(rows),
                  TRBConfig(launch_date="2024-01-01", bucket_column="yw",
                            analysis_date="2024-12-31"))
    labels = [res.label(p) for p, _ in res.buying_index_series]
    assert labels == ["2024-01", "2024-02"]


# --------------------------------------------------------------------------- #
# Cohorts remain weekly regardless of the calendar axis
# --------------------------------------------------------------------------- #
def test_cohorts_stay_weekly_under_month_axis():
    rows = [
        row("e", "2024-01-03", True, True, 1),    # week 1 -> cohort 1-6w
        row("e", "2024-03-01", True, True, 1),
        row("l", "2024-03-20", True, True, 1),    # week ~12 -> cohort 7-12w
        row("c1", "2024-01-04", False, True, 1),
    ]
    res = run_trb(make_df(rows),
                  TRBConfig(launch_date="2024-01-01", period_unit="month",
                            analysis_date="2024-06-30"), project_penetration=False)
    labels = {c.label for c in res.cohorts}
    assert "1-6w" in labels and "7-12w" in labels


def test_invalid_period_unit_rejected():
    with pytest.raises(ValueError):
        TRBConfig(period_unit="day")
