"""Calendar-axis tests: the penetration / realised-share / per-period buying
index can be computed and displayed at week or month granularity, or driven by a
precomputed bucket-label column (which also handles non-daily feeds and labels
that cross a year boundary). Cohorts stay weekly (Parfitt Table 2). All runs go
through the Spark engine."""
from __future__ import annotations

import pytest

from parfitt_trb import TRBConfig, run_trb
from tests.helpers import approx, make_sdf, row


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


def test_month_axis_dynamic_vs_static(spark):
    base = dict(launch_date="2024-01-01", analysis_date="2024-03-31", period_unit="month")
    dyn = run_trb(make_sdf(spark, _month_rows()),
                  TRBConfig(penetration_denominator="dynamic", **base),
                  project_penetration=False)
    sta = run_trb(make_sdf(spark, _month_rows()),
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


def test_month_axis_share_labels(spark):
    rows = [
        row("s1", "2024-01-05", True, True, 2), row("s1", "2024-01-20", False, True, 8),  # 2024-01 .2
        row("s1", "2024-02-05", True, True, 6), row("s1", "2024-02-20", False, True, 4),  # 2024-02 .6
    ]
    res = run_trb(make_sdf(spark, rows), TRBConfig(launch_date="2024-01-01", period_unit="month"))
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


def test_bucket_column_dense_ordinal_crosses_year(spark):
    # A weekly feed labelled by ISO year-week straddling 2023 -> 2024.
    rows = [
        _bucket_row("a", "2023-12-26", True, True, 1, "2023-W52"),   # brand trial
        _bucket_row("c", "2023-12-26", False, True, 1, "2023-W52"),  # cat entrant
        _bucket_row("a", "2024-01-02", True, True, 1, "2024-W01"),
        _bucket_row("c", "2024-01-02", False, True, 1, "2024-W01"),
    ]
    res = run_trb(make_sdf(spark, rows),
                  TRBConfig(launch_date="2023-12-25", bucket_column="yw",
                            analysis_date="2024-02-01"), project_penetration=False)
    assert res.period_unit == "bucket"
    # non-contiguous labels collapse to consecutive periods 1, 2
    periods = sorted(p for p, *_ in res.share_series)
    assert periods == [1, 2]
    assert res.label(1) == "2023-W52" and res.label(2) == "2024-W01"


def test_bucket_column_buying_index_on_calendar_axis(spark):
    rows = [
        _bucket_row("r1", "2024-01-05", True, True, 10, "2024-01"),
        _bucket_row("r1", "2024-02-05", False, True, 10, "2024-02"),
        _bucket_row("n1", "2024-01-06", True, True, 5, "2024-01"),
        _bucket_row("n2", "2024-02-07", False, True, 5, "2024-02"),
    ]
    res = run_trb(make_sdf(spark, rows),
                  TRBConfig(launch_date="2024-01-01", bucket_column="yw",
                            analysis_date="2024-12-31"))
    labels = [res.label(p) for p, _ in res.buying_index_series]
    assert labels == ["2024-01", "2024-02"]


# --------------------------------------------------------------------------- #
# Cohorts remain weekly regardless of the calendar axis
# --------------------------------------------------------------------------- #
def test_cohorts_stay_weekly_under_month_axis(spark):
    rows = [
        row("e", "2024-01-03", True, True, 1),    # week 1 -> cohort 1-6w
        row("e", "2024-03-01", True, True, 1),
        row("l", "2024-03-20", True, True, 1),    # week ~12 -> cohort 7-12w
        row("c1", "2024-01-04", False, True, 1),
    ]
    res = run_trb(make_sdf(spark, rows),
                  TRBConfig(launch_date="2024-01-01", period_unit="month",
                            analysis_date="2024-06-30"), project_penetration=False)
    labels = {c.label for c in res.cohorts}
    assert "1-6w" in labels and "7-12w" in labels


def test_invalid_period_unit_rejected():
    with pytest.raises(ValueError):
        TRBConfig(period_unit="day")


def test_invalid_share_period_unit_rejected():
    with pytest.raises(ValueError):
        TRBConfig(share_period_unit="quarter")


# --------------------------------------------------------------------------- #
# Mixed calendar axes: penetration on ISOWEEKYEAR, share on YEARMONTH
# --------------------------------------------------------------------------- #
def _mixed_axis_rows():
    """A few purchase rows spanning two calendar months; brand trier is in Jan,
    repeat purchases in Jan and Feb. Category-only buyer also in both months."""
    return [
        # brand trier: trial 2024-01-08, repeat 2024-01-22 and 2024-02-05
        dict(shopper_id="t1", txn_date="2024-01-08", is_new_product=True,
             is_category=True, volume=1.0, units=1.0, value=1.0,
             yw="2024-W02", ym="2024-01"),
        dict(shopper_id="t1", txn_date="2024-01-22", is_new_product=True,
             is_category=True, volume=2.0, units=2.0, value=2.0,
             yw="2024-W04", ym="2024-01"),
        dict(shopper_id="t1", txn_date="2024-02-05", is_new_product=False,
             is_category=True, volume=3.0, units=3.0, value=3.0,
             yw="2024-W06", ym="2024-02"),
        # category-only buyer: purchases in Jan and Feb
        dict(shopper_id="c1", txn_date="2024-01-10", is_new_product=False,
             is_category=True, volume=4.0, units=4.0, value=4.0,
             yw="2024-W02", ym="2024-01"),
        dict(shopper_id="c1", txn_date="2024-02-12", is_new_product=False,
             is_category=True, volume=5.0, units=5.0, value=5.0,
             yw="2024-W07", ym="2024-02"),
    ]


def test_mixed_axes_pen_weekly_share_monthly(spark):
    res = run_trb(
        make_sdf(spark, _mixed_axis_rows()),
        TRBConfig(launch_date="2024-01-01",
                  bucket_column="yw",
                  share_bucket_column="ym",
                  analysis_date="2024-03-31"),
        project_penetration=False,
    )
    # Period axes are correctly set
    assert res.period_unit == "bucket"
    assert res.share_period_unit == "bucket"

    # Penetration labels are ISO weeks
    pen_labels = {res.label(p) for p, _ in res.penetration.series}
    assert all(lbl.startswith("2024-W") for lbl in pen_labels), pen_labels

    # Share labels are calendar months
    share_labels = {res.label_share(p) for p, *_ in res.share_series}
    assert share_labels == {"2024-01", "2024-02"}, share_labels

    # label() and label_share() differ for the same period ordinal
    share_periods = [p for p, *_ in res.share_series]
    # at least one share period has a YYYY-MM label via label_share
    assert any(res.label_share(p).count("-") == 1 and
               not res.label_share(p).startswith("2024-W")
               for p in share_periods)

    # K and segmented share are unaffected by the share axis
    assert res.penetration.snapshot > 0
    assert res.segmented_share() >= 0


def test_mixed_axes_share_period_unit(spark):
    """share_period_unit='month' on a derived-date (non-bucket) panel."""
    res = run_trb(
        make_sdf(spark, _mixed_axis_rows()),
        TRBConfig(launch_date="2024-01-01",
                  period_unit="week",
                  share_period_unit="month",
                  analysis_date="2024-03-31"),
        project_penetration=False,
    )
    # Penetration uses weeks
    assert res.period_unit == "week"
    pen_label_1 = res.label(res.penetration.series[0][0])
    assert pen_label_1.startswith("2024-W"), pen_label_1

    # Share uses months
    share_labels = {res.label_share(p) for p, *_ in res.share_series}
    assert all(lbl.startswith("2024-") and not lbl.startswith("2024-W")
               for lbl in share_labels), share_labels
