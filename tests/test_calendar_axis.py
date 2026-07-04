"""Calendar-axis tests: the penetration / realised-share / per-period buying
index can be computed and displayed on week / fortnight / month / iso_week /
iso_fortnight / fiscal_445 granularities. The calendar-anchored units (iso_week,
iso_fortnight, fiscal_445) live on the real calendar grid via a date dimension,
so empty buckets (out-of-stock weeks) keep their slot instead of collapsing.
Cohorts stay weekly (Parfitt Table 2). All runs go through the Spark engine."""
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
# ISO-week axis: real calendar grid, cross-year, gap-safe (out-of-stock weeks)
# --------------------------------------------------------------------------- #
def test_iso_week_axis_crosses_year(spark):
    # A weekly panel straddling 2023 -> 2024 (Dec 25 2023 is a Monday = 2023-W52).
    rows = [
        row("a", "2023-12-26", True, True, 1),    # brand trial, 2023-W52
        row("c", "2023-12-26", False, True, 1),   # cat entrant, 2023-W52
        row("a", "2024-01-02", True, True, 1),    # 2024-W01
        row("c", "2024-01-02", False, True, 1),
    ]
    res = run_trb(make_sdf(spark, rows),
                  TRBConfig(launch_date="2023-12-25", period_unit="iso_week",
                            analysis_date="2024-02-01"), project_penetration=False)
    assert res.period_unit == "iso_week"
    # consecutive ISO weeks -> consecutive periods 1, 2, with the cross-year labels
    periods = sorted(p for p, *_ in res.share_series)
    assert periods == [1, 2]
    assert res.label(1) == "2023-W52" and res.label(2) == "2024-W01"


def test_extended_labels_match_spark_axis(spark):
    """The python-side derived labels (used for projected future periods) must
    agree with the Spark-built dense map on every OBSERVED period, or the
    projected region of a chart would be mislabelled."""
    from parfitt_trb.periods import extended_period_label
    rows = [
        row("a", "2023-12-26", True, True, 1),    # 2023-W52
        row("c", "2023-12-26", False, True, 1),
        row("a", "2024-01-02", True, True, 1),    # 2024-W01
        row("c", "2024-01-10", False, True, 1),   # 2024-W02
    ]
    for unit in ("iso_week", "iso_fortnight", "fiscal_445"):
        res = run_trb(make_sdf(spark, rows),
                      TRBConfig(launch_date="2023-12-25", period_unit=unit,
                                analysis_date="2024-02-01"),
                      project_penetration=False)
        for p, spark_label in res.period_labels.items():
            assert extended_period_label(p, res.origin, unit) == spark_label, \
                (unit, p, spark_label)


def test_iso_week_gap_preserved_out_of_stock(spark):
    """An out-of-stock 2024-W02 (no sales) must keep its slot: the W03 brand
    trier lands on period 3, NOT collapsed onto period 2 (points 2 & 4)."""
    rows = [
        row("c1", "2024-01-01", False, True, 1),   # cat-only entrant, 2024-W01
        row("b1", "2024-01-15", True, True, 1),    # brand+cat trier, 2024-W03 (W02 empty)
    ]
    res = run_trb(make_sdf(spark, rows),
                  TRBConfig(launch_date="2024-01-01", period_unit="iso_week",
                            analysis_date="2024-03-31"), project_penetration=False)
    d = dict(res.penetration.series)
    # If the empty W02 collapsed, b1 would sit at period 2 (d[2]==0.5). It must not.
    assert approx(d[2], 0.0) and approx(d[3], 0.5)
    # the empty week still carries its true calendar label
    assert res.label(2) == "2024-W02" and res.label(3) == "2024-W03"


def test_iso_week_buying_index_on_calendar_axis(spark):
    rows = [
        row("r1", "2024-01-01", True, True, 10),   # 2024-W01
        row("r1", "2024-01-15", False, True, 10),  # 2024-W03
        row("n1", "2024-01-02", True, True, 5),    # 2024-W01
        row("n2", "2024-01-16", False, True, 5),   # 2024-W03
    ]
    res = run_trb(make_sdf(spark, rows),
                  TRBConfig(launch_date="2024-01-01", period_unit="iso_week",
                            analysis_date="2024-12-31"))
    labels = [res.label(p) for p, _ in res.buying_index_series]
    assert labels == ["2024-W01", "2024-W03"]


# --------------------------------------------------------------------------- #
# Fortnight axes: derived 14-day buckets and epoch-aligned ISO-week pairs
# --------------------------------------------------------------------------- #
def test_fortnight_axis_derived(spark):
    rows = [
        row("b1", "2024-01-01", True, True, 1),    # day 0  -> period 1
        row("c1", "2024-01-14", False, True, 1),   # day 13 -> period 1
        row("c2", "2024-01-15", False, True, 1),   # day 14 -> period 2
    ]
    res = run_trb(make_sdf(spark, rows),
                  TRBConfig(launch_date="2024-01-01", period_unit="fortnight",
                            analysis_date="2024-03-31"), project_penetration=False)
    assert res.period_unit == "fortnight"
    d = dict(res.penetration.series)
    # period 1: brand b1 over cat {b1, c1} -> 1/2 ; period 2: +c2 -> 1/3
    assert approx(d[1], 1 / 2) and approx(d[2], 1 / 3)
    # labels: the ISO week of each bucket's first day
    assert res.label(1) == "2024-W01" and res.label(2) == "2024-W03"


def test_iso_fortnight_axis_crosses_year(spark):
    """Epoch-aligned pairs may straddle the year boundary: (2023-W52, 2024-W01)
    is ONE bucket labelled after its first week ('2023-F52'), the next pair
    (2024-W02, 2024-W03) is '2024-F02'."""
    rows = [
        row("a", "2023-12-26", True, True, 1),    # 2023-W52 -> period 1
        row("c", "2023-12-26", False, True, 1),
        row("a", "2024-01-02", False, True, 1),   # 2024-W01 -> STILL period 1
        row("c2", "2024-01-10", False, True, 1),  # 2024-W02 -> period 2
    ]
    res = run_trb(make_sdf(spark, rows),
                  TRBConfig(launch_date="2023-12-25", period_unit="iso_fortnight",
                            analysis_date="2024-02-01"), project_penetration=False)
    assert res.period_unit == "iso_fortnight"
    periods = sorted(p for p, *_ in res.share_series)
    assert periods == [1, 2]
    assert res.label(1) == "2023-F52" and res.label(2) == "2024-F02"
    d = dict(res.penetration.series)
    # period 1: brand a over cat {a, c} -> 1/2 ; period 2: +c2 -> 1/3
    assert approx(d[1], 1 / 2) and approx(d[2], 1 / 3)


def test_iso_fortnight_gap_preserved(spark):
    """An empty fortnight keeps its calendar slot (no collapsing)."""
    rows = [
        row("c1", "2024-01-01", False, True, 1),   # 2024-W01 -> period 1 (2023-F52)
        row("b1", "2024-01-29", True, True, 1),    # 2024-W05 -> period 3 (F02 empty)
    ]
    res = run_trb(make_sdf(spark, rows),
                  TRBConfig(launch_date="2024-01-01", period_unit="iso_fortnight",
                            analysis_date="2024-03-31"), project_penetration=False)
    d = dict(res.penetration.series)
    assert approx(d[2], 0.0) and approx(d[3], 0.5)
    assert res.label(2) == "2024-F02" and res.label(3) == "2024-F04"


def test_fortnight_pipeline_with_smoothing(spark):
    """End-to-end smoke: a fortnightly axis with fit-side smoothing enabled."""
    cat_dates = ["2024-01-03", "2024-01-20", "2024-02-05", "2024-02-19",
                 "2024-03-04", "2024-03-18", "2024-04-02", "2024-04-16"]
    rows = [row("b1", "2024-01-02", True, True, 1)]
    rows += [row(f"c{i}", d, False, True, 1) for i, d in enumerate(cat_dates)]
    res = run_trb(make_sdf(spark, rows),
                  TRBConfig(launch_date="2024-01-01", period_unit="fortnight",
                            penetration_smoothing_window=3,
                            analysis_date="2024-06-30"))
    assert res.period_unit == "fortnight"
    assert res.penetration.series                      # built and (maybe) fitted
    assert res.penetration.series == sorted(res.penetration.series)


# --------------------------------------------------------------------------- #
# Retail 4-4-5 axis: ISO weeks rolled into 'YYYY-Pnn' periods
# --------------------------------------------------------------------------- #
def test_fiscal_445_axis_labels(spark):
    rows = [
        row("b1", "2024-01-08", True, True, 1),    # 2024-W02 -> 2024-P01
        row("c1", "2024-01-08", False, True, 1),
        row("c2", "2024-02-05", False, True, 1),   # 2024-W06 -> 2024-P02
    ]
    res = run_trb(make_sdf(spark, rows),
                  TRBConfig(launch_date="2024-01-01", period_unit="fiscal_445",
                            analysis_date="2024-03-31"), project_penetration=False)
    assert res.period_unit == "fiscal_445"
    assert res.label(1) == "2024-P01" and res.label(2) == "2024-P02"
    assert res.segmented_share() >= 0


# --------------------------------------------------------------------------- #
# Defensive collapse (point 1): duplicate-grain lines are SUMMED, not dropped
# --------------------------------------------------------------------------- #
def test_prepare_collapses_duplicate_grain(spark):
    rows = [
        row("s1", "2024-01-05", True, True, 2),    # same (card, day, brand, cat)
        row("s1", "2024-01-05", True, True, 3),    # grain -> collapse to brand qty 5
        row("s1", "2024-01-05", False, True, 5),   # cat-only line, qty 5
    ]
    res = run_trb(make_sdf(spark, rows), TRBConfig(launch_date="2024-01-01"))
    sr = dict(res.share_ratio_series())
    # brand_qty = 2+3 = 5 (summed, not dropped); cat_qty = 5 (brand-as-cat) + 5 = 10
    assert approx(sr[1], 0.5)


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
                  period_unit="iso_week",
                  share_period_unit="month",
                  analysis_date="2024-03-31"),
        project_penetration=False,
    )
    # Period axes are correctly set
    assert res.period_unit == "iso_week"
    assert res.share_period_unit == "month"

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
