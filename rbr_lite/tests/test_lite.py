"""Hand-computed anchors for rbr_lite: interval assignment and eligibility in
Spark, the engine-free stability helpers, the cohort split, plot smoke tests
and the end-to-end recovery of a planted RBR curve."""
from __future__ import annotations

import pathlib
from datetime import date

import pandas as pd
import pytest

from rbr_lite import (
    RBRCurve, RBRPoint, boundary_end, build_rbr, detect_plateau, label_after,
    label_last_day, last_available_rbr, period_label, period_of, plot_rbr,
    plot_rbr_cohorts, stable_rbr,
)


# --------------------------------------------------------------------------- #
# Calendar sanity (the shared functions are a copy of penetration_lite's,
# which owns the exhaustive tests; the boundary helpers are owned here)
# --------------------------------------------------------------------------- #
def test_calendar_anchors():
    origin = "2024-01-01"
    assert period_of("2024-01-08", origin, "iso_week") == 2
    assert period_label(1, origin, "iso_week") == "2024-W01"
    # fortnights are named after the pair's FIRST ISO week
    assert period_label(1, origin, "iso_fortnight") == "2023-F52"
    assert period_label(1, origin, "month") == "2024-01"


def test_boundary_helpers():
    assert label_last_day("2023-W31") == date(2023, 8, 6)     # Sunday
    assert label_last_day("2023-F31") == date(2023, 8, 13)    # Monday + 13
    assert label_last_day("2023-07") == date(2023, 7, 31)
    assert label_last_day("2024-02") == date(2024, 2, 29)     # leap month end
    assert label_last_day("2020-W53") == date(2021, 1, 3)     # 53-week ISO year
    assert label_after("2023-W52") == "2024-W01"
    assert label_after("2020-W53") == "2021-W01"
    assert label_after("2023-F52") == "2024-F02"
    assert label_after("2023-12") == "2024-01"
    # non-label boundaries coerce like dates, inclusive as themselves
    assert boundary_end("2024-01-15") == date(2024, 1, 15)
    assert boundary_end(date(2024, 1, 15)) == date(2024, 1, 15)
    assert boundary_end(pd.Timestamp("2024-01-15")) == date(2024, 1, 15)
    assert boundary_end("2024-W02") == date(2024, 1, 14)      # label -> last day
    with pytest.raises(ValueError, match="cannot interpret cohort boundary"):
        boundary_end("nope")
    with pytest.raises(ValueError, match="cannot interpret cohort boundary"):
        boundary_end(3.14)
    with pytest.raises(ValueError, match="invalid ISO week"):
        boundary_end("2024-W60")


# --------------------------------------------------------------------------- #
# build_rbr (Spark, hand-computed volumes)
# --------------------------------------------------------------------------- #
def _rows():
    # launch 2024-01-01 (Monday), analysis 2024-01-29, 7-day intervals.
    # Trials: h1 = 01-02 (max_interval 3), h2 = 01-15 (2), h3 = 01-28 (0).
    return [
        ("h1", "2024-01-02", True, True, 1.0),    # trial h1
        ("h1", "2024-01-02", False, True, 2.0),   # trial-day category: never a repeat
        ("h1", "2024-01-05", True, False, 1.0),   # datediff 3 -> t1; brand-only counts as cat
        ("h1", "2024-01-09", False, True, 1.0),   # datediff 7 -> t1 (ceil boundary)
        ("h1", "2024-01-10", False, True, 3.0),   # datediff 8 -> t2
        ("h2", "2024-01-15", True, True, 1.0),    # trial h2
        ("h2", "2024-01-18", False, True, 3.0),   # datediff 3 -> t1
        ("h3", "2024-01-28", True, True, 1.0),    # trial h3 (too recent: 0 elapsed intervals)
        ("h3", "2024-01-29", False, True, 5.0),   # datediff 1 -> t1 but not fully elapsed
    ]


def _sdf(spark, rows):
    return spark.createDataFrame(
        pd.DataFrame(rows, columns=["shopper_id", "txn_date", "is_new_product",
                                    "is_category", "volume"]))


def _build(spark, rows=None, **kw):
    kw.setdefault("period_length_days", 7)
    kw.setdefault("launch_date", "2024-01-01")
    kw.setdefault("analysis_date", "2024-01-29")
    return build_rbr(_sdf(spark, rows if rows is not None else _rows()), **kw)


def test_intervals_eligibility_and_ratio_of_sums(spark):
    curve = _build(spark)
    assert curve.n_triers == 3
    assert curve.origin == date(2024, 1, 1)
    assert curve.analysis_date == date(2024, 1, 29)
    f = {p.interval: p for p in curve.points}
    assert sorted(f) == [1, 2, 3]                       # gap-free axis
    # t1: brand 1.0 (brand-only line, in cat too); cat 1.0+1.0+3.0 = 5.0.
    # Trial-day rows (datediff 0) and h3's line (interval > max_interval 0)
    # contribute nothing. Ratio of sums: 1/5, NOT the mean of ratios (0.25).
    assert f[1].brand_qty == 1.0 and f[1].cat_qty == 5.0
    assert f[1].rbr == pytest.approx(0.2)
    assert f[2].brand_qty == 0.0 and f[2].cat_qty == 3.0 and f[2].rbr == 0.0
    # t3: h1 eligible but lapsed -- zero-filled row, rate unobserved.
    assert f[3].brand_qty == 0.0 and f[3].cat_qty == 0.0 and f[3].rbr is None
    assert {t: p.n_eligible for t, p in f.items()} == {1: 2, 2: 2, 3: 1}
    assert curve.rbr_at(1) == pytest.approx(0.2) and curve.rbr_at(9) is None


def test_duplicate_lines_are_summed(spark):
    rows = _rows() + [("h2", "2024-01-20", True, True, 1.0),
                      ("h2", "2024-01-20", True, True, 1.0)]
    f = {p.interval: p for p in _build(spark, rows).points}
    assert f[1].brand_qty == 3.0 and f[1].cat_qty == 7.0


def test_launch_floor_redates_trial_and_origin_inference(spark):
    rows = [
        ("h4", "2023-12-20", True, True, 1.0),    # pre-launch brand history
        ("h4", "2024-01-16", True, True, 1.0),    # first POST-launch brand buy
        ("h4", "2024-01-19", False, True, 2.0),
    ]
    floored = build_rbr(_sdf(spark, rows), period_length_days=7,
                        launch_date="2024-01-01", analysis_date="2024-01-30")
    # trial re-dated to 01-16 (max_interval 2); the pre-launch line never counts
    f = {p.interval: p for p in floored.points}
    assert floored.n_triers == 1 and sorted(f) == [1, 2]
    assert f[1].brand_qty == 0.0 and f[1].cat_qty == 2.0
    # without launch_date: origin/trial = first brand ts, analysis = max ts
    inferred = build_rbr(_sdf(spark, rows), period_length_days=7)
    assert inferred.origin == date(2023, 12, 20)
    assert inferred.analysis_date == date(2024, 1, 19)
    g = {p.interval: p for p in inferred.points}
    assert sorted(g) == [1, 2, 3, 4]
    assert g[4].brand_qty == 1.0                       # 01-16, datediff 27 -> t4
    # 01-19 (datediff 30 -> t5) exceeds max_interval 4: not fully elapsed
    assert g[4].cat_qty == 1.0 and g[1].rbr is None


def test_max_interval_caps_axis_but_not_n_eligible(spark):
    curve = _build(spark, max_interval=2)
    f = {p.interval: p for p in curve.points}
    assert sorted(f) == [1, 2]                          # horizon capped
    # h1 (max_interval 3, beyond the cap) still backs every base: the seeded
    # top-down cumulation counts it (the parfitt_trb undercount, fixed here).
    assert {t: p.n_eligible for t, p in f.items()} == {1: 2, 2: 2}
    assert f[1].rbr == pytest.approx(0.2)               # sums untouched by the cap


def test_cohort_split_boundaries(spark):
    # one boundary ending W02 (2024-01-14): h1 (trial 01-02) closes the first
    # band; h2 (01-15) and h3 (01-28) fall in the open tail band.
    curve = _build(spark, cohort_boundaries=["2024-W02"])
    assert curve.cohort_labels == ["≤2024-W02", "2024-W03+"]
    series = curve.cohort_series()
    # band order, NOT label sort (sorted() would put '2024-W03+' first)
    assert list(series) == ["≤2024-W02", "2024-W03+"]
    assert series["≤2024-W02"] == [(1, pytest.approx(0.5)), (2, 0.0)]
    assert series["2024-W03+"] == [(1, 0.0)]
    # the pooled curve is identical to the cohort-free run
    plain = _build(spark)
    assert [(p.interval, p.rbr) for p in curve.points] == \
           [(p.interval, p.rbr) for p in plain.points]
    frame = curve.cohort_frame()
    assert list(frame.columns) == ["cohort", "interval", "brand_qty",
                                   "cat_qty", "rbr"]


def test_cohort_table_respects_the_cap(spark):
    curve = _build(spark, cohort_boundaries=["2024-W02"], max_interval=1)
    assert (curve.cohort_table["interval"] <= 1).all()


def test_cohort_boundary_inclusive_edge(spark):
    # e1 trials exactly on the boundary bucket's LAST day -> first band;
    # e2 trials the day after -> tail band.
    rows = [("e1", "2024-01-14", True, True, 1.0),
            ("e1", "2024-01-16", False, True, 2.0),
            ("e2", "2024-01-15", True, True, 1.0),
            ("e2", "2024-01-17", False, True, 3.0)]
    curve = build_rbr(_sdf(spark, rows), period_length_days=7,
                      launch_date="2024-01-01", analysis_date="2024-01-29",
                      cohort_boundaries=["2024-W02"])
    by_cohort = {r.cohort: float(r.cat_qty)
                 for r in curve.cohort_frame().itertuples(index=False)}
    assert by_cohort == {"≤2024-W02": 2.0, "2024-W03+": 3.0}


def test_cohort_mixed_grammar_and_dates(spark):
    # a week label (ends 01-07), a plain date (01-20) and a month label (ends
    # 02-29, inclusive leap edge) -> four bands, one trier each.
    rows = [("g1", "2024-01-03", True, True, 1.0),
            ("g1", "2024-01-06", False, True, 2.0),
            ("g2", "2024-01-14", True, True, 1.0),
            ("g2", "2024-01-17", False, True, 2.0),
            ("g3", "2024-02-29", True, True, 1.0),
            ("g3", "2024-03-03", False, True, 2.0),
            ("g4", "2024-03-05", True, True, 1.0),
            ("g4", "2024-03-08", False, True, 2.0)]
    curve = build_rbr(_sdf(spark, rows), period_length_days=7,
                      launch_date="2024-01-01", analysis_date="2024-03-31",
                      cohort_boundaries=["2024-W01", "2024-01-20", "2024-02"])
    assert curve.cohort_labels == ["≤2024-W01", "2024-W02–2024-01-20",
                                   "2024-01-21–2024-02", "2024-03+"]
    series = curve.cohort_series()
    assert list(series) == curve.cohort_labels
    assert all(pts == [(1, 0.0)] for pts in series.values())


def test_cohort_boundary_validation(spark):
    sdf = _sdf(spark, _rows())

    def go(bounds):
        return build_rbr(sdf, period_length_days=7, launch_date="2024-01-01",
                         analysis_date="2024-01-29", cohort_boundaries=bounds)

    with pytest.raises(ValueError, match="at least one boundary"):
        go([])
    with pytest.raises(ValueError, match="strictly increasing"):
        go(["2024-W03", "2024-W02"])
    with pytest.raises(ValueError, match="strictly increasing"):
        go(["2024-W02", "2024-01-14"])   # distinct tokens, same end date
    with pytest.raises(ValueError, match="before the origin"):
        go(["2023-W01"])
    with pytest.raises(ValueError, match="on/after the analysis date"):
        go(["2024-01-29"])


def test_validation_errors(spark):
    sdf = _sdf(spark, _rows())
    with pytest.raises(ValueError, match="unit must be one of"):
        build_rbr(sdf, interval_unit="week")
    with pytest.raises(ValueError, match="unit must be one of"):
        build_rbr(sdf, interval_unit="foo")
    with pytest.raises(ValueError, match="period_length_days"):
        build_rbr(sdf, period_length_days=0)
    with pytest.raises(ValueError, match="period_length_days does not apply"):
        build_rbr(sdf, interval_unit="iso_week", period_length_days=7)
    with pytest.raises(ValueError, match="max_interval"):
        build_rbr(sdf, max_interval=0)
    with pytest.raises(ValueError, match="without cohorts"):
        _build(spark).cohort_series()


def test_degenerate_windows(spark):
    sdf = _sdf(spark, _rows())
    with pytest.raises(ValueError, match="no transactions"):
        build_rbr(sdf, analysis_date="2023-01-01")
    with pytest.raises(ValueError, match="no brand triers"):
        build_rbr(sdf, launch_date="2024-06-01", analysis_date="2024-06-30")
    cat_only = [("h9", "2024-01-05", False, True, 1.0)]
    with pytest.raises(ValueError, match="launch origin"):
        build_rbr(_sdf(spark, cat_only))
    # triers exist but no interval has fully elapsed: a valid, empty curve
    young = build_rbr(_sdf(spark, [("h3", "2024-01-28", True, True, 1.0)]),
                      period_length_days=7, analysis_date="2024-01-28")
    assert young.n_triers == 1 and young.points == []
    assert young.last_available() is None and young.to_frame().empty


# --------------------------------------------------------------------------- #
# Bucket-interval mode (interval = calendar-bucket difference from the trial;
# only FULLY elapsed buckets count -- stricter than parfitt_trb)
# --------------------------------------------------------------------------- #
def _bucket_rows():
    # launch Mon 2024-01-01; W02 = Jan 8-14, W03 = Jan 15-21, W04 = Jan 22-28.
    return [
        ("b1", "2024-01-10", True, True, 1.0),    # trial b1 (W02)
        ("b1", "2024-01-12", False, True, 2.0),   # same bucket -> interval 0
        ("b1", "2024-01-15", False, True, 4.0),   # W03 -> t1
        ("b1", "2024-01-21", True, False, 1.0),   # W03 -> t1 (brand is cat too)
        ("b1", "2024-01-23", False, True, 7.0),   # W04 -> t2
        ("b1", "2024-01-28", False, True, 3.0),   # W04 -> t2
        ("b2", "2024-01-22", True, True, 1.0),    # trial b2 (W04)
        ("b2", "2024-01-25", False, True, 9.0),   # same bucket -> interval 0
        ("b3", "2024-01-16", True, True, 1.0),    # trial b3 (W03), no repeats
    ]


def _build_bucket(spark, **kw):
    kw.setdefault("interval_unit", "iso_week")
    kw.setdefault("launch_date", "2024-01-01")
    kw.setdefault("analysis_date", "2024-01-28")
    return build_rbr(_sdf(spark, _bucket_rows()), **kw)


def test_bucket_interval_iso_week(spark):
    # adate Sun 01-28 closes W04: maxes b1 (trial W02) = 2, b3 (W03) = 1,
    # b2 (W04) = 0.
    curve = _build_bucket(spark)
    assert curve.n_triers == 3 and curve.interval_unit == "iso_week"
    assert curve.period_length_days is None
    f = {p.interval: p for p in curve.points}
    assert sorted(f) == [1, 2]
    # t1 = W03 for b1: brand 1.0, cat 4.0 + 1.0. The same-bucket rows (01-12,
    # 01-25) are interval 0 and contribute NOTHING -- exact mode with P=7
    # would have counted 01-12 in t1.
    assert f[1].brand_qty == 1.0 and f[1].cat_qty == 5.0
    assert f[1].rbr == pytest.approx(0.2)
    assert f[2].brand_qty == 0.0 and f[2].cat_qty == 10.0 and f[2].rbr == 0.0
    assert {t: p.n_eligible for t, p in f.items()} == {1: 2, 2: 1}


def test_bucket_interval_partial_last_bucket_excluded(spark):
    # adate Thu 01-25: W04 has NOT fully elapsed -> b1 max 1 (W03 only, its
    # observed W04 row is excluded), b3 max 0, and b2 max -1 (trial inside
    # adate's own unfinished bucket -- absorbed safely by dist/seed/filters).
    curve = _build_bucket(spark, analysis_date="2024-01-25")
    f = {p.interval: p for p in curve.points}
    assert sorted(f) == [1]
    assert f[1].brand_qty == 1.0 and f[1].cat_qty == 5.0
    assert f[1].n_eligible == 1
    assert curve.n_triers == 3


def test_bucket_interval_month_boundaries(spark):
    rows = [("m1", "2024-01-31", True, True, 1.0),
            ("m1", "2024-02-01", False, True, 2.0)]   # next month -> t1
    full = build_rbr(_sdf(spark, rows), interval_unit="month",
                     launch_date="2024-01-01", analysis_date="2024-02-29")
    f = {p.interval: p for p in full.points}
    assert sorted(f) == [1] and f[1].cat_qty == 2.0 and f[1].rbr == 0.0
    # one day earlier February is not complete: no fully-elapsed bucket
    partial = build_rbr(_sdf(spark, rows), interval_unit="month",
                        launch_date="2024-01-01", analysis_date="2024-02-28")
    assert partial.n_triers == 1 and partial.points == []


def test_bucket_interval_fortnight_grid(spark):
    # The fortnight grid is epoch-aligned, not trial-anchored: Dec 25 2023 -
    # Jan 7 2024 is ONE fortnight, Jan 8-21 the next.
    rows = [("f1", "2024-01-02", True, True, 1.0),
            ("f1", "2024-01-07", False, True, 2.0),   # same fortnight (other ISO week)
            ("f1", "2024-01-08", False, True, 3.0)]   # next fortnight -> t1
    full = build_rbr(_sdf(spark, rows), interval_unit="iso_fortnight",
                     launch_date="2024-01-01", analysis_date="2024-01-21")
    f = {p.interval: p for p in full.points}
    assert sorted(f) == [1] and f[1].cat_qty == 3.0    # 01-07 excluded
    partial = build_rbr(_sdf(spark, rows), interval_unit="iso_fortnight",
                        launch_date="2024-01-01", analysis_date="2024-01-20")
    assert partial.points == []


def test_bucket_mode_cap_keeps_seeded_eligibility(spark):
    # bucket-mode replica of test_max_interval_caps_axis_but_not_n_eligible:
    # b1 (max 2, beyond the cap) still backs the base at t1.
    curve = _build_bucket(spark, max_interval=1)
    f = {p.interval: p for p in curve.points}
    assert sorted(f) == [1]
    assert f[1].n_eligible == 2 and f[1].rbr == pytest.approx(0.2)


def test_bucket_mode_with_cohorts(spark):
    curve = _build_bucket(spark, cohort_boundaries=["2024-W02"])
    assert curve.cohort_labels == ["≤2024-W02", "2024-W03+"]
    series = curve.cohort_series()
    # b2/b3 (tail band) have no eligible repeat rows: the band is omitted
    # from the series but stays in cohort_labels
    assert list(series) == ["≤2024-W02"]
    assert series["≤2024-W02"] == [(1, pytest.approx(0.2)), (2, 0.0)]


# --------------------------------------------------------------------------- #
# Engine-free stability helpers
# --------------------------------------------------------------------------- #
def _pts(*vals):
    return [RBRPoint(interval=i + 1, rbr=v, brand_qty=0.0, cat_qty=0.0,
                     n_eligible=0) for i, v in enumerate(vals)]


def test_detect_plateau():
    pts = _pts(0.5, 0.42, 0.40, 0.401, 0.399, 0.4)
    assert detect_plateau(pts, tol=0.005, k=3) == (3, 0.40)
    assert detect_plateau(_pts(0.5, 0.3, 0.45, 0.2), tol=0.005, k=3) is None
    assert detect_plateau(_pts(0.4, 0.4), tol=0.005, k=3) is None  # too short


def test_stable_rbr_and_last_available():
    pts = _pts(0.5, 0.4, None, 0.3, 0.35, None)
    assert stable_rbr(pts, 4) == pytest.approx((0.3 + 0.35) / 2)
    assert stable_rbr(pts, 2) == pytest.approx((0.4 + 0.3 + 0.35) / 3)  # None skipped
    assert stable_rbr(pts, 6) is None
    assert last_available_rbr(pts) == (5, 0.35)         # trailing None skipped
    assert last_available_rbr(_pts(None, None)) is None


def test_to_frame_columns():
    curve = RBRCurve(origin=date(2024, 1, 1), analysis_date=date(2024, 1, 29),
                     period_length_days=7, points=_pts(0.5, None), n_triers=2)
    frame = curve.to_frame()
    assert list(frame.columns) == ["interval", "rbr", "brand_qty", "cat_qty",
                                   "n_eligible"]
    assert frame["rbr"].isna().tolist() == [False, True]
    with pytest.raises(ValueError, match="without cohorts"):
        curve.cohort_frame()


# --------------------------------------------------------------------------- #
# Plot smoke (Agg)
# --------------------------------------------------------------------------- #
def _dummy_curve(with_cohorts=False):
    table = labels = None
    if with_cohorts:
        labels = ["≤2024-W01", "2024-W02+"]
        table = pd.DataFrame({"cohort": ["≤2024-W01", "≤2024-W01", "2024-W02+"],
                              "interval": [1, 2, 1],
                              "brand_qty": [1.0, 2.0, 1.0],
                              "cat_qty": [2.0, 4.0, 4.0]})
    return RBRCurve(origin=date(2024, 1, 1), analysis_date=date(2024, 3, 1),
                    period_length_days=7,
                    points=_pts(0.5, 0.42, 0.40, 0.401, 0.399), n_triers=10,
                    cohort_labels=labels, cohort_table=table)


def test_plot_rbr_smoke():
    import matplotlib
    matplotlib.use("Agg")
    ax = plot_rbr(_dummy_curve(), mark_plateau=True)
    assert ax.get_lines() and "7-day" in ax.get_xlabel()
    bucket = RBRCurve(origin=date(2024, 1, 1), analysis_date=date(2024, 3, 1),
                      period_length_days=None, interval_unit="iso_week",
                      points=_pts(0.5, 0.4), n_triers=5)
    assert "iso_week" in plot_rbr(bucket).get_xlabel()
    empty = RBRCurve(origin=date(2024, 1, 1), analysis_date=date(2024, 1, 2),
                     period_length_days=7, points=[], n_triers=1)
    with pytest.raises(ValueError, match="no RBR points"):
        plot_rbr(empty)


def test_plot_rbr_cohorts_smoke():
    import matplotlib
    matplotlib.use("Agg")
    ax = plot_rbr_cohorts(_dummy_curve(with_cohorts=True))
    labels = [line.get_label() for line in ax.get_lines()]
    # band order, not label sort ('2024-W02+' sorts before '≤2024-W01')
    assert labels == ["pooled", "≤2024-W01", "2024-W02+"]
    with pytest.raises(ValueError, match="without cohorts"):
        plot_rbr_cohorts(_dummy_curve())


# --------------------------------------------------------------------------- #
# End-to-end via Spark + synth (planted curve recovered exactly)
# --------------------------------------------------------------------------- #
def test_end_to_end_synth_recovers_planted_curve(spark):
    from rbr_lite.synth import planted_rate, simulate_transactions
    pdf = simulate_transactions(n_households=120, trial_weeks=4,
                                horizon_weeks=20, launch="2024-01-01", seed=3)
    curve = build_rbr(spark.createDataFrame(pdf), period_length_days=7,
                      launch_date="2024-01-01", analysis_date="2024-05-20")
    assert curve.n_triers == 120 and len(curve.points) == 20
    for t in range(1, 21):
        assert curve.rbr_at(t) == pytest.approx(planted_rate(t), abs=1e-9)
    # 30 households per trial week -> deterministic eligibility bases
    elig = {p.interval: p.n_eligible for p in curve.points}
    assert elig[20] == 30 and elig[18] == 90 and elig[17] == 120
    assert curve.last_available()[0] == 20
    assert curve.stable(15) == pytest.approx(0.3, abs=1e-3)       # ~r_inf
    plat = curve.plateau()
    assert plat is not None and 8 <= plat[0] <= 12


def test_end_to_end_cohort_effect(spark):
    from rbr_lite.synth import planted_rate, simulate_transactions
    pdf = simulate_transactions(n_households=80, trial_weeks=2,
                                horizon_weeks=10, cohort_effect=0.05,
                                launch="2024-01-01", seed=1)
    # the boundary at W01 splits the two synth trial weeks into two bands
    curve = build_rbr(spark.createDataFrame(pdf), period_length_days=7,
                      launch_date="2024-01-01", analysis_date="2024-03-11",
                      cohort_boundaries=["2024-W01"])
    series = curve.cohort_series()
    assert list(series) == ["≤2024-W01", "2024-W02+"]
    for w, label in ((1, "≤2024-W01"), (2, "2024-W02+")):
        for t, r in series[label]:
            assert r == pytest.approx(planted_rate(t, w, cohort_effect=0.05),
                                      abs=1e-9)
    # the pooled rate sits between the two cohort rates wherever both exist
    w2 = dict(series["2024-W02+"])
    for t, r1 in series["≤2024-W01"]:
        if t in w2:
            lo, hi = sorted((r1, w2[t]))
            assert lo - 1e-9 <= curve.rbr_at(t) <= hi + 1e-9


# --------------------------------------------------------------------------- #
# Independence from the parent library
# --------------------------------------------------------------------------- #
def test_no_parfitt_trb_imports():
    import re
    pkg = pathlib.Path(__file__).resolve().parents[1]
    pattern = re.compile(r"^\s*(?:from|import)\s+parfitt_trb", re.MULTILINE)
    offenders = [p.name for p in pkg.rglob("*.py")
                 if pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == [], offenders
