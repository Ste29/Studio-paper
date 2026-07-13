"""Hand-computed anchors for buying_lite: the windowed headline B on both
bases, the growing-base per-bucket series (bucket-granularity membership,
gap-free axis), cohorts, the partial-last-bucket note, plot smoke tests and
the end-to-end recovery of a planted index."""
from __future__ import annotations

import pathlib
from datetime import date

import pandas as pd
import pytest

from buying_lite import (
    BuyingIndex, BuyingPoint, build_buying_index, period_label, period_of,
    plot_buying_cohorts, plot_buying_index,
)


# --------------------------------------------------------------------------- #
# Calendar sanity (the module is a verbatim copy of penetration_lite's, which
# owns the exhaustive tests; these anchors protect the copy from drift)
# --------------------------------------------------------------------------- #
def test_calendar_anchors():
    origin = "2024-01-01"
    assert period_of("2024-01-08", origin, "iso_week") == 2
    assert period_label(1, origin, "iso_week") == "2024-W01"
    # fortnights are named after the pair's FIRST ISO week
    assert period_label(1, origin, "iso_fortnight") == "2023-F52"
    assert period_label(1, origin, "month") == "2024-01"


# --------------------------------------------------------------------------- #
# build_buying_index (Spark, hand-computed volumes)
# --------------------------------------------------------------------------- #
def _rows():
    # launch 2024-01-01 (Monday), analysis 2024-01-21 (Sunday, end of W3).
    # Triers: h1 (01-02, W1; repeater from its 2nd brand line on 01-16, W3)
    # and h3 (01-10, W2). h2/h5 are category-only buyers (first cat W1 / W3).
    return [
        ("h1", "2024-01-02", True, True, 1.0),    # trial h1 (W1)
        ("h1", "2024-01-03", False, True, 2.0),
        ("h1", "2024-01-08", False, True, 1.0),   # W2
        ("h1", "2024-01-16", True, True, 1.0),    # 2nd brand line -> repeater (W3)
        ("h1", "2024-01-17", False, True, 1.0),
        ("h2", "2024-01-04", False, True, 4.0),   # never a trier
        ("h2", "2024-01-10", False, True, 2.0),
        ("h3", "2024-01-08", False, True, 2.0),   # W2, BEFORE h3's trial
        ("h3", "2024-01-10", True, True, 1.0),    # trial h3 (mid-bucket, W2)
        ("h3", "2024-01-15", False, True, 1.0),   # W3
        ("h5", "2024-01-15", False, True, 3.0),   # buyer seen only from W3
    ]


def _sdf(spark, rows):
    return spark.createDataFrame(
        pd.DataFrame(rows, columns=["shopper_id", "txn_date", "is_new_product",
                                    "is_category", "volume"]))


def _build(spark, rows=None, **kw):
    kw.setdefault("window_days", 14)
    kw.setdefault("launch_date", "2024-01-01")
    kw.setdefault("analysis_date", "2024-01-21")
    return build_buying_index(_sdf(spark, rows if rows is not None else _rows()),
                              **kw)


def test_headline_windowed_b(spark):
    bi = _build(spark)                       # window (01-07, 01-21] = W2+W3
    assert bi.origin == date(2024, 1, 1)
    assert bi.analysis_date == date(2024, 1, 21)
    assert bi.window_start == date(2024, 1, 8)
    assert (bi.n_buyers, bi.n_triers, bi.n_repeaters) == (4, 2, 1)
    # volumes in the window: all 12.0; triers h1 3.0 + h3 4.0; repeater h1 3.0
    assert bi.window_cat_qty == 12.0
    assert bi.window_cat_qty_triers == 7.0
    assert bi.window_cat_qty_repeaters == 3.0
    # per-capita: (7/2)/(12/4) and (3/1)/(12/4); silent members weigh 0
    assert bi.b_triers == pytest.approx(7 / 6)
    assert bi.b_repeaters == pytest.approx(1.0)


def test_all_history_window(spark):
    bi = _build(spark, window_days=None)
    assert bi.window_start == date(2024, 1, 1)
    # all 19.0; triers 6.0 + 4.0; repeater 6.0 -> (10/2)/(19/4), (6/1)/(19/4)
    assert bi.b_triers == pytest.approx(20 / 19)
    assert bi.b_repeaters == pytest.approx(24 / 19)


def test_series_growing_bases(spark):
    bi = _build(spark)
    f = {p.period: p for p in bi.points}
    assert sorted(f) == [1, 2, 3]
    # growing bases: members seen up to the END of each bucket
    assert {t: p.n_buyers for t, p in f.items()} == {1: 2, 2: 3, 3: 4}
    assert {t: p.n_triers for t, p in f.items()} == {1: 1, 2: 2, 3: 2}
    assert {t: p.n_repeaters for t, p in f.items()} == {1: 0, 2: 0, 3: 1}
    # W1: (3/1)/(7/2); W2: (4/2)/(6/3) -- h3's 2.0 bought BEFORE its trial but
    # in the trial bucket counts (bucket-granularity membership); W3: (3/2)/(6/4)
    assert f[1].b_triers == pytest.approx(6 / 7)
    assert f[2].b_triers == pytest.approx(1.0)
    assert f[3].b_triers == pytest.approx(1.0)
    # repeaters exist only from W3: (2/1)/(6/4)
    assert f[1].b_repeaters is None and f[2].b_repeaters is None
    assert f[3].b_repeaters == pytest.approx(4 / 3)
    assert f[2].cat_qty == 6.0 and f[2].cat_qty_triers == 4.0


def test_series_gap_free_zero_filled(spark):
    bi = _build(spark, analysis_date="2024-01-28")   # W4 has no transactions
    f = {p.period: p for p in bi.points}
    assert sorted(f) == [1, 2, 3, 4]
    assert f[4].cat_qty == 0.0 and f[4].b_triers is None
    assert f[4].b_repeaters is None
    # the bases carry over into the empty bucket
    assert (f[4].n_buyers, f[4].n_triers, f[4].n_repeaters) == (4, 2, 1)


def test_cohorts(spark):
    bi = _build(spark, cohort_unit="iso_week")
    frame = bi.cohort_frame()
    assert list(frame.columns) == ["cohort", "n_triers", "cat_qty", "b"]
    assert frame["cohort"].tolist() == ["2024-W01", "2024-W02"]
    assert frame["n_triers"].tolist() == [1, 1]
    # window volumes vs the all-buyer average 12/4: (3/1)/3 and (4/1)/3
    assert frame["b"].tolist() == pytest.approx([1.0, 4 / 3])
    # cohorts never change the pooled numbers
    plain = _build(spark)
    assert bi.b_triers == plain.b_triers


def test_repeater_threshold_and_lines_not_days(spark):
    # threshold 3: h1 has only 2 brand lines -> no repeaters anywhere
    bi = _build(spark, repeater_min_purchases=3)
    assert bi.n_repeaters == 0 and bi.b_repeaters is None
    assert all(p.b_repeaters is None for p in bi.points)
    # two brand LINES on the same day cross the threshold that day
    rows = [("g1", "2024-01-02", True, True, 1.0),
            ("g1", "2024-01-02", True, True, 1.0),
            ("g2", "2024-01-04", False, True, 2.0)]
    two = build_buying_index(_sdf(spark, rows), window_days=None,
                             launch_date="2024-01-01",
                             analysis_date="2024-01-07")
    assert two.n_repeaters == 1
    assert two.points[0].n_repeaters == 1
    assert two.b_repeaters == pytest.approx(1.0)     # (2/1)/(4/2)


def test_launch_floor_and_inference(spark):
    rows = [
        ("h4", "2023-12-20", True, True, 1.0),    # pre-launch: never counts
        ("h4", "2024-01-16", True, True, 1.0),
        ("h4", "2024-01-19", False, True, 2.0),
    ]
    floored = build_buying_index(_sdf(spark, rows), window_days=None,
                                 launch_date="2024-01-01",
                                 analysis_date="2024-01-21")
    # trial re-dated to 01-16 (W3); only 1 post-launch brand line -> no repeater
    assert (floored.n_buyers, floored.n_triers, floored.n_repeaters) == (1, 1, 0)
    f = {p.period: p for p in floored.points}
    assert f[1].cat_qty == 0.0 and f[1].b_triers is None    # empty early bucket
    assert f[3].b_triers == pytest.approx(1.0)              # (3/1)/(3/1)
    assert floored.b_triers == pytest.approx(1.0)
    assert floored.b_repeaters is None
    # without launch/analysis: origin = first brand ts, analysis = max ts, and
    # the pre-launch line now counts -- h4 turns repeater at its 2nd line
    inferred = build_buying_index(_sdf(spark, rows), window_days=None)
    assert inferred.origin == date(2023, 12, 20)
    assert inferred.analysis_date == date(2024, 1, 19)
    assert len(inferred.points) == 5
    assert inferred.points[0].n_repeaters == 0
    assert inferred.points[4].n_repeaters == 1              # 01-16 is bucket 5


def test_validation_errors(spark):
    sdf = _sdf(spark, _rows())
    with pytest.raises(ValueError, match="unit must be one of"):
        build_buying_index(sdf, window_days=None, unit="week")
    with pytest.raises(ValueError, match="unit must be one of"):
        build_buying_index(sdf, window_days=None, cohort_unit="foo")
    with pytest.raises(ValueError, match="window_days"):
        build_buying_index(sdf, window_days=0)
    with pytest.raises(ValueError, match="at least 2 brand purchases"):
        build_buying_index(sdf, window_days=None, repeater_min_purchases=1)
    with pytest.raises(ValueError, match="no transactions"):
        build_buying_index(sdf, window_days=None, analysis_date="2023-01-01")
    with pytest.raises(ValueError, match="no brand triers"):
        build_buying_index(sdf, window_days=None, launch_date="2024-06-01")
    cat_only = [("h9", "2024-01-05", False, True, 1.0)]
    with pytest.raises(ValueError, match="launch origin"):
        build_buying_index(_sdf(spark, cat_only), window_days=None)
    # a window past the last purchase has no volume to compare
    with pytest.raises(ValueError, match="no category volume"):
        _build(spark, analysis_date="2024-02-25", window_days=7)
    with pytest.raises(ValueError, match="without cohorts"):
        _build(spark).cohort_frame()


def test_frames_and_partial_note(spark, capsys):
    bi = _build(spark)
    capsys.readouterr()                      # drop anything Spark printed
    frame = bi.to_frame()
    assert list(frame.columns) == [
        "period", "label", "b_triers", "b_repeaters", "cat_qty",
        "cat_qty_triers", "cat_qty_repeaters", "n_buyers", "n_triers",
        "n_repeaters"]
    assert frame["label"].tolist() == ["2024-W01", "2024-W02", "2024-W03"]
    assert frame["b_repeaters"].isna().tolist() == [True, True, False]
    # analysis on the bucket's last day (Sunday): complete, no note printed
    assert not bi.last_bucket_partial
    assert capsys.readouterr().out == ""
    summary = bi.summary()
    assert list(summary.columns) == ["scope", "n_members", "cat_qty",
                                     "avg_per_member", "b"]
    assert summary["scope"].tolist() == ["all", "triers", "repeaters"]
    assert summary["b"].tolist() == pytest.approx([1.0, 7 / 6, 1.0])
    assert summary["avg_per_member"].tolist() == pytest.approx([3.0, 3.5, 3.0])
    # mid-bucket analysis date: the note is printed before the frame is shown
    partial = _build(spark, analysis_date="2024-01-19")
    assert partial.last_bucket_partial
    partial.to_frame()
    out = capsys.readouterr().out
    assert "PARTIAL" in out and "2024-W03" in out


# --------------------------------------------------------------------------- #
# Plot smoke (Agg)
# --------------------------------------------------------------------------- #
def _dummy_bi(with_cohorts=False, empty_series=False):
    points = ([BuyingPoint(1, None, None, 0.0, 0.0, 0.0, 0, 0, 0)]
              if empty_series else
              [BuyingPoint(1, 0.9, None, 5.0, 2.0, 0.0, 2, 1, 0),
               BuyingPoint(2, 1.1, 1.3, 6.0, 3.0, 1.5, 3, 2, 1)])
    table = None
    if with_cohorts:
        table = pd.DataFrame({"cohort": ["2024-W01", "2024-W02"],
                              "n_triers": [1, 1], "cat_qty": [3.0, 4.0],
                              "b": [1.0, 4 / 3]})
    return BuyingIndex(
        origin=date(2024, 1, 1), analysis_date=date(2024, 1, 14),
        unit="iso_week", window_days=14, repeater_min_purchases=2,
        b_triers=1.05, b_repeaters=1.3, n_buyers=3, n_triers=2, n_repeaters=1,
        window_cat_qty=11.0, window_cat_qty_triers=5.0,
        window_cat_qty_repeaters=1.5, points=points,
        cohort_unit="iso_week" if with_cohorts else None, cohort_table=table)


def test_plot_buying_index_smoke():
    import matplotlib
    matplotlib.use("Agg")
    ax = plot_buying_index(_dummy_bi())
    labels = [line.get_label() for line in ax.get_lines()]
    assert "B(t) triers" in labels and "B(t) repeaters" in labels
    assert any(str(lb).startswith("B triers (window)") for lb in labels)
    assert "ISO week" in ax.get_xlabel()
    with pytest.raises(ValueError, match="no buying-index series"):
        plot_buying_index(_dummy_bi(empty_series=True))


def test_plot_buying_cohorts_smoke():
    import matplotlib
    matplotlib.use("Agg")
    ax = plot_buying_cohorts(_dummy_bi(with_cohorts=True))
    assert len(ax.patches) == 2
    assert [t.get_text() for t in ax.get_xticklabels()] == ["2024-W01",
                                                            "2024-W02"]
    with pytest.raises(ValueError, match="without cohorts"):
        plot_buying_cohorts(_dummy_bi())


# --------------------------------------------------------------------------- #
# End-to-end via Spark + synth (planted index recovered exactly)
# --------------------------------------------------------------------------- #
def test_end_to_end_synth_recovers_planted_index(spark):
    from buying_lite.synth import planted_index, simulate_transactions
    kw = dict(n_households=100, trier_share=0.3, repeater_share=0.5,
              heavy_trier=1.5, heavy_repeater=2.5)
    pdf = simulate_transactions(**kw, trial_weeks=1, horizon_weeks=8,
                                launch="2024-01-01", seed=2)
    b_t, b_r = planted_index(**kw)
    # analysis on the Sunday closing week 8; 28-day window = weeks 5..8
    bi = build_buying_index(spark.createDataFrame(pdf), window_days=28,
                            launch_date="2024-01-01",
                            analysis_date="2024-02-25",
                            cohort_unit="iso_week")
    assert (bi.n_buyers, bi.n_triers, bi.n_repeaters) == (100, 30, 15)
    assert bi.b_triers == pytest.approx(b_t)
    assert bi.b_repeaters == pytest.approx(b_r)
    # weekly volumes are constant, so all-history gives the same index
    full = build_buying_index(spark.createDataFrame(pdf), window_days=None,
                              launch_date="2024-01-01",
                              analysis_date="2024-02-25")
    assert full.b_triers == pytest.approx(b_t)
    # series: triers steady from W1; repeaters join at their 2nd line in W2
    f = {p.period: p for p in bi.points}
    assert f[1].b_repeaters is None
    for t in range(1, 9):
        assert f[t].b_triers == pytest.approx(b_t)
        if t >= 2:
            assert f[t].b_repeaters == pytest.approx(b_r)
    # a single entry cohort == the trier base itself
    frame = bi.cohort_frame()
    assert frame["cohort"].tolist() == ["2024-W01"]
    assert frame["b"].tolist() == pytest.approx([b_t])


def test_end_to_end_staggered_trials_grow_the_bases(spark):
    from buying_lite.synth import simulate_transactions
    pdf = simulate_transactions(n_households=100, trier_share=0.3,
                                repeater_share=0.5, trial_weeks=2,
                                horizon_weeks=6, launch="2024-01-01", seed=5)
    bi = build_buying_index(spark.createDataFrame(pdf), window_days=28,
                            launch_date="2024-01-01",
                            analysis_date="2024-02-11")
    f = {p.period: p for p in bi.points}
    # 30 triers split over 2 trial weeks; every household buys from W1
    assert {t: f[t].n_buyers for t in (1, 2)} == {1: 100, 2: 100}
    assert (f[1].n_triers, f[2].n_triers) == (15, 30)
    # repeaters (15) turn members the week AFTER their trial: 8 even-indexed
    # trial in W1 -> repeater in W2, the 7 odd-indexed follow in W3
    assert (f[1].n_repeaters, f[2].n_repeaters, f[3].n_repeaters) == (0, 8, 15)


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
