"""Hand-computed anchors for penetration_lite: calendar ordinals/labels, the
Spark aggregation, the fit, the piecewise promo composition, validation,
stability, and a plot smoke test."""
from __future__ import annotations

import math
import pathlib
from datetime import date

import numpy as np
import pandas as pd
import pytest

from penetration_lite import (
    PenetrationCurve, build_penetration, fit, fit_piecewise, period_label,
    period_of, plot_penetration, pwsd, smoothed_series, stability, validate,
)
from penetration_lite.calendar import bucket_start


# --------------------------------------------------------------------------- #
# Calendar (pure Python, hand-computed)
# --------------------------------------------------------------------------- #
def test_iso_week_ordinals_and_labels():
    origin = "2024-01-03"                       # Wednesday, ISO 2024-W01
    assert period_of("2024-01-01", origin, "iso_week") == 1   # Monday, same week
    assert period_of("2024-01-07", origin, "iso_week") == 1   # Sunday, same week
    assert period_of("2024-01-08", origin, "iso_week") == 2
    assert period_label(1, origin, "iso_week") == "2024-W01"
    assert period_label(2, origin, "iso_week") == "2024-W02"


def test_iso_week_year_boundary():
    origin = "2023-12-27"                       # Wednesday, ISO 2023-W52
    assert period_of("2024-01-02", origin, "iso_week") == 2   # ISO 2024-W01
    assert period_label(1, origin, "iso_week") == "2023-W52"
    assert period_label(2, origin, "iso_week") == "2024-W01"


def test_iso_fortnight_pairs_and_first_week_naming():
    # abs_week(2024-01-01) = 2817 (odd) -> 2024-W01 is the SECOND week of its
    # pair: the fortnight is (2023-W52, 2024-W01), named after the first week.
    origin = "2024-01-01"
    assert period_of("2023-12-27", origin, "iso_fortnight") == 1  # 2023-W52, same pair
    assert period_of("2024-01-08", origin, "iso_fortnight") == 2  # 2024-W02 opens the next
    assert period_label(1, origin, "iso_fortnight") == "2023-F52"
    assert period_label(2, origin, "iso_fortnight") == "2024-F02"
    assert period_label(3, origin, "iso_fortnight") == "2024-F04"
    # bucket 1 starts on the Monday of the pair's first week and contains origin
    start = bucket_start(1, origin, "iso_fortnight")
    assert start == date(2023, 12, 25) and start.weekday() == 0


def test_month_ordinals_and_labels():
    origin = "2023-11-05"
    assert period_of("2023-11-30", origin, "month") == 1
    assert period_of("2024-02-01", origin, "month") == 4      # cross-year
    assert period_label(1, origin, "month") == "2023-11"
    assert period_label(4, origin, "month") == "2024-02"


def test_future_labels_exist():
    # labels are pure arithmetic: any ordinal works (projection region).
    # origin 2024-01-03 is ISO 2024-W01 (Monday 2024-01-01); +52 weeks = 2025-W01
    assert period_label(53, "2024-01-03", "iso_week") == "2025-W01"
    assert period_label(13, "2024-01-15", "month") == "2025-01"


# --------------------------------------------------------------------------- #
# build_penetration (Spark, hand-computed counts)
# --------------------------------------------------------------------------- #
def _rows():
    # launch 2024-01-01 (Monday). Weeks: W01 = Jan 1-7, W02 = Jan 8-14, W03 = ...
    return [
        ("h1", "2024-01-02", False, True, 1.0),   # cat trier W01
        ("h1", "2024-01-09", True, True, 1.0),    # brand trier W02
        ("h2", "2024-01-03", False, True, 1.0),   # cat trier W01
        ("h3", "2024-01-10", False, True, 1.0),   # cat trier W02
        ("h3", "2024-01-16", True, True, 1.0),    # brand trier W03
        ("h4", "2024-01-17", False, True, 1.0),   # cat trier W03
        ("h5", "2024-01-09", True, False, 1.0),   # brand-only line W02: counts as cat too
    ]


def _sdf(spark, rows):
    return spark.createDataFrame(
        pd.DataFrame(rows, columns=["shopper_id", "txn_date", "is_new_product",
                                    "is_category", "volume"]))


def test_build_penetration_dynamic_and_static(spark):
    sdf = _sdf(spark, _rows())
    dyn = build_penetration(sdf, unit="iso_week", launch_date="2024-01-01")
    # cat_new: W01=2 (h1,h2), W02=2 (h3,h5 via brand-implies-category), W03=1 (h4)
    # brand_new: W02=2 (h1,h5), W03=1 (h3)
    assert dyn.n_brand_triers == 3 and dyn.n_category_triers == 5
    d = dict(dyn.series)
    assert d[1] == 0.0 and d[2] == 2 / 4 and d[3] == 3 / 5
    assert math.isclose(dyn.snapshot, 0.6)
    sta = build_penetration(sdf, unit="iso_week", launch_date="2024-01-01",
                            denominator="static")
    s = dict(sta.series)
    assert s[1] == 0.0 and s[2] == 2 / 5 and s[3] == 3 / 5


def test_build_penetration_analysis_date_and_origin_inference(spark):
    sdf = _sdf(spark, _rows())
    cut = build_penetration(sdf, unit="iso_week", launch_date="2024-01-01",
                            analysis_date="2024-01-10")
    assert cut.n_brand_triers == 2 and cut.n_category_triers == 4
    assert dict(cut.series)[2] == 2 / 4
    inferred = build_penetration(sdf, unit="iso_week")   # origin = first brand ts
    assert inferred.origin == date(2024, 1, 9)


def test_build_penetration_labels_on_frame(spark):
    curve = build_penetration(_sdf(spark, _rows()), unit="month",
                              launch_date="2024-01-01")
    frame = curve.to_frame()
    assert list(frame.columns) == ["period", "label", "P_observed", "P_fitted"]
    assert frame["label"].iloc[0] == "2024-01"


# --------------------------------------------------------------------------- #
# Fit (engine-free)
# --------------------------------------------------------------------------- #
def _curve(series, unit="iso_week"):
    return PenetrationCurve(origin=date(2024, 1, 1), unit=unit,
                            denominator="dynamic", series=series,
                            n_brand_triers=100, n_category_triers=300)


def _exp_series(K, a, periods, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for t in range(1, periods + 1):
        p = K * (1 - math.exp(-a * t)) + (rng.normal(0, noise) if noise else 0.0)
        out.append((t, max(p, 1e-6)))
    return out


def test_fit_recovers_planted_K_a():
    c = fit(_curve(_exp_series(0.40, 0.20, 25)))
    assert abs(c.K - 0.40) <= 0.02 and abs(c.a - 0.20) <= 0.03


def test_smoothing_improves_noisy_fit():
    raw = fit(_curve(_exp_series(0.40, 0.20, 25, noise=0.004, seed=7)))
    smooth = fit(_curve(_exp_series(0.40, 0.20, 25, noise=0.004, seed=7)),
                 smoothing_window=5)
    # smoothing recovers a good K; the raw fit is no better (here it fails
    # outright -- the differencing amplified the noise past the a>0 guardrail).
    assert smooth.K is not None and abs(smooth.K - 0.40) <= 0.03
    assert raw.K is None or abs(raw.K - 0.40) >= abs(smooth.K - 0.40)
    assert smooth.series == raw.series          # raw series never mutated


def test_smoothed_series_edges_and_validation():
    sm = smoothed_series([(1, 0.1), (2, 0.2), (3, 0.3), (4, 0.4)], 3)
    assert len(sm) == 4
    assert math.isclose(sm[0][1], 0.15) and math.isclose(sm[-1][1], 0.35)
    with pytest.raises(ValueError):
        smoothed_series([(1, 0.1), (2, 0.2)], 4)


def test_fit_guardrail_accelerating_curve():
    c = fit(_curve([(t, 0.001 * t * t) for t in range(1, 12)]))
    assert c.K is None and "a<=0" in c.note


# --------------------------------------------------------------------------- #
# Piecewise promo composition (engine-free)
# --------------------------------------------------------------------------- #
def _promo_series(K1=0.24, a1=0.30, promo=18, K2=0.12, a2=0.35, periods=32):
    series = _exp_series(K1, a1, promo)
    base = series[-1][1]
    for t in range(promo + 1, periods + 1):
        series.append((t, base + K2 * (1 - math.exp(-a2 * (t - promo)))))
    return series


def test_piecewise_continuity_and_ceiling():
    series = _promo_series()
    pw = fit_piecewise(_curve(series), [18])
    obs18 = dict(series)[18]
    assert pw.fitted(18) == obs18                       # anchored by construction
    assert abs(pw.segments[0].K_inc - 0.24) <= 0.02     # launch K recovered
    assert pw.ultimate_penetration > pw.segments[0].ceiling
    assert abs(pw.ultimate_penetration - (obs18 + 0.12)) <= 0.02


def test_piecewise_short_segment_falls_back():
    pw = fit_piecewise(_curve(_promo_series()), [30])   # only 2 points after
    assert "not fitted" in pw.note
    assert pw.fitted(31) is not None                    # falls back to segment 0


def test_piecewise_to_frame_has_labels():
    pw = fit_piecewise(_curve(_promo_series(), unit="month"), [18])
    frame = pw.to_frame()
    assert "label" in frame.columns and frame["label"].iloc[0] == "2024-01"


# --------------------------------------------------------------------------- #
# Validation and stability (engine-free)
# --------------------------------------------------------------------------- #
def test_validate_clean_series():
    v = validate(_curve(_exp_series(0.40, 0.20, 30)), cutoff_period=18)
    assert v.pwsd_holdout is not None and v.pwsd_holdout < 0.05
    assert len(v.actual) == len(v.forecast) == 30
    frame = v.to_frame()
    assert list(frame.columns) == ["period", "label", "actual", "forecast"]


def test_validate_promo_aware_beats_plain():
    c = _curve(_promo_series())
    plain = validate(c, cutoff_period=26)
    aware = validate(c, cutoff_period=26, promo_periods=[18])
    assert aware.pwsd_full < plain.pwsd_full


def test_validate_requires_train_and_holdout():
    c = _curve(_exp_series(0.40, 0.20, 10))
    with pytest.raises(ValueError):
        validate(c, cutoff_period=2)
    with pytest.raises(ValueError):
        validate(c, cutoff_period=10)


def test_stability_converges_and_has_labels():
    tab = stability(_curve(_exp_series(0.40, 0.20, 30)))
    assert list(tab.columns) == ["cutoff", "label", "K", "a", "observed_P", "note"]
    tail = tab["K"].tail(3)
    assert (tail - 0.40).abs().max() <= 0.02 and tail.std() <= 0.01


def test_pwsd_hand_computed():
    # single-step case: sqrt((0.6*0.25^2 + 1*0^2)/1.6) with actual=[0.4,0.5]
    val = pwsd([0.4, 0.5], [0.3, 0.5], w=0.6)
    assert math.isclose(val, math.sqrt(0.6 * 0.25 ** 2 / 1.6))


# --------------------------------------------------------------------------- #
# Plot smoke (Agg) and end-to-end via Spark + synth
# --------------------------------------------------------------------------- #
def test_plot_smoke_with_promo_boost():
    import matplotlib
    matplotlib.use("Agg")
    curve = _curve(_promo_series())
    fit(curve)
    ax = plot_penetration(curve, promo_periods=[18])
    assert ax is not None
    from matplotlib.collections import PolyCollection
    assert any(isinstance(c, PolyCollection) for c in ax.collections)  # green fill
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert 0 < len(labels) <= 14 and labels[0].startswith("2024-W")


def test_end_to_end_synth_fortnight(spark):
    from penetration_lite.synth import simulate_transactions
    pdf = simulate_transactions(n_households=1500, weeks=30, K=0.35, a=0.20,
                                launch="2024-01-01", seed=3)
    curve = build_penetration(spark.createDataFrame(pdf), unit="iso_fortnight",
                              launch_date="2024-01-01")
    fit(curve, smoothing_window=3)
    assert curve.K is not None and abs(curve.K - 0.35) <= 0.05
    assert curve.to_frame()["label"].str.match(r"^\d{4}-F\d{2}$").all()


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
