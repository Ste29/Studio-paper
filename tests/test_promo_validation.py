"""Engine-free tests for the promo-aware piecewise penetration curve, the
out-of-sample pwsd validation, the K/a stability table, the stabilised-mean RBR
and the derived (future-capable) calendar labels -- plus smoke tests for the
new diagnostic plots. No Spark session needed."""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from parfitt_trb import TRBConfig, TRBResult
from parfitt_trb.core import (
    Penetration, RBRPoint, build_cohorts, fit_penetration,
    fit_piecewise_penetration, penetration_stability, rbr_cohort_series,
    smoothed_series, stable_rbr, validate_penetration,
)
from parfitt_trb.periods import extended_period_label
from tests.helpers import approx


# --------------------------------------------------------------------------- #
# Synthetic series helpers
# --------------------------------------------------------------------------- #
def _exp_series(K=0.30, a=0.25, n=16, t0=1):
    return [(t, K * (1 - math.exp(-a * t))) for t in range(t0, n + 1)]


def _promo_series(K=0.30, a=0.25, t_promo=18, K2=0.10, a2=0.30, n=36):
    """Clean exponential up to t_promo, then a fresh incremental exponential
    anchored on the observed level (exactly the piecewise model's shape)."""
    series = _exp_series(K, a, t_promo)
    base = series[-1][1]
    series += [(t, base + K2 * (1 - math.exp(-a2 * (t - t_promo))))
               for t in range(t_promo + 1, n + 1)]
    return series


def _pen(series):
    return Penetration("dynamic", date(2024, 1, 1), series, 300, 1000)


# --------------------------------------------------------------------------- #
# Piecewise promo-aware penetration
# --------------------------------------------------------------------------- #
def test_piecewise_recovers_segments():
    series = _promo_series()
    pw = fit_piecewise_penetration(_pen(series), [18], method="ols")
    assert len(pw.segments) == 2
    seg0, seg1 = pw.segments
    assert seg0.K_inc is not None and abs(seg0.K_inc - 0.30) <= 0.05
    obs = dict(series)
    # the promo segment is anchored exactly on the observed P at the promo
    assert seg1.base == obs[18]
    assert approx(pw.fitted(18), obs[18])
    # continuity: right after the promo the curve tracks the observed bump
    assert abs(pw.fitted(19) - obs[19]) <= 0.01
    # the composed ultimate exceeds the launch-only ceiling
    assert pw.ultimate_penetration > seg0.ceiling


def test_piecewise_two_promos():
    series = _promo_series(t_promo=14, n=28)
    base2 = dict(series)[22]
    series = [(t, p) for t, p in series if t <= 22]
    series += [(t, base2 + 0.05 * (1 - math.exp(-0.35 * (t - 22))))
               for t in range(23, 34)]
    pw = fit_piecewise_penetration(_pen(series), [14, 22], method="ols")
    assert len(pw.segments) == 3
    assert all(s.K_inc is not None for s in pw.segments), pw.note
    assert approx(pw.fitted(22), dict(series)[22])


def test_piecewise_short_tail_falls_back():
    series = _promo_series(n=20)                 # promo at 18, only 2 post points
    pw = fit_piecewise_penetration(_pen(series), [18])
    assert pw.segments[1].K_inc is None
    assert "not fitted" in pw.note
    # .fitted stays total: falls back to the (fitted) launch segment
    assert pw.fitted(20) is not None


def test_piecewise_validates_inputs():
    pen = _pen(_exp_series(n=20))
    with pytest.raises(ValueError):
        fit_piecewise_penetration(pen, [12, 12])          # not strictly increasing
    with pytest.raises(ValueError):
        fit_piecewise_penetration(pen, [99])              # not an observed period


# --------------------------------------------------------------------------- #
# Out-of-sample pwsd validation
# --------------------------------------------------------------------------- #
def test_validate_penetration_clean_series():
    pen = _pen(_exp_series(K=0.30, a=0.25, n=30))
    val = validate_penetration(pen, 18, method="ols")
    assert len(val.forecast) == len(val.actual) == 30
    assert val.pwsd_full is not None and val.pwsd_full < 0.05
    assert val.pwsd_holdout is not None and val.pwsd_holdout < 0.05


def test_validate_penetration_promo_aware():
    series = _promo_series()
    val = validate_penetration(_pen(series), 30, method="ols", promo_periods=[18])
    assert val.pwsd_holdout is not None and val.pwsd_holdout < 0.05
    # a promo after the cutoff is unknowable at forecast time: dropped + noted
    val2 = validate_penetration(_pen(series), 12, method="ols", promo_periods=[18])
    assert "dropped" in val2.note
    assert isinstance(val2.curve, Penetration)


def test_validate_penetration_guards():
    pen = _pen(_exp_series(n=10))
    with pytest.raises(ValueError):
        validate_penetration(pen, 3)      # < 4 training points
    with pytest.raises(ValueError):
        validate_penetration(pen, 10)     # nothing held out


# --------------------------------------------------------------------------- #
# K / a stability vs estimation cutoff
# --------------------------------------------------------------------------- #
def test_penetration_stability_converges():
    pen = _pen(_exp_series(K=0.40, a=0.20, n=25))
    df = penetration_stability(pen, method="ols")
    assert list(df.columns) == ["cutoff", "K", "a", "observed_P", "note"]
    tail_K = df["K"].tail(3)
    assert (abs(tail_K - 0.40) <= 0.02).all()
    assert float(tail_K.std()) < 0.01


def test_penetration_stability_notes_unstable_fit():
    # still-accelerating curve: ΔP grows with P -> a <= 0 -> noted, K is NaN
    pen = _pen([(t, 0.001 * t * t) for t in range(1, 9)])
    df = penetration_stability(pen, cutoffs=[8])
    assert np.isnan(df.loc[0, "K"]) and df.loc[0, "note"] != ""


# --------------------------------------------------------------------------- #
# Stabilised-mean RBR
# --------------------------------------------------------------------------- #
def _rbr_points():
    vals = [(1, 0.50), (2, 0.30), (3, 0.25), (4, None), (5, 0.23)]
    return [RBRPoint(interval=t, rbr=v, brand_qty=0.0, cat_qty=0.0, n_eligible=1)
            for t, v in vals]


def test_stable_rbr_mean_skips_missing():
    pts = _rbr_points()
    assert approx(stable_rbr(pts, 3), (0.25 + 0.23) / 2)
    assert stable_rbr(pts, 6) is None


def test_predict_share_projected_uses_stable_mean():
    pen = Penetration("dynamic", date(2024, 1, 1), [(1, 0.1)], 100, 1000,
                      ultimate_penetration=0.40, growth_rate=0.2)
    base = dict(trial_index=0.1, buying_index=1.0, rbr_series=_rbr_points(),
                analysis_date=date(2024, 6, 1), penetration=pen)
    res = TRBResult(config=TRBConfig(rbr_stable_from=3), **base)
    assert approx(res.predict_share_projected(), 0.40 * 0.24 * 1.0)
    res_default = TRBResult(config=TRBConfig(), **base)
    assert approx(res_default.predict_share_projected(), 0.40 * 0.23 * 1.0)
    # an explicit rbr_value always wins
    assert approx(res.predict_share_projected(0.30), 0.40 * 0.30 * 1.0)


def _cohort_fixture():
    cohort_counts = {"1-6w": 100, "7-12w": 50}
    rbr_cohort = pd.DataFrame([
        {"cohort": "1-6w", "interval": 1, "brand_qty": 30, "cat_qty": 100},
        {"cohort": "1-6w", "interval": 2, "brand_qty": 20, "cat_qty": 100},
        {"cohort": "7-12w", "interval": 1, "brand_qty": 15, "cat_qty": 100},
    ])
    scopes = pd.DataFrame([
        {"scope": "__all__", "sum_cat": 1000, "n_buyers": 1000},
        {"scope": "1-6w", "sum_cat": 100, "n_buyers": 100},
        {"scope": "7-12w", "sum_cat": 50, "n_buyers": 50},
    ])
    return cohort_counts, rbr_cohort, scopes


def test_build_cohorts_stable_mean_and_fallback():
    counts, rbr_c, scopes = _cohort_fixture()
    order = ["1-6w", "7-12w", "13-24w", "25+w"]
    cohorts = build_cohorts(counts, rbr_c, scopes, 1000, order, rbr_stable_from=1)
    by = {c.label: c for c in cohorts}
    assert approx(by["1-6w"].rbr, 0.25)          # mean(0.30, 0.20)
    assert approx(by["7-12w"].rbr, 0.15)
    # young cohort with no points past the threshold -> furthest-available
    cohorts = build_cohorts(counts, rbr_c, scopes, 1000, order, rbr_stable_from=2)
    by = {c.label: c for c in cohorts}
    assert approx(by["1-6w"].rbr, 0.20)          # only interval 2
    assert approx(by["7-12w"].rbr, 0.15)         # fallback: furthest (interval 1)
    # None -> unchanged behaviour (furthest-available)
    cohorts = build_cohorts(counts, rbr_c, scopes, 1000, order)
    by = {c.label: c for c in cohorts}
    assert approx(by["1-6w"].rbr, 0.20)


def test_rbr_cohort_series_builder():
    _, rbr_c, _ = _cohort_fixture()
    rbr_c = pd.concat([rbr_c, pd.DataFrame(
        [{"cohort": "7-12w", "interval": 2, "brand_qty": 5, "cat_qty": 0}])],
        ignore_index=True)
    out = rbr_cohort_series(rbr_c, ["1-6w", "7-12w", "13-24w"])
    assert list(out) == ["1-6w", "7-12w"]        # cohort order kept, empty omitted
    assert out["1-6w"] == [(1, 0.30), (2, 0.20)]
    assert out["7-12w"][1] == (2, None)          # no category volume yet


# --------------------------------------------------------------------------- #
# Centred smoothing of the penetration series (fit-only)
# --------------------------------------------------------------------------- #
def test_smoothed_series_shape_and_edges():
    series = [(1, 0.1), (2, 0.2), (3, 0.6), (4, 0.8), (5, 1.0)]
    sm = smoothed_series(series, 3)
    assert [t for t, _ in sm] == [1, 2, 3, 4, 5]          # same grid, same length
    assert approx(sm[0][1], (0.1 + 0.2) / 2)              # clipped left edge
    assert approx(sm[1][1], (0.1 + 0.2 + 0.6) / 3)
    assert approx(sm[4][1], (0.8 + 1.0) / 2)              # clipped right edge
    # a linear series is a fixed point of the interior average
    lin = [(t, 0.1 * t) for t in range(1, 8)]
    assert all(approx(v, 0.1 * t) for t, v in smoothed_series(lin, 3)[1:-1])
    with pytest.raises(ValueError):
        smoothed_series(series, 4)                        # even window
    with pytest.raises(ValueError):
        smoothed_series(series, 1)


def test_smoothing_window_config_validated():
    with pytest.raises(ValueError):
        TRBConfig(penetration_smoothing_window=4)
    with pytest.raises(ValueError):
        TRBConfig(penetration_smoothing_window=1)
    assert TRBConfig(penetration_smoothing_window=5).penetration_smoothing_window == 5


def test_smoothing_improves_noisy_fit():
    """On a noisy exponential the smoothed fit recovers K closer to the truth
    than the raw fit (the centred differencing amplifies the noise)."""
    K_true, a_true, n = 0.30, 0.25, 30
    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 0.008, n)
    series = [(t, K_true * (1 - math.exp(-a_true * t)) + noise[t - 1])
              for t in range(1, n + 1)]

    raw = fit_penetration(_pen(series), method="ols")
    smooth = fit_penetration(_pen(series), method="ols", smoothing_window=5)
    assert smooth.ultimate_penetration is not None
    assert (abs(smooth.ultimate_penetration - K_true)
            <= abs(raw.ultimate_penetration - K_true))
    # the observed series itself is untouched by the smoothing
    assert smooth.series == series


def test_smoothing_none_matches_current_behaviour():
    series = _exp_series(K=0.40, a=0.20, n=20)
    plain = fit_penetration(_pen(series), method="ols")
    explicit = fit_penetration(_pen(series), method="ols", smoothing_window=None)
    assert approx(plain.ultimate_penetration, explicit.ultimate_penetration)
    assert approx(plain.growth_rate, explicit.growth_rate)


def test_validate_penetration_accepts_smoothing():
    pen = _pen(_exp_series(K=0.30, a=0.25, n=30))
    val = validate_penetration(pen, 18, method="ols", smoothing_window=3)
    assert val.pwsd_holdout is not None and val.pwsd_holdout < 0.05


# --------------------------------------------------------------------------- #
# Derived calendar labels (future-capable)
# --------------------------------------------------------------------------- #
def test_extended_period_label_all_units():
    assert extended_period_label(4, date(2024, 11, 15), "month") == "2025-02"
    assert extended_period_label(3, date(2024, 1, 1), "week") == "2024-W03"
    # fortnight: 14-day origin-relative buckets, labelled by their first day's ISO week
    assert extended_period_label(1, date(2024, 1, 1), "fortnight") == "2024-W01"
    assert extended_period_label(2, date(2024, 1, 1), "fortnight") == "2024-W03"
    # iso_week straddling the year boundary (2024-12-25 is ISO 2024-W52)
    assert extended_period_label(1, date(2024, 12, 25), "iso_week") == "2024-W52"
    assert extended_period_label(2, date(2024, 12, 25), "iso_week") == "2025-W01"
    # iso_fortnight: epoch-aligned ISO-week pairs named after the pair's FIRST
    # week; (2023-W52, 2024-W01) is one pair -> '2023-F52', then '2024-F02'
    assert extended_period_label(1, date(2023, 12, 25), "iso_fortnight") == "2023-F52"
    assert extended_period_label(2, date(2023, 12, 25), "iso_fortnight") == "2024-F02"
    # an origin on the pair's SECOND week still labels from the pair's first
    assert extended_period_label(1, date(2024, 1, 2), "iso_fortnight") == "2023-F52"
    # fiscal 4-4-5 from a P11 origin rolls into the next year after P12
    assert extended_period_label(1, date(2024, 11, 5), "fiscal_445") == "2024-P11"
    assert extended_period_label(2, date(2024, 11, 5), "fiscal_445") == "2024-P12"
    assert extended_period_label(3, date(2024, 11, 5), "fiscal_445") == "2025-P01"


def test_result_label_falls_back_to_derived():
    res = TRBResult(trial_index=0.1, buying_index=1.0, rbr_series=[],
                    analysis_date=date(2024, 6, 30), origin=date(2024, 1, 1),
                    period_unit="month", period_labels={1: "2024-01"})
    assert res.label(1) == "2024-01"             # observed map wins
    assert res.label(8) == "2024-08"             # future ordinal: derived
    res.origin = None
    assert res.label(8) == "8"                   # no origin: bare ordinal


# --------------------------------------------------------------------------- #
# Plot smoke tests (Agg backend, hand-built results -- no Spark)
# --------------------------------------------------------------------------- #
def _fake_result(series, unit="week", fit=True, **kw):
    pen = _pen(series)
    if fit:
        from parfitt_trb.core import fit_penetration
        fit_penetration(pen, method="ols")
    return TRBResult(trial_index=0.2, buying_index=1.0,
                     rbr_series=_rbr_points(), analysis_date=date(2024, 12, 31),
                     origin=date(2024, 1, 1), penetration=pen,
                     period_unit=unit, **kw)


def test_plot_penetration_caps_projection_and_thins_labels():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from parfitt_trb import plots

    res = _fake_result(_exp_series(K=0.30, a=0.05, n=30), unit="month")
    ax = plots.plot_penetration(res)
    # projected tail capped at last + 12 on a monthly axis
    assert max(max(line.get_xdata()) for line in ax.lines) == 30 + 12
    ticks = ax.get_xticks()
    assert len(ticks) <= 13                       # thinned calendar labels
    assert max(ticks) > 30                        # ...extending past the data
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert all("-" in t for t in labels if t)     # real calendar labels
    plt.close("all")


def test_plot_penetration_piecewise_fills_boost():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from parfitt_trb import plots

    res = _fake_result(_promo_series(), unit="week", fit=False)
    ax = plots.plot_penetration_piecewise(res, [18])
    assert any(isinstance(c, PolyCollection) for c in ax.collections)
    assert any("bought" in t.get_text() for t in ax.texts)
    plt.close("all")


def test_plot_dp_vs_p_line_matches_fit():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from parfitt_trb import plots

    res = _fake_result(_exp_series(K=0.30, a=0.25, n=20))
    pen = res.penetration
    ax = plots.plot_dp_vs_p(pen, as_percent=False)
    fit_lines = [l for l in ax.lines if l.get_label().startswith("fit:")]
    assert fit_lines
    (x0, x1), (y0, y1) = fit_lines[0].get_xdata(), fit_lines[0].get_ydata()
    assert approx((y1 - y0) / (x1 - x0), -pen.growth_rate, 1e-9)
    # segmented variant: one scatter set + line per segment
    ax2 = plots.plot_dp_vs_p(_pen(_promo_series()), promo_periods=[18], method="ols")
    assert len(ax2.collections) >= 2
    # smoothing variant: smoothed points solid, raw points faded behind
    ax3 = plots.plot_dp_vs_p(pen, as_percent=False, smoothing_window=3)
    legend3 = [t.get_text() for t in ax3.get_legend().get_texts()]
    assert "raw (unsmoothed)" in legend3
    assert len(ax3.collections) >= 2
    plt.close("all")


def test_plot_rbr_cohorts_smoke():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from parfitt_trb import plots

    res = _fake_result(_exp_series(), rbr_cohort_series={
        "1-6w": [(1, 0.5), (2, 0.4)], "7-12w": [(1, 0.3), (2, None)]})
    ax = plots.plot_rbr_cohorts(res)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "1-6w" in labels and "7-12w" in labels and "pooled" in labels
    plt.close("all")
