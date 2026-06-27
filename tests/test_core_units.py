"""Unit tests for the backend-free core: cohorts (Table 2), p.w.s.d., the promo
penetration comparison, RBR bucket mode, and a plotting smoke test."""
from __future__ import annotations

import math
from datetime import date

import pandas as pd

from parfitt_trb import TRBConfig, pwsd, run_trb
from parfitt_trb.core import (
    Penetration, blended_rbr, build_cohorts, penetration_vs_actual, segmented_share,
)
from tests.helpers import approx, make_df, row


# --------------------------------------------------------------------------- #
# Segmented model (Table 2 mechanics)
# --------------------------------------------------------------------------- #
def test_cohort_reconstruction():
    trials = pd.DataFrame({
        "card": [f"a{i}" for i in range(150)],
        "cohort": ["1-6w"] * 100 + ["7-12w"] * 50,
    })
    rbr_cohort = pd.DataFrame([
        {"cohort": "1-6w", "interval": 1, "brand_qty": 30, "cat_qty": 100},
        {"cohort": "1-6w", "interval": 2, "brand_qty": 20, "cat_qty": 100},   # last -> 0.20
        {"cohort": "7-12w", "interval": 1, "brand_qty": 15, "cat_qty": 100},
        {"cohort": "7-12w", "interval": 2, "brand_qty": 10, "cat_qty": 100},  # last -> 0.10
    ])
    scopes = pd.DataFrame([
        {"scope": "__all__", "sum_cat": 1000, "n_buyers": 1000},   # avg 1.0
        {"scope": "1-6w", "sum_cat": 100, "n_buyers": 100},        # B = 1.0
        {"scope": "7-12w", "sum_cat": 50, "n_buyers": 50},          # B = 1.0
    ])
    cohorts = build_cohorts(trials, rbr_cohort, scopes, n_category_triers=1000,
                            cohort_order=["1-6w", "7-12w", "13-24w", "25+w"],
                            ultimate_penetration=0.30)
    by = {c.label: c for c in cohorts}
    assert approx(by["1-6w"].penetration, 0.10) and approx(by["1-6w"].rbr, 0.20)
    assert approx(by["7-12w"].penetration, 0.05) and approx(by["7-12w"].rbr, 0.10)
    fut = by["future (estimated)"]
    assert fut.is_future and approx(fut.penetration, 0.15) and approx(fut.rbr, 0.10)
    assert approx(fut.buying_index, 1.0)
    # Share = Σ Pᵢ Rᵢ Bᵢ = 0.02 + 0.005 + 0.015 = 0.04
    assert approx(segmented_share(cohorts), 0.04)
    # blended RBR is the penetration-weighted average, NOT a sum
    assert approx(blended_rbr(cohorts), 0.04 / 0.30)


# --------------------------------------------------------------------------- #
# Percentage weighted standard deviation (appendix / Fig 19)
# --------------------------------------------------------------------------- #
def test_pwsd():
    assert approx(pwsd([0.30, 0.31, 0.32], [0.30, 0.31, 0.32]), 0.0)
    # actual=[.2,.3], forecast=[.25,.3], w=.6 ; latest (last) weighted most
    expected = math.sqrt((0.6 * 0.25 ** 2) / (0.6 + 1.0))
    assert approx(pwsd([0.2, 0.3], [0.25, 0.3], w=0.6), expected, 1e-9)


# --------------------------------------------------------------------------- #
# Promotion: actual diverges above the pre-promo baseline projection
# --------------------------------------------------------------------------- #
def test_penetration_vs_actual_promo():
    K, a = 0.30, 0.25
    series = [(t, K * (1 - math.exp(-a * t))) for t in range(1, 11)]
    series += [(t, K * (1 - math.exp(-a * t)) + 0.05) for t in range(11, 17)]  # promo bump
    pen = Penetration("dynamic", date(2024, 1, 1), series, 0, 1)
    promo = penetration_vs_actual(pen, cutoff_period=10, method="ols")
    assert promo.baseline_K is not None and abs(promo.baseline_K - 0.30) <= 0.05
    assert promo.bought_penetration is not None and promo.bought_penetration > 0.02
    assert promo.refit_K is not None and promo.refit_K > promo.baseline_K


# --------------------------------------------------------------------------- #
# RBR bucket mode (calendar weeks / months after the trial bucket)
# --------------------------------------------------------------------------- #
def test_rbr_bucket_week():
    rows = [
        row("x", "2024-01-01", True, True, 5),     # trial (Monday)
        row("x", "2024-02-05", True, True, 4),     # exactly 5 calendar weeks later
        row("x", "2024-02-05", False, True, 6),
    ]
    res = run_trb(make_df(rows),
                  TRBConfig(rbr_interval_mode="bucket", rbr_bucket_unit="week",
                            analysis_date="2024-06-01"))
    assert approx(res.rbr_at(5), 0.4), res.rbr_at(5)


def test_rbr_bucket_month():
    rows = [
        row("x", "2024-01-10", True, True, 5),     # trial in month 2024-01
        row("x", "2024-06-15", True, True, 3),     # 5 months later
        row("x", "2024-06-15", False, True, 7),
    ]
    res = run_trb(make_df(rows),
                  TRBConfig(rbr_interval_mode="bucket", rbr_bucket_unit="month",
                            analysis_date="2024-12-31"))
    assert approx(res.rbr_at(5), 0.3), res.rbr_at(5)


# --------------------------------------------------------------------------- #
def test_plotting_smoke():
    import matplotlib
    matplotlib.use("Agg")
    from parfitt_trb import plots

    res = run_trb(make_df([
        row("anita", "2024-01-01", True, True, 5),
        row("anita", "2024-05-15", True, True, 10),
        row("anita", "2024-05-15", False, True, 10),
        row("claire", "2024-04-01", True, True, 5),
        row("claire", "2024-08-14", True, True, 5),
        row("claire", "2024-08-14", False, True, 15),
    ]), TRBConfig(period_length_days=30, analysis_date="2024-09-30"))

    plots.plot_rbr(res)
    plots.plot_share_bars(res)
    plots.plot_share_over_time(res)
    plots.plot_predicted_share(res)
    try:
        plots.plot_penetration(res)
    except ValueError:
        pass
    plots.plot_dashboard(res)
    import matplotlib.pyplot as plt
    plt.close("all")
