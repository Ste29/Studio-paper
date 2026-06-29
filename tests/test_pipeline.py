"""End-to-end Spark anchors, ported from the original Spark test suite and
re-derived for the weekly-period core. Each asserts a number that can be traced
back to the paper or hand-computed from the rows. With a single Spark engine
these anchors are the oracle that pins the Spark column expressions."""
from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from parfitt_trb import TRBConfig, TRBResult, run_trb
from tests.helpers import approx, make_sdf, row


# --------------------------------------------------------------------------- #
# RBR -- Anita/Betty/Claire pooled RBR(5) = 15% (date-anchored, lapsed kept)
# --------------------------------------------------------------------------- #
def rbr_rows():
    return [
        row("anita", "2024-01-01", True, True, 5),     # trial (excluded)
        row("anita", "2024-05-15", True, True, 10),    # interval 5: brand 10
        row("anita", "2024-05-15", False, True, 10),   # interval 5: +cat 10 -> 20
        row("betty", "2024-03-01", True, True, 5),     # trial
        row("betty", "2024-07-14", False, True, 60),   # interval 5: brand 0, cat 60 (lapsed)
        row("claire", "2024-04-01", True, True, 5),    # trial
        row("claire", "2024-08-14", True, True, 5),    # interval 5: brand 5
        row("claire", "2024-08-14", False, True, 15),  # interval 5: +cat 15 -> 20
        row("dora", "2024-07-01", True, True, 5),      # trial; only reaches interval 3
        row("dora", "2024-08-15", True, True, 100),
        row("dora", "2024-08-15", False, True, 100),
    ]


def test_rbr_parfitt_example(spark):
    res = run_trb(make_sdf(spark, rbr_rows()),
                  TRBConfig(period_length_days=30, analysis_date="2024-09-30"))
    assert approx(res.rbr_at(5), 0.15), res.rbr_at(5)


def test_eligibility_inherent(spark):
    res = run_trb(make_sdf(spark, rbr_rows()),
                  TRBConfig(period_length_days=30, analysis_date="2024-09-30"))
    by_t = {p.interval: p for p in res.rbr_series}
    assert by_t[3].n_eligible == 4    # incl. Dora
    assert by_t[4].n_eligible == 3    # Dora dropped
    assert by_t[5].n_eligible == 3
    assert approx(res.rbr_at(5), 0.15)


def test_analysis_date_shrinks_cohort(spark):
    """Re-running on 2024-06-30, only Anita has reached interval 5 -> RBR=0.5."""
    res = run_trb(make_sdf(spark, rbr_rows()),
                  TRBConfig(period_length_days=30, analysis_date="2024-06-30"))
    assert approx(res.rbr_at(5), 0.5), res.rbr_at(5)


# --------------------------------------------------------------------------- #
# RBR -- Parfitt Table 1 grid -> pooled 60/50/40/40 (period = 2 weeks)
# --------------------------------------------------------------------------- #
TABLE1 = {
    "b1": "TTRTRRRRTRR", "b2": "T-S-S-S-S-S", "b3": "-TTTTTTTTTT",
    "b4": "-T-T-R-T-R-", "b5": "--T-T-T-T-T", "b6": "--TTRTRRSRR",
    "b7": "--TS-S-S-S-",
}


def table1_rows():
    origin = date(2024, 1, 1)
    rows = []
    for sid, seq in TABLE1.items():
        for i, ch in enumerate(seq):
            if ch == "-":
                continue
            d = (origin + timedelta(days=i * 7)).isoformat()
            rows.append(row(sid, d, ch == "T", True, 1))
    return rows


def test_rbr_table1_anchor(spark):
    res = run_trb(make_sdf(spark, table1_rows()),
                  TRBConfig(period_length_days=14, analysis_date="2024-03-31"))
    expected = {1: 0.60, 2: 0.50, 3: 0.40, 4: 0.40}
    for t, e in expected.items():
        assert approx(res.rbr_at(t), e), f"RBR({t})={res.rbr_at(t)} != {e}"


# --------------------------------------------------------------------------- #
# Buying index & trial index
# --------------------------------------------------------------------------- #
def buying_rows():
    return [
        row("r1", "2023-01-05", True, True, 25),
        row("r1", "2023-02-05", True, True, 25),
        row("r1", "2023-02-20", False, True, 50),     # cat vol 100, repeater
        row("r2", "2023-01-10", True, True, 20),
        row("r2", "2023-02-10", True, True, 20),
        row("r2", "2023-03-10", True, True, 20),
        row("r2", "2023-03-15", False, True, 80),      # cat vol 140, repeater
        row("n1", "2023-01-15", True, True, 30),
        row("n1", "2023-01-20", False, True, 30),      # cat vol 60, non-repeater trier
        row("n2", "2023-01-25", False, True, 40),      # category-only buyer
    ]


def test_buying_index_triers_default(spark):
    res = run_trb(make_sdf(spark, buying_rows()), TRBConfig(analysis_date="2023-12-31"))
    assert approx(res.buying_index, 100.0 / 85.0), res.buying_index


def test_buying_index_repeaters(spark):
    res = run_trb(make_sdf(spark, buying_rows()),
                  TRBConfig(buying_index_base="repeaters", analysis_date="2023-12-31"))
    assert approx(res.buying_index, 120.0 / 85.0), res.buying_index


def test_trial_index_ratio(spark):
    res = run_trb(make_sdf(spark, buying_rows()), TRBConfig(analysis_date="2023-12-31"))
    assert approx(res.trial_index, 0.75), res.trial_index   # 3 brand / 4 category triers


def test_predict_share_formula():
    """Parfitt p.133 worked example: 34% × 25% × 1.00 = 8.5% (pure, engine-free)."""
    res = TRBResult(trial_index=0.34, buying_index=1.00, rbr_series=[],
                    analysis_date=date(2024, 1, 1))
    assert approx(res.predict_share(0.25), 0.085)


# --------------------------------------------------------------------------- #
# Penetration projection (weekly) recovers the planted K, a
# --------------------------------------------------------------------------- #
def projection_rows(K=0.40, a=0.20, n_cat=8000, periods=25):
    origin = date(2024, 1, 1)
    rows = [row(f"c{i}", "2024-01-01", False, True, 1) for i in range(n_cat)]
    tid, cum_prev = 0, 0
    for t in range(1, periods + 1):
        cum = round(K * (1 - math.exp(-a * t)) * n_cat)
        d = (origin + timedelta(days=(t - 1) * 7 + 1)).isoformat()
        for _ in range(cum - cum_prev):                 # reuse existing cat ids
            rows.append(row(f"c{tid}", d, True, True, 1))
            tid += 1
        cum_prev = cum
    return rows


@pytest.mark.parametrize("method", ["discounted", "ols"])
def test_penetration_projection_recovers_K(spark, method):
    cfg = TRBConfig(launch_date="2024-01-01", penetration_method=method)
    pen = run_trb(make_sdf(spark, projection_rows()), cfg).penetration
    assert pen.ultimate_penetration is not None and abs(pen.ultimate_penetration - 0.40) <= 0.03
    assert pen.growth_rate is not None and abs(pen.growth_rate - 0.20) <= 0.05


# --------------------------------------------------------------------------- #
# Penetration denominator: dynamic vs static (hand-computed, weekly)
# --------------------------------------------------------------------------- #
def test_penetration_dynamic_vs_static(spark):
    rows = [
        row("b1", "2024-01-05", True, True, 1),    # week1 trial (repeater)
        row("b1", "2024-01-20", True, True, 1),
        row("b2", "2024-01-06", True, True, 1),    # week1
        row("c1", "2024-01-07", False, True, 1),   # cat entrant week1
        row("c2", "2024-02-06", False, True, 1),   # cat entrant week6
        row("c3", "2024-03-07", False, True, 1),   # cat entrant week10
    ]
    base = dict(launch_date="2024-01-01", analysis_date="2024-03-31")
    dyn = run_trb(make_sdf(spark, rows), TRBConfig(penetration_denominator="dynamic", **base),
                  project_penetration=False).penetration
    sta = run_trb(make_sdf(spark, rows), TRBConfig(penetration_denominator="static", **base),
                  project_penetration=False).penetration
    d = dict(dyn.series)
    s = dict(sta.series)
    assert approx(d[1], 2 / 3) and approx(d[6], 0.5) and approx(d[10], 0.4)
    assert approx(s[1], 0.4) and approx(s[10], 0.4)
    assert approx(dyn.snapshot, 0.4) and approx(sta.snapshot, 0.4)


# --------------------------------------------------------------------------- #
# Realised share over calendar weeks (overshoot then decline)
# --------------------------------------------------------------------------- #
def test_share_over_time(spark):
    rows = [
        row("s1", "2024-01-05", True, True, 2), row("s1", "2024-01-05", False, True, 8),   # wk1 .2
        row("s1", "2024-02-05", True, True, 6), row("s1", "2024-02-05", False, True, 4),   # wk6 .6
        row("s1", "2024-03-06", True, True, 4), row("s1", "2024-03-06", False, True, 6),   # wk10 .4
        row("s1", "2024-04-06", True, True, 3), row("s1", "2024-04-06", False, True, 7),   # wk14 .3
    ]
    res = run_trb(make_sdf(spark, rows), TRBConfig(launch_date="2024-01-01"))
    sr = dict(res.share_ratio_series())
    assert approx(sr[1], 0.2) and approx(sr[6], 0.6) and approx(sr[10], 0.4) and approx(sr[14], 0.3)
    peak = max(sr, key=lambda k: sr[k])
    assert peak == 6


def test_buying_index_series_present(spark):
    res = run_trb(make_sdf(spark, buying_rows()), TRBConfig(analysis_date="2023-12-31"))
    assert len(res.buying_index_series) >= 1
    assert all(b is None or b > 0 for _, b in res.buying_index_series)
