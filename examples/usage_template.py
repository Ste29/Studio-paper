"""TEMPLATE for a TRB brand-share analysis. Copy this and swap the data source.

Run:  uv run python -m examples.usage_template

The flow mirrors how an analyst reads a launch:
  1. realised market share over time (all we have up to the analysis date);
  2. penetration so far + the projected ultimate level (and, if a promotion ran,
     how the realised curve diverged from the baseline and the re-fitted ultimate);
  3. repeat-buying rate by interval and by entry cohort;
  4. the final expected (equilibrium) share = Σ Pᵢ × Rᵢ × Bᵢ.

PRODUCTION NOTE: the only change for the real single-retailer panel is the data
source and the SparkSession. Here we build a small *local* session (the engine is
the same as production); on the cluster you pass your own session's DataFrame:
    res = run_trb(spark_df, cfg)
The Spark aggregator runs every heavy group-by on the cluster and only collects
the small, card-collapsed series; everything below is identical.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from parfitt_trb import (
    TRBConfig, penetration_stability, penetration_vs_actual, pwsd, run_trb,
    validate_penetration,
)
from parfitt_trb import plots
from parfitt_trb.display import rollup_ratio
from parfitt_trb.local_spark import build_local_spark
from examples.synth import simulate_panel

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)


def main():
    # --- 0. Data -----------------------------------------------------------
    # Here: a simulated launch with a price promotion at week 18. In production,
    # replace this with your transaction DataFrame.
    PROMO_WEEK = 18
    pdf = simulate_panel(n_households=6000, weeks=40, K=0.24, a=0.22,
                         rbr_start=0.42, rbr_stable=0.24, cat_interval=2, seed=11,
                         promo_week=PROMO_WEEK, promo_K=0.12, promo_rbr=0.07)
    spark = build_local_spark("trb-usage-template")
    df = spark.createDataFrame(pdf)            # in production: your cluster DataFrame
    cfg = TRBConfig(launch_date="2024-01-01", period_length_days=14,
                    buying_index_base="triers")
    # Calendar granularity of penetration / share / per-period buying index:
    #   period_unit="month"      -> compute & label these on calendar months
    #   period_unit="iso_week"   -> ISO calendar weeks ('YYYY-Www', cross-year safe)
    #   period_unit="fiscal_445" -> retail 4-4-5 periods ('YYYY-Pnn')
    # The iso_week / fiscal_445 axes live on the real calendar grid, so an
    # out-of-stock week with no sales keeps its slot instead of collapsing.
    # res.label(period) gives the calendar label; the default below is weekly,
    # then coarsened to months for the printout via rollup_ratio.
    res = run_trb(df, cfg)                      
    pen = res.penetration

    # --- 1. Realised market share over time --------------------------------
    print("=" * 64)
    print("1. MARKET SHARE OVER TIME (realised, to the analysis date)")
    monthly_share = rollup_ratio(res.share_series, res.origin)
    for m, s in monthly_share[-6:]:
        print(f"   {m}: {s*100:5.1f}%" if s is not None else f"   {m}:   n/a")

    # --- 2. Penetration: observed, projected, promotion divergence ---------
    print("\n2. PENETRATION")
    print(f"   observed trial index (snapshot)   = {res.trial_index*100:5.1f}% "
          f"of category buyers")
    if pen.ultimate_penetration:
        print(f"   projected ultimate penetration K  = {pen.ultimate_penetration*100:5.1f}% "
              f"(growth a = {pen.growth_rate:.3f}/week)")
        # model fit quality on the penetration curve (appendix p.w.s.d.)
        actual = [p for _, p in pen.series]
        fitted = [pen.fitted(t) for t, _ in pen.series]
        print(f"   penetration fit p.w.s.d.          = {pwsd(actual, fitted)*100:.2f}%")
    if pen.note:
        print(f"   note: {pen.note}")

    promo = penetration_vs_actual(pen, cutoff_period=PROMO_WEEK, method="discounted")
    if promo.baseline_K is not None and promo.refit_K is not None:
        print(f"   promotion @week {PROMO_WEEK}: baseline K={promo.baseline_K*100:.1f}% "
              f"-> re-fit K={promo.refit_K*100:.1f}% "
              f"(bought ~ {promo.bought_penetration*100:+.1f}pp of penetration)")

    # composed promo-aware curve: each segment re-anchored on the observed
    # penetration at its promo (the 'true' theoretical curve).
    pw = res.piecewise_penetration([PROMO_WEEK])
    if pw.ultimate_penetration is not None:
        print(f"   piecewise (promo-aware) ultimate K = {pw.ultimate_penetration*100:.1f}%")
    if pw.note:
        print(f"   note: {pw.note}")

    # out-of-sample validation: fit up to week 30 only, project, and score the
    # whole curve (old + predicted weeks) against the full observed series.
    val = validate_penetration(pen, cutoff_period=30, promo_periods=[PROMO_WEEK])
    if val.pwsd_full is not None:
        print(f"   validation @wk30: pwsd full = {val.pwsd_full*100:.2f}%  "
              f"held-out only = {val.pwsd_holdout*100:.2f}%")
    if val.note:
        print(f"   note: {val.note}")

    # does the estimated K stabilise as the estimation window grows?
    print("\n   K / a stability vs estimation cutoff (last 5 windows):")
    print(penetration_stability(pen).tail(5).to_string(index=False, formatters={
        "K": "{:.3f}".format, "a": "{:.3f}".format, "observed_P": "{:.3f}".format}))

    # --- 3. Repeat-buying rate by interval and by cohort -------------------
    print("\n3. REPEAT-BUYING RATE")
    ur = res.ultimate_rbr()
    if ur:
        print(f"   furthest-available RBR = {ur[1]*100:.1f}% (interval {ur[0]})")
    plat = res.detect_plateau()
    if plat:
        print(f"   (diagnostic) levels off near {plat[1]*100:.1f}% from interval {plat[0]}")
        # stabilised mean: set rbr_stable_from in TRBConfig to make this the
        # share default; here we just show the diagnostic value.
        st = res.stable_rbr(from_interval=plat[0])
        if st is not None:
            print(f"   stabilised mean RBR (from interval {plat[0]}) = {st*100:.1f}%")
    print("\n   By entry cohort (segmented / Table 2 model):")
    print(res.cohort_table().to_string(index=False, formatters={
        "penetration": "{:.3f}".format,
        "rbr": (lambda v: "" if v is None else f"{v:.3f}"),
        "buying_index": (lambda v: "" if v is None else f"{v:.3f}"),
        "contribution": "{:.4f}".format}))

    # --- 4. Final expected share ------------------------------------------
    print("\n4. EXPECTED EQUILIBRIUM SHARE")
    print(f"   simple   K x R(last) x B = {res.predict_share_projected()*100:5.2f}%")
    print(f"   segmented  Σ Pᵢ Rᵢ Bᵢ    = {res.segmented_share()*100:5.2f}%")
    print(f"   buying index (overall)   = {res.buying_index:.3f}")
    print("=" * 64)

    # --- Dashboard figure --------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    plots.plot_share_over_time(res, ax=axes[0][0])
    plots.plot_penetration_promo(promo, ax=axes[0][1])
    plots.plot_rbr(res, ax=axes[1][0], mark_plateau=True)
    plots.plot_cohort_contributions(res, ax=axes[1][1])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "usage_template_dashboard.png"), dpi=110)
    plt.close(fig)
    print("wrote figures/usage_template_dashboard.png")

    # --- Diagnostics figure: promo piecewise / ΔP vs P / cohort RBR ---------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    plots.plot_penetration_piecewise(res, [PROMO_WEEK], ax=axes[0], pw=pw)
    plots.plot_dp_vs_p(pen, ax=axes[1], promo_periods=[PROMO_WEEK])
    plots.plot_rbr_cohorts(res, ax=axes[2])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "usage_template_diagnostics.png"), dpi=110)
    plt.close(fig)
    print("wrote figures/usage_template_diagnostics.png")
    spark.stop()


if __name__ == "__main__":
    main()
