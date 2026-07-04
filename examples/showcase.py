"""SHOWCASE — every analysis the `parfitt_trb` engine can do, in one run.

    uv run python -m examples.showcase

Where `usage_template.py` is the *one flow you copy for a real analysis* and
`replicate_paper_figures.py` reproduces the *paper's figures*, this script is the
guided tour of the toolbox: each numbered section drives one capability, prints
the numbers it produces, and the whole thing ends in a chart gallery so you can
see every plot the library ships. Nothing here is production code — it exists to
answer "what can this thing tell me about a launch?".

Sections
  1. Penetration & ultimate K — the trial curve, projected K, fit quality
  2. Estimation method — discounted (Gilchrist) vs OLS fit of K
  3. Calendar axes — week / month / iso_week / fiscal_445 (out-of-stock safe)
  4. Repeat-buying rate — pooled curve, plateau diagnostic, stabilised mean
  5. RBR interval mode — exact P-day windows vs calendar buckets
  6. Buying index — triers (Parfitt) vs repeaters (Charan) base, over time
  7. Entry cohorts — the segmented Σ Pᵢ·Rᵢ·Bᵢ model (Table 2)
  8. Promotion analysis — baseline vs realised vs re-fit, "bought" penetration
  9. Piecewise promo curve — the composed promo-aware theoretical penetration
 10. Validation & stability — out-of-sample p.w.s.d. and does K settle?
 11. Share prediction — the three headline numbers, side by side
 12. Chart gallery — every plotting function on one figure
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:                                   # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from parfitt_trb import (
    TRBConfig, penetration_stability, penetration_vs_actual, pwsd, run_trb,
    validate_penetration,
)
from parfitt_trb import plots
from parfitt_trb.display import label_ratio, rollup_ratio
from parfitt_trb.local_spark import build_local_spark
from examples.synth import simulate_panel

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

RULE = "=" * 72
PROMO_WEEK = 18


def hdr(n: int, title: str) -> None:
    print(f"\n{RULE}\n{n:>2}. {title}\n{RULE}")


def fmt_pct(x, nd: int = 1) -> str:
    return "  n/a" if x is None else f"{x * 100:.{nd}f}%"


# --------------------------------------------------------------------------- #
# Panels — three engineered launches, so every analysis has data that shows it
# off. Everything downstream is real engine output on these.
# --------------------------------------------------------------------------- #
def make_panels(spark):
    # A clear success: high trial, repeat settles on a plateau.
    success = simulate_panel(n_households=6000, weeks=44, K=0.30, a=0.18,
                             rbr_start=0.46, rbr_stable=0.25, cat_interval=2,
                             heavy_frac=0.25, heavy_mult=1.8, seed=1)
    # A failure: fine trial, but repeat collapses (the launch that "sells in").
    failure = simulate_panel(n_households=6000, weeks=32, K=0.17, a=0.30,
                             rbr_start=0.22, rbr_stable=0.06, cat_interval=2, seed=2)
    # A promoted launch: a price cut at week 18 lifts penetration off baseline
    # but recruits low-repeat deal-seekers (cohort-dependent repeat).
    def stable(wt):                    # earlier cohorts repeat better
        return 0.28 if wt <= 6 else (0.18 if wt <= 12 else 0.12)
    promoted = simulate_panel(n_households=6000, weeks=40, K=0.22, a=0.24,
                              rbr_start=0.42, rbr_stable=0.20, decay=0.6,
                              cat_interval=2, seed=11, stable_by_week=stable,
                              heavy_frac=0.2, heavy_mult=1.7,
                              promo_week=PROMO_WEEK, promo_K=0.12, promo_rbr=0.07)
    return (spark.createDataFrame(success),
            spark.createDataFrame(failure),
            spark.createDataFrame(promoted))


def main():
    spark = build_local_spark("trb-showcase")
    try:
        success_df, failure_df, promo_df = make_panels(spark)
        base = dict(launch_date="2024-01-01", period_length_days=14)  # 2-week interval

        # ---- 1. Penetration & ultimate K -------------------------------- #
        hdr(1, "PENETRATION & ULTIMATE K  (trial curve, projected ceiling)")
        res_ok = run_trb(success_df, TRBConfig(**base))
        res_bad = run_trb(failure_df, TRBConfig(**base))
        for tag, r in (("success", res_ok), ("failure", res_bad)):
            pen = r.penetration
            actual = [p for _, p in pen.series]
            fitted = [pen.fitted(t) for t, _ in pen.series]
            print(f"  {tag:8s}: snapshot trial index = {fmt_pct(r.trial_index)}"
                  f"  ultimate K = {fmt_pct(pen.ultimate_penetration)}"
                  f"  a = {pen.growth_rate:.3f}/period"
                  f"  fit p.w.s.d. = {fmt_pct(pwsd(actual, fitted), 2)}")
        print("  -> same trial, opposite verdict: penetration alone never decides")

        # ---- 2. Estimation method: discounted vs OLS -------------------- #
        hdr(2, "ESTIMATION METHOD  (Gilchrist discounted LS vs plain OLS for K)")
        for method in ("discounted", "ols"):
            r = run_trb(success_df, TRBConfig(penetration_method=method, **base))
            print(f"  {method:11s}: K = {fmt_pct(r.penetration.ultimate_penetration)}"
                  f"   a = {r.penetration.growth_rate:.3f}")
        print("  -> discounted weights recent periods (w=0.6); OLS weights all equally")

        # ---- 3. Calendar axes ------------------------------------------- #
        hdr(3, "CALENDAR AXES  (period_unit: derived vs calendar-anchored)")
        for unit in ("week", "month", "iso_week", "fiscal_445"):
            r = run_trb(success_df, TRBConfig(period_unit=unit, **base))
            labelled = label_ratio(r.share_series, r.share_period_labels)
            sample = ", ".join(f"{lab}:{fmt_pct(v)}"
                               for lab, v in labelled[:3] if v is not None)
            print(f"  {unit:11s}: {len(r.penetration.series):>2d} periods"
                  f"   first labels -> {sample}")
        print("  -> iso_week / fiscal_445 live on the real calendar grid, so an"
              "\n     out-of-stock period keeps its slot instead of collapsing")

        # ---- 4. Repeat-buying rate -------------------------------------- #
        hdr(4, "REPEAT-BUYING RATE  (pooled curve, plateau, stabilised mean)")
        ur = res_ok.ultimate_rbr()
        print(f"  furthest-available RBR      = {fmt_pct(ur[1])} (interval {ur[0]})")
        plat = res_ok.detect_plateau()
        if plat:
            print(f"  plateau diagnostic          = {fmt_pct(plat[1])} "
                  f"from interval {plat[0]}")
            print(f"  stabilised mean (from {plat[0]})   = "
                  f"{fmt_pct(res_ok.stable_rbr(from_interval=plat[0]))}")
        print("  RBR(t):  " + "  ".join(
            f"t{p.interval}={fmt_pct(p.rbr, 0)}" for p in res_ok.rbr_series[:8]
            if p.rbr is not None))

        # ---- 5. RBR interval mode --------------------------------------- #
        hdr(5, "RBR INTERVAL MODE  (exact P-day windows vs calendar buckets)")
        r_exact = run_trb(success_df, TRBConfig(rbr_interval_mode="exact", **base))
        r_bucket = run_trb(success_df, TRBConfig(
            rbr_interval_mode="bucket", rbr_bucket_unit="week",
            launch_date="2024-01-01", period_length_days=14))
        for tag, r in (("exact", r_exact), ("bucket/week", r_bucket)):
            u = r.ultimate_rbr()
            print(f"  {tag:11s}: {len(r.rbr_series):>2d} intervals"
                  f"   furthest RBR = {fmt_pct(u[1])} (t{u[0]})")

        # ---- 6. Buying index -------------------------------------------- #
        hdr(6, "BUYING INDEX  (triers=Parfitt vs repeaters=Charan; 1.0 = average)")
        for basis in ("triers", "repeaters"):
            r = run_trb(success_df, TRBConfig(buying_index_base=basis, **base))
            print(f"  base={basis:9s}: B = {r.buying_index:.3f}")
        bidx = [(p, b) for p, b in res_ok.buying_index_series if b is not None]
        print("  B(t) over time: " + "  ".join(
            f"{res_ok.label(p)}={b:.2f}" for p, b in bidx[:6]))

        # ---- 7. Entry cohorts / segmented model ------------------------- #
        hdr(7, "ENTRY COHORTS  (segmented Σ Pᵢ·Rᵢ·Bᵢ model — Table 2)")
        res_promo = run_trb(promo_df, TRBConfig(**base))
        print(res_promo.cohort_table().to_string(index=False, formatters={
            "penetration": "{:.3f}".format,
            "rbr": (lambda v: "  -  " if v is None else f"{v:.3f}"),
            "buying_index": (lambda v: "  -  " if v is None else f"{v:.3f}"),
            "contribution": "{:.4f}".format}))
        print(f"  blended RBR (penetration-weighted) = "
              f"{fmt_pct(res_promo.blended_rbr())}")

        # ---- 8. Promotion analysis -------------------------------------- #
        hdr(8, "PROMOTION ANALYSIS  (baseline vs realised vs re-fit)")
        promo = penetration_vs_actual(res_promo.penetration,
                                      cutoff_period=PROMO_WEEK, method="discounted")
        if promo.baseline_K is not None:
            print(f"  baseline K (fit before wk{PROMO_WEEK}) = {fmt_pct(promo.baseline_K)}")
            print(f"  re-fit  K (after the promo)   = {fmt_pct(promo.refit_K)}")
            print(f"  'bought' penetration          = {fmt_pct(promo.bought_penetration)} "
                  f"of the category (extra trial the deal pulled forward)")

        # ---- 9. Piecewise promo-aware curve ----------------------------- #
        hdr(9, "PIECEWISE PROMO CURVE  (one re-anchored K(1-e^-at) per promo)")
        pw = res_promo.piecewise_penetration([PROMO_WEEK])
        print(f"  composed ultimate K = {fmt_pct(pw.ultimate_penetration)}")
        if pw.note:
            print(f"  note: {pw.note}")

        # ---- 10. Validation & stability --------------------------------- #
        hdr(10, "VALIDATION & STABILITY  (out-of-sample p.w.s.d.; does K settle?)")
        val = validate_penetration(res_promo.penetration, cutoff_period=30,
                                   promo_periods=[PROMO_WEEK])
        if val.pwsd_full is not None:
            print(f"  fit to wk30, project the rest:"
                  f"  p.w.s.d. full = {fmt_pct(val.pwsd_full, 2)}"
                  f"   held-out only = {fmt_pct(val.pwsd_holdout, 2)}")
        print("  K / a as the estimation window grows (last 5 cutoffs):")
        tail = penetration_stability(res_promo.penetration).tail(5)
        print(tail.to_string(index=False, formatters={
            "K": "{:.3f}".format, "a": "{:.3f}".format,
            "observed_P": "{:.3f}".format}))

        # ---- 11. Share prediction --------------------------------------- #
        hdr(11, "SHARE PREDICTION  (the three headline numbers)")
        print(f"  simple, observed trial   Kobs·R·B = {fmt_pct(res_promo.predict_share(res_promo.ultimate_rbr()[1]), 2)}")
        print(f"  simple, projected K       K·R·B    = {fmt_pct(res_promo.predict_share_projected(), 2)}")
        print(f"  segmented   Σ Pᵢ·Rᵢ·Bᵢ  (headline) = {fmt_pct(res_promo.segmented_share(), 2)}")
        print(f"  components: K={fmt_pct(res_promo.penetration.ultimate_penetration)}"
              f"  R={fmt_pct(res_promo.ultimate_rbr()[1])}"
              f"  B={res_promo.buying_index:.3f}")

        # ---- 12. Chart gallery ------------------------------------------ #
        hdr(12, "CHART GALLERY  (every plotting function on one figure)")
        gallery(res_ok, res_promo, promo, pw)

    finally:
        spark.stop()
    print(f"\nDone. Figures written to {OUT}")


def gallery(res_ok, res_promo, promo, pw) -> None:
    """Every plots.* function on a 4x3 grid. Each panel is wrapped so a missing
    series degrades to a note instead of killing the whole figure."""
    panels = [
        ("plot_penetration",            lambda ax: plots.plot_penetration(res_ok, ax=ax)),
        ("plot_rbr (+plateau)",         lambda ax: plots.plot_rbr(res_ok, ax=ax, mark_plateau=True)),
        ("plot_buying_index_series",    lambda ax: plots.plot_buying_index_series(res_ok, ax=ax)),
        ("plot_share_over_time",        lambda ax: plots.plot_share_over_time(res_ok, ax=ax)),
        ("plot_share_bars",             lambda ax: plots.plot_share_bars(res_ok, ax=ax)),
        ("plot_predicted_share",        lambda ax: plots.plot_predicted_share(res_ok, ax=ax, base="projected")),
        ("plot_cohort_contributions",   lambda ax: plots.plot_cohort_contributions(res_promo, ax=ax)),
        ("plot_rbr_cohorts",            lambda ax: plots.plot_rbr_cohorts(res_promo, ax=ax)),
        ("plot_penetration_promo",      lambda ax: plots.plot_penetration_promo(promo, ax=ax)),
        ("plot_penetration_piecewise",  lambda ax: plots.plot_penetration_piecewise(res_promo, [PROMO_WEEK], ax=ax, pw=pw)),
        ("plot_dp_vs_p",                lambda ax: plots.plot_dp_vs_p(res_promo.penetration, ax=ax, promo_periods=[PROMO_WEEK])),
        ("plot_dashboard (composite)",  None),   # dashboard is its own figure below
    ]
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    flat = axes.ravel()
    for ax, (name, fn) in zip(flat, panels):
        if fn is None:
            ax.axis("off")
            ax.text(0.5, 0.5, "plots.plot_dashboard(res)\n-> separate figure",
                    ha="center", va="center", fontsize=10, transform=ax.transAxes)
            continue
        try:
            fn(ax)
        except Exception as e:                     # keep the gallery robust
            ax.set_title(f"{name} (n/a)", fontsize=9)
            ax.text(0.5, 0.5, str(e), ha="center", va="center", fontsize=7,
                    wrap=True, transform=ax.transAxes)
    fig.suptitle("parfitt_trb — chart gallery", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(os.path.join(OUT, "showcase_gallery.png"), dpi=100)
    plt.close(fig)
    print("  wrote figures/showcase_gallery.png")

    # The packaged 3-panel dashboard, as its own file.
    fig = plots.plot_dashboard(res_ok)
    fig.savefig(os.path.join(OUT, "showcase_dashboard.png"), dpi=110)
    plt.close(fig)
    print("  wrote figures/showcase_dashboard.png")


if __name__ == "__main__":
    main()
