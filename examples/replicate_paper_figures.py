"""Replicate the single-dataset figures of Parfitt & Collins (1968) with the
library, to demonstrate its analytical range. Run:

    uv run python -m examples.replicate_paper_figures

PNGs are written to examples/figures/. Each function notes the paper figure(s)
it reproduces. Figures requiring multi-case aggregates (8, 11, 19) or
psychographic segmentation (16) are intentionally out of scope -- see README.
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta

try:                                  # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from parfitt_trb import TRBConfig, penetration_vs_actual, run_trb
from parfitt_trb import plots
from parfitt_trb.aggregation import SparkAggregator
from parfitt_trb.cohorts import cohort_order
from parfitt_trb.local_spark import build_local_spark
from examples.synth import simulate_panel

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)
CFG = dict(launch_date="2024-01-01", period_length_days=14)   # 2-week RBR interval


def save(fig, name: str) -> None:
    fig.savefig(os.path.join(OUT, name), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


# --- Fig 1/3 + 2/4 + 5 + 18: a successful launch (Brand T / Signal) --------- #
def successful_brand(spark):
    df = simulate_panel(n_households=5000, weeks=40, K=0.34, a=0.18,
                        rbr_start=0.46, rbr_stable=0.25, cat_interval=2, seed=1)
    sdf = spark.createDataFrame(df)
    res = run_trb(sdf, TRBConfig(**CFG))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plots.plot_penetration(res, ax=ax, title="Fig 1/3 — Cumulative penetration (success)")
    save(fig, "fig01_03_penetration_success.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plots.plot_rbr(res, ax=ax, mark_plateau=True, title="Fig 2/4 — Repeat-buying rate (success)")
    save(fig, "fig02_04_rbr_success.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plots.plot_share_bars(res, ax=ax, title="Fig 5 — Realised brand share by period")
    save(fig, "fig05_share_bars.png")

    # Fig 18: fit on the first 12 weeks only, project forward (early prediction)
    early = run_trb(sdf, TRBConfig(analysis_date="2024-03-25", **CFG))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plots.plot_penetration(early, ax=ax, project_to=40,
                           title="Fig 18 — Early projection from 12 weeks of data")
    save(fig, "fig18_early_projection.png")
    return res


# --- Fig 6/7: a failure (good penetration, very low repeat) ------------------ #
def failed_brand(spark):
    df = simulate_panel(n_households=5000, weeks=32, K=0.17, a=0.30,
                        rbr_start=0.20, rbr_stable=0.06, cat_interval=2, seed=2)
    res = run_trb(spark.createDataFrame(df), TRBConfig(**CFG))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plots.plot_penetration(res, ax=ax, title="Fig 6 — Cumulative penetration (Brand Y, failure)")
    save(fig, "fig06_penetration_failure.png")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plots.plot_rbr(res, ax=ax, title="Fig 7 — Repeat-buying rate (Brand Y, failure)")
    save(fig, "fig07_rbr_failure.png")


# --- Fig 9: penetration analysed by entry cohort (Table 2) ------------------- #
def cohorts_brand_b(spark):
    def stable(wt):
        return 0.30 if wt <= 6 else (0.18 if wt <= 12 else 0.10)
    df = simulate_panel(n_households=5000, weeks=44, entry_weeks=24, K=0.30, a=0.16,
                        rbr_start=0.42, rbr_stable=0.10, decay=0.7,
                        cat_interval=2, seed=4, stable_by_week=stable)
    cfg = TRBConfig(**CFG)
    sdf = spark.createDataFrame(df)
    res = run_trb(sdf, cfg)
    agg = SparkAggregator(sdf, cfg)
    F = agg.n_category_triers
    tcw = agg.trier_counts_by_entry_week()        # small: entry_week, cohort, n
    order = cohort_order(cfg.cohort_boundaries_weeks, False)
    maxw = int(tcw["entry_week"].max())
    xs = list(range(1, maxw + 1))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bottoms = np.zeros(maxw)
    for label in order:
        counts = np.zeros(maxw)
        sub = tcw[tcw["cohort"] == label]
        for w, n in zip(sub["entry_week"], sub["n"]):
            if 1 <= w <= maxw:
                counts[int(w) - 1] += int(n)
        cumc = np.cumsum(counts) / F * 100
        ax.fill_between(xs, bottoms, bottoms + cumc, alpha=0.75, label=label)
        bottoms = bottoms + cumc
    K = res.penetration.ultimate_penetration
    if K:
        ax.axhline(K * 100, ls=":", color="grey")
        ax.annotate(f"ultimate {K*100:.1f}%", xy=(maxw, K * 100), ha="right",
                    va="bottom", fontsize=8, color="grey")
    ax.set(xlabel="Weeks after launch", ylabel="Penetration %",
           title="Fig 9 — Cumulative penetration by entry cohort")
    ax.legend(fontsize=8, title="entry cohort")
    save(fig, "fig09_cohort_penetration.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plots.plot_cohort_contributions(res, ax=ax,
                                    title="Table 2 — cohort contributions to share")
    save(fig, "fig09b_cohort_contributions.png")

    print("\n  Table 2 (segmented model) reconstruction:")
    print(res.cohort_table().to_string(index=False,
          formatters={"penetration": "{:.3f}".format, "rbr": (lambda v: "" if v is None else f"{v:.3f}"),
                      "buying_index": (lambda v: "" if v is None else f"{v:.3f}"),
                      "contribution": "{:.4f}".format}))
    print(f"  segmented share = {res.segmented_share():.4f} "
          f"(blended RBR = {res.blended_rbr():.3f})\n")


# --- Fig 10: seasonality inflates apparent penetration ---------------------- #
def seasonal_brand_j(spark):
    rng = np.random.default_rng(3)
    H, weeks = 5000, 32
    origin = date(2024, 1, 1)
    dstr = lambda w: (origin + timedelta(days=(w - 1) * 7 + 2)).isoformat()
    probs = np.ones(weeks)
    probs[12:20] = 5.0                                   # high season weeks 13-20
    probs /= probs.sum()
    entry = rng.choice(np.arange(1, weeks + 1), size=H, p=probs)
    is_trier = rng.random(H) < 0.18
    rows = []
    for h in range(H):
        ew = int(entry[h])
        if is_trier[h]:
            rows.append((f"h{h}", dstr(ew), True, True, 1.0))
        for w in range(ew, weeks + 1, 4):
            b = bool(is_trier[h] and rng.random() < 0.45)
            rows.append((f"h{h}", dstr(w), b, True, 1.0))
    df = pd.DataFrame(rows, columns=["shopper_id", "txn_date", "is_new_product",
                                     "is_category", "volume"])
    cfg = TRBConfig(launch_date="2024-01-01", period_length_days=14)
    sdf = spark.createDataFrame(df)
    res = run_trb(sdf, cfg, project_penetration=False)
    agg = SparkAggregator(sdf, cfg)
    ent = agg.entrants().sort_values("period")
    cb = ent["n_brand_new"].cumsum().to_numpy() / H * 100        # brand % households
    cf = ent["n_cat_new"].cumsum().to_numpy() / H * 100          # field % households
    per = ent["period"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(per, cf, "-o", ms=3, color="tab:blue", label="Total field (% households)")
    ax.plot(per, cb, "-o", ms=3, color="tab:green", label="Brand J (% households)")
    bp = [(p, v * 100) for p, v in res.penetration.series]
    ax.plot([p for p, _ in bp], [v for _, v in bp], "--", color="tab:red",
            label="Brand J (% category buyers = true penetration)")
    ax.axvspan(13, 20, color="orange", alpha=0.12)
    ax.set(xlabel="Weeks after launch", ylabel="Penetration %",
           title="Fig 10 — Seasonality lifts apparent (household) penetration")
    ax.legend(fontsize=8)
    save(fig, "fig10_seasonality.png")


# --- Fig 12-15: a promotion lifts penetration above the baseline projection -- #
def promotion_effect(spark):
    df = simulate_panel(n_households=5000, weeks=36, K=0.22, a=0.30,
                        rbr_start=0.40, rbr_stable=0.22, cat_interval=2, seed=7,
                        promo_week=18, promo_K=0.14, promo_rbr=0.06)
    cfg = TRBConfig(**CFG)
    res = run_trb(spark.createDataFrame(df), cfg)
    promo = penetration_vs_actual(res.penetration, cutoff_period=18, method="discounted")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plots.plot_penetration_promo(promo, ax=ax,
                                 title="Fig 12-15 — Promotion vs baseline projection")
    save(fig, "fig12_15_promotion.png")
    print(f"  promotion: baseline K={promo.baseline_K:.3f} -> re-fit K={promo.refit_K:.3f}, "
          f"bought penetration ~ {promo.bought_penetration*100:+.1f}pp")


# --- Fig 17: the P × B × R decomposition ------------------------------------ #
def concept_waterfall():
    P, B, R = 0.34, 1.20, 0.25
    ceiling = P * B                                     # effective volume ceiling
    share = ceiling * R
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(["Penetration\n(P)", "× Buying index\n(ceiling P·B)", "× Repeat\n(share)"],
           [P * 100, ceiling * 100, share * 100],
           color=["tab:blue", "tab:cyan", "tab:green"], alpha=0.85)
    for i, v in enumerate([P * 100, ceiling * 100, share * 100]):
        ax.annotate(f"{v:.1f}%", xy=(i, v), ha="center", va="bottom", fontsize=9)
    ax.set(ylabel="%", title="Fig 17 — Concept: share = P × B × R (= 34% × 1.20 × 25% = 10.2%)")
    save(fig, "fig17_concept_waterfall.png")


def main():
    print("Replicating Parfitt-Collins single-dataset figures -> examples/figures/")
    spark = build_local_spark("trb-figures")
    try:
        successful_brand(spark)
        failed_brand(spark)
        cohorts_brand_b(spark)
        seasonal_brand_j(spark)
        promotion_effect(spark)
        concept_waterfall()                 # pure plot, no data engine needed
    finally:
        spark.stop()
    print("Done. (Figures 8, 11, 16, 19 are out of scope -- see README.)")


if __name__ == "__main__":
    main()
