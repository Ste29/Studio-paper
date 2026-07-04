"""Paper-style charts. matplotlib is imported lazily so the model core has no
hard dependency on it. Every function takes an optional Axes and returns it, so
callers can compose / restyle / save. Solid = observed/raw, dashed = projected.
"""
from __future__ import annotations

from typing import Optional, Sequence

from .core import (
    Penetration, PenetrationPromo, PiecewisePenetration, centred_differences,
    fit_penetration, fit_piecewise_penetration, smoothed_series,
)
from .model import TRBResult

# Fixed per-segment styles for promo-segmented charts (assigned in order, never
# cycled): launch = blue circles, promo 1 = red triangles, ... The marker is a
# second identity encoding on top of hue (CVD/print safe).
_SEGMENT_COLORS = ("tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple")
_SEGMENT_MARKERS = ("o", "^", "s", "D", "v")


def _mpl():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("plotting requires matplotlib (uv add matplotlib)") from e


def _ax(ax, figsize=(7, 4.5)):
    if ax is None:
        _, ax = _mpl().subplots(figsize=figsize)
    return ax


def _calendar_xlabel(result: TRBResult, share: bool = False) -> str:
    """X-axis label for a calendar-axis chart, matching the period unit."""
    unit = getattr(result, "share_period_unit" if share else "period_unit", "week")
    return {"week": "Weeks after launch", "fortnight": "Fortnights after launch",
            "month": "Months after launch", "iso_week": "ISO calendar week",
            "iso_fortnight": "ISO fortnight", "fiscal_445": "Retail 4-4-5 period",
            }.get(unit, "Periods after launch")


# Projected-tail horizon (max periods past the last observation) and minimum tail
# length per calendar unit -- ~50 weeks / 12 months whatever the bucket size.
_PROJECTION_SPAN = {"week": (50, 6), "iso_week": (50, 6),
                    "fortnight": (25, 3), "iso_fortnight": (25, 3),
                    "month": (12, 3), "fiscal_445": (12, 3)}


def _calendar_ticks(ax, result: TRBResult, periods: Sequence[int],
                    share: bool = False, max_labels: int = 13) -> None:
    """For non-weekly axes (fortnight / month / iso_week / iso_fortnight /
    fiscal_445), label the ticks with calendar labels instead of bare ordinals,
    thinned to at most `max_labels`. Weekly axes keep numeric ticks."""
    unit = getattr(result, "share_period_unit" if share else "period_unit", "week")
    if unit == "week" or not periods:
        return
    labeller = result.label_share if share else result.label
    ps = sorted({int(p) for p in periods})
    step = max(1, -(-len(ps) // max_labels))          # ceil division
    ticks = ps[::step]
    ax.set_xticks(ticks)
    ax.set_xticklabels([labeller(p) for p in ticks], rotation=45, ha="right",
                       fontsize=7)


# --------------------------------------------------------------------------- #
def plot_penetration(result: TRBResult, ax=None, project_to: Optional[int] = None,
                     as_percent: bool = True, title: str = "Cumulative penetration"):
    """Figure 1 / 3 / 6 style: observed penetration (solid) + the projected
    K(1-e^{-a t}) tail (dashed) and a dotted line at the estimated ultimate K."""
    ax = _ax(ax)
    pen = result.penetration
    if pen is None or not pen.series:
        raise ValueError("no penetration series to plot")
    scale = 100.0 if as_percent else 1.0
    unit = "%" if as_percent else ""
    ts = [t for t, _ in pen.series]
    ax.plot(ts, [p * scale for _, p in pen.series], "-o", color="tab:blue",
            ms=3, label="Observed (raw data)")
    K, a = pen.ultimate_penetration, pen.growth_rate
    tp: list = []
    if K is not None and a is not None:
        last = ts[-1]
        if project_to is None:
            # projected tail capped at ~50 weeks' worth of periods past the last
            # observation whatever the unit; stop early near the ceiling.
            horizon, floor_ = _PROJECTION_SPAN.get(
                getattr(result, "period_unit", "week"), (12, 3))
            t = last
            while (t < last + horizon and pen.fitted(t) is not None
                   and pen.fitted(t) < 0.99 * K):
                t += 1
            project_to = max(t, last + floor_)
        tp = list(range(last, project_to + 1))
        ax.plot(tp, [pen.fitted(t) * scale for t in tp], "--", color="tab:blue",
                label="Projection (expected)")
        ax.axhline(K * scale, ls=":", color="grey", lw=1)
        ax.annotate(f"Ultimate {K * scale:.1f}{unit}", xy=(project_to, K * scale),
                    ha="right", va="bottom", fontsize=8, color="grey")
    ax.set(xlabel=_calendar_xlabel(result), ylabel=f"Penetration {unit}".strip(), title=title)
    _calendar_ticks(ax, result, list(ts) + tp)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    return ax


def plot_penetration_promo(promo: PenetrationPromo, ax=None, as_percent: bool = True,
                           title: str = "Penetration: actual vs projected"):
    """Figures 12-15 style: observed penetration, the baseline projection fitted
    BEFORE the disturbance (dashed), and the re-fitted ultimate AFTER it (dotted).
    The gap between observed and baseline at the end is the 'bought' penetration."""
    ax = _ax(ax)
    scale = 100.0 if as_percent else 1.0
    unit = "%" if as_percent else ""
    ax.plot([t for t, _ in promo.series], [p * scale for _, p in promo.series],
            "-o", color="tab:blue", ms=3, label="Observed")
    if promo.baseline_curve:
        ax.plot([t for t, _ in promo.baseline_curve],
                [p * scale for _, p in promo.baseline_curve], "--", color="grey",
                label=f"Baseline projection (K={promo.baseline_K * scale:.1f}{unit})")
    if promo.refit_curve and promo.refit_K is not None:
        ax.axhline(promo.refit_K * scale, ls=":", color="tab:red", lw=1)
        ax.annotate(f"Re-fit ultimate {promo.refit_K * scale:.1f}{unit}",
                    xy=(promo.series[-1][0], promo.refit_K * scale), ha="right",
                    va="bottom", fontsize=8, color="tab:red")
    ax.axvline(promo.cutoff_period, ls="-.", color="black", lw=0.8, alpha=0.6)
    if promo.bought_penetration is not None:
        ax.annotate(f"bought ≈ {promo.bought_penetration * scale:+.1f}{unit}",
                    xy=(promo.series[-1][0], promo.series[-1][1] * scale),
                    fontsize=8, color="tab:green")
    ax.set(xlabel="Weeks after launch", ylabel=f"Penetration {unit}".strip(), title=title)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    return ax


def plot_rbr(result: TRBResult, ax=None, mark_plateau: bool = False,
             as_percent: bool = True, title: str = "Repeat-buying rate"):
    """Figure 2 / 4 / 7 style: pooled RBR(t). Optionally mark the diagnostic
    plateau (NOT used by the share formula)."""
    ax = _ax(ax)
    scale = 100.0 if as_percent else 1.0
    unit = "%" if as_percent else ""
    pts = sorted((p.interval, p.rbr) for p in result.rbr_series if p.rbr is not None)
    if not pts:
        raise ValueError("no RBR points to plot")
    ax.plot([t for t, _ in pts], [v * scale for _, v in pts], "-o",
            color="tab:green", ms=3, label="RBR(t)")
    if mark_plateau:
        plat = result.detect_plateau()
        if plat:
            ax.axhline(plat[1] * scale, ls=":", color="grey", lw=1)
            ax.annotate(f"plateau {plat[1] * scale:.1f}{unit}",
                        xy=(pts[-1][0], plat[1] * scale), ha="right", va="bottom",
                        fontsize=8, color="grey")
    ax.set(xlabel="Interval after first purchase (t)",
           ylabel=f"RBR {unit}".strip(), title=title)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    return ax


def plot_penetration_piecewise(result: TRBResult, promo_periods: Sequence[int],
                               ax=None, pw: Optional[PiecewisePenetration] = None,
                               baseline_extension: Optional[int] = None,
                               as_percent: bool = True,
                               title: str = "Penetration: piecewise promo model"):
    """Observed penetration + the composed promo-aware theoretical curve.

    Each promo gets a vertical dash-dot marker; the PRE-promo segment continues
    (grey dashed) for `baseline_extension` periods past the promo (default 3 on
    weekly axes, 2 on fortnightly, 1 on monthly-scale) and the observed-vs-baseline
    gap over that window is filled green with the 'bought' penetration annotated."""
    ax = _ax(ax)
    pen = result.penetration
    if pen is None or not pen.series:
        raise ValueError("no penetration series to plot")
    cfg = result.config
    if pw is None:
        pw = fit_piecewise_penetration(
            pen, promo_periods,
            method=cfg.penetration_method if cfg else "discounted",
            discount_weight=cfg.discount_weight if cfg else 0.6)
    scale = 100.0 if as_percent else 1.0
    unit = "%" if as_percent else ""
    if baseline_extension is None:
        # ~3 weeks of baseline continuation past the promo whatever the unit.
        baseline_extension = {"week": 3, "iso_week": 3,
                              "fortnight": 2, "iso_fortnight": 2,
                              }.get(getattr(result, "period_unit", "week"), 1)

    ts = [t for t, _ in pen.series]
    obs = dict(pen.series)
    last = ts[-1]
    ax.plot(ts, [p * scale for _, p in pen.series], "-o", color="tab:blue",
            ms=3, label="Observed")
    fit_pts = [(t, pw.fitted(t)) for t in range(ts[0], last + 1)]
    fit_pts = [(t, v) for t, v in fit_pts if v is not None]
    if fit_pts:
        ax.plot([t for t, _ in fit_pts], [v * scale for _, v in fit_pts], "--",
                color="tab:red", label="Piecewise fit")

    for i, t_i in enumerate(pw.promo_periods):
        ax.axvline(t_i, ls="-.", color="black", lw=0.8, alpha=0.6)
        prev = pw.segments[i]                       # segment before this promo
        t_end = min(t_i + baseline_extension, last)
        if prev.fitted(t_i) is None or t_end <= t_i:
            continue                                # unfitted baseline: no boost fill
        span = list(range(t_i, t_end + 1))
        base_vals = [prev.fitted(t) for t in span]
        ax.plot(span, [v * scale for v in base_vals], "--", color="grey", lw=1,
                label="Pre-promo baseline" if i == 0 else "_nolegend_")
        pairs = [(t, obs[t], b) for t, b in zip(span, base_vals) if t in obs]
        if len(pairs) >= 2:
            ft = [t for t, _, _ in pairs]
            fo = [o for _, o, _ in pairs]
            fb = [b for _, _, b in pairs]
            ax.fill_between(ft, [b * scale for b in fb], [o * scale for o in fo],
                            where=[o >= b for o, b in zip(fo, fb)],
                            color="tab:green", alpha=0.3, interpolate=True)
            ax.annotate(f"bought ≈ {(fo[-1] - fb[-1]) * scale:+.1f}{unit}",
                        xy=(ft[-1], fo[-1] * scale), fontsize=8, color="tab:green")

    ax.set(xlabel=_calendar_xlabel(result), ylabel=f"Penetration {unit}".strip(),
           title=title)
    _calendar_ticks(ax, result, ts)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    return ax


def plot_dp_vs_p(pen: Penetration, ax=None,
                 promo_periods: Optional[Sequence[int]] = None,
                 as_percent: bool = True, method: str = "discounted",
                 discount_weight: float = 0.6,
                 smoothing_window: Optional[int] = None,
                 title: str = "ΔP vs P (difference model)"):
    """Scatter of the difference model the fit regresses: x=P(t), y=ΔP(t) with
    ΔP the centred difference. The fitted line ΔP = a(K - P) is overlaid. With
    `promo_periods`, each segment gets its own marker/colour and its own dashed
    line (in the original coordinates a promo segment's line is still straight:
    ΔP = a'(base + K' - P)). With `smoothing_window`, the solid points are the
    SMOOTHED ones the fit actually regresses, with the raw points faded behind."""
    ax = _ax(ax)
    scale = 100.0 if as_percent else 1.0
    unit = "%" if as_percent else ""
    series = list(pen.series)
    if len(series) < 3:
        raise ValueError("need >=3 observed periods for centred differences")
    raw_in_legend = [False]                    # one 'raw' legend entry overall

    def _scatter(sub, color, marker, label):
        """Scatter the points the fit sees (smoothed when requested, raw faded
        behind); returns the P values used for the fitted-line span."""
        if smoothing_window is not None:
            _, P_raw, dP_raw = centred_differences(sub)
            ax.scatter([p * scale for p in P_raw], [d * scale for d in dP_raw],
                       marker=marker, color=color, s=16, alpha=0.25,
                       label="_nolegend_" if raw_in_legend[0] else "raw (unsmoothed)")
            raw_in_legend[0] = True
            sub = smoothed_series(sub, smoothing_window)
        _, P, dP = centred_differences(sub)
        ax.scatter([p * scale for p in P], [d * scale for d in dP],
                   marker=marker, color=color, s=22, label=label)
        return P

    if promo_periods:
        pw = fit_piecewise_penetration(pen, promo_periods, method=method,
                                       discount_weight=discount_weight,
                                       smoothing_window=smoothing_window)
        promos = pw.promo_periods
        bounds = [series[0][0]] + promos + [series[-1][0]]
        for i, seg in enumerate(pw.segments):
            lo, hi = bounds[i], bounds[i + 1]
            sub = [(t, p) for t, p in series if lo <= t <= hi]
            color = _SEGMENT_COLORS[i % len(_SEGMENT_COLORS)]
            marker = _SEGMENT_MARKERS[i % len(_SEGMENT_MARKERS)]
            label = "launch" if i == 0 else f"promo @{seg.t0}"
            if len(sub) >= 3:
                P = _scatter(sub, color, marker, label)
                if seg.a is not None and seg.ceiling is not None:
                    lo_p, hi_p = float(min(P)), float(max(P))
                    ax.plot([lo_p * scale, hi_p * scale],
                            [seg.a * (seg.ceiling - lo_p) * scale,
                             seg.a * (seg.ceiling - hi_p) * scale],
                            "--", color=color, lw=1)
            else:
                ax.scatter([], [], marker=marker, color=color, s=22,
                           label=f"{label} (too few points)")
    else:
        P = _scatter(series, "tab:blue", "o", "observed")
        fitted = pen
        if fitted.ultimate_penetration is None or fitted.growth_rate is None:
            fitted = Penetration(pen.denominator, pen.origin, series,
                                 pen.n_brand_triers, pen.n_category_triers)
            fit_penetration(fitted, method=method, discount_weight=discount_weight,
                            smoothing_window=smoothing_window)
        K, a = fitted.ultimate_penetration, fitted.growth_rate
        if K is not None and a is not None:
            lo_p, hi_p = float(min(P)), float(max(P))
            ax.plot([lo_p * scale, hi_p * scale],
                    [a * (K - lo_p) * scale, a * (K - hi_p) * scale],
                    "--", color="tab:blue", lw=1,
                    label=f"fit: ΔP = {a:.3f}(K - P)")

    ax.axhline(0.0, color="grey", lw=0.5)
    ax.set(xlabel=f"Penetration P {unit}".strip(),
           ylabel=f"ΔP per period {unit}".strip(), title=title)
    ax.legend(fontsize=8)
    return ax


def plot_rbr_cohorts(result: TRBResult, ax=None, as_percent: bool = True,
                     show_pooled: bool = True,
                     title: str = "Repeat-buying rate by entry cohort"):
    """Diagnostic RBR(t) per entry cohort (one line each, cohort order fixed),
    with the pooled curve as a thick grey reference. Diverging cohort curves
    flag entry-wave differences and inform the choice of cohort boundaries."""
    ax = _ax(ax)
    scale = 100.0 if as_percent else 1.0
    unit = "%" if as_percent else ""
    series = result.rbr_cohort_series
    if not series:
        raise ValueError("no per-cohort RBR series to plot")
    if show_pooled:
        pooled = sorted((p.interval, p.rbr) for p in result.rbr_series
                        if p.rbr is not None)
        if pooled:
            ax.plot([t for t, _ in pooled], [v * scale for _, v in pooled], "-",
                    color="grey", lw=2.5, alpha=0.5, label="pooled")
    for i, (label, pts) in enumerate(series.items()):
        pts = [(t, v) for t, v in pts if v is not None]
        if not pts:
            continue
        color = _SEGMENT_COLORS[i % len(_SEGMENT_COLORS)]
        marker = _SEGMENT_MARKERS[i % len(_SEGMENT_MARKERS)]
        ax.plot([t for t, _ in pts], [v * scale for _, v in pts],
                linestyle="-", marker=marker, color=color, ms=3, label=label)
    ax.set(xlabel="Interval after first purchase (t)",
           ylabel=f"RBR {unit}".strip(), title=title)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    return ax


def plot_share_bars(result: TRBResult, ax=None, as_percent: bool = True,
                    title: str = "Realised brand share by period"):
    """Figure 5 style: realised calendar-period share as bars."""
    ax = _ax(ax)
    scale = 100.0 if as_percent else 1.0
    pts = [(p, r) for p, r in result.share_ratio_series() if r is not None]
    if not pts:
        raise ValueError("no share series to plot")
    ax.bar([t for t, _ in pts], [r * scale for _, r in pts], color="tab:blue", alpha=0.8)
    ax.set(xlabel=_calendar_xlabel(result, share=True), ylabel="Share %" if as_percent else "Share",
           title=title)
    _calendar_ticks(ax, result, [t for t, _ in pts], share=True)
    ax.set_ylim(bottom=0)
    return ax


def plot_share_over_time(result: TRBResult, ax=None, as_percent: bool = True,
                         show_equilibrium: bool = True,
                         title: str = "Realised brand share over time"):
    """Line of realised share over calendar weeks, with the Parfitt equilibrium
    (projected K × ultimate RBR × B) as a dotted reference if estimable."""
    ax = _ax(ax)
    scale = 100.0 if as_percent else 1.0
    unit = "%" if as_percent else ""
    pts = [(p, r) for p, r in result.share_ratio_series() if r is not None]
    if not pts:
        raise ValueError("no share series to plot")
    ax.plot([t for t, _ in pts], [r * scale for _, r in pts], "-o",
            color="tab:orange", ms=3, label="Realised share")
    if show_equilibrium:
        eq = result.predict_share_projected()
        if eq is not None:
            ax.axhline(eq * scale, ls=":", color="grey", lw=1)
            ax.annotate(f"Parfitt equilibrium {eq * scale:.1f}{unit}",
                        xy=(pts[-1][0], eq * scale), ha="right", va="bottom",
                        fontsize=8, color="grey")
    ax.set(xlabel=_calendar_xlabel(result, share=True),
           ylabel=f"Share {unit}".strip(), title=title)
    _calendar_ticks(ax, result, [t for t, _ in pts], share=True)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    return ax


def plot_predicted_share(result: TRBResult, ax=None, base: str = "observed",
                         as_percent: bool = True,
                         title: str = "Predicted share by RBR maturity"):
    """Predicted share = Trial × RBR(t) × Buying, plotted across RBR interval t,
    showing the prediction stabilising as RBR matures. base: 'observed' (trial
    snapshot) or 'projected' (ultimate K)."""
    ax = _ax(ax)
    scale = 100.0 if as_percent else 1.0
    if base == "projected":
        trial = result.penetration.ultimate_penetration if result.penetration else None
        if trial is None:
            raise ValueError("base='projected' needs an estimable ultimate penetration")
    else:
        trial = result.trial_index
    pts = sorted((p.interval, p.rbr) for p in result.rbr_series if p.rbr is not None)
    if not pts:
        raise ValueError("no RBR points to plot")
    ax.plot([t for t, _ in pts], [trial * v * result.buying_index * scale for _, v in pts],
            "-o", color="tab:purple", ms=3, label=f"Predicted share ({base})")
    ax.set(xlabel="RBR interval used (t)",
           ylabel="Predicted share %" if as_percent else "Predicted share", title=title)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    return ax


def plot_cohort_contributions(result: TRBResult, ax=None, as_percent: bool = True,
                              title: str = "Cohort contributions to share (Table 2)"):
    """Stacked picture of each entry cohort's Pᵢ×Rᵢ×Bᵢ contribution."""
    ax = _ax(ax)
    scale = 100.0 if as_percent else 1.0
    cohorts = [c for c in result.cohorts]
    if not cohorts:
        raise ValueError("no cohorts to plot")
    labels = [c.label for c in cohorts]
    contribs = [c.contribution * scale for c in cohorts]
    colors = ["tab:gray" if c.is_future else "tab:blue" for c in cohorts]
    ax.bar(labels, contribs, color=colors, alpha=0.85)
    ax.set(ylabel="Share contribution %" if as_percent else "contribution", title=title)
    ax.tick_params(axis="x", rotation=30)
    return ax


def plot_buying_index_series(result: TRBResult, ax=None,
                             title: str = "Buying index over time"):
    """Per-period buying index B(t) (point 1)."""
    ax = _ax(ax)
    pts = [(p, b) for p, b in result.buying_index_series if b is not None]
    if not pts:
        raise ValueError("no buying-index series to plot")
    ax.plot([t for t, _ in pts], [b for _, b in pts], "-o", color="tab:brown", ms=3)
    ax.axhline(1.0, ls=":", color="grey", lw=1)
    ax.set(xlabel=_calendar_xlabel(result), ylabel="Buying index (1.0 = average)", title=title)
    _calendar_ticks(ax, result, [t for t, _ in pts])
    return ax


def plot_lines(series_map: dict, ax=None, xlabel="Weeks after launch",
               ylabel="Penetration %", title="", as_percent: bool = True):
    """Generic multi-line helper (Figures 9/10/18): {label: [(x, y), ...]}.
    Used by the example script for cohort / seasonal / multi-field charts."""
    ax = _ax(ax)
    scale = 100.0 if as_percent else 1.0
    for label, pts in series_map.items():
        style = "--" if "proj" in label.lower() or "expected" in label.lower() else "-o"
        ax.plot([x for x, _ in pts], [y * scale for _, y in pts], style, ms=3, label=label)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    return ax


def plot_dashboard(result: TRBResult, as_percent: bool = True):
    """penetration | RBR | predicted share, side by side. Returns the Figure."""
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    try:
        plot_penetration(result, ax=axes[0], as_percent=as_percent)
    except ValueError as e:
        axes[0].set_title("penetration n/a")
        axes[0].text(0.5, 0.5, str(e), ha="center", va="center", fontsize=8,
                     wrap=True, transform=axes[0].transAxes)
    plot_rbr(result, ax=axes[1], as_percent=as_percent)
    plot_predicted_share(result, ax=axes[2], as_percent=as_percent)
    fig.tight_layout()
    return fig
