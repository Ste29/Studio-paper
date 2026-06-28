"""Paper-style charts. matplotlib is imported lazily so the model core has no
hard dependency on it. Every function takes an optional Axes and returns it, so
callers can compose / restyle / save. Solid = observed/raw, dashed = projected.
"""
from __future__ import annotations

from typing import Optional, Sequence

from .core import PenetrationPromo
from .model import TRBResult


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
    return {"week": "Weeks after launch", "month": "Months after launch",
            "bucket": "Calendar bucket"}.get(unit, "Periods after launch")


def _calendar_ticks(ax, result: TRBResult, periods: Sequence[int],
                    share: bool = False) -> None:
    """For month / bucket axes, label the ticks with calendar labels instead of
    bare ordinals. Weekly axes keep numeric ticks (there can be many)."""
    unit = getattr(result, "share_period_unit" if share else "period_unit", "week")
    if unit == "week" or not periods:
        return
    labeller = result.label_share if share else result.label
    ax.set_xticks(list(periods))
    ax.set_xticklabels([labeller(p) for p in periods], rotation=45, ha="right",
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
    if K is not None and a is not None:
        last = ts[-1]
        if project_to is None:
            t = last
            while t < last + 200 and pen.fitted(t) is not None and pen.fitted(t) < 0.99 * K:
                t += 1
            project_to = max(t, last + 6)
        tp = list(range(last, project_to + 1))
        ax.plot(tp, [pen.fitted(t) * scale for t in tp], "--", color="tab:blue",
                label="Projection (expected)")
        ax.axhline(K * scale, ls=":", color="grey", lw=1)
        ax.annotate(f"Ultimate {K * scale:.1f}{unit}", xy=(project_to, K * scale),
                    ha="right", va="bottom", fontsize=8, color="grey")
    ax.set(xlabel=_calendar_xlabel(result), ylabel=f"Penetration {unit}".strip(), title=title)
    _calendar_ticks(ax, result, ts)
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
