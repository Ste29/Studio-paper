"""Penetration chart: observed curve, theoretical overlay (single or piecewise
promo-aware) and the promo boost highlighted against the extended pre-promo
baseline. matplotlib is imported lazily; solid = observed, dashed = theoretical.
"""
from __future__ import annotations

from typing import Optional, Sequence

from .penetration import PenetrationCurve, PiecewiseCurve, fit_piecewise

AXIS_LABELS = {"iso_week": "ISO week", "iso_fortnight": "ISO fortnight",
               "month": "Month"}
PROJECTION_CAP = {"iso_week": 50, "iso_fortnight": 25, "month": 12}
# Pre-promo baseline extension: ~3 weeks of visibility after each promo
# (3 weekly buckets / 2 fortnights ≈ 1 month / 1 month).
BOOST_EXTENSION = {"iso_week": 3, "iso_fortnight": 2, "month": 1}


def _mpl():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("plotting requires matplotlib") from e


def _calendar_ticks(ax, curve, periods: Sequence[int], max_labels: int = 13) -> None:
    """Calendar labels on the x ticks, thinned to at most ~max_labels."""
    ps = sorted({int(p) for p in periods})
    if not ps:
        return
    step = max(1, -(-len(ps) // max_labels))          # ceil division
    ticks = ps[::step]
    ax.set_xticks(ticks)
    ax.set_xticklabels([curve.label(p) for p in ticks], rotation=45, ha="right",
                       fontsize=7)


def plot_penetration(curve: PenetrationCurve, *,
                     piecewise: Optional[PiecewiseCurve] = None,
                     promo_periods: Optional[Sequence[int | str]] = None,
                     baseline_extension: Optional[int] = None,
                     project_to: Optional[int] = None,
                     ax=None, as_percent: bool = True,
                     title: str = "Cumulative penetration"):
    """Observed penetration (solid) with the theoretical curve overlaid (dashed)
    and, per promo, the pre-promo baseline extended a few buckets so the bought
    boost shows as a green filled area with its delta annotated.

    Pass a fitted `curve` (single model) or a `piecewise` composition; with
    `promo_periods` and no `piecewise`, the piecewise fit is run with defaults.
    """
    if ax is None:
        _, ax = _mpl().subplots(figsize=(8, 4.5))
    if not curve.series:
        raise ValueError("no penetration series to plot")
    if piecewise is None and promo_periods:
        piecewise = fit_piecewise(curve, promo_periods)

    scale = 100.0 if as_percent else 1.0
    unit_str = "%" if as_percent else ""
    ts = [t for t, _ in curve.series]
    obs = dict(curve.series)
    last = ts[-1]
    ax.plot(ts, [p * scale for _, p in curve.series], "-o", color="tab:blue",
            ms=3, label="Observed")

    theoretical = piecewise if piecewise is not None else curve
    K_ult = (piecewise.ultimate_penetration if piecewise is not None else curve.K)
    if theoretical.fitted(last) is not None:
        if project_to is None:
            cap = last + PROJECTION_CAP[curve.unit]
            t = last
            while t < cap:
                v = theoretical.fitted(t)
                if v is None or (K_ult is not None and v >= 0.99 * K_ult):
                    break
                t += 1
            project_to = max(t, last + 3)
        tp = [t for t in range(1, project_to + 1)]
        ax.plot(tp, [theoretical.fitted(t) * scale for t in tp], "--",
                color="tab:red", lw=1.4,
                label="Theoretical fit" + (" (piecewise)" if piecewise else ""))
        if K_ult is not None:
            ax.axhline(K_ult * scale, ls=":", color="grey", lw=1)
            ax.annotate(f"Ultimate {K_ult * scale:.1f}{unit_str}",
                        xy=(project_to, K_ult * scale), ha="right", va="bottom",
                        fontsize=8, color="grey")
    else:
        project_to = last

    # Promo boost: previous segment's baseline extended a few buckets, the gap
    # to the observed curve filled green with the bought delta annotated.
    if piecewise is not None and piecewise.promo_periods:
        ext = (BOOST_EXTENSION[curve.unit]
               if baseline_extension is None else int(baseline_extension))
        for t_i in piecewise.promo_periods:
            ax.axvline(t_i, ls="-.", color="black", lw=0.8, alpha=0.6)
            idx = piecewise.segment_index_for(t_i)
            prev = piecewise.segments[idx - 1] if idx > 0 else None
            if prev is None or prev.fitted(t_i) is None:
                continue
            t_end = min(t_i + ext, last)
            span = [t for t in ts if t_i <= t <= t_end]
            if len(span) < 2:
                continue
            base_v = [prev.fitted(t) for t in span]
            obs_v = [obs[t] for t in span]
            ax.plot(span, [v * scale for v in base_v], "--", color="grey", lw=1.2,
                    label="Pre-promo baseline" if t_i == piecewise.promo_periods[0]
                    else None)
            ax.fill_between(span, [v * scale for v in base_v],
                            [v * scale for v in obs_v],
                            where=[o >= b for o, b in zip(obs_v, base_v)],
                            color="tab:green", alpha=0.3, interpolate=True)
            bought = obs_v[-1] - base_v[-1]
            ax.annotate(f"bought ≈ {bought * scale:+.1f}{unit_str}",
                        xy=(t_end, obs_v[-1] * scale), fontsize=8,
                        color="tab:green", ha="left", va="bottom")

    ax.set(xlabel=AXIS_LABELS[curve.unit],
           ylabel=f"Penetration {unit_str}".strip(), title=title)
    _calendar_ticks(ax, curve, list(range(1, project_to + 1)))
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    return ax
