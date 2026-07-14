"""RBR charts: the pooled repeat-buying-rate curve (with the optional
diagnostic plateau marked) and the per-cohort curves overlaid on the pooled
reference. matplotlib is imported lazily.
"""
from __future__ import annotations

from .rbr import RBRCurve

# Fixed per-cohort styles (assigned in label order, cycled when cohorts exceed
# the palette). The marker is a second identity encoding on top of hue
# (CVD/print safe).
_COHORT_COLORS = ("tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple")
_COHORT_MARKERS = ("o", "^", "s", "D", "v")


def _mpl():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("plotting requires matplotlib") from e


def _ax(ax):
    if ax is None:
        _, ax = _mpl().subplots(figsize=(7, 4.5))
    return ax


def _xlabel(curve: RBRCurve) -> str:
    if curve.interval_unit is not None:
        return f"Interval after first purchase (t, {curve.interval_unit} buckets)"
    return f"Interval after first purchase (t, {curve.period_length_days}-day)"


def plot_rbr(curve: RBRCurve, *, ax=None, mark_plateau: bool = False,
             as_percent: bool = True, title: str = "Repeat-buying rate"):
    """Figure 2 / 4 / 7 style: pooled RBR(t). Optionally mark the diagnostic
    plateau (an aid for choosing `stable(from_interval)`, nothing more)."""
    ax = _ax(ax)
    scale = 100.0 if as_percent else 1.0
    unit = "%" if as_percent else ""
    pts = sorted((p.interval, p.rbr) for p in curve.points if p.rbr is not None)
    if not pts:
        raise ValueError("no RBR points to plot")
    ax.plot([t for t, _ in pts], [v * scale for _, v in pts], "-o",
            color="tab:green", ms=3, label="RBR(t)")
    if mark_plateau:
        plat = curve.plateau()
        if plat:
            ax.axhline(plat[1] * scale, ls=":", color="grey", lw=1)
            ax.annotate(f"plateau {plat[1] * scale:.1f}{unit}",
                        xy=(pts[-1][0], plat[1] * scale), ha="right", va="bottom",
                        fontsize=8, color="grey")
    ax.set(xlabel=_xlabel(curve), ylabel=f"RBR {unit}".strip(), title=title)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    return ax


def plot_rbr_cohorts(curve: RBRCurve, *, ax=None, as_percent: bool = True,
                     show_pooled: bool = True,
                     title: str = "Repeat-buying rate by entry cohort"):
    """Diagnostic RBR(t) per entry cohort (one line each, band order), with
    the pooled curve as a thick grey reference. Diverging cohort curves flag
    entry-wave differences (late triers repeating unlike the early ones)."""
    ax = _ax(ax)
    scale = 100.0 if as_percent else 1.0
    unit = "%" if as_percent else ""
    series = curve.cohort_series()
    if not series:
        raise ValueError("no per-cohort RBR series to plot")
    if show_pooled:
        pooled = sorted((p.interval, p.rbr) for p in curve.points
                        if p.rbr is not None)
        if pooled:
            ax.plot([t for t, _ in pooled], [v * scale for _, v in pooled], "-",
                    color="grey", lw=2.5, alpha=0.5, label="pooled")
    for i, (label, pts) in enumerate(series.items()):
        pts = [(t, v) for t, v in pts if v is not None]
        if not pts:
            continue
        color = _COHORT_COLORS[i % len(_COHORT_COLORS)]
        marker = _COHORT_MARKERS[i % len(_COHORT_MARKERS)]
        ax.plot([t for t, _ in pts], [v * scale for _, v in pts],
                linestyle="-", marker=marker, color=color, ms=3, label=label)
    ax.set(xlabel=_xlabel(curve), ylabel=f"RBR {unit}".strip(), title=title)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    return ax
