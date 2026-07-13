"""Buying-index charts: the growing-base B(t) series for both bases (with the
headline windowed B optionally marked) and the per-entry-cohort B_i bars.
matplotlib is imported lazily. A printed note flags a partial last bucket.
"""
from __future__ import annotations

from typing import Sequence

from .buying import BuyingIndex

AXIS_LABELS = {"iso_week": "ISO week", "iso_fortnight": "ISO fortnight",
               "month": "Month"}


def _mpl():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("plotting requires matplotlib") from e


def _ax(ax):
    if ax is None:
        _, ax = _mpl().subplots(figsize=(8, 4.5))
    return ax


def _calendar_ticks(ax, bi: BuyingIndex, periods: Sequence[int],
                    max_labels: int = 13) -> None:
    """Calendar labels on the x ticks, thinned to at most ~max_labels."""
    ps = sorted({int(p) for p in periods})
    if not ps:
        return
    step = max(1, -(-len(ps) // max_labels))          # ceil division
    ticks = ps[::step]
    ax.set_xticks(ticks)
    ax.set_xticklabels([bi.label(p) for p in ticks], rotation=45, ha="right",
                       fontsize=7)


def plot_buying_index(bi: BuyingIndex, *, ax=None, mark_window: bool = True,
                      title: str = "Buying-rate index"):
    """Growing-base B(t) for both bases, the 1.0 parity line (brand buyers
    behave like the average category buyer) and, with `mark_window`, the
    headline windowed B of each base as a horizontal reference."""
    ax = _ax(ax)
    tri = [(p.period, p.b_triers) for p in bi.points if p.b_triers is not None]
    rep = [(p.period, p.b_repeaters) for p in bi.points
           if p.b_repeaters is not None]
    if not tri and not rep:
        raise ValueError("no buying-index series to plot")
    bi._print_partial_note()
    if tri:
        ax.plot([t for t, _ in tri], [v for _, v in tri], "-o",
                color="tab:blue", ms=3, label="B(t) triers")
    if rep:
        ax.plot([t for t, _ in rep], [v for _, v in rep], "--^",
                color="tab:orange", ms=3, label="B(t) repeaters")
    ax.axhline(1.0, ls=":", color="grey", lw=1)
    if mark_window:
        ax.axhline(bi.b_triers, ls="-.", color="tab:blue", lw=1, alpha=0.5,
                   label=f"B triers (window) = {bi.b_triers:.2f}")
        if bi.b_repeaters is not None:
            ax.axhline(bi.b_repeaters, ls="-.", color="tab:orange", lw=1,
                       alpha=0.5,
                       label=f"B repeaters (window) = {bi.b_repeaters:.2f}")
    ax.set(xlabel=AXIS_LABELS[bi.unit], ylabel="Buying index", title=title)
    _calendar_ticks(ax, bi, [p.period for p in bi.points])
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    return ax


def plot_buying_cohorts(bi: BuyingIndex, *, ax=None,
                        title: str = "Buying index by entry cohort"):
    """B_i per entry cohort (bars, chronological order) against the 1.0 parity
    line -- do late triers buy the category as heavily as the early ones?"""
    table = bi.cohort_frame()                 # raises when built without cohorts
    if table.empty:
        raise ValueError("no per-cohort buying index to plot")
    ax = _ax(ax)
    xs = range(len(table))
    ax.bar(xs, table["b"], color="tab:blue", alpha=0.85)
    ax.axhline(1.0, ls=":", color="grey", lw=1)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(table["cohort"], rotation=45, ha="right", fontsize=7)
    ax.set(xlabel=f"Entry cohort ({AXIS_LABELS[bi.cohort_unit]})",
           ylabel="Buying index", title=title)
    ax.set_ylim(bottom=0)
    return ax
