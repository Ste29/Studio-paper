"""Display-time presentation of the period-indexed series.

Two ways to label periods for output:

* ``label_ratio`` / ``label_cumulative`` take a period-ordinal -> calendar-label
  map (``TRBResult.period_labels``) and present the series on whatever calendar
  axis the model computed it (week / month / iso_week / fiscal_445). Use these
  when the axis is already at the granularity you want to show.
* ``rollup_ratio`` / ``rollup_cumulative`` additionally COARSEN a weekly axis to
  ``YYYY-MM`` months (the original point-8 rollup). They assume a weekly base
  axis; with a calendar-anchored ``period_unit`` (month / iso_week / fiscal_445)
  use the ``label_*`` helpers instead, since the series is already bucketed.

Ratio series (share, buying index) are aggregated by re-summing their
numerator/denominator components; cumulative series (penetration) take the last
value observed within each label.
"""
from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional, Tuple

from .periods import month_label, week_repr_date


def label_ratio(series4: List[Tuple[int, Optional[float], float, float]],
                labels: Dict[int, str]) -> List[Tuple[str, Optional[float]]]:
    """[(period, ratio, num, den)] -> [(label, ratio)] by re-summing num/den per
    calendar label (collapses any periods that share a label)."""
    agg: dict = {}
    order: List[str] = []
    for period, _ratio, num, den in series4:
        m = labels.get(int(period), str(int(period)))
        if m not in agg:
            agg[m] = [0.0, 0.0]
            order.append(m)
        agg[m][0] += num
        agg[m][1] += den
    return [(m, (agg[m][0] / agg[m][1]) if agg[m][1] > 0 else None) for m in order]


def label_cumulative(series2: List[Tuple[int, float]],
                     labels: Dict[int, str]) -> List[Tuple[str, float]]:
    """[(period, value)] -> [(label, last value within the label)]."""
    agg: dict = {}
    order: List[str] = []
    for period, val in series2:
        m = labels.get(int(period), str(int(period)))
        if m not in agg:
            order.append(m)
        agg[m] = val
    return [(m, agg[m]) for m in order]


def period_month(period: int, origin: date) -> str:
    return month_label(week_repr_date(period, origin))


def rollup_ratio(series4: List[Tuple[int, Optional[float], float, float]],
                 origin: date) -> List[Tuple[str, Optional[float]]]:
    """[(period, ratio, num, den)] -> [(YYYY-MM, ratio)] by re-summing num/den."""
    agg: dict = {}
    order: List[str] = []
    for period, _ratio, num, den in series4:
        m = period_month(period, origin)
        if m not in agg:
            agg[m] = [0.0, 0.0]
            order.append(m)
        agg[m][0] += num
        agg[m][1] += den
    return [(m, (agg[m][0] / agg[m][1]) if agg[m][1] > 0 else None) for m in order]


def rollup_cumulative(series2: List[Tuple[int, float]],
                      origin: date) -> List[Tuple[str, float]]:
    """[(period, value)] -> [(YYYY-MM, last value in month)] (for penetration)."""
    agg: dict = {}
    order: List[str] = []
    for period, val in series2:
        m = period_month(period, origin)
        if m not in agg:
            order.append(m)
        agg[m] = val
    return [(m, agg[m]) for m in order]
