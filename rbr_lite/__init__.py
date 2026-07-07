"""rbr_lite -- standalone lite edition of the Parfitt-Collins repeat-buying
rate (pooled RBR(t) on exact-day intervals from each shopper's trial, optional
per-cohort curves, plateau / stabilised-mean diagnostics, two plots).

Fully independent of `parfitt_trb`; Spark is touched only by `build_rbr`.
Cohort calendar buckets: iso_week / iso_fortnight / month.
"""
from .calendar import UNITS, parse_period_label, period_label, period_of
from .plots import plot_rbr, plot_rbr_cohorts
from .rbr import (
    RBRCurve, RBRPoint, build_rbr, detect_plateau, last_available_rbr,
    stable_rbr,
)

__all__ = [
    "UNITS", "parse_period_label", "period_label", "period_of",
    "RBRPoint", "RBRCurve", "build_rbr",
    "last_available_rbr", "detect_plateau", "stable_rbr",
    "plot_rbr", "plot_rbr_cohorts",
]
