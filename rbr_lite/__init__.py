"""rbr_lite -- standalone lite edition of the Parfitt-Collins repeat-buying
rate (pooled RBR(t) on exact-day or calendar-bucket intervals from each
shopper's trial, optional entry-cohort bands with custom boundaries, plateau /
stabilised-mean diagnostics, two plots).

Fully independent of `parfitt_trb`; Spark is touched only by `build_rbr`.
Calendar units: iso_week / iso_fortnight / month.
"""
from .calendar import (
    UNITS, boundary_end, label_after, label_last_day, parse_period_label,
    period_label, period_of,
)
from .plots import plot_rbr, plot_rbr_cohorts
from .rbr import (
    RBRCurve, RBRPoint, build_rbr, detect_plateau, last_available_rbr,
    stable_rbr,
)

__all__ = [
    "UNITS", "parse_period_label", "period_label", "period_of",
    "boundary_end", "label_after", "label_last_day",
    "RBRPoint", "RBRCurve", "build_rbr",
    "last_available_rbr", "detect_plateau", "stable_rbr",
    "plot_rbr", "plot_rbr_cohorts",
]
