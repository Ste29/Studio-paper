"""buying_lite -- standalone lite edition of the Parfitt-Collins buying-rate
index (headline B on a window ending at the analysis date, triers and
repeaters bases, growing-base per-bucket series B(t), optional per-entry-
cohort B_i, two plots).

Fully independent of `parfitt_trb`; Spark is touched only by
`build_buying_index`. Calendar buckets: iso_week / iso_fortnight / month.
"""
from .buying import BuyingIndex, BuyingPoint, build_buying_index
from .calendar import UNITS, parse_period_label, period_label, period_of
from .plots import plot_buying_cohorts, plot_buying_index

__all__ = [
    "UNITS", "parse_period_label", "period_label", "period_of",
    "BuyingIndex", "BuyingPoint", "build_buying_index",
    "plot_buying_index", "plot_buying_cohorts",
]
