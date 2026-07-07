"""penetration_lite -- standalone lite edition of the Parfitt-Collins
penetration model (observed curve, ultimate K, piecewise promo-aware
theoretical curve, p.w.s.d. validation, K stability, one plot).

Fully independent of `parfitt_trb`; Spark is touched only by
`build_penetration`. Calendar buckets: iso_week / iso_fortnight / month.
"""
from .calendar import UNITS, parse_period_label, period_label, period_of
from .penetration import (
    PenetrationCurve, PiecewiseCurve, Segment, ValidationResult,
    build_penetration, centred_differences, fit, fit_piecewise, pwsd,
    smoothed_series, stability, stability_piecewise, validate,
)
from .plots import plot_penetration

__all__ = [
    "UNITS", "parse_period_label", "period_label", "period_of",
    "PenetrationCurve", "PiecewiseCurve", "Segment", "ValidationResult",
    "build_penetration", "centred_differences", "fit", "fit_piecewise",
    "pwsd", "smoothed_series", "stability", "stability_piecewise", "validate",
    "plot_penetration",
]
