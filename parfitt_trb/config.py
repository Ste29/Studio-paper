"""Configuration for the Parfitt-Collins (TRB) brand-share model.

Everything tunable lives in :class:`TRBConfig` so nothing is a hidden constant.
The defaults are chosen so the small worked examples in the tests run unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

# Canonical calendar-axis unit sets (single source of truth, shared with the
# aggregation layer). DERIVED units are origin-relative date arithmetic;
# CALENDAR (anchored) units live on the real calendar grid via a date dimension.
DERIVED_UNITS = ("week", "fortnight", "month")
CALENDAR_UNITS = ("iso_week", "iso_fortnight", "fiscal_445")
PERIOD_UNITS = DERIVED_UNITS + CALENDAR_UNITS


@dataclass(frozen=True)
class TRBConfig:
    # --- input schema (column names of the transaction DataFrame) ----------- #
    card_column: str = "shopper_id"        # loyalty-card / household id
    date_column: str = "txn_date"          # purchase date (string or date)
    brand_column: str = "is_new_product"   # boolean: line is the studied brand
    category_column: str = "is_category"   # boolean: line is in the category
    measure: str = "volume"                # quantity column used as `qty`

    # --- calendar axis (penetration / share / per-period buying index) ------ #
    # Granularity of the calendar-time periods. RBR has its own axis (see below)
    # and entry COHORTS are always weekly (Parfitt Table 2); this only sets the
    # axis the penetration curve, the realised-share series and the per-period
    # buying index are computed and displayed on.
    #   'week' / 'fortnight' / 'month' : derived by date arithmetic from the launch
    #                            origin (origin-relative 7-day / 14-day /
    #                            calendar-month buckets).
    #   'iso_week'             : ISO calendar weeks ('YYYY-Www', cross-year safe).
    #   'iso_fortnight'        : pairs of consecutive ISO weeks on a fixed
    #                            epoch-aligned grid (each bucket exactly 14 days; a
    #                            pair may straddle a year boundary). Labelled
    #                            'YYYY-Fww' after the pair's FIRST ISO week.
    #   'fiscal_445'           : retail 4-4-5 periods ('YYYY-Pnn').
    # The calendar-anchored axes (iso_week / iso_fortnight / fiscal_445) live on the
    # real calendar grid (built from a date dimension), so a bucket with no sales
    # (e.g. an out-of-stock week) keeps its slot instead of collapsing onto its
    # neighbour.
    period_unit: str = "week"              # one of PERIOD_UNITS

    # --- share axis (optional override) ------------------------------------- #
    # When set, the REALISED SHARE series uses a different calendar axis from the
    # penetration / buying-index series -- e.g. penetration on ISO weeks but the
    # share reported on calendar months. None = share uses the main axis.
    share_period_unit: Optional[str] = None     # same domain as period_unit; None = same

    # --- window of the analysis -------------------------------------------- #
    launch_date: Optional[str] = None      # if set, the calendar origin & trial floor
    analysis_date: Optional[str] = None    # "as of" date; default = max purchase date

    # --- category handling -------------------------------------------------- #
    treat_brand_as_category: bool = True   # OR the brand flag into the category flag

    # --- penetration model -------------------------------------------------- #
    penetration_method: str = "discounted"     # 'discounted' (Gilchrist) | 'ols'
    discount_weight: float = 0.6               # Gilchrist lambda (paper uses 0.6)
    penetration_denominator: str = "dynamic"   # 'dynamic' (cum. F) | 'static' (F_tot)
    # Centred moving-average window applied to a COPY of the observed penetration
    # before the K/a fit differences it (noise in P is amplified by the centred
    # differencing). The stored/plotted series and pwsd comparisons stay raw.
    # None = no smoothing; otherwise an odd integer >= 3.
    penetration_smoothing_window: Optional[int] = None

    # --- repeat-buying rate (RBR) ------------------------------------------ #
    rbr_interval_mode: str = "exact"       # 'exact' (P-day windows) | 'bucket'
    period_length_days: int = 28           # length s of one RBR interval (exact mode)
    rbr_bucket_unit: str = "week"          # 'week' | 'month' (bucket mode)
    max_interval: Optional[int] = None     # cap on interval t (default = max feasible)
    # Interval from which the analyst judges the RBR curve stabilised. When set,
    # the share estimates use the MEAN of the observed RBR from this interval on
    # instead of the furthest-available value. None = furthest-available (paper).
    rbr_stable_from: Optional[int] = None

    # --- buying-rate index -------------------------------------------------- #
    buying_index_base: str = "triers"      # 'triers' (Parfitt) | 'repeaters' (Charan)
    repeater_min_purchases: int = 2        # >= this many brand buys => repeater
    # The window narrows the VOLUME being compared, never the membership bases:
    # scope members (triers / repeaters / category triers) stay all-time, and a
    # member with no purchases in the window weighs 0 in the per-capita average.
    buying_index_window_days: Optional[int] = None  # None = all history up to analysis

    # --- entry cohorts (Table 2) ------------------------------------------- #
    # week boundaries; (6, 12, 24) => [1-6], [7-12], [13-24], [25+]
    cohort_boundaries_weeks: Tuple[int, ...] = (6, 12, 24)
    include_prelaunch_cohort: bool = False  # treat pre-launch brand buyers as a cohort

    def __post_init__(self) -> None:
        if not 0.0 < self.discount_weight <= 1.0:
            raise ValueError("discount_weight must be in (0, 1]")
        if self.period_length_days <= 0:
            raise ValueError("period_length_days must be positive")
        if self.repeater_min_purchases < 2:
            raise ValueError("a repeater needs at least 2 brand purchases")
        if self.penetration_method not in ("discounted", "ols"):
            raise ValueError("penetration_method must be 'discounted' or 'ols'")
        if self.penetration_denominator not in ("dynamic", "static"):
            raise ValueError("penetration_denominator must be 'dynamic' or 'static'")
        if self.period_unit not in PERIOD_UNITS:
            raise ValueError(f"period_unit must be one of {PERIOD_UNITS}")
        if self.share_period_unit is not None and self.share_period_unit not in PERIOD_UNITS:
            raise ValueError(f"share_period_unit must be one of {PERIOD_UNITS} or None")
        if self.rbr_interval_mode not in ("exact", "bucket"):
            raise ValueError("rbr_interval_mode must be 'exact' or 'bucket'")
        if self.rbr_bucket_unit not in ("week", "month"):
            raise ValueError("rbr_bucket_unit must be 'week' or 'month'")
        if self.buying_index_base not in ("triers", "repeaters"):
            raise ValueError("buying_index_base must be 'triers' or 'repeaters'")
        if self.rbr_stable_from is not None and self.rbr_stable_from < 1:
            raise ValueError("rbr_stable_from must be >= 1 (an RBR interval)")
        w = self.penetration_smoothing_window
        if w is not None and (w < 3 or w % 2 == 0):
            raise ValueError("penetration_smoothing_window must be an odd int >= 3 "
                             "(a centred window needs symmetric arms)")
        if tuple(self.cohort_boundaries_weeks) != tuple(sorted(set(self.cohort_boundaries_weeks))):
            raise ValueError("cohort_boundaries_weeks must be strictly increasing & unique")
