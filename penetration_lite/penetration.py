"""Penetration model of Parfitt & Collins (1968), lite edition.

Observed cumulative penetration per calendar bucket, the fitted theoretical
curve P(t) = K(1 - e^{-a t}) (K = ultimate expected penetration), the piecewise
promo-aware composition (each post-promo segment re-anchored on the observed
penetration at its promo via a change of coordinates), p.w.s.d. validation and
the K-stability diagnostic.

Spark is used ONLY inside :func:`build_penetration` (two tiny group-bys whose
collected result is one row per bucket); everything else is numpy/pandas on the
small series. Every tabular output carries a calendar `label` column alongside
the period ordinal.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from datetime import date
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .calendar import _check_unit, as_date, period_col, period_label

_DENOMINATORS = ("dynamic", "static")


def _observed_fitted_frame(series: Sequence[Tuple[int, float]],
                           label_fn: Callable[[int], str],
                           fitted_fn: Callable[[float], Optional[float]]
                           ) -> pd.DataFrame:
    """Table of a penetration curve: period, label, P_observed, P_fitted
    (NaN where the fit is unavailable). `fitted_fn` is evaluated once per
    period. Shared by the single and piecewise curves."""
    fitted = [fitted_fn(t) for t, _ in series]
    return pd.DataFrame({
        "period": [t for t, _ in series],
        "label": [label_fn(t) for t, _ in series],
        "P_observed": [p for _, p in series],
        "P_fitted": [np.nan if f is None else f for f in fitted],
    })


# --------------------------------------------------------------------------- #
# Observed curve
# --------------------------------------------------------------------------- #
@dataclass
class PenetrationCurve:
    """Observed cumulative penetration P(t) plus the fitted model, on one of
    the calendar-bucket axes (iso_week / iso_fortnight / month)."""
    origin: date                           # launch (bucket 1 contains it)
    unit: str                              # 'iso_week' | 'iso_fortnight' | 'month'
    denominator: str                       # 'dynamic' (cumΣF) | 'static' (F_tot)
    series: List[Tuple[int, float]]        # [(period, P), ...] raw, never smoothed
    n_brand_triers: int                    # N_tot
    n_category_triers: int                 # F_tot
    K: Optional[float] = None              # ultimate expected penetration
    a: Optional[float] = None              # growth rate
    note: str = ""

    @property
    def snapshot(self) -> float:
        """End-of-window penetration N_tot / F_tot (the observed trial index)."""
        return self.n_brand_triers / self.n_category_triers

    def fitted(self, t: float) -> Optional[float]:
        """Theoretical P(t) = K(1 - e^{-a t}); None when not fitted."""
        if self.K is None or self.a is None:
            return None
        return self.K * (1.0 - math.exp(-self.a * t))

    def label(self, period: int) -> str:
        """Calendar label of a period ordinal (works for future periods too)."""
        return period_label(period, self.origin, self.unit)

    def to_frame(self) -> pd.DataFrame:
        """The curve as a table: period, label, P_observed, P_fitted."""
        return _observed_fitted_frame(self.series, self.label, self.fitted)


def build_penetration(sdf, *, card_col: str = "shopper_id",
                      date_col: str = "txn_date",
                      brand_col: str = "is_new_product",
                      category_col: str = "is_category",
                      unit: str = "iso_week", denominator: str = "dynamic",
                      launch_date=None, analysis_date=None) -> PenetrationCurve:
    """Build the observed penetration curve from a Spark transaction log.

    The brand is treated as part of the category (a brand purchase is also a
    category purchase). Only the FIRST brand / first category purchase per card
    matters; both group-bys collapse to one row per bucket before collection.
    dynamic: P(t) = cumΣN / cumΣF ; static: P(t) = cumΣN / F_tot.
    """
    _check_unit(unit)
    if denominator not in _DENOMINATORS:
        raise ValueError(f"denominator must be one of {_DENOMINATORS}")
    from pyspark.sql import functions as F

    p = (sdf.withColumn("_ts", F.to_date(F.col(date_col)))
            .withColumn("_card", F.col(card_col).cast("string"))
            .withColumn("_brand", F.col(brand_col).cast("boolean"))
            .withColumn("_cat", F.col(category_col).cast("boolean")
                        | F.col(brand_col).cast("boolean")))
    if analysis_date is not None:
        p = p.filter(F.col("_ts") <= F.lit(as_date(analysis_date).isoformat()).cast("date"))

    if launch_date is not None:
        origin = as_date(launch_date)
    else:
        o = (p.filter(F.col("_brand")).agg(F.min("_ts").alias("o")).collect()[0]["o"])
        if o is None:
            raise ValueError("no brand purchase on/before the analysis date: set "
                             "launch_date or widen the window")
        origin = as_date(o)
    on_axis = F.col("_ts") >= F.lit(origin.isoformat()).cast("date")

    # One shuffle over the log: each card's FIRST brand and FIRST category
    # purchase date on the axis (min ignores the nulls of the non-matching
    # rows). Cached so the two per-bucket counts reuse the single group-by.
    per_card = (p.filter(on_axis).groupBy("_card").agg(
        F.min(F.when(F.col("_brand"), F.col("_ts"))).alias("_fb"),
        F.min(F.when(F.col("_cat"), F.col("_ts"))).alias("_fc"))).cache()

    def _bucket_counts(first_col: str) -> dict:
        """{period: #cards whose FIRST such purchase falls in that bucket}."""
        rows = (per_card.filter(F.col(first_col).isNotNull())
                .withColumn("_period", period_col(F, F.col(first_col), unit, origin))
                .filter(F.col("_period") >= 1)
                .groupBy("_period").count().collect())
        return {int(r["_period"]): int(r["count"]) for r in rows}

    try:
        brand_new = _bucket_counts("_fb")
        cat_new = _bucket_counts("_fc")
    finally:
        per_card.unpersist()
    n_brand, n_cat = sum(brand_new.values()), sum(cat_new.values())
    if n_cat == 0:
        raise ValueError("no category triers on/after the launch date")

    dynamic = denominator == "dynamic"
    cb = cc = 0
    series: List[Tuple[int, float]] = []
    for t in range(1, max(max(cat_new, default=1), max(brand_new, default=1)) + 1):
        cb += brand_new.get(t, 0)
        cc += cat_new.get(t, 0)
        if cc == 0:                          # nobody in the market yet
            continue
        series.append((t, cb / cc if dynamic else cb / n_cat))
    return PenetrationCurve(origin=origin, unit=unit, denominator=denominator,
                            series=series, n_brand_triers=n_brand,
                            n_category_triers=n_cat)


# --------------------------------------------------------------------------- #
# Fit (discounted least squares on the difference model)
# --------------------------------------------------------------------------- #
def smoothed_series(series: Sequence[Tuple[int, float]],
                    window: int) -> List[Tuple[int, float]]:
    """Centred moving average of the P values (same grid, same length). At the
    edges the window is clipped, so the first/last points survive. `window`
    must be an odd int >= 3."""
    if window < 3 or window % 2 == 0:
        raise ValueError("smoothing window must be an odd int >= 3")
    ps = pd.Series([p for _, p in series], dtype=float)
    sm = ps.rolling(window, center=True, min_periods=1).mean()
    return [(t, float(v)) for (t, _), v in zip(series, sm)]


def centred_differences(series: Sequence[Tuple[int, float]]
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interior centred differences: (t, P, ΔP) with ΔP=(P(t+1)-P(t-1))/2 --
    exactly what :func:`fit` regresses."""
    ts = np.array([t for t, _ in series], dtype=float)
    ps = np.array([p for _, p in series], dtype=float)
    return ts[1:-1], ps[1:-1], (ps[2:] - ps[:-2]) / 2.0


def fit(curve: PenetrationCurve, *, discount_weight: float = 0.6,
        smoothing_window: Optional[int] = None) -> PenetrationCurve:
    """Estimate K, a from the difference model ΔP(t) = a(K - P(t)) + ε by
    discounted least squares (recent points weighted w^age).

    With `smoothing_window`, the differencing runs on a centrally-smoothed COPY
    of the series (differencing amplifies noise in P); `curve.series` stays
    raw. Mutates `curve` in place and returns it.
    """
    series = curve.series
    if len(series) < 4:
        curve.note = "need >=4 periods to fit; too few observed"
        return curve
    fit_series = (smoothed_series(series, smoothing_window)
                  if smoothing_window is not None else series)
    _, x, dP = centred_differences(fit_series)

    _rw = (getattr(getattr(np, "exceptions", None), "RankWarning", None)
           or getattr(np, "RankWarning", Warning))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", _rw)
        try:
            ages = np.arange(len(x) - 1, -1, -1, dtype=float)   # latest -> age 0
            w = np.sqrt(discount_weight ** ages)
            slope, intercept = np.polyfit(x, dP, 1, w=w)
        except np.linalg.LinAlgError:
            curve.note = "degenerate penetration series (no variation): cannot fit"
            return curve
    a = -slope
    if a <= 0:
        curve.note = ("rate of increase not yet declining (a<=0): cannot project "
                      "the ultimate penetration -- wait for the curve to decelerate")
        return curve
    curve.a = float(a)
    curve.K = float(intercept / a)
    if curve.K < max(p for _, p in series):
        curve.note = ("estimated K below the latest observed penetration; fit "
                      "unstable (e.g. a dynamic denominator still growing) -- caution")
    return curve


def _truncated(curve: PenetrationCurve, cutoff_period: int) -> PenetrationCurve:
    """Unfitted copy restricted to periods <= cutoff_period."""
    return PenetrationCurve(curve.origin, curve.unit, curve.denominator,
                            [(t, p) for t, p in curve.series if t <= cutoff_period],
                            curve.n_brand_triers, curve.n_category_triers)


# --------------------------------------------------------------------------- #
# Piecewise promo-aware composition
# --------------------------------------------------------------------------- #
@dataclass
class Segment:
    """One piece of the composed curve: base + K_inc(1 - e^{-a(t - t0)})."""
    t0: int                       # 0 for the launch segment, else the promo period
    base: float                   # observed P at t0 (0.0 for the launch segment)
    K_inc: Optional[float]        # incremental ceiling K' (launch: K itself)
    a: Optional[float]
    note: str = ""

    def fitted(self, t: float) -> Optional[float]:
        if self.K_inc is None or self.a is None:
            return None
        return self.base + self.K_inc * (1.0 - math.exp(-self.a * (t - self.t0)))

    @property
    def ceiling(self) -> Optional[float]:
        return None if self.K_inc is None else self.base + self.K_inc


@dataclass
class PiecewiseCurve:
    """Composed theoretical penetration across promos. Segment i governs
    t in [promo_i, promo_{i+1}); `fitted` is total over the whole axis (an
    unfitted segment falls back to the nearest earlier fitted one)."""
    origin: date
    unit: str
    promo_periods: List[int]
    segments: List[Segment]
    series: List[Tuple[int, float]]        # full observed series
    note: str = ""

    def segment_index_for(self, t: float) -> int:
        """Index of the segment governing time t (the last anchored at/before t)."""
        idx = 0
        for i, s in enumerate(self.segments):
            if s.t0 <= t:
                idx = i
        return idx

    def segment_for(self, t: float) -> Segment:
        return self.segments[self.segment_index_for(t)]

    def fitted(self, t: float) -> Optional[float]:
        idx = self.segment_index_for(t)
        for s in reversed(self.segments[:idx + 1]):
            v = s.fitted(t)
            if v is not None:
                return v
        return None

    @property
    def ultimate_penetration(self) -> Optional[float]:
        for s in reversed(self.segments):
            if s.ceiling is not None:
                return s.ceiling
        return None

    def label(self, period: int) -> str:
        return period_label(period, self.origin, self.unit)

    def to_frame(self) -> pd.DataFrame:
        """Observed vs composed-fitted per period: period, label, P_observed,
        P_fitted."""
        return _observed_fitted_frame(self.series, self.label, self.fitted)


def fit_piecewise(curve: PenetrationCurve, promo_periods: Sequence[int], *,
                  discount_weight: float = 0.6,
                  smoothing_window: Optional[int] = None,
                  min_segment_points: int = 4) -> PiecewiseCurve:
    """Fit the composed promo-aware curve.

    Segment 0 is fitted on the series up to the first promo. For each promo at
    t_i with observed penetration P0 = P_obs(t_i), the post-promo points are
    re-expressed from a fresh origin -- (t - t_i, P(t) - P0) -- and re-fitted;
    the change of coordinates back gives P_i(t) = P0 + K'(1 - e^{-a'(t - t_i)}),
    so every piece starts exactly where the observed curve stood at its promo.
    An empty `promo_periods` reduces to the plain single fit in one segment.
    """
    series = list(curve.series)
    if not series:
        raise ValueError("no penetration series to fit")
    obs = dict(series)
    promos = [int(p) for p in promo_periods]
    if promos != sorted(set(promos)):
        raise ValueError("promo_periods must be strictly increasing")
    missing = [p for p in promos if p not in obs]
    if missing:
        raise ValueError(f"promo periods {missing} not in the observed series "
                         "(the observed penetration at each promo anchors its segment)")
    last = series[-1][0]

    def _fit_sub(sub_series: List[Tuple[int, float]], t0: int, base: float,
                 label: str) -> Segment:
        if len(sub_series) < min_segment_points:
            return Segment(t0=t0, base=base, K_inc=None, a=None,
                           note=f"{label}: {len(sub_series)} points < "
                                f"min_segment_points={min_segment_points}; not fitted")
        sub = PenetrationCurve(curve.origin, curve.unit, curve.denominator,
                               sub_series, curve.n_brand_triers,
                               curve.n_category_triers)
        # Smoothing the shifted sub-series equals smoothing P inside the
        # segment (the change of coordinates is affine).
        fit(sub, discount_weight=discount_weight, smoothing_window=smoothing_window)
        return Segment(t0=t0, base=base, K_inc=sub.K, a=sub.a,
                       note=f"{label}: {sub.note}" if sub.note else "")

    segments: List[Segment] = []
    first_end = promos[0] if promos else last
    segments.append(_fit_sub([(t, p) for t, p in series if t <= first_end],
                             t0=0, base=0.0, label="launch segment"))
    for i, t_i in enumerate(promos):
        end = promos[i + 1] if i + 1 < len(promos) else last
        base = obs[t_i]
        shifted = ([(0, 0.0)]
                   + [(t - t_i, p - base) for t, p in series if t_i < t <= end])
        segments.append(_fit_sub(shifted, t0=t_i, base=base, label=f"promo @{t_i}"))

    notes = "; ".join(s.note for s in segments if s.note)
    return PiecewiseCurve(origin=curve.origin, unit=curve.unit,
                          promo_periods=promos, segments=segments,
                          series=series, note=notes)


# --------------------------------------------------------------------------- #
# Validation (p.w.s.d.) and K-stability diagnostics
# --------------------------------------------------------------------------- #
def pwsd(actual: Sequence[float], forecast: Sequence[float],
         w: float = 0.6) -> float:
    """Percentage weighted standard deviation (paper appendix): weighted RMS of
    the relative error (P - P̂)/P, the LAST element weighted most."""
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    if a.shape != f.shape or a.size == 0:
        raise ValueError("actual and forecast must be aligned, non-empty")
    ages = np.arange(a.size - 1, -1, -1, dtype=float)   # last element age 0
    wts = w ** ages
    rel = (a - f) / a
    return float(np.sqrt(np.sum(wts * rel ** 2) / wts.sum()))


@dataclass
class ValidationResult:
    """Held-out validation: the curve is fitted on periods <= cutoff only,
    projected over the full horizon and scored against the whole observed
    series (pwsd_full) and the held-out tail alone (pwsd_holdout)."""
    cutoff_period: int
    pwsd_full: Optional[float]
    pwsd_holdout: Optional[float]
    curve: object                                  # PenetrationCurve | PiecewiseCurve
    actual: List[Tuple[int, float]]
    forecast: List[Tuple[int, Optional[float]]]
    note: str = ""

    def to_frame(self) -> pd.DataFrame:
        """Aligned comparison table: period, label, actual, forecast."""
        return pd.DataFrame({
            "period": [t for t, _ in self.actual],
            "label": [self.curve.label(t) for t, _ in self.actual],
            "actual": [p for _, p in self.actual],
            "forecast": [np.nan if f is None else f for _, f in self.forecast],
        })


def validate(curve: PenetrationCurve, cutoff_period: int, *,
             promo_periods: Optional[Sequence[int]] = None,
             discount_weight: float = 0.6,
             smoothing_window: Optional[int] = None,
             w: float = 0.6) -> ValidationResult:
    """Fit on data up to `cutoff_period`, project the future periods, and score
    the whole theoretical curve against the full observed series. With
    `promo_periods` the truncated fit is piecewise promo-aware (promos after
    the cutoff are unknowable at forecast time: dropped and noted)."""
    series = list(curve.series)
    cutoff = int(cutoff_period)
    train_pts = [(t, p) for t, p in series if t <= cutoff]
    holdout = [(t, p) for t, p in series if t > cutoff]
    if len(train_pts) < 4:
        raise ValueError("need >=4 observed periods on/before the cutoff to fit")
    if not holdout:
        raise ValueError("no held-out periods after the cutoff: nothing to validate")

    train = _truncated(curve, cutoff)
    note_parts: List[str] = []
    pre_promos = [int(p) for p in (promo_periods or []) if int(p) <= cutoff]
    dropped = [int(p) for p in (promo_periods or []) if int(p) > cutoff]
    if dropped:
        note_parts.append(f"promos after the cutoff dropped from the fit: {dropped}")
    fitted_curve: object
    if pre_promos:
        fitted_curve = fit_piecewise(train, pre_promos,
                                     discount_weight=discount_weight,
                                     smoothing_window=smoothing_window)
    else:
        fitted_curve = fit(train, discount_weight=discount_weight,
                           smoothing_window=smoothing_window)
    if fitted_curve.note:
        note_parts.append(fitted_curve.note)

    forecast = [(t, fitted_curve.fitted(t)) for t, _ in series]
    pwsd_full = pwsd_holdout = None
    if all(f is not None for _, f in forecast):
        pwsd_full = pwsd([p for _, p in series], [f for _, f in forecast], w=w)
        pwsd_holdout = pwsd([p for t, p in series if t > cutoff],
                            [f for t, f in forecast if t > cutoff], w=w)
    else:
        note_parts.append("truncated fit could not project: pwsd unavailable")
    return ValidationResult(cutoff_period=cutoff, pwsd_full=pwsd_full,
                            pwsd_holdout=pwsd_holdout, curve=fitted_curve,
                            actual=series, forecast=forecast,
                            note="; ".join(note_parts))


def stability(curve: PenetrationCurve, *, cutoffs: Optional[Sequence[int]] = None,
              min_periods: int = 6, discount_weight: float = 0.6,
              smoothing_window: Optional[int] = None) -> pd.DataFrame:
    """One row per estimation cutoff -- cutoff, label, K, a, observed_P, note --
    so the analyst can see whether K stabilises as the window grows. Default
    cutoffs = every observed period from `min_periods` to the last."""
    series = curve.series
    if not series:
        raise ValueError("no penetration series")
    obs = dict(series)
    if cutoffs is None:
        cutoffs = [t for t, _ in series if t >= min_periods]
    rows = []
    for c in cutoffs:
        sub = _truncated(curve, int(c))
        fit(sub, discount_weight=discount_weight, smoothing_window=smoothing_window)
        rows.append({
            "cutoff": int(c),
            "label": curve.label(int(c)),
            "K": np.nan if sub.K is None else sub.K,
            "a": np.nan if sub.a is None else sub.a,
            "observed_P": obs.get(int(c), sub.series[-1][1] if sub.series else np.nan),
            "note": sub.note,
        })
    return pd.DataFrame(rows, columns=["cutoff", "label", "K", "a",
                                       "observed_P", "note"])
