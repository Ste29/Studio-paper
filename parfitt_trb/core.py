"""Backend-free modelling core for the Parfitt-Collins (TRB) model.

Every function here consumes the small, card-collapsed tables produced by
:class:`~parfitt_trb.aggregation.SparkAggregator` and returns plain
numpy / pandas / dataclasses. There is no DataFrame-engine dependency: the
maths is written once, engine-free.

References are to Parfitt & Collins (1968), JMR 5(2):131-145 and its appendix.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Result structures
# --------------------------------------------------------------------------- #
@dataclass
class RBRPoint:
    """One pooled repeat-buying-rate observation R(interval)."""
    interval: int
    rbr: Optional[float]       # None when no category volume observed yet
    brand_qty: float
    cat_qty: float
    n_eligible: int            # shoppers whose whole interval has elapsed


@dataclass
class Penetration:
    """Cumulative penetration series P(t) = ΣN(r)/ΣF(r) plus the fitted model."""
    denominator: str                       # 'dynamic' | 'static'
    origin: date
    series: List[Tuple[int, float]]        # [(week period, P), ...]
    n_brand_triers: int                    # N_tot
    n_category_triers: int                 # F_tot
    ultimate_penetration: Optional[float] = None   # K
    growth_rate: Optional[float] = None            # a
    note: str = ""

    @property
    def snapshot(self) -> float:
        """End-of-window penetration N_tot/F_tot (== the observed Trial Index)."""
        return self.n_brand_triers / self.n_category_triers

    def fitted(self, t: float) -> Optional[float]:
        """Theoretical P(t) = K(1 - e^{-a t}). None when not fitted."""
        if self.ultimate_penetration is None or self.growth_rate is None:
            return None
        return self.ultimate_penetration * (1.0 - math.exp(-self.growth_rate * t))


@dataclass
class Cohort:
    """One entry-cohort row of the segmented (Table 2) model."""
    label: str
    penetration: float            # P_i  (share of category triers)
    rbr: Optional[float]          # R_i  (furthest available interval)
    buying_index: float           # B_i
    n_triers: int
    is_future: bool = False

    @property
    def contribution(self) -> float:
        """P_i × R_i × B_i (0 when R_i is unknown)."""
        if self.rbr is None:
            return 0.0
        return self.penetration * self.rbr * self.buying_index


# --------------------------------------------------------------------------- #
# Penetration
# --------------------------------------------------------------------------- #
def build_penetration(entrants: pd.DataFrame, origin: date,
                      denominator: str) -> Penetration:
    """Cumulative penetration from per-period entrant counts.

    `entrants` has columns: period (1-based week), n_brand_new, n_cat_new.
    dynamic: P(t)=cumΣN / cumΣF ; static: P(t)=cumΣN / F_tot. Both end at
    N_tot/F_tot (the snapshot / observed trial index).
    """
    if entrants.empty:
        raise ValueError("no entrants: cannot build penetration")
    df = entrants.sort_values("period")
    bmap: Dict[int, int] = dict(zip(df["period"].astype(int), df["n_brand_new"].astype(int)))
    cmap: Dict[int, int] = dict(zip(df["period"].astype(int), df["n_cat_new"].astype(int)))
    n_brand = int(df["n_brand_new"].sum())
    n_cat = int(df["n_cat_new"].sum())
    if n_cat == 0:
        raise ValueError("no category triers on/after the launch date")

    dynamic = denominator == "dynamic"
    T = int(df["period"].max())
    cb = cc = 0
    series: List[Tuple[int, float]] = []
    for t in range(1, T + 1):
        cb += bmap.get(t, 0)
        cc += cmap.get(t, 0)
        if cc == 0:                       # nobody in the market yet
            continue
        series.append((t, cb / cc if dynamic else cb / n_cat))
    return Penetration(denominator=denominator, origin=origin, series=series,
                       n_brand_triers=n_brand, n_category_triers=n_cat)


def _truncated(pen: Penetration, cutoff_period: int) -> Penetration:
    """Unfitted copy of `pen` restricted to periods <= cutoff_period."""
    return Penetration(pen.denominator, pen.origin,
                       [(t, p) for t, p in pen.series if t <= cutoff_period],
                       pen.n_brand_triers, pen.n_category_triers)


def centred_differences(series: Sequence[Tuple[int, float]]
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interior centred differences of a penetration series: (t, P, ΔP) with
    ΔP(t)=(P(t+1)-P(t-1))/2 -- the exact quantities `fit_penetration` regresses,
    exposed so diagnostics (the ΔP-vs-P plot) can never drift from the fit."""
    ts = np.array([t for t, _ in series], dtype=float)
    ps = np.array([p for _, p in series], dtype=float)
    return ts[1:-1], ps[1:-1], (ps[2:] - ps[:-2]) / 2.0


def smoothed_series(series: Sequence[Tuple[int, float]],
                    window: int) -> List[Tuple[int, float]]:
    """Centred moving average of the P values (same period grid, same length).

    At the edges the window is clipped to the series (mean over the in-range
    part of [i-h, i+h], as pandas rolling(center=True, min_periods=1)), so the
    first and last points survive -- the differences near the end carry the
    most weight in the discounted fit. `window` must be an odd int >= 3."""
    if window < 3 or window % 2 == 0:
        raise ValueError("smoothing window must be an odd int >= 3")
    ts = [t for t, _ in series]
    ps = np.array([p for _, p in series], dtype=float)
    h = window // 2
    sm = [float(ps[max(0, i - h):i + h + 1].mean()) for i in range(len(ps))]
    return list(zip(ts, sm))


def fit_penetration(pen: Penetration, *, method: str = "discounted",
                    discount_weight: float = 0.6,
                    smoothing_window: Optional[int] = None) -> Penetration:
    """Estimate K, a from the printed difference model ΔP(t)=a(K-P(t))+ε.

    Centred difference ΔP(t)=(P(t+1)-P(t-1))/2; regress ΔP on P (slope=-a,
    intercept=aK) by discounted least squares (recent points weighted w^age) or
    plain OLS. When `smoothing_window` is set, the differencing runs on a
    centrally-smoothed COPY of the series (noise in P is amplified by the
    differencing); `pen.series` itself stays raw. Mutates `pen` in place and
    returns it.
    """
    series = pen.series
    if len(series) < 4:
        pen.note = "need >=4 periods to fit; too few observed"
        return pen
    fit_series = (smoothed_series(series, smoothing_window)
                  if smoothing_window is not None else series)
    _, x, dP = centred_differences(fit_series)
    if len(x) < 2:
        pen.note = "too few interior points to fit"
        return pen

    _rw = (getattr(getattr(np, "exceptions", None), "RankWarning", None)
           or getattr(np, "RankWarning", Warning))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", _rw)
        try:
            if method == "discounted":
                ages = np.arange(len(x) - 1, -1, -1, dtype=float)   # latest -> age 0
                w = np.sqrt(discount_weight ** ages)
                slope, intercept = np.polyfit(x, dP, 1, w=w)
            else:
                slope, intercept = np.polyfit(x, dP, 1)
        except np.linalg.LinAlgError:
            pen.note = "degenerate penetration series (no variation): cannot fit"
            return pen
    a = -slope
    if a <= 0:
        pen.note = ("rate of increase not yet declining (a<=0): cannot project "
                    "ultimate penetration -- wait for the curve to decelerate")
        return pen
    pen.growth_rate = float(a)
    pen.ultimate_penetration = float(intercept / a)
    if pen.ultimate_penetration < max(p for _, p in series):
        pen.note = ("estimated K below the latest observed penetration; fit "
                    "unstable (e.g. a dynamic denominator still growing) -- caution")
    return pen


def pwsd(actual: Sequence[float], forecast: Sequence[float],
         w: float = 0.6) -> float:
    """Percentage weighted standard deviation (paper appendix, Fig 19).

    s = sqrt( Σ_{i=0..t} (w^i / W)·((P(t-i)-P̂(t-i))/P(t-i))² ), W = Σ w^i,
    with i=0 the LATEST observation (so the last array element is weighted most).
    `actual` / `forecast` are aligned chronological sequences.
    """
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    if a.shape != f.shape or a.size == 0:
        raise ValueError("actual and forecast must be aligned, non-empty")
    ages = np.arange(a.size - 1, -1, -1, dtype=float)   # last element age 0
    wts = w ** ages
    rel = (a - f) / a
    return float(np.sqrt(np.sum(wts * rel ** 2) / wts.sum()))


@dataclass
class PenetrationPromo:
    """Result of comparing a baseline projection with the realised penetration
    after a marketing disturbance (price cut, new flavour, promotion)."""
    cutoff_period: int
    baseline_K: Optional[float]
    baseline_a: Optional[float]
    refit_K: Optional[float]
    refit_a: Optional[float]
    series: List[Tuple[int, float]]                 # observed (full)
    baseline_curve: List[Tuple[int, float]]         # projected from pre-cutoff fit
    refit_curve: List[Tuple[int, float]]            # projected from full/refit
    bought_penetration: Optional[float]             # observed - baseline at last period


def penetration_vs_actual(pen: Penetration, cutoff_period: int, *,
                          method: str = "discounted",
                          discount_weight: float = 0.6,
                          smoothing_window: Optional[int] = None,
                          project_to: Optional[int] = None) -> PenetrationPromo:
    """Fit K,a on periods <= cutoff (pre-promo), project forward, and compare
    with the realised curve; also re-fit on the full series (post-promo).

    Quantifies the 'bought' penetration = realised - baseline-projected at the
    last observed period (Parfitt's Figures 12-15 / appendix refinement).
    For the composed multi-promo theoretical curve (each segment re-anchored on
    the observed penetration at its promo), see :func:`fit_piecewise_penetration`.
    """
    full = pen.series
    last = full[-1][0]
    if project_to is None:
        project_to = last

    pre = _truncated(pen, cutoff_period)
    fit_penetration(pre, method=method, discount_weight=discount_weight,
                    smoothing_window=smoothing_window)
    post = Penetration(pen.denominator, pen.origin, list(full),
                       pen.n_brand_triers, pen.n_category_triers)
    fit_penetration(post, method=method, discount_weight=discount_weight,
                    smoothing_window=smoothing_window)

    base_curve = ([(t, pre.fitted(t)) for t in range(cutoff_period, project_to + 1)]
                  if pre.fitted(cutoff_period) is not None else [])
    refit_curve = ([(t, post.fitted(t)) for t in range(1, project_to + 1)]
                   if post.fitted(1) is not None else [])
    bought = None
    obs_last = full[-1][1]
    if pre.fitted(last) is not None:
        bought = obs_last - pre.fitted(last)
    return PenetrationPromo(
        cutoff_period=cutoff_period,
        baseline_K=pre.ultimate_penetration, baseline_a=pre.growth_rate,
        refit_K=post.ultimate_penetration, refit_a=post.growth_rate,
        series=list(full), baseline_curve=base_curve, refit_curve=refit_curve,
        bought_penetration=bought)


# --------------------------------------------------------------------------- #
# Piecewise promo-aware penetration
# --------------------------------------------------------------------------- #
@dataclass
class PenetrationSegment:
    """One piece of the composed theoretical curve: from its anchor `t0` on,
    P(t) = base + K_inc·(1 - e^{-a·(t - t0)}). The launch segment has t0=0,
    base=0 (so K_inc = K); a promo segment is anchored on the OBSERVED
    penetration at the promo period, so the pieces join continuously."""
    t0: int                        # 0 for the launch segment, else the promo period
    base: float                    # observed P at t0 (0.0 for the launch segment)
    K_inc: Optional[float]         # incremental ceiling K' = K_segment - base
    a: Optional[float]             # growth rate of this segment
    n_points: int                  # observed points the segment was fitted on
    note: str = ""

    def fitted(self, t: float) -> Optional[float]:
        if self.K_inc is None or self.a is None:
            return None
        return self.base + self.K_inc * (1.0 - math.exp(-self.a * (t - self.t0)))

    @property
    def ceiling(self) -> Optional[float]:
        """Ultimate penetration this segment tends to (base + K')."""
        return None if self.K_inc is None else self.base + self.K_inc


@dataclass
class PiecewisePenetration:
    """The composed theoretical penetration curve across promo disturbances.

    Segment i governs t in [promo_i, promo_{i+1}); `fitted` is total over the
    whole axis (an unfitted segment falls back to the nearest earlier fitted
    one), so the curve can always be projected and compared via `pwsd`."""
    origin: date
    promo_periods: List[int]
    segments: List[PenetrationSegment]
    series: List[Tuple[int, float]]        # full observed series
    note: str = ""

    def segment_for(self, t: float) -> PenetrationSegment:
        """The segment governing time t (the last one anchored at or before t)."""
        seg = self.segments[0]
        for s in self.segments:
            if s.t0 <= t:
                seg = s
        return seg

    def fitted(self, t: float) -> Optional[float]:
        idx = self.segments.index(self.segment_for(t))
        for s in reversed(self.segments[:idx + 1]):
            v = s.fitted(t)
            if v is not None:
                return v
        return None

    @property
    def ultimate_penetration(self) -> Optional[float]:
        """Ceiling of the last fitted segment (the curve's ultimate level)."""
        for s in reversed(self.segments):
            if s.ceiling is not None:
                return s.ceiling
        return None


def fit_piecewise_penetration(pen: Penetration, promo_periods: Sequence[int], *,
                              method: str = "discounted",
                              discount_weight: float = 0.6,
                              smoothing_window: Optional[int] = None,
                              min_segment_points: int = 4) -> PiecewisePenetration:
    """Fit the composed promo-aware curve (the 'true' theoretical penetration).

    Segment 0 is fitted on the observed series up to the first promo. For each
    promo at t_i with observed penetration P0 = P_obs(t_i), the post-promo data
    is re-expressed from a fresh origin -- (t - t_i, P(t) - P0) -- and re-fitted;
    the change of coordinates back gives P_i(t) = P0 + K'(1 - e^{-a'(t - t_i)})
    with K' the incremental ceiling, so every piece starts exactly where the
    observed curve stood at its promo. An empty `promo_periods` reduces to the
    plain single fit wrapped in one segment.
    """
    series = list(pen.series)
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
                 label: str) -> PenetrationSegment:
        if len(sub_series) < min_segment_points:
            return PenetrationSegment(
                t0=t0, base=base, K_inc=None, a=None, n_points=len(sub_series),
                note=f"{label}: {len(sub_series)} points < min_segment_points="
                     f"{min_segment_points}; not fitted")
        sub = Penetration(pen.denominator, pen.origin, sub_series,
                          pen.n_brand_triers, pen.n_category_triers)
        # Smoothing the shifted sub-series equals smoothing P within the segment
        # (the change of coordinates is affine).
        fit_penetration(sub, method=method, discount_weight=discount_weight,
                        smoothing_window=smoothing_window)
        return PenetrationSegment(
            t0=t0, base=base, K_inc=sub.ultimate_penetration, a=sub.growth_rate,
            n_points=len(sub_series),
            note=f"{label}: {sub.note}" if sub.note else "")

    segments: List[PenetrationSegment] = []
    first_end = promos[0] if promos else last
    segments.append(_fit_sub([(t, p) for t, p in series if t <= first_end],
                             t0=0, base=0.0, label="launch segment"))
    for i, t_i in enumerate(promos):
        end = promos[i + 1] if i + 1 < len(promos) else last
        base = obs[t_i]
        shifted = ([(0, 0.0)]
                   + [(t - t_i, p - base) for t, p in series if t_i < t <= end])
        segments.append(_fit_sub(shifted, t0=t_i, base=base,
                                 label=f"promo @{t_i}"))

    notes = "; ".join(s.note for s in segments if s.note)
    return PiecewisePenetration(origin=pen.origin, promo_periods=promos,
                                segments=segments, series=series, note=notes)


# --------------------------------------------------------------------------- #
# Out-of-sample validation (pwsd on a truncated fit vs the full observed series)
# --------------------------------------------------------------------------- #
@dataclass
class PenetrationValidation:
    """Held-out validation of the penetration fit: the curve is fitted on the
    periods up to `cutoff_period` only, projected over the full horizon, and
    compared (pwsd) against the complete observed series."""
    cutoff_period: int
    pwsd_full: Optional[float]              # over the whole series (train + held-out)
    pwsd_holdout: Optional[float]           # over t > cutoff only
    curve: object                           # Penetration | PiecewisePenetration
    actual: List[Tuple[int, float]]         # full observed series
    forecast: List[Tuple[int, Optional[float]]]   # curve.fitted at the same t
    note: str = ""


def validate_penetration(pen: Penetration, cutoff_period: int, *,
                         method: str = "discounted", discount_weight: float = 0.6,
                         smoothing_window: Optional[int] = None,
                         w: float = 0.6,
                         promo_periods: Optional[Sequence[int]] = None
                         ) -> PenetrationValidation:
    """Fit on data up to `cutoff_period`, project the future periods, and score
    the whole curve (old + predicted periods) against the full observed series.

    With `promo_periods`, the truncated fit uses the piecewise promo-aware curve
    (promos after the cutoff are unknowable at forecast time and are dropped,
    noted). Returns the aligned actual/forecast pairs for plotting alongside the
    two pwsd scores (full series and held-out tail only).
    """
    series = list(pen.series)
    cutoff = int(cutoff_period)
    train_pts = [(t, p) for t, p in series if t <= cutoff]
    holdout = [(t, p) for t, p in series if t > cutoff]
    if len(train_pts) < 4:
        raise ValueError("need >=4 observed periods on/before the cutoff to fit")
    if not holdout:
        raise ValueError("no held-out periods after the cutoff: nothing to validate")

    train = _truncated(pen, cutoff)
    note_parts: List[str] = []
    curve: object
    pre_promos = [int(p) for p in (promo_periods or []) if int(p) <= cutoff]
    dropped = [int(p) for p in (promo_periods or []) if int(p) > cutoff]
    if dropped:
        note_parts.append(f"promos after the cutoff dropped from the fit: {dropped}")
    if pre_promos:
        curve = fit_piecewise_penetration(train, pre_promos, method=method,
                                          discount_weight=discount_weight,
                                          smoothing_window=smoothing_window)
        if curve.note:
            note_parts.append(curve.note)
    else:
        curve = fit_penetration(train, method=method, discount_weight=discount_weight,
                                smoothing_window=smoothing_window)
        if curve.note:
            note_parts.append(curve.note)

    forecast = [(t, curve.fitted(t)) for t, _ in series]
    pwsd_full = pwsd_holdout = None
    if all(f is not None for _, f in forecast):
        pwsd_full = pwsd([p for _, p in series], [f for _, f in forecast], w=w)
        pwsd_holdout = pwsd([p for t, p in series if t > cutoff],
                            [f for t, f in forecast if t > cutoff], w=w)
    else:
        note_parts.append("truncated fit could not project: pwsd unavailable")
    return PenetrationValidation(
        cutoff_period=cutoff, pwsd_full=pwsd_full, pwsd_holdout=pwsd_holdout,
        curve=curve, actual=series, forecast=forecast,
        note="; ".join(note_parts))


def penetration_stability(pen: Penetration, *, cutoffs: Optional[Sequence[int]] = None,
                          min_periods: int = 6, method: str = "discounted",
                          discount_weight: float = 0.6,
                          smoothing_window: Optional[int] = None) -> pd.DataFrame:
    """Diagnostic table of the fit as the estimation window grows: one row per
    cutoff with the K and a fitted on periods <= cutoff, the observed P at the
    cutoff and the fit note -- so the analyst can see whether K stabilises.

    Default cutoffs = every observed period from `min_periods` to the last.
    """
    series = pen.series
    if not series:
        raise ValueError("no penetration series")
    obs = dict(series)
    if cutoffs is None:
        cutoffs = [t for t, _ in series if t >= min_periods]
    rows = []
    for c in cutoffs:
        sub = _truncated(pen, int(c))
        fit_penetration(sub, method=method, discount_weight=discount_weight,
                        smoothing_window=smoothing_window)
        rows.append({
            "cutoff": int(c),
            "K": (np.nan if sub.ultimate_penetration is None
                  else sub.ultimate_penetration),
            "a": np.nan if sub.growth_rate is None else sub.growth_rate,
            "observed_P": obs.get(int(c),
                                  sub.series[-1][1] if sub.series else np.nan),
            "note": sub.note,
        })
    return pd.DataFrame(rows, columns=["cutoff", "K", "a", "observed_P", "note"])


# --------------------------------------------------------------------------- #
# Repeat-buying rate
# --------------------------------------------------------------------------- #
def build_rbr(rbr_long: pd.DataFrame) -> List[RBRPoint]:
    """Pooled RBR series from per-interval summed volumes. Columns:
    interval, brand_qty, cat_qty, n_eligible."""
    pts: List[RBRPoint] = []
    for row in rbr_long.sort_values("interval").itertuples(index=False):
        cat = float(row.cat_qty)
        rbr = (float(row.brand_qty) / cat) if cat > 0 else None
        pts.append(RBRPoint(interval=int(row.interval), rbr=rbr,
                            brand_qty=float(row.brand_qty), cat_qty=cat,
                            n_eligible=int(row.n_eligible)))
    return pts


def last_available_rbr(points: Sequence[RBRPoint]) -> Optional[Tuple[int, float]]:
    """The furthest interval with an observed rate -> the r→∞ proxy used for
    the ultimate RBR (no auto plateau selection; the analyst reads the plot)."""
    avail = [(p.interval, p.rbr) for p in points if p.rbr is not None]
    return max(avail, key=lambda kv: kv[0]) if avail else None


def detect_plateau(points: Sequence[RBRPoint], tol: float = 0.005,
                   k: int = 3) -> Optional[Tuple[int, float]]:
    """DIAGNOSTIC only (not used by the share path): first interval where the
    rate stays within `tol` for `k` consecutive observations."""
    pts = sorted((p.interval, p.rbr) for p in points if p.rbr is not None)
    for i in range(len(pts) - k + 1):
        vals = [v for _, v in pts[i:i + k]]
        if max(vals) - min(vals) <= tol:
            return pts[i]
    return None


def stable_rbr(points: Sequence[RBRPoint], from_interval: int) -> Optional[float]:
    """Mean of the observed RBR over intervals >= `from_interval` -- the
    stabilised-rate estimate used when the analyst judges the curve flat from
    that interval on. None when no rate is observed there yet."""
    vals = [p.rbr for p in points
            if p.rbr is not None and p.interval >= from_interval]
    return float(np.mean(vals)) if vals else None


def rbr_cohort_series(rbr_cohort: pd.DataFrame, cohort_order: Sequence[str]
                      ) -> Dict[str, List[Tuple[int, Optional[float]]]]:
    """Ordered {cohort: [(interval, rbr | None), ...]} from the per-cohort RBR
    table (columns: cohort, interval, brand_qty, cat_qty) -- the full curves
    behind Table 2's single per-cohort rates, for the cohort-RBR diagnostic
    plot. rbr is None when no category volume is observed yet; cohorts with no
    rows at all are omitted."""
    out: Dict[str, List[Tuple[int, Optional[float]]]] = {}
    if rbr_cohort.empty:
        return out
    grouped = dict(tuple(rbr_cohort.groupby("cohort")))
    for label in cohort_order:
        grp = grouped.get(label)
        if grp is None:
            continue
        pts: List[Tuple[int, Optional[float]]] = []
        for r in grp.sort_values("interval").itertuples(index=False):
            cat = float(r.cat_qty)
            pts.append((int(r.interval),
                        (float(r.brand_qty) / cat) if cat > 0 else None))
        out[label] = pts
    return out


# --------------------------------------------------------------------------- #
# Buying-rate index
# --------------------------------------------------------------------------- #
def buying_index_from_scopes(scopes: pd.DataFrame, scope: str) -> Optional[float]:
    """avg category volume of `scope` buyers / avg of all category buyers.

    `scopes` has columns scope, sum_cat, n_buyers; the all-category row is
    keyed '__all__'. Returns None when the scope has no buyers in the window.
    """
    rows = {r.scope: (float(r.sum_cat), int(r.n_buyers))
            for r in scopes.itertuples(index=False)}
    if "__all__" not in rows or rows["__all__"][1] == 0:
        raise ValueError("no category buyers in the buying-index window")
    if scope not in rows or rows[scope][1] == 0:
        return None
    avg_all = rows["__all__"][0] / rows["__all__"][1]
    avg_sel = rows[scope][0] / rows[scope][1]
    return avg_sel / avg_all


def buying_index_series(buying_series: pd.DataFrame) -> List[Tuple[int, Optional[float]]]:
    """Per-period diagnostic buying index. Columns:
    period, sel_sum, sel_n, all_sum, all_n. (point 1: each period its own B)."""
    out: List[Tuple[int, Optional[float]]] = []
    for r in buying_series.sort_values("period").itertuples(index=False):
        if r.sel_n and r.all_n and r.all_sum > 0:
            out.append((int(r.period),
                        (r.sel_sum / r.sel_n) / (r.all_sum / r.all_n)))
        else:
            out.append((int(r.period), None))
    return out


# --------------------------------------------------------------------------- #
# Share over calendar time
# --------------------------------------------------------------------------- #
def share_series(share_long: pd.DataFrame) -> List[Tuple[int, Optional[float], float, float]]:
    """Realised brand share per calendar week. Columns: period, brand_qty,
    cat_qty. Returns (period, ratio, brand_qty, cat_qty) so monthly rollups can
    re-sum the components rather than averaging ratios."""
    out: List[Tuple[int, Optional[float], float, float]] = []
    for r in share_long.sort_values("period").itertuples(index=False):
        c = float(r.cat_qty)
        out.append((int(r.period), (float(r.brand_qty) / c) if c > 0 else None,
                    float(r.brand_qty), c))
    return out


# --------------------------------------------------------------------------- #
# Segmented (cohort) model -- Table 2
# --------------------------------------------------------------------------- #
def build_cohorts(cohort_counts: Mapping[str, int], rbr_cohort: pd.DataFrame,
                  buying_scopes: pd.DataFrame, n_category_triers: int,
                  cohort_order: Sequence[str], *,
                  ultimate_penetration: Optional[float] = None,
                  rbr_stable_from: Optional[int] = None) -> List[Cohort]:
    """Assemble the per-cohort Pᵢ/Rᵢ/Bᵢ rows and the estimated future cohort.

    Pᵢ = brand triers entering in cohort i / F_tot (so Σ observed Pᵢ = snapshot).
    Rᵢ = cohort's furthest-available RBR, or -- when `rbr_stable_from` is set --
    the mean of its rates from that interval on (young cohorts with no points
    there yet fall back to their furthest-available rate). Bᵢ = cohort buying
    index. Future cohort: P=K-Σobserved, R=last cohort's R, B=1.0 (Table 2).

    `cohort_counts` maps each entry-cohort label to its number of brand triers.
    """
    counts = dict(cohort_counts)
    # per-cohort RBR estimate (furthest-available or stabilised mean)
    rbr_by_cohort: Dict[str, float] = {}
    if not rbr_cohort.empty:
        for label, grp in rbr_cohort.groupby("cohort"):
            avail = [(int(r.interval), float(r.brand_qty) / float(r.cat_qty))
                     for r in grp.itertuples(index=False) if float(r.cat_qty) > 0]
            if not avail:
                continue
            stable = ([v for i, v in avail if i >= rbr_stable_from]
                      if rbr_stable_from is not None else [])
            rbr_by_cohort[label] = (float(np.mean(stable)) if stable
                                    else max(avail, key=lambda kv: kv[0])[1])

    cohorts: List[Cohort] = []
    observed_pen = 0.0
    last_rbr: Optional[float] = None
    for label in cohort_order:
        n = int(counts.get(label, 0))
        if n == 0:
            continue
        p_i = n / n_category_triers
        observed_pen += p_i
        r_i = rbr_by_cohort.get(label)
        b_i = buying_index_from_scopes(buying_scopes, label)
        if b_i is None:
            b_i = 1.0
        if r_i is not None:
            last_rbr = r_i
        cohorts.append(Cohort(label=label, penetration=p_i, rbr=r_i,
                              buying_index=b_i, n_triers=n))

    # estimated future entrants (Table 2 row 5)
    if ultimate_penetration is not None and ultimate_penetration > observed_pen:
        cohorts.append(Cohort(label="future (estimated)",
                              penetration=ultimate_penetration - observed_pen,
                              rbr=last_rbr, buying_index=1.0, n_triers=0,
                              is_future=True))
    return cohorts


def segmented_share(cohorts: Sequence[Cohort]) -> float:
    """Σ Pᵢ × Rᵢ × Bᵢ (the Table 2 brand-share prediction)."""
    return float(sum(c.contribution for c in cohorts))


def blended_rbr(cohorts: Sequence[Cohort]) -> Optional[float]:
    """Penetration-weighted average RBR (the single 'total' RBR the paper
    reports, e.g. 15.5% in Table 2). NOT a sum of cohort RBRs."""
    num = sum(c.penetration * c.rbr for c in cohorts if c.rbr is not None)
    den = sum(c.penetration for c in cohorts if c.rbr is not None)
    return (num / den) if den > 0 else None
