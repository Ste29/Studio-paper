"""Backend-free modelling core for the Parfitt-Collins (TRB) model.

Every function here consumes the small, card-collapsed tables produced by an
:class:`~parfitt_trb.aggregation.Aggregator` (pandas or Spark) and returns plain
numpy / pandas / dataclasses. There is no DataFrame-engine dependency, so the
maths is written once and shared by both backends.

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


def fit_penetration(pen: Penetration, *, method: str = "discounted",
                    discount_weight: float = 0.6) -> Penetration:
    """Estimate K, a from the printed difference model ΔP(t)=a(K-P(t))+ε.

    Centred difference ΔP(t)=(P(t+1)-P(t-1))/2; regress ΔP on P (slope=-a,
    intercept=aK) by discounted least squares (recent points weighted w^age) or
    plain OLS. Mutates `pen` in place and returns it.
    """
    series = pen.series
    if len(series) < 4:
        pen.note = "need >=4 periods to fit; too few observed"
        return pen
    ps = np.array([p for _, p in series], dtype=float)
    dP = (ps[2:] - ps[:-2]) / 2.0          # interior centred difference
    x = ps[1:-1]
    if len(x) < 2:
        pen.note = "too few interior points to fit"
        return pen

    _rw = (getattr(getattr(np, "exceptions", None), "RankWarning", None)
           or getattr(np, "RankWarning", Warning))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", _rw)
        if method == "discounted":
            ages = np.arange(len(x) - 1, -1, -1, dtype=float)   # latest -> age 0
            w = np.sqrt(discount_weight ** ages)
            slope, intercept = np.polyfit(x, dP, 1, w=w)
        else:
            slope, intercept = np.polyfit(x, dP, 1)
    a = -slope
    if a <= 0:
        pen.note = ("rate of increase not yet declining (a<=0): cannot project "
                    "ultimate penetration -- wait for the curve to decelerate")
        return pen
    pen.growth_rate = float(a)
    pen.ultimate_penetration = float(intercept / a)
    if pen.ultimate_penetration < max(ps):
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
                          project_to: Optional[int] = None) -> PenetrationPromo:
    """Fit K,a on periods <= cutoff (pre-promo), project forward, and compare
    with the realised curve; also re-fit on the full series (post-promo).

    Quantifies the 'bought' penetration = realised - baseline-projected at the
    last observed period (Parfitt's Figures 12-15 / appendix refinement).
    """
    full = pen.series
    last = full[-1][0]
    if project_to is None:
        project_to = last

    pre = Penetration(pen.denominator, pen.origin,
                      [(t, p) for t, p in full if t <= cutoff_period],
                      pen.n_brand_triers, pen.n_category_triers)
    fit_penetration(pre, method=method, discount_weight=discount_weight)
    post = Penetration(pen.denominator, pen.origin, list(full),
                       pen.n_brand_triers, pen.n_category_triers)
    fit_penetration(post, method=method, discount_weight=discount_weight)

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
                  ultimate_penetration: Optional[float] = None) -> List[Cohort]:
    """Assemble the per-cohort Pᵢ/Rᵢ/Bᵢ rows and the estimated future cohort.

    Pᵢ = brand triers entering in cohort i / F_tot (so Σ observed Pᵢ = snapshot).
    Rᵢ = cohort's furthest-available RBR. Bᵢ = cohort buying index.
    Future cohort: P=K-Σobserved, R=last cohort's R, B=1.0 (point 5 / Table 2).

    `cohort_counts` maps each entry-cohort label to its number of brand triers.
    """
    counts = dict(cohort_counts)
    # furthest available RBR per cohort
    rbr_by_cohort: Dict[str, Tuple[int, float]] = {}
    if not rbr_cohort.empty:
        for label, grp in rbr_cohort.groupby("cohort"):
            avail = [(int(r.interval), float(r.brand_qty) / float(r.cat_qty))
                     for r in grp.itertuples(index=False) if float(r.cat_qty) > 0]
            if avail:
                rbr_by_cohort[label] = max(avail, key=lambda kv: kv[0])

    cohorts: List[Cohort] = []
    observed_pen = 0.0
    last_rbr: Optional[float] = None
    for label in cohort_order:
        n = int(counts.get(label, 0))
        if n == 0:
            continue
        p_i = n / n_category_triers
        observed_pen += p_i
        r_i = rbr_by_cohort.get(label, (None, None))[1]
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
