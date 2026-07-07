"""Repeat-buying rate of Parfitt & Collins (1968), lite edition.

RBR(t) is the brand's share of the category volume bought by its triers in
their t-th interval after trial: Σ brand_qty(t) / Σ cat_qty(t), a RATIO OF
SUMS (never a mean of per-shopper ratios). Intervals are exact
``period_length_days``-day windows counted from each shopper's OWN trial date
(interval t covers days (t-1)*P+1 .. t*P after the trial, 1-based), so the
axis is repeat-buying time, not calendar time. Only fully-elapsed intervals
count: a shopper contributes to interval t only when its whole window fits
before the analysis date, and lapsed buyers stay in the denominator base
(``n_eligible`` comes from elapsed time, not from purchasing).

Entry cohorts (optional) are the calendar bucket of the trial date -- iso_week
/ iso_fortnight / month -- for the "do late triers repeat like the early
ones?" diagnostic.

Spark is used ONLY inside :func:`build_rbr` (per-card trial identification and
two per-interval group-bys whose collected result is one row per interval);
everything else is numpy/pandas on the small series.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .calendar import _check_unit, as_date, period_col, period_label


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
    n_eligible: int            # triers whose whole interval has elapsed


@dataclass
class RBRCurve:
    """Pooled RBR(t) on exact-day intervals from each shopper's trial, plus
    the optional per-cohort volumes behind the cohort diagnostic."""
    origin: date                          # launch (trial floor when launch_date given)
    analysis_date: date
    period_length_days: int
    points: List[RBRPoint]                # gap-free 1..upper (zero-filled)
    n_triers: int                         # ALL triers, incl. max_interval == 0
    cohort_unit: Optional[str] = None     # 'iso_week' | 'iso_fortnight' | 'month'
    cohort_table: Optional[pd.DataFrame] = None  # cohort, interval, brand_qty, cat_qty

    def to_frame(self) -> pd.DataFrame:
        """The pooled curve as a table: interval, rbr (NaN where unobserved),
        brand_qty, cat_qty, n_eligible."""
        return pd.DataFrame({
            "interval": [p.interval for p in self.points],
            "rbr": [np.nan if p.rbr is None else p.rbr for p in self.points],
            "brand_qty": [p.brand_qty for p in self.points],
            "cat_qty": [p.cat_qty for p in self.points],
            "n_eligible": [p.n_eligible for p in self.points],
        })

    def rbr_at(self, interval: int) -> Optional[float]:
        """Observed rate at one interval; None when unobserved / off the axis."""
        for p in self.points:
            if p.interval == int(interval):
                return p.rbr
        return None

    def last_available(self) -> Optional[Tuple[int, float]]:
        """(interval, rbr) at the furthest observed interval -- the r→∞ proxy."""
        return last_available_rbr(self.points)

    def plateau(self, tol: float = 0.005, k: int = 3) -> Optional[Tuple[int, float]]:
        """Diagnostic plateau, see :func:`detect_plateau`."""
        return detect_plateau(self.points, tol, k)

    def stable(self, from_interval: int) -> Optional[float]:
        """Stabilised mean rate, see :func:`stable_rbr`."""
        return stable_rbr(self.points, from_interval)

    def _require_cohorts(self) -> pd.DataFrame:
        if self.cohort_table is None:
            raise ValueError("curve built without cohorts: pass "
                             "cohort_unit=... to build_rbr")
        return self.cohort_table

    def cohort_series(self) -> Dict[str, List[Tuple[int, Optional[float]]]]:
        """Ordered {cohort label: [(interval, rbr | None), ...]} -- the full
        per-cohort curves behind the cohort diagnostic plot. rbr is None when
        no category volume is observed yet; cohorts with no rows are omitted.
        Labels ('YYYY-Www' / 'YYYY-Fww' / 'YYYY-MM') sort chronologically."""
        table = self._require_cohorts()
        out: Dict[str, List[Tuple[int, Optional[float]]]] = {}
        if table.empty:
            return out
        for label in sorted(table["cohort"].unique()):
            grp = table[table["cohort"] == label]
            pts: List[Tuple[int, Optional[float]]] = []
            for r in grp.sort_values("interval").itertuples(index=False):
                cat = float(r.cat_qty)
                pts.append((int(r.interval),
                            (float(r.brand_qty) / cat) if cat > 0 else None))
            out[str(label)] = pts
        return out

    def cohort_frame(self) -> pd.DataFrame:
        """Per-cohort volumes with the derived rate: cohort, interval,
        brand_qty, cat_qty, rbr (NaN where no category volume)."""
        out = self._require_cohorts().copy()
        out["rbr"] = [b / c if c > 0 else np.nan
                      for b, c in zip(out["brand_qty"], out["cat_qty"])]
        return out


# --------------------------------------------------------------------------- #
# Engine-free stability helpers
# --------------------------------------------------------------------------- #
def last_available_rbr(points: Sequence[RBRPoint]) -> Optional[Tuple[int, float]]:
    """The furthest interval with an observed rate -> the r→∞ proxy used for
    the ultimate RBR (no auto plateau selection; the analyst reads the plot)."""
    avail = [(p.interval, p.rbr) for p in points if p.rbr is not None]
    return max(avail, key=lambda kv: kv[0]) if avail else None


def detect_plateau(points: Sequence[RBRPoint], tol: float = 0.005,
                   k: int = 3) -> Optional[Tuple[int, float]]:
    """DIAGNOSTIC only: first interval where the rate stays within `tol` for
    `k` consecutive observations."""
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


# --------------------------------------------------------------------------- #
# The single Spark entry point
# --------------------------------------------------------------------------- #
def build_rbr(sdf, *, card_col: str = "shopper_id",
              date_col: str = "txn_date",
              brand_col: str = "is_new_product",
              category_col: str = "is_category",
              qty_col: str = "volume",
              period_length_days: int = 28,
              max_interval: Optional[int] = None,
              cohort_unit: Optional[str] = None,
              launch_date=None, analysis_date=None) -> RBRCurve:
    """Build the pooled (and optionally per-cohort) RBR curve from a Spark
    transaction log.

    The brand is treated as part of the category (a brand purchase is also a
    category purchase). Trial = each card's first brand purchase on/after the
    launch (with `launch_date` set, earlier brand history is ignored and the
    trial re-dated to the first post-launch brand purchase). Purchases on the
    trial day itself never count as repeats; interval t only enters the curve
    once its whole window has elapsed for that shopper. `max_interval` caps
    the horizon of the WHOLE analysis (pooled and cohort curves alike);
    `n_eligible` still counts every trier whose window has elapsed, including
    those observable beyond the cap.
    """
    if period_length_days <= 0:
        raise ValueError("period_length_days must be positive")
    if max_interval is not None and max_interval < 1:
        raise ValueError("max_interval must be >= 1 (an RBR interval)")
    if cohort_unit is not None:
        _check_unit(cohort_unit)
    from pyspark.sql import functions as F

    p = (sdf.withColumn("_ts", F.to_date(F.col(date_col)))
            .withColumn("_card", F.col(card_col).cast("string"))
            .withColumn("_brand", F.col(brand_col).cast("boolean"))
            .withColumn("_cat", F.col(category_col).cast("boolean")
                        | F.col(brand_col).cast("boolean"))
            .withColumn("_qty", F.col(qty_col).cast("double")))

    if analysis_date is not None:
        adate = as_date(analysis_date)
    else:
        m = p.agg(F.max("_ts").alias("m")).collect()[0]["m"]
        if m is None:
            raise ValueError("no transactions on/before the analysis date")
        adate = as_date(m)
    p = p.filter(F.col("_ts") <= F.lit(adate.isoformat()).cast("date"))
    if p.head(1) == []:
        raise ValueError("no transactions on/before the analysis date")

    if launch_date is not None:
        origin = as_date(launch_date)
    else:
        o = p.filter(F.col("_brand")).agg(F.min("_ts").alias("o")).collect()[0]["o"]
        if o is None:
            raise ValueError("cannot determine launch origin: set launch_date "
                             "or ensure there is at least one brand purchase")
        origin = as_date(o)

    # Trials: first brand purchase per card, floored at the launch. Everything
    # stays a Spark column; only per-interval aggregates are collected.
    brand = p.filter(F.col("_brand"))
    if launch_date is not None:
        brand = brand.filter(F.col("_ts") >= F.lit(origin.isoformat()).cast("date"))
    adate_lit = F.lit(adate.isoformat()).cast("date")
    trials = (brand.groupBy("_card").agg(F.min("_ts").alias("_trial_ts"))
              .withColumn("_max_interval",
                          F.floor(F.datediff(adate_lit, F.col("_trial_ts"))
                                  / period_length_days)))
    if cohort_unit is not None:
        trials = trials.withColumn(
            "_cohort_ord", period_col(F, F.col("_trial_ts"), cohort_unit, origin))
    trials = trials.cache()

    joined = None
    try:
        # One collect gives the max_interval distribution, the trier total and
        # the feasible horizon at once (one row per distinct value).
        dist = {int(r["_max_interval"]): int(r["count"]) for r in
                trials.groupBy("_max_interval").count().collect()}
        n_triers = sum(dist.values())
        if n_triers == 0:
            raise ValueError("no brand triers on/after the launch date")
        feasible = max(dist)
        upper = min(int(max_interval), feasible) if max_interval is not None else feasible

        # n_eligible(t) = #triers whose max_interval >= t. The top-down
        # cumulation is SEEDED with the triers observable beyond the cap:
        # their early intervals are inside the sums, so they belong in every
        # base at t <= upper.
        running = sum(c for v, c in dist.items() if v > upper)
        n_eligible: Dict[int, int] = {}
        for t in range(upper, 0, -1):
            running += dist.get(t, 0)
            n_eligible[t] = running

        # Post-trial lines of every trier with their interval, the per-trier
        # eligibility filter (only fully-elapsed windows; lapsed buyers stay
        # in the base) and the analysis-horizon cap.
        cols = ["_card", "_trial_ts", "_max_interval"]
        if cohort_unit is not None:
            cols.append("_cohort_ord")
        joined = (p.join(trials.select(*cols), on="_card", how="inner")
                  .filter(F.datediff(F.col("_ts"), F.col("_trial_ts")) > 0)
                  .withColumn("_interval",
                              F.ceil(F.datediff(F.col("_ts"), F.col("_trial_ts"))
                                     / period_length_days))
                  .filter((F.col("_interval") >= 1)
                          & (F.col("_interval") <= F.col("_max_interval"))
                          & (F.col("_interval") <= F.lit(upper)))
                  .withColumn("_bq", F.when(F.col("_brand"), F.col("_qty")).otherwise(0.0))
                  .withColumn("_cq", F.when(F.col("_cat"), F.col("_qty")).otherwise(0.0))
                  ).cache()

        agg = {int(r["_interval"]): (float(r["b"]), float(r["c"])) for r in
               joined.groupBy("_interval").agg(F.sum("_bq").alias("b"),
                                               F.sum("_cq").alias("c")).collect()}
        points: List[RBRPoint] = []
        for t in range(1, upper + 1):
            b, c = agg.get(t, (0.0, 0.0))
            points.append(RBRPoint(interval=t, rbr=(b / c) if c > 0 else None,
                                   brand_qty=b, cat_qty=c,
                                   n_eligible=n_eligible.get(t, 0)))

        cohort_table = None
        if cohort_unit is not None:
            rows = (joined.groupBy("_cohort_ord", "_interval")
                    .agg(F.sum("_bq").alias("brand_qty"),
                         F.sum("_cq").alias("cat_qty")).toPandas())
            if rows.empty:
                cohort_table = pd.DataFrame(
                    columns=["cohort", "interval", "brand_qty", "cat_qty"])
            else:
                rows["cohort"] = [period_label(int(o), origin, cohort_unit)
                                  for o in rows["_cohort_ord"]]
                rows["interval"] = rows["_interval"].astype(int)
                cohort_table = (rows[["cohort", "interval", "brand_qty", "cat_qty"]]
                                .sort_values(["cohort", "interval"])
                                .reset_index(drop=True))
    finally:
        trials.unpersist()
        if joined is not None:
            joined.unpersist()

    return RBRCurve(origin=origin, analysis_date=adate,
                    period_length_days=int(period_length_days), points=points,
                    n_triers=n_triers, cohort_unit=cohort_unit,
                    cohort_table=cohort_table)
