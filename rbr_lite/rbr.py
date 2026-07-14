"""Repeat-buying rate of Parfitt & Collins (1968), lite edition.

RBR(t) is the brand's share of the category volume bought by its triers in
their t-th interval after trial: Σ brand_qty(t) / Σ cat_qty(t), a RATIO OF
SUMS (never a mean of per-shopper ratios). The axis is repeat-buying time,
not calendar time, in one of two flavours: exact ``period_length_days``-day
windows counted from each shopper's OWN trial date (interval t covers days
(t-1)*P+1 .. t*P after the trial, 1-based), or -- with ``interval_unit`` set
-- calendar-bucket differences from the trial's bucket (the next iso_week /
iso_fortnight / month is interval 1; purchases in the trial's own bucket are
never repeats). Only fully-elapsed intervals count: a shopper contributes to
interval t only when its whole window (or bucket) fits before the analysis
date, and lapsed buyers stay in the denominator base (``n_eligible`` comes
from elapsed time, not from purchasing).

Entry cohorts (optional) are custom bands of the trial date delimited by
``cohort_boundaries`` -- period labels or dates, each closing a band
inclusively -- for the "do late triers repeat like the early ones?"
diagnostic.

Spark is used ONLY inside :func:`build_rbr` (per-card trial identification and
two per-interval group-bys whose collected result is one row per interval);
everything else is numpy/pandas on the small series.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .calendar import (
    _check_unit, as_date, boundary_end, is_period_label, label_after,
    period_col, period_of,
)


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
    """Pooled RBR(t) on exact-day or calendar-bucket intervals from each
    shopper's trial, plus the optional per-band volumes behind the cohort
    diagnostic."""
    origin: date                          # launch (trial floor when launch_date given)
    analysis_date: date
    period_length_days: Optional[int]     # exact-day window; None in bucket mode
    points: List[RBRPoint]                # gap-free 1..upper (zero-filled)
    n_triers: int                         # ALL triers, incl. max_interval == 0
    interval_unit: Optional[str] = None   # 'iso_week' | 'iso_fortnight' | 'month'
    cohort_labels: Optional[List[str]] = None    # ALL bands, band order
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
                             "cohort_boundaries=[...] to build_rbr")
        return self.cohort_table

    def cohort_series(self) -> Dict[str, List[Tuple[int, Optional[float]]]]:
        """Ordered {band label: [(interval, rbr | None), ...]} -- the full
        per-band curves behind the cohort diagnostic plot. rbr is None when
        no category volume is observed yet; bands with no rows are omitted.
        Iteration follows band order (`cohort_labels`), never label sort."""
        table = self._require_cohorts()
        out: Dict[str, List[Tuple[int, Optional[float]]]] = {}
        if table.empty:
            return out
        order = (self.cohort_labels if self.cohort_labels is not None
                 else list(pd.unique(table["cohort"])))
        for label in order:
            grp = table[table["cohort"] == label]
            if grp.empty:
                continue
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
# Cohort boundary resolution (pure Python, fail-fast before Spark is touched)
# --------------------------------------------------------------------------- #
def _resolve_cohort_boundaries(boundaries) -> Tuple[List[date], List[str]]:
    """Boundary tokens -> (inclusive end dates, band labels). N boundaries
    delimit N+1 bands: band k holds triers with end_{k-1} < trial <= end_k
    (the first band starts at the origin, the last is open-ended)."""
    bs = list(boundaries)
    if not bs:
        raise ValueError("cohort_boundaries must contain at least one "
                         "boundary (pass None to skip cohorts)")
    ends = [boundary_end(b) for b in bs]
    for prev_b, prev_e, cur_b, cur_e in zip(bs, ends, bs[1:], ends[1:]):
        if prev_e >= cur_e:
            raise ValueError(
                "cohort_boundaries must resolve to strictly increasing end "
                f"dates: {prev_b!r} -> {prev_e} is not before "
                f"{cur_b!r} -> {cur_e}")
    return ends, _band_labels(bs)


def _band_labels(bs) -> List[str]:
    """Human-readable band labels, e.g. ['≤2023-W31', '2023-W32–2023-W38',
    '2023-W39+']: each boundary rendered as typed (labels keep their grammar,
    dates -> ISO), band starts derived from the previous boundary."""
    def tok(b):
        return b.strip() if is_period_label(b) else as_date(b).isoformat()

    def nxt(b):   # first token AFTER the boundary: next bucket, or end + 1 day
        return (label_after(b) if is_period_label(b)
                else (as_date(b) + timedelta(days=1)).isoformat())

    labels = [f"≤{tok(bs[0])}"]
    labels += [f"{nxt(p)}–{tok(c)}" for p, c in zip(bs, bs[1:])]
    labels.append(f"{nxt(bs[-1])}+")
    return labels


# --------------------------------------------------------------------------- #
# The single Spark entry point
# --------------------------------------------------------------------------- #
def build_rbr(sdf, *, card_col: str = "shopper_id",
              date_col: str = "txn_date",
              brand_col: str = "is_new_product",
              category_col: str = "is_category",
              qty_col: str = "volume",
              period_length_days: Optional[int] = None,
              interval_unit: Optional[str] = None,
              max_interval: Optional[int] = None,
              cohort_boundaries: Optional[Sequence] = None,
              launch_date=None, analysis_date=None) -> RBRCurve:
    """Build the pooled (and optionally per-cohort) RBR curve from a Spark
    transaction log.

    The brand is treated as part of the category (a brand purchase is also a
    category purchase). Trial = each card's first brand purchase on/after the
    launch (with `launch_date` set, earlier brand history is ignored and the
    trial re-dated to the first post-launch brand purchase). Purchases on the
    trial day itself never count as repeats; interval t only enters the curve
    once its whole window has elapsed for that shopper.

    Two interval axes: with `interval_unit=None` intervals are exact
    `period_length_days`-day windows from the trial (default 28); with
    `interval_unit` set ('iso_week' / 'iso_fortnight' / 'month') the interval
    is the calendar-bucket difference from the trial's bucket -- purchases in
    the trial's own bucket are never repeats, and a bucket counts only once
    it has FULLY elapsed by the analysis date (a partial current bucket is
    excluded). `period_length_days` must not be passed in bucket mode.

    `cohort_boundaries` (optional) splits triers into entry bands: each
    boundary -- a period label ('YYYY-Www' / 'YYYY-Fww' / 'YYYY-MM', mixed
    grammars allowed) or a date -- closes a band at the last day of its
    bucket (labels) or at itself (dates), inclusive. N boundaries give N+1
    bands; the first starts at the origin, the last stays open until the
    analysis date.

    The horizon is chosen with `analysis_date` (observation cutoff, default =
    last transaction) and `max_interval`, which caps the axis of the WHOLE
    analysis (pooled and cohort curves alike) -- e.g. interval_unit='month',
    max_interval=20 gives at most 20 monthly points. `n_eligible` still
    counts every trier whose window has elapsed, including those observable
    beyond the cap.
    """
    if interval_unit is not None:
        _check_unit(interval_unit)
        if period_length_days is not None:
            raise ValueError("period_length_days does not apply when "
                             "interval_unit is set: bucket intervals have "
                             "calendar length")
        plen = None
    else:
        plen = 28 if period_length_days is None else int(period_length_days)
        if plen <= 0:
            raise ValueError("period_length_days must be positive")
    if max_interval is not None and max_interval < 1:
        raise ValueError("max_interval must be >= 1 (an RBR interval)")
    ends = band_labels = None
    if cohort_boundaries is not None:
        ends, band_labels = _resolve_cohort_boundaries(cohort_boundaries)
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

    if ends is not None:
        bs = list(cohort_boundaries)
        if ends[0] < origin:
            raise ValueError(
                f"cohort boundary {bs[0]!r} resolves to {ends[0]}, before "
                f"the origin {origin}: its band would be empty")
        if ends[-1] >= adate:
            raise ValueError(
                f"cohort boundary {bs[-1]!r} resolves to {ends[-1]}, on/after "
                f"the analysis date {adate}: the open last band would be empty")

    # Trials: first brand purchase per card, floored at the launch. Everything
    # stays a Spark column; only per-interval aggregates are collected.
    brand = p.filter(F.col("_brand"))
    if launch_date is not None:
        brand = brand.filter(F.col("_ts") >= F.lit(origin.isoformat()).cast("date"))
    adate_lit = F.lit(adate.isoformat()).cast("date")
    trials = brand.groupBy("_card").agg(F.min("_ts").alias("_trial_ts"))
    if interval_unit is None:
        trials = trials.withColumn(
            "_max_interval",
            F.floor(F.datediff(adate_lit, F.col("_trial_ts")) / plen))
    else:
        # Buckets FULLY elapsed by the analysis date: (B(adate+1d) - 1) -
        # B(trial). The +1 day rolls the bucket index forward exactly when
        # adate closes its bucket (month ends included), so a partial current
        # bucket never counts. Can be negative for a trial inside adate's
        # unfinished bucket; the dist / seed / filters below absorb that.
        k = period_of(adate + timedelta(days=1), origin, interval_unit) - 1
        trials = trials.withColumn(
            "_max_interval",
            F.lit(int(k)) - period_col(F, F.col("_trial_ts"),
                                       interval_unit, origin))
    if ends is not None:
        band = F.when(F.col("_trial_ts")
                      <= F.lit(ends[0].isoformat()).cast("date"), F.lit(0))
        for i in range(1, len(ends)):
            band = band.when(F.col("_trial_ts")
                             <= F.lit(ends[i].isoformat()).cast("date"), F.lit(i))
        trials = trials.withColumn("_cohort_idx",
                                   band.otherwise(F.lit(len(ends))))
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
        if ends is not None:
            cols.append("_cohort_idx")
        if interval_unit is None:
            ival = F.ceil(F.datediff(F.col("_ts"), F.col("_trial_ts")) / plen)
        else:
            ival = (period_col(F, F.col("_ts"), interval_unit, origin)
                    - period_col(F, F.col("_trial_ts"), interval_unit, origin))
        joined = (p.join(trials.select(*cols), on="_card", how="inner")
                  .filter(F.datediff(F.col("_ts"), F.col("_trial_ts")) > 0)
                  .withColumn("_interval", ival)
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
        if ends is not None:
            rows = (joined.groupBy("_cohort_idx", "_interval")
                    .agg(F.sum("_bq").alias("brand_qty"),
                         F.sum("_cq").alias("cat_qty")).toPandas())
            if rows.empty:
                cohort_table = pd.DataFrame(
                    columns=["cohort", "interval", "brand_qty", "cat_qty"])
            else:
                rows["cohort"] = [band_labels[int(i)]
                                  for i in rows["_cohort_idx"]]
                rows["interval"] = rows["_interval"].astype(int)
                cohort_table = (rows.sort_values(["_cohort_idx", "interval"])
                                [["cohort", "interval", "brand_qty", "cat_qty"]]
                                .reset_index(drop=True))
    finally:
        trials.unpersist()
        if joined is not None:
            joined.unpersist()

    return RBRCurve(origin=origin, analysis_date=adate,
                    period_length_days=plen, points=points,
                    n_triers=n_triers, interval_unit=interval_unit,
                    cohort_labels=band_labels, cohort_table=cohort_table)
