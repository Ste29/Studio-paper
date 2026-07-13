"""Buying-rate index of Parfitt & Collins (1968), lite edition.

B compares how heavily the brand's buyers use the CATEGORY against the average
category buyer: B = (category volume per capita of the brand's buyers) /
(category volume per capita of all category buyers), both measured on the SAME
time window. Two membership bases are always computed: the brand triers
(Parfitt's original) and the repeaters (Charan's restatement: >=
``repeater_min_purchases`` brand purchase LINES).

Following the paper ("the amount of Field A purchased in period s beginning at
time t - s"), the headline B is measured on a window of ``window_days`` days
ending at the analysis date. The window narrows the VOLUME being averaged,
never the membership bases: members are everyone seen up to the analysis date,
and a member with no purchases inside the window weighs 0 instead of dropping
out. ``window_days`` is required; pass ``None`` explicitly for the
all-history-from-launch variant (the parfitt_trb / monolith default).

The per-bucket series B(t) is the same index evaluated on each calendar bucket
with GROWING bases: at bucket t the triers / repeaters / category buyers are
the members seen up to the END of bucket t (the paper's reading) -- NOT the
whole dataset's members as in parfitt_trb's fixed-panel diagnostic, which also
counts the pre-trial category volume of future triers in the early buckets.
Membership is resolved at bucket granularity: a card trialling mid-bucket
counts as a trier for that whole bucket. B(t) doubles as the stability
diagnostic -- there is no fit to converge, so watching B(t) settle IS watching
the estimate stabilise.

Spark is used ONLY inside :func:`build_buying_index` (per-card firsts and a
handful of per-bucket group-bys, each collected as one row per bucket);
everything else is numpy/pandas on the small series.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .calendar import (_check_unit, as_date, bucket_start, period_col,
                       period_label, period_of)


# --------------------------------------------------------------------------- #
# Result structures
# --------------------------------------------------------------------------- #
@dataclass
class BuyingPoint:
    """One per-bucket buying-index observation B(t) on growing bases."""
    period: int
    b_triers: Optional[float]      # None before the first trier / empty bucket
    b_repeaters: Optional[float]   # None before the first repeater / empty bucket
    cat_qty: float                 # bucket category volume, all buyers
    cat_qty_triers: float          # ... bought by cards already triers at t
    cat_qty_repeaters: float       # ... bought by cards already repeaters at t
    n_buyers: int                  # category buyers seen up to the end of t
    n_triers: int                  # triers seen up to the end of t
    n_repeaters: int               # repeaters seen up to the end of t


@dataclass
class BuyingIndex:
    """The windowed headline B (triers and repeaters bases) plus the growing-
    base per-bucket series B(t), on one of the calendar-bucket axes."""
    origin: date                          # launch (bucket 1 contains it)
    analysis_date: date
    unit: str                             # 'iso_week' | 'iso_fortnight' | 'month'
    window_days: Optional[int]            # None = all history since launch
    repeater_min_purchases: int
    b_triers: float                       # headline B on the window (Parfitt)
    b_repeaters: Optional[float]          # Charan variant; None if no repeaters
    n_buyers: int                         # category buyers up to the analysis date
    n_triers: int
    n_repeaters: int
    window_cat_qty: float                 # category volume inside the window
    window_cat_qty_triers: float
    window_cat_qty_repeaters: float
    points: List[BuyingPoint]             # gap-free 1..P (zero-filled)
    cohort_unit: Optional[str] = None     # 'iso_week' | 'iso_fortnight' | 'month'
    cohort_table: Optional[pd.DataFrame] = None  # cohort, n_triers, cat_qty, b

    @property
    def window_start(self) -> date:
        """First day whose volume enters the headline B (never before launch)."""
        if self.window_days is None:
            return self.origin
        return max(self.origin,
                   self.analysis_date - timedelta(days=self.window_days - 1))

    @property
    def last_bucket_partial(self) -> bool:
        """True when the analysis date falls mid-bucket, so the last B(t)
        point observes only part of its bucket."""
        last = self.points[-1].period
        last_day = bucket_start(last + 1, self.origin, self.unit) - timedelta(days=1)
        return self.analysis_date < last_day

    def label(self, period: int) -> str:
        """Calendar label of a period ordinal (works for future periods too)."""
        return period_label(period, self.origin, self.unit)

    def _print_partial_note(self) -> None:
        if self.last_bucket_partial:
            last = self.points[-1].period
            print(f"note: the last bucket {self.label(last)} is PARTIAL -- "
                  f"observed only through {self.analysis_date.isoformat()}")

    def to_frame(self) -> pd.DataFrame:
        """The B(t) series as a table: period, label, b_triers, b_repeaters,
        the bucket volumes and the growing bases (None -> NaN)."""
        self._print_partial_note()
        return pd.DataFrame({
            "period": [p.period for p in self.points],
            "label": [self.label(p.period) for p in self.points],
            "b_triers": [np.nan if p.b_triers is None else p.b_triers
                         for p in self.points],
            "b_repeaters": [np.nan if p.b_repeaters is None else p.b_repeaters
                            for p in self.points],
            "cat_qty": [p.cat_qty for p in self.points],
            "cat_qty_triers": [p.cat_qty_triers for p in self.points],
            "cat_qty_repeaters": [p.cat_qty_repeaters for p in self.points],
            "n_buyers": [p.n_buyers for p in self.points],
            "n_triers": [p.n_triers for p in self.points],
            "n_repeaters": [p.n_repeaters for p in self.points],
        })

    def summary(self) -> pd.DataFrame:
        """Ingredients of the headline B, one row per scope: the all-time
        member counts, the window volume, the per-capita average and the index
        (1.0 for 'all' by definition)."""
        rows = []
        for scope, n, q, b in (
                ("all", self.n_buyers, self.window_cat_qty, 1.0),
                ("triers", self.n_triers, self.window_cat_qty_triers,
                 self.b_triers),
                ("repeaters", self.n_repeaters, self.window_cat_qty_repeaters,
                 self.b_repeaters)):
            rows.append((scope, n, q, (q / n) if n else np.nan,
                         np.nan if b is None else b))
        return pd.DataFrame(rows, columns=["scope", "n_members", "cat_qty",
                                           "avg_per_member", "b"])

    def cohort_frame(self) -> pd.DataFrame:
        """Per-entry-cohort B_i on the SAME window as the headline B, with the
        cohort's FULL trier membership as base: cohort, n_triers, cat_qty, b."""
        if self.cohort_table is None:
            raise ValueError("built without cohorts: pass "
                             "cohort_unit=... to build_buying_index")
        return self.cohort_table.copy()


# --------------------------------------------------------------------------- #
# The single Spark entry point
# --------------------------------------------------------------------------- #
def build_buying_index(sdf, *, window_days: Optional[int],
                       card_col: str = "shopper_id",
                       date_col: str = "txn_date",
                       brand_col: str = "is_new_product",
                       category_col: str = "is_category",
                       qty_col: str = "volume",
                       unit: str = "iso_week",
                       repeater_min_purchases: int = 2,
                       cohort_unit: Optional[str] = None,
                       launch_date=None, analysis_date=None) -> BuyingIndex:
    """Build the buying index (headline windowed B + growing-base B(t) series)
    from a Spark transaction log.

    The brand is treated as part of the category (a brand purchase is also a
    category purchase). Trial = each card's first brand purchase on/after the
    launch; a repeater is a card with >= `repeater_min_purchases` brand
    purchase LINES, member from the day of the threshold-crossing line.
    Pre-launch history never counts -- not as volume, not as membership.
    `window_days` is deliberately REQUIRED (the paper measures B on a recent
    period ending at the analysis date); pass None for all history since
    launch. The series always uses one bucket per point; membership within a
    bucket is resolved at bucket granularity.
    """
    if window_days is not None and int(window_days) < 1:
        raise ValueError("window_days must be >= 1 (days), or None for all "
                         "history since the launch")
    if repeater_min_purchases < 2:
        raise ValueError("a repeater needs at least 2 brand purchases")
    _check_unit(unit)
    if cohort_unit is not None:
        _check_unit(cohort_unit)
    from pyspark.sql import Window, functions as F

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
    # Pre-launch history never counts: volumes, trials, repeater lines and
    # buyer membership all start at the origin.
    d = p.filter(F.col("_ts") >= F.lit(origin.isoformat()).cast("date"))

    # Per-card membership dates: first brand line (trial), first category line
    # (buyer), and the line crossing the repeater threshold -- everything as
    # period ordinals; only one row per card survives, never collected.
    firsts = d.groupBy("_card").agg(
        F.min(F.when(F.col("_brand"), F.col("_ts"))).alias("_trial_ts"),
        F.min(F.when(F.col("_cat"), F.col("_ts"))).alias("_cat_ts"))
    kth = (d.filter(F.col("_brand"))
           .withColumn("_rn", F.row_number().over(
               Window.partitionBy("_card").orderBy("_ts")))
           .filter(F.col("_rn") == int(repeater_min_purchases))
           .select("_card", F.col("_ts").alias("_rep_ts")))
    members = (firsts.join(kth, on="_card", how="left")
               .withColumn("_trial_p", period_col(F, F.col("_trial_ts"), unit, origin))
               .withColumn("_cat_p", period_col(F, F.col("_cat_ts"), unit, origin))
               .withColumn("_rep_p", period_col(F, F.col("_rep_ts"), unit, origin)))
    if cohort_unit is not None:
        members = members.withColumn(
            "_coh", period_col(F, F.col("_trial_ts"), cohort_unit, origin))
    members = members.cache()

    cat = None
    try:
        # Membership distributions (tiny: one row per bucket) -> totals and,
        # cumulated in Python, the growing bases of the series.
        def _dist(col: str) -> Dict[int, int]:
            rows = (members.filter(F.col(col).isNotNull())
                    .groupBy(col).count().collect())
            return {int(r[col]): int(r["count"]) for r in rows}

        trial_dist, cat_dist, rep_dist = (_dist("_trial_p"), _dist("_cat_p"),
                                          _dist("_rep_p"))
        n_triers = sum(trial_dist.values())
        if n_triers == 0:
            raise ValueError("no brand triers on/after the launch date")
        n_buyers, n_repeaters = sum(cat_dist.values()), sum(rep_dist.values())

        upper = period_of(adate, origin, unit)      # bucket containing analysis
        upto: Dict[str, List[int]] = {}             # cumulative bases, index t-1
        for key, dist in (("buyers", cat_dist), ("triers", trial_dist),
                          ("repeaters", rep_dist)):
            running, cums = 0, []
            for t in range(1, upper + 1):
                running += dist.get(t, 0)
                cums.append(running)
            upto[key] = cums

        # Category lines with their bucket and their card's membership buckets:
        # one shuffle feeds the series, the window aggregate and the cohorts.
        mcols = ["_card", "_trial_p", "_rep_p"] + \
                (["_coh"] if cohort_unit is not None else [])
        cat = (d.filter(F.col("_cat"))
               .withColumn("_p", period_col(F, F.col("_ts"), unit, origin))
               .join(members.select(*mcols), on="_card", how="left")).cache()

        ser: Dict[int, Tuple[float, float, float]] = {
            int(r["_p"]): (float(r["aq"] or 0.0), float(r["tq"] or 0.0),
                           float(r["rq"] or 0.0))
            for r in cat.groupBy("_p").agg(
                F.sum("_qty").alias("aq"),
                F.sum(F.when(F.col("_trial_p") <= F.col("_p"),
                             F.col("_qty"))).alias("tq"),
                F.sum(F.when(F.col("_rep_p") <= F.col("_p"),
                             F.col("_qty"))).alias("rq")).collect()}

        points: List[BuyingPoint] = []
        for t in range(1, upper + 1):
            aq, tq, rq = ser.get(t, (0.0, 0.0, 0.0))
            nb, nt, nr = (upto["buyers"][t - 1], upto["triers"][t - 1],
                          upto["repeaters"][t - 1])
            avg_all = (aq / nb) if aq > 0 and nb else None
            points.append(BuyingPoint(
                period=t,
                b_triers=(tq / nt) / avg_all if avg_all and nt else None,
                b_repeaters=(rq / nr) / avg_all if avg_all and nr else None,
                cat_qty=aq, cat_qty_triers=tq, cat_qty_repeaters=rq,
                n_buyers=nb, n_triers=nt, n_repeaters=nr))

        # Headline B: window volume over ALL-TIME bases (a member silent in
        # the window weighs 0). The floor at the origin is already in `d`.
        win = cat
        if window_days is not None:
            wstart = adate - timedelta(days=int(window_days))
            win = cat.filter(F.col("_ts") > F.lit(wstart.isoformat()).cast("date"))
        row = win.agg(
            F.sum("_qty").alias("aq"),
            F.sum(F.when(F.col("_trial_p").isNotNull(), F.col("_qty"))).alias("tq"),
            F.sum(F.when(F.col("_rep_p").isNotNull(), F.col("_qty"))).alias("rq"),
        ).collect()[0]
        w_all = float(row["aq"] or 0.0)
        w_tri, w_rep = float(row["tq"] or 0.0), float(row["rq"] or 0.0)
        if w_all <= 0:
            raise ValueError("no category volume inside the buying-index "
                             "window: widen window_days")
        avg_all_w = w_all / n_buyers
        b_triers = (w_tri / n_triers) / avg_all_w
        b_repeaters = ((w_rep / n_repeaters) / avg_all_w
                       if n_repeaters else None)

        cohort_table = None
        if cohort_unit is not None:
            counts = {int(r["_coh"]): int(r["count"]) for r in
                      (members.filter(F.col("_trial_p").isNotNull())
                       .groupBy("_coh").count().collect())}
            vols = {int(r["_coh"]): float(r["s"] or 0.0) for r in
                    (win.filter(F.col("_coh").isNotNull())
                     .groupBy("_coh").agg(F.sum("_qty").alias("s")).collect())}
            rows = [(period_label(o, origin, cohort_unit), counts[o],
                     vols.get(o, 0.0),
                     (vols.get(o, 0.0) / counts[o]) / avg_all_w)
                    for o in sorted(counts)]
            cohort_table = pd.DataFrame(
                rows, columns=["cohort", "n_triers", "cat_qty", "b"])
    finally:
        members.unpersist()
        if cat is not None:
            cat.unpersist()

    return BuyingIndex(
        origin=origin, analysis_date=adate, unit=unit,
        window_days=None if window_days is None else int(window_days),
        repeater_min_purchases=int(repeater_min_purchases),
        b_triers=b_triers, b_repeaters=b_repeaters,
        n_buyers=n_buyers, n_triers=n_triers, n_repeaters=n_repeaters,
        window_cat_qty=w_all, window_cat_qty_triers=w_tri,
        window_cat_qty_repeaters=w_rep, points=points,
        cohort_unit=cohort_unit, cohort_table=cohort_table)
