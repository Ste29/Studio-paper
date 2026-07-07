"""The single Spark aggregator: turns a raw transaction log into the small,
card-collapsed modelling tables.

EVERY heavy join / group-by runs in Spark, and only the already-aggregated result
(one row per interval / period / cohort / scope) is ``.toPandas()``-ed. No
transaction-level or per-card frame is ever pulled to the driver, so it scales to
a real single-retailer panel of millions of lines.

Contract (all small pandas DataFrames):
  entrants()    : period, n_brand_new, n_cat_new
  rbr_pooled()  : interval, brand_qty, cat_qty, n_eligible
  rbr_cohort()  : cohort, interval, brand_qty, cat_qty
  buying_scopes(): scope, sum_cat, n_buyers   ('__all__' = everyone; n_buyers is
                   the scope's FULL membership, not the window-active buyers)
  buying_series(): period, sel_sum, sel_n, all_sum, all_n   (fixed panel: the
                   n columns are the constant all-time base sizes)
  share_long()  : period, brand_qty, cat_qty
  cohort_counts(): {cohort label -> n triers}
plus the calendar-label maps period_labels() / share_period_labels().
"""
from __future__ import annotations

from typing import Dict

import pandas as pd

from ..cohorts import cohort_order
from ..config import TRBConfig
from ..periods import as_date, month_index
from ._expr import _cohort_col, _interval_col, _max_interval_col, _period_col
from .calendar import _period_labels, axis_maps


class SparkAggregator:
    """Turn a Spark transaction DataFrame into the small modelling tables.

    Heavy work (trial identification, interval/period assignment, the RBR and
    buying group-bys) all runs in Spark; only aggregated results are collected.
    """

    def __init__(self, sdf, cfg: TRBConfig):
        from pyspark.sql import functions as F

        self.cfg = cfg
        self._F = F
        p = self._prepare(sdf)
        adate = (cfg.analysis_date if cfg.analysis_date
                 else p.agg(F.max("ts").alias("m")).collect()[0]["m"])
        self.analysis_date = as_date(adate)
        self._adate = self.analysis_date.isoformat()
        self._adate_month = month_index(self.analysis_date)
        p = p.filter(F.col("ts") <= F.lit(self._adate).cast("date")).cache()
        if p.head(1) == []:
            raise ValueError("no transactions on/before the analysis date")
        self._p = p

        if cfg.launch_date:
            origin = as_date(cfg.launch_date)
        else:
            o = (p.filter(F.col("is_brand")).agg(F.min("ts").alias("o"))
                 .collect()[0]["o"])
            origin = as_date(o) if o is not None else None
        if origin is None:
            raise ValueError("cannot determine launch origin: set launch_date "
                             "or ensure there is at least one brand purchase")
        self.origin = origin
        self._origin = origin.isoformat()
        self._build_calendar_axis()
        self._trials = self._build_trials().cache()       # Spark DF, never collected
        self.n_category_triers = self._n_cat_triers()
        self._tj = None                                    # lazily built RBR join

    def close(self) -> None:
        """Release the cached Spark DataFrames once the small tables are out."""
        for df in (self._tj, getattr(self, "_trials", None), getattr(self, "_p", None)):
            if df is not None:
                df.unpersist()

    # -- preparation -------------------------------------------------------- #
    def _prepare(self, sdf):
        F, c = self._F, self.cfg
        out = (sdf.withColumn("ts", F.to_date(F.col(c.date_column)))
                  .withColumn("card", F.col(c.card_column).cast("string"))
                  .withColumn("is_brand", F.col(c.brand_column).cast("boolean"))
                  .withColumn("qty", F.col(c.measure).cast("double")))
        cat = F.col(c.category_column).cast("boolean")
        if c.treat_brand_as_category:
            cat = cat | F.col("is_brand")
        out = out.withColumn("is_cat", cat)
        # Collapse to the (card, day, brand/category) grain, summing qty. Sums and
        # distinct-card counts are unchanged (the downstream aggregations are all
        # sums or countDistinct), so this never alters a result; it just makes the
        # working grain explicit and avoids carrying duplicate transaction lines.
        # NOTE: this is not de-duplication -- genuinely duplicated lines have their
        # qty summed (collapsed), not dropped.
        grain = ["card", "ts", "is_brand", "is_cat"]
        return out.groupBy(*grain).agg(F.sum("qty").alias("qty"))

    # -- calendar axis ------------------------------------------------------ #
    def _build_calendar_axis(self) -> None:
        """Prepare the calendar-time period axes. Cohorts and RBR are NOT on them.

        Main axis (penetration / buying_series / entrants): period_unit.
        Share axis (share_long only): mirrors the main axis unless share_period_unit
        overrides it. Calendar-anchored units (iso_week / fiscal_445) get a gap-free
        ordinal map -- built for both axes in a SINGLE date-dimension pass; the
        derived week/month units need no map."""
        c = self.cfg
        self.period_unit = c.period_unit
        self.share_period_unit = c.share_period_unit or c.period_unit
        self._origin_month = month_index(self.origin)
        maps = axis_maps(self._p.sparkSession, self._origin, self._adate,
                         (self.period_unit, self.share_period_unit))
        self._bucket_to_period = maps.get(self.period_unit, {})
        proxy = (c if c.share_period_unit is None
                 else TRBConfig(**{**c.__dict__, "period_unit": c.share_period_unit,
                                   "share_period_unit": None}))
        self._share_axis = {"cfg_proxy": proxy,
                            "bucket_to_period": maps.get(self.share_period_unit, {})}

    def _main_period_col(self, ts_col=None):
        F = self._F
        return _period_col(F, ts_col if ts_col is not None else F.col("ts"),
                           self.cfg, self._origin, self._origin_month,
                           self._bucket_to_period)

    def period_labels(self, periods) -> dict:
        return _period_labels(periods, self.cfg, self.origin, self._bucket_to_period)

    def share_period_labels(self, periods) -> dict:
        sa = self._share_axis
        return _period_labels(periods, sa["cfg_proxy"], self.origin, sa["bucket_to_period"])

    # -- trials (kept in Spark) --------------------------------------------- #
    def _build_trials(self):
        """One row per brand trier with its trial date, entry week/period, entry
        cohort and furthest feasible RBR interval — all as Spark columns."""
        F, c = self._F, self.cfg
        brand = self._p.filter(F.col("is_brand"))
        if c.launch_date and not c.include_prelaunch_cohort:
            brand = brand.filter(F.col("ts") >= F.lit(self._origin).cast("date"))
        tr = brand.groupBy("card").agg(F.min("ts").alias("trial_ts"))
        origin = F.lit(self._origin).cast("date")
        # entry_week: weekly stage-of-entry for Table 2 cohorts (always weekly,
        # regardless of the calendar-axis period_unit).
        tr = tr.withColumn("entry_week", F.floor(F.datediff(F.col("trial_ts"), origin) / 7) + 1)
        # entry_period on the calendar axis -- derived from trial_ts, so it tracks
        # the trial's real calendar position (no collapsing across empty buckets).
        tr = tr.withColumn("entry_period", _period_col(
            F, F.col("trial_ts"), c, self._origin, self._origin_month,
            self._bucket_to_period))
        tr = tr.withColumn("cohort", _cohort_col(
            F, F.col("entry_week"), c.cohort_boundaries_weeks, c.include_prelaunch_cohort))
        tr = tr.withColumn("max_interval", _max_interval_col(
            F, F.col("trial_ts"), self._adate, self._adate_month, c))
        keep = ["card", "trial_ts", "entry_week", "entry_period", "cohort", "max_interval"]
        return tr.select(*keep)

    def _n_cat_triers(self) -> int:
        F = self._F
        return int(self._p.filter(F.col("is_cat") & (F.col("ts") >= F.lit(self._origin).cast("date")))
                   .select("card").distinct().count())

    def cohort_counts(self) -> Dict[str, int]:
        """Triers per entry cohort (tiny: one row per cohort label)."""
        rows = self._trials.groupBy("cohort").count().collect()
        return {r["cohort"]: int(r["count"]) for r in rows}

    def trier_counts_by_entry_week(self) -> pd.DataFrame:
        """Triers per (entry_week, cohort) — small, for the Fig 9 cohort plot."""
        return (self._trials.groupBy("entry_week", "cohort").count()
                .toPandas().rename(columns={"count": "n"}))

    # -- entrants (per-period counts) --------------------------------------- #
    def entrants(self) -> pd.DataFrame:
        F = self._F
        bw = (self._trials.filter(F.col("entry_period") >= 1)
              .groupBy("entry_period").count()
              .toPandas().rename(columns={"entry_period": "period", "count": "n_brand_new"}))
        cat = self._p.filter(F.col("is_cat") & (F.col("ts") >= F.lit(self._origin).cast("date")))
        first = cat.groupBy("card").agg(F.min("ts").alias("ts"))
        cw = (first.withColumn("period", self._main_period_col())
              .filter(F.col("period") >= 1).groupBy("period").count()
              .toPandas().rename(columns={"count": "n_cat_new"}))
        out = (bw.set_index("period").join(cw.set_index("period"), how="outer")
               .fillna(0).astype(int).reset_index().sort_values("period"))
        return out[["period", "n_brand_new", "n_cat_new"]]

    # -- RBR ---------------------------------------------------------------- #
    def _joined_trialists(self):
        """Post-trial lines of every trier, with their RBR interval and the
        eligibility filter applied — built once in Spark and cached. Stays a
        Spark DataFrame; only the per-interval group-bys below are collected."""
        if self._tj is not None:
            return self._tj
        F = self._F
        tr = self._trials.select("card", "trial_ts", "max_interval", "cohort")
        j = (self._p.join(tr, on="card", how="inner")
             .filter(F.datediff(F.col("ts"), F.col("trial_ts")) > 0)
             .withColumn("interval", _interval_col(F, F.col("ts"), F.col("trial_ts"), self.cfg)))
        j = j.filter((F.col("interval") >= 1) & (F.col("interval") <= F.col("max_interval")))
        j = (j.withColumn("bq", F.when(F.col("is_brand"), F.col("qty")).otherwise(0.0))
              .withColumn("cq", F.when(F.col("is_cat"), F.col("qty")).otherwise(0.0)))
        self._tj = j.select("interval", "cohort", "bq", "cq").cache()
        return self._tj

    def _upper_interval(self) -> int:
        F = self._F
        mt = self._trials.agg(F.max("max_interval").alias("m")).collect()[0]["m"]
        mt = int(mt) if mt is not None else 0
        return min(self.cfg.max_interval, mt) if self.cfg.max_interval is not None else mt

    def _n_eligible(self, upper: int) -> Dict[int, int]:
        """n_eligible(t) = #triers whose max_interval >= t, for t in 1..upper.
        From the tiny max_interval distribution (one row per distinct value).
        The cumulation is seeded with the triers observable BEYOND `upper`:
        when cfg.max_interval caps the axis below the feasible horizon, their
        early intervals are still inside the brand/cat sums, so they belong in
        every base at t <= upper."""
        dist = {int(r["max_interval"]): int(r["count"]) for r in
                self._trials.groupBy("max_interval").count().collect()}
        out = {}
        running = sum(c for v, c in dist.items() if v > upper)
        for t in range(upper, 0, -1):                       # cumulate from the top
            running += dist.get(t, 0)
            out[t] = running
        return out

    def rbr_pooled(self) -> pd.DataFrame:
        F = self._F
        upper = self._upper_interval()
        agg = {int(r["interval"]): (float(r["b"]), float(r["c"])) for r in
               self._joined_trialists().groupBy("interval")
               .agg(F.sum("bq").alias("b"), F.sum("cq").alias("c")).collect()}
        elig = self._n_eligible(upper)
        rows = [(t, agg.get(t, (0.0, 0.0))[0], agg.get(t, (0.0, 0.0))[1], elig.get(t, 0))
                for t in range(1, upper + 1)]
        return pd.DataFrame(rows, columns=["interval", "brand_qty", "cat_qty", "n_eligible"])

    def rbr_cohort(self) -> pd.DataFrame:
        F = self._F
        rows = (self._joined_trialists().groupBy("cohort", "interval")
                .agg(F.sum("bq").alias("brand_qty"), F.sum("cq").alias("cat_qty"))
                .toPandas())
        if rows.empty:
            return pd.DataFrame(columns=["cohort", "interval", "brand_qty", "cat_qty"])
        return rows[["cohort", "interval", "brand_qty", "cat_qty"]]

    # -- buying index ------------------------------------------------------- #
    def buying_scopes(self) -> pd.DataFrame:
        """Fixed-base scopes: ``sum_cat`` is the category volume inside the
        (optional) window, but ``n_buyers`` is the scope's FULL all-time
        membership -- a member with no purchases in the window weighs 0 in the
        per-capita average instead of dropping out. The window narrows the
        volume, never the base."""
        F, c = self._F, self.cfg
        # Category volume is always scoped to on/after the launch origin (pre-launch
        # history must not dilute the buying index); the optional rolling window
        # narrows it further.
        cat = self._p.filter(F.col("is_cat")
                             & (F.col("ts") >= F.lit(self._origin).cast("date")))
        if c.buying_index_window_days is not None:
            start = (pd.Timestamp(self.analysis_date)
                     - pd.Timedelta(days=c.buying_index_window_days)).date()
            cat = cat.filter(F.col("ts") > F.lit(start.isoformat()).cast("date"))
        cbc = cat.groupBy("card").agg(F.sum("qty").alias("v"))   # per-card window volume
        brand_counts = self._p.filter(F.col("is_brand")).groupBy("card").count()
        is_rep = F.coalesce(F.col("count"), F.lit(0)) >= c.repeater_min_purchases
        # Window sums per entry cohort (null = non-trier), with the repeater-only
        # sums alongside; the member counts come from the membership tables, not
        # from who happened to buy in the window.
        rows = (cbc.join(self._trials.select("card", "cohort"), "card", "left")
                .join(brand_counts, "card", "left")
                .groupBy("cohort")
                .agg(F.sum("v").alias("s"),
                     F.sum(F.when(is_rep, F.col("v"))).alias("rs"))
                .collect())
        sums = {r["cohort"]: float(r["s"] or 0.0) for r in rows}
        counts = self.cohort_counts()                      # {cohort: total triers}
        n_rep = int(brand_counts
                    .filter(F.col("count") >= c.repeater_min_purchases).count())
        scopes = [
            ("__all__", float(sum(sums.values())), int(self.n_category_triers)),
            ("__triers__",
             float(sum(v for k, v in sums.items() if k is not None)),
             int(sum(counts.values()))),
            ("__repeaters__",
             float(sum(float(r["rs"] or 0.0) for r in rows)), n_rep),
        ]
        for label in cohort_order(c.cohort_boundaries_weeks, c.include_prelaunch_cohort):
            scopes.append((label, sums.get(label, 0.0), int(counts.get(label, 0))))
        return pd.DataFrame(scopes, columns=["scope", "sum_cat", "n_buyers"])

    def buying_series(self) -> pd.DataFrame:
        """Fixed-panel per-period diagnostic (deliberate design): the trier base
        is the WHOLE dataset's triers in every period -- early periods include
        category volume bought by shoppers who only trial later -- and ``sel_n``
        / ``all_n`` are the constant base sizes, so a member who skips a period
        weighs 0 in that period's per-capita average."""
        F = self._F
        triers = (self._trials.select("card").distinct()
                  .withColumn("_is_trier", F.lit(True)))
        # Single group-by over the category lines: the all-buyer and trier-only
        # sums come out of one shuffle on `period`; the constant denominators
        # are attached afterwards from the membership tables.
        cat = (self._p.filter(F.col("is_cat") & (F.col("ts") >= F.lit(self._origin).cast("date")))
               .withColumn("period", self._main_period_col())
               .join(triers, "card", "left"))
        out = (cat.groupBy("period").agg(
                   F.sum("qty").alias("all_sum"),
                   F.sum(F.when(F.col("_is_trier"), F.col("qty"))).alias("sel_sum"))
               .toPandas().fillna(0.0).sort_values("period"))
        if out.empty:
            return pd.DataFrame(columns=["period", "sel_sum", "sel_n", "all_sum", "all_n"])
        out["sel_n"] = int(self._trials.count())
        out["all_n"] = int(self.n_category_triers)
        return out[["period", "sel_sum", "sel_n", "all_sum", "all_n"]]

    # -- realised share ----------------------------------------------------- #
    def share_long(self) -> pd.DataFrame:
        F = self._F
        sa = self._share_axis
        d = self._p.filter(F.col("ts") >= F.lit(self._origin).cast("date"))
        period = _period_col(F, F.col("ts"), sa["cfg_proxy"], self._origin,
                             self._origin_month, sa["bucket_to_period"])
        g = (d.withColumn("period", period)
             .withColumn("bq", F.when(F.col("is_brand"), F.col("qty")).otherwise(0.0))
             .withColumn("cq", F.when(F.col("is_cat"), F.col("qty")).otherwise(0.0))
             .groupBy("period")
             .agg(F.sum("bq").alias("brand_qty"), F.sum("cq").alias("cat_qty"))
             .toPandas().sort_values("period"))
        return g[["period", "brand_qty", "cat_qty"]]
