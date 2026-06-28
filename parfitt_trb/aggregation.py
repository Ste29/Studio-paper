"""Backend aggregation layer (Spark only).

This is the ONLY place the DataFrame engine appears. It turns a raw transaction
log into a handful of small, card-collapsed tables with a fixed schema;
everything downstream (``core``, ``model``, ``plots``) is engine-free and works
on those small pandas tables.

The whole point of this module is *where* the per-shopper dimension collapses:
EVERY heavy join / group-by runs in Spark, and only the already-aggregated
result (one row per interval / period / cohort / scope) is `.toPandas()`-ed. No
transaction-level or per-card frame is ever pulled to the driver, so it scales
to a real single-retailer panel of millions of lines.

Contract (all small pandas DataFrames):
  entrants()    : period, n_brand_new, n_cat_new
  rbr_pooled()  : interval, brand_qty, cat_qty, n_eligible
  rbr_cohort()  : cohort, interval, brand_qty, cat_qty
  buying_scopes(): scope, sum_cat, n_buyers          (scope '__all__' = everyone)
  buying_series(): period, sel_sum, sel_n, all_sum, all_n
  share_long()  : period, brand_qty, cat_qty
  cohort_counts(): {cohort label -> n triers}
plus the calendar-label maps period_labels() / share_period_labels().
"""
from __future__ import annotations

from typing import Dict

import pandas as pd

from .cohorts import cohort_order
from .config import TRBConfig
from .periods import _MONDAY_EPOCH, as_date, month_index, period_label


# --------------------------------------------------------------------------- #
# Calendar / share axis descriptors (operate on the small set of distinct
# labels, so they stay in pandas and are shared with the label rendering).
# --------------------------------------------------------------------------- #
def _build_bucket_map(first_dates: pd.Series) -> dict:
    """Dense chronological ordinal map {label: 1..N} from a Series indexed by
    bucket label holding each label's first observed date."""
    return {b: i for i, b in enumerate(first_dates.sort_values().index, start=1)}


def _build_share_axis_from_dates(cfg: TRBConfig, first_dates, origin,
                                  fallback_bucket_to_period: dict = None) -> dict:
    """Build the share-axis descriptor used by share_long / share_period_labels.

    `first_dates` is a pandas Series indexed by bucket label -> first observed
    date (one row per distinct label, so it is tiny). Pass None when no separate
    share bucket column is configured.

    Returns a dict with keys: cfg_proxy, origin_month, bucket_to_period, bucket_col.
    When no separate share axis is configured the dict mirrors the main axis so
    callers never need to branch.
    """
    has_share_bucket = cfg.share_bucket_column is not None and first_dates is not None
    has_share_unit = cfg.share_period_unit is not None

    if not has_share_bucket and not has_share_unit:
        # Same axis as main — pass through using the main bucket map.
        return {"cfg_proxy": cfg, "origin_month": month_index(origin),
                "bucket_to_period": fallback_bucket_to_period or {}, "bucket_col": "bucket"}

    if has_share_bucket:
        bucket_map = _build_bucket_map(first_dates)
        # cfg_proxy: present the share_bucket_column as bucket_column so the
        # period column / label code take the bucket-mode paths.
        frame_col = ("share_bucket" if cfg.share_bucket_column != cfg.bucket_column
                     else "bucket")
        proxy = TRBConfig(**{**cfg.__dict__,
                             "bucket_column": cfg.share_bucket_column,
                             "share_bucket_column": None})
        return {"cfg_proxy": proxy, "origin_month": month_index(origin),
                "bucket_to_period": bucket_map, "bucket_col": frame_col}

    # Derived-date mode with a different period_unit.
    proxy = TRBConfig(**{**cfg.__dict__,
                         "period_unit": cfg.share_period_unit,
                         "bucket_column": None,
                         "share_period_unit": None})
    return {"cfg_proxy": proxy, "origin_month": month_index(origin),
            "bucket_to_period": {}, "bucket_col": "bucket"}


def _period_labels(periods, cfg: TRBConfig, origin, bucket_to_period: dict) -> dict:
    """Map calendar-axis period ordinals -> display labels. Bucket mode inverts
    the dense ordinal map; week/month modes derive the label from the ordinal and
    the origin. `periods` is the small set of ordinals actually used."""
    if cfg.bucket_column is not None:
        inv = {p: b for b, p in bucket_to_period.items()}
        return {int(p): inv.get(int(p), str(int(p))) for p in periods}
    return {int(p): period_label(int(p), origin, cfg.period_unit) for p in periods}


# --------------------------------------------------------------------------- #
# Spark column expressions for the calendar period and the RBR interval. These
# mirror the calendar maths so the heavy reductions can run entirely in Spark.
# --------------------------------------------------------------------------- #
def _period_col(F, ts_col, cfg: TRBConfig, origin_iso: str, origin_month: int,
                bucket_to_period: dict, bucket_colname: str):
    """1-based calendar-axis period ordinal as a Spark Column. Rows before the
    origin / with an unknown bucket map to <= 0 (callers keep period >= 1)."""
    if cfg.bucket_column is not None:
        if not bucket_to_period:
            return F.lit(0)
        pairs = []
        for label, period in bucket_to_period.items():
            pairs += [F.lit(label), F.lit(int(period))]
        return F.coalesce(F.create_map(*pairs)[F.col(bucket_colname)], F.lit(0))
    origin = F.lit(origin_iso).cast("date")
    if cfg.period_unit == "week":
        return F.floor(F.datediff(ts_col, origin) / 7) + 1
    # month: (year*12 + month - 1) - origin_month + 1
    return (F.year(ts_col) * 12 + F.month(ts_col) - 1) - F.lit(origin_month) + 1


def _abs_week_col(F, col):
    """floor(days since the Monday epoch / 7) — the calendar-aligned week index."""
    return F.floor(F.datediff(col, F.lit(_MONDAY_EPOCH.isoformat()).cast("date")) / 7)


def _month_idx_col(F, col):
    return F.year(col) * 12 + F.month(col) - 1


def _interval_col(F, ts_col, ref_col, cfg: TRBConfig):
    """1-based RBR interval index of `ts_col` relative to the trial `ref_col`."""
    if cfg.rbr_interval_mode == "exact":
        return F.ceil(F.datediff(ts_col, ref_col) / cfg.period_length_days)
    if cfg.rbr_bucket_unit == "week":
        return _abs_week_col(F, ts_col) - _abs_week_col(F, ref_col)
    return _month_idx_col(F, ts_col) - _month_idx_col(F, ref_col)


def _max_interval_col(F, trial_col, analysis_iso: str, analysis_month: int,
                      cfg: TRBConfig):
    """Highest RBR interval whose whole window has elapsed by the analysis date."""
    if cfg.rbr_interval_mode == "exact":
        adate = F.lit(analysis_iso).cast("date")
        return F.floor(F.datediff(adate, trial_col) / cfg.period_length_days)
    if cfg.rbr_bucket_unit == "week":
        adate = F.lit(analysis_iso).cast("date")
        return _abs_week_col(F, adate) - _abs_week_col(F, trial_col)
    return F.lit(analysis_month) - _month_idx_col(F, trial_col)


def _cohort_col(F, entry_week_col, boundaries, include_prelaunch: bool):
    """Entry-cohort label as a Spark Column (pure when-chain, no Python UDF).
    The label strings come from :func:`cohorts.cohort_order` so the three label
    encodings (here, ``cohort_label``, ``cohort_order``) cannot drift."""
    labels = cohort_order(boundaries, include_prelaunch)
    # cohort_order: [PRELAUNCH?, one bounded label per boundary, final '+w'].
    first_label = labels[0]                          # the <=0 / earliest bucket
    bounded = labels[1:] if include_prelaunch else labels
    col = F.when(entry_week_col <= 0, F.lit(first_label))
    for b, label in zip(boundaries, bounded):        # bounded[-1] is the '+w' tail
        col = col.when(entry_week_col <= b, F.lit(label))
    return col.otherwise(F.lit(bounded[-1]))


# --------------------------------------------------------------------------- #
# Spark aggregator — the single backend.
# --------------------------------------------------------------------------- #
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
        cols = ["card", "ts", "is_brand", "is_cat", "qty"]
        if c.bucket_column is not None:
            out = out.withColumn("bucket", F.col(c.bucket_column).cast("string"))
            cols.append("bucket")
        if c.share_bucket_column is not None and c.share_bucket_column != c.bucket_column:
            out = out.withColumn("share_bucket", F.col(c.share_bucket_column).cast("string"))
            cols.append("share_bucket")
        return out.select(*cols)

    # -- calendar axis ------------------------------------------------------ #
    def _build_calendar_axis(self) -> None:
        """Prepare the calendar-time period axes. Cohorts and RBR are NOT on them.

        Main axis (penetration / buying_series / entrants):
          'week'/'month': derived from dates; bucket_column: dense ordinal.
        Share axis (share_long only):
          defaults to the main axis; share_bucket_column / share_period_unit
          override it so the share can live on a different granularity."""
        F, c = self._F, self.cfg
        self.period_unit = "bucket" if c.bucket_column is not None else c.period_unit
        self._origin_month = month_index(self.origin)
        self._bucket_to_period: dict = {}
        obs_filter = F.col("ts") >= F.lit(self._origin).cast("date")
        if c.bucket_column is not None:
            # one row per distinct label -> tiny
            first = (self._p.filter(obs_filter)
                     .groupBy("bucket").agg(F.min("ts").alias("first")).toPandas()
                     .set_index("bucket")["first"])
            self._bucket_to_period = _build_bucket_map(first)
        if c.share_bucket_column is not None and c.share_bucket_column != c.bucket_column:
            # `_prepare` already renamed the share-bucket column to 'share_bucket'.
            first_dates = (self._p.filter(obs_filter)
                           .groupBy(F.col("share_bucket").alias("_sb"))
                           .agg(F.min("ts").alias("first")).toPandas()
                           .set_index("_sb")["first"])
        else:
            first_dates = None
        self._share_axis = _build_share_axis_from_dates(
            c, first_dates, self.origin, fallback_bucket_to_period=self._bucket_to_period)
        sp = self._share_axis["cfg_proxy"]
        self.share_period_unit = ("bucket" if sp.bucket_column is not None
                                  else (sp.period_unit or "week"))

    def _main_period_col(self, ts_col=None):
        F = self._F
        return _period_col(F, ts_col if ts_col is not None else F.col("ts"),
                           self.cfg, self._origin, self._origin_month,
                           self._bucket_to_period, "bucket")

    def period_labels(self, periods) -> dict:
        return _period_labels(periods, self.cfg, self.origin, self._bucket_to_period)

    def share_period_labels(self, periods) -> dict:
        sa = self._share_axis
        if not sa["bucket_to_period"] and sa["cfg_proxy"] is self.cfg:
            return self.period_labels(periods)
        return _period_labels(periods, sa["cfg_proxy"], self.origin, sa["bucket_to_period"])

    # -- trials (kept in Spark) --------------------------------------------- #
    def _build_trials(self):
        """One row per brand trier with its trial date, entry week/period, entry
        cohort and furthest feasible RBR interval — all as Spark columns."""
        F, c = self._F, self.cfg
        brand = self._p.filter(F.col("is_brand"))
        if c.launch_date and not c.include_prelaunch_cohort:
            brand = brand.filter(F.col("ts") >= F.lit(self._origin).cast("date"))
        if c.bucket_column is not None:
            # F.min(struct(ts, bucket)) keeps the bucket of the earliest line.
            m = brand.groupBy("card").agg(F.min(F.struct("ts", "bucket")).alias("m"))
            tr = m.select("card", F.col("m.ts").alias("trial_ts"),
                          F.col("m.bucket").alias("bucket"))
        else:
            tr = brand.groupBy("card").agg(F.min("ts").alias("trial_ts"))
        origin = F.lit(self._origin).cast("date")
        # entry_week: weekly stage-of-entry for Table 2 cohorts (always weekly,
        # regardless of the calendar-axis period_unit).
        tr = tr.withColumn("entry_week", F.floor(F.datediff(F.col("trial_ts"), origin) / 7) + 1)
        tr = tr.withColumn("entry_period", _period_col(
            F, F.col("trial_ts"), c, self._origin, self._origin_month,
            self._bucket_to_period, "bucket"))
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
        if self.cfg.bucket_column is not None:
            m = cat.groupBy("card").agg(F.min(F.struct("ts", "bucket")).alias("m"))
            first = m.select(F.col("m.ts").alias("ts"), F.col("m.bucket").alias("bucket"))
        else:
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
        From the tiny max_interval distribution (one row per distinct value)."""
        dist = {int(r["max_interval"]): int(r["count"]) for r in
                self._trials.groupBy("max_interval").count().collect()}
        out, running = {}, 0
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
        F, c = self._F, self.cfg
        cat = self._p.filter(F.col("is_cat"))
        if c.buying_index_window_days is not None:
            start = (pd.Timestamp(self.analysis_date)
                     - pd.Timedelta(days=c.buying_index_window_days)).date()
            cat = cat.filter(F.col("ts") > F.lit(start.isoformat()).cast("date"))
        cbc = cat.groupBy("card").agg(F.sum("qty").alias("v"))   # per-card category volume
        brand_counts = self._p.filter(F.col("is_brand")).groupBy("card").count()
        is_rep = F.coalesce(F.col("count"), F.lit(0)) >= c.repeater_min_purchases
        # One per-card-collapse pass: group category buyers by entry cohort
        # (null = non-trier) and carry the repeater-only sums alongside, so every
        # scope below is read off one tiny collected result.
        rows = (cbc.join(self._trials.select("card", "cohort"), "card", "left")
                .join(brand_counts, "card", "left")
                .groupBy("cohort")
                .agg(F.sum("v").alias("s"), F.count("v").alias("n"),
                     F.sum(F.when(is_rep, F.col("v"))).alias("rs"),
                     F.count(F.when(is_rep, F.col("v"))).alias("rn"))
                .collect())
        by_cohort = {r["cohort"]: (float(r["s"]), int(r["n"]))
                     for r in rows if r["cohort"] is not None}
        triers = [r for r in rows if r["cohort"] is not None]
        scopes = [
            ("__all__", float(sum(r["s"] or 0.0 for r in rows)),
             int(sum(r["n"] or 0 for r in rows))),
            ("__triers__", float(sum(r["s"] or 0.0 for r in triers)),
             int(sum(r["n"] or 0 for r in triers))),
            ("__repeaters__", float(sum(r["rs"] or 0.0 for r in rows)),
             int(sum(r["rn"] or 0 for r in rows))),
        ]
        for label in cohort_order(c.cohort_boundaries_weeks, c.include_prelaunch_cohort):
            scopes.append((label, *by_cohort.get(label, (0.0, 0))))
        return pd.DataFrame(scopes, columns=["scope", "sum_cat", "n_buyers"])

    def buying_series(self) -> pd.DataFrame:
        F = self._F
        triers = self._trials.select("card").distinct().withColumn("_is_trier", F.lit(True))
        # Single group-by over the category lines: the all-buyer and trier-only
        # sums/distinct-counts come out of one shuffle on `period`.
        cat = (self._p.filter(F.col("is_cat") & (F.col("ts") >= F.lit(self._origin).cast("date")))
               .withColumn("period", self._main_period_col())
               .join(triers, "card", "left"))
        out = (cat.groupBy("period").agg(
                   F.sum("qty").alias("all_sum"),
                   F.countDistinct("card").alias("all_n"),
                   F.sum(F.when(F.col("_is_trier"), F.col("qty"))).alias("sel_sum"),
                   F.countDistinct(F.when(F.col("_is_trier"), F.col("card"))).alias("sel_n"))
               .toPandas().fillna(0.0).sort_values("period"))
        if out.empty:
            return pd.DataFrame(columns=["period", "sel_sum", "sel_n", "all_sum", "all_n"])
        return out[["period", "sel_sum", "sel_n", "all_sum", "all_n"]]

    # -- realised share ----------------------------------------------------- #
    def share_long(self) -> pd.DataFrame:
        F = self._F
        sa = self._share_axis
        d = self._p.filter(F.col("ts") >= F.lit(self._origin).cast("date"))
        period = _period_col(F, F.col("ts"), sa["cfg_proxy"], self._origin,
                             sa["origin_month"], sa["bucket_to_period"], sa["bucket_col"])
        g = (d.withColumn("period", period)
             .withColumn("bq", F.when(F.col("is_brand"), F.col("qty")).otherwise(0.0))
             .withColumn("cq", F.when(F.col("is_cat"), F.col("qty")).otherwise(0.0))
             .groupBy("period")
             .agg(F.sum("bq").alias("brand_qty"), F.sum("cq").alias("cat_qty"))
             .toPandas().sort_values("period"))
        return g[["period", "brand_qty", "cat_qty"]]
