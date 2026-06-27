"""Backend aggregation layer.

This is the ONLY place the DataFrame engine appears. Each aggregator turns a raw
transaction log into a handful of small, card-collapsed tables with an identical
schema; everything downstream (``core``, ``plots``) is backend-free.

Contract (all small pandas DataFrames):
  trials        : card, trial_date, entry_week, cohort, max_interval
  entrants()    : period, n_brand_new, n_cat_new
  rbr_pooled()  : interval, brand_qty, cat_qty, n_eligible
  rbr_cohort()  : cohort, interval, brand_qty, cat_qty
  buying_scopes(): scope, sum_cat, n_buyers          (scope '__all__' = everyone)
  buying_series(): period, sel_sum, sel_n, all_sum, all_n
  share_long()  : period, brand_qty, cat_qty

`PandasAggregator` runs locally (tests, figures). `SparkAggregator` runs the same
logical reductions in Spark and `.toPandas()`-es the (small) results for prod.
"""
from __future__ import annotations

import abc
from datetime import date
from typing import List, Optional

import numpy as np
import pandas as pd

from .cohorts import cohort_label, cohort_order
from .config import TRBConfig
from .periods import _MONDAY_EPOCH, as_date, month_index, period_label

_DAY = np.timedelta64(1, "D")


def _periods_pandas(ts, bucket, cfg: TRBConfig, origin, origin_month: int,
                    bucket_to_period: dict) -> np.ndarray:
    """1-based calendar-axis period ordinal for timestamps `ts` (and matching
    `bucket` labels in bucket mode). Shared by both backends so they cannot drift
    on the axis. Rows before the origin map to <= 0 (callers keep period >= 1)."""
    if cfg.bucket_column is not None:
        return (pd.Series(bucket).map(bucket_to_period)
                .fillna(0).astype(int).to_numpy())
    t = pd.to_datetime(pd.Series(np.asarray(ts)))
    if cfg.period_unit == "week":
        return (np.floor((t - pd.Timestamp(origin)) / _DAY / 7).astype(int) + 1).to_numpy()
    # `origin_month` is month_index(origin); keep the same convention per row.
    return ((t.dt.year * 12 + t.dt.month - 1) - origin_month + 1).to_numpy()


def _build_bucket_map(first_dates: pd.Series) -> dict:
    """Dense chronological ordinal map {label: 1..N} from a Series indexed by
    bucket label holding each label's first observed date."""
    return {b: i for i, b in enumerate(first_dates.sort_values().index, start=1)}


def _period_labels(periods, cfg: TRBConfig, origin, bucket_to_period: dict) -> dict:
    """Map calendar-axis period ordinals -> display labels. Shared by both
    backends. Bucket mode inverts the dense ordinal map; week/month modes derive
    the label from the ordinal and the origin."""
    if cfg.bucket_column is not None:
        inv = {p: b for b, p in bucket_to_period.items()}
        return {int(p): inv.get(int(p), str(int(p))) for p in periods}
    return {int(p): period_label(int(p), origin, cfg.period_unit) for p in periods}


class Aggregator(abc.ABC):
    """Strategy interface. Subclasses fill `trials`, `origin`, `analysis_date`
    and the table methods below."""
    origin: date
    analysis_date: date
    trials: pd.DataFrame
    n_category_triers: int

    @abc.abstractmethod
    def entrants(self) -> pd.DataFrame: ...
    @abc.abstractmethod
    def rbr_pooled(self) -> pd.DataFrame: ...
    @abc.abstractmethod
    def rbr_cohort(self) -> pd.DataFrame: ...
    @abc.abstractmethod
    def buying_scopes(self) -> pd.DataFrame: ...
    @abc.abstractmethod
    def buying_series(self) -> pd.DataFrame: ...
    @abc.abstractmethod
    def share_long(self) -> pd.DataFrame: ...


# --------------------------------------------------------------------------- #
# Pandas backend
# --------------------------------------------------------------------------- #
class PandasAggregator(Aggregator):
    def __init__(self, df: pd.DataFrame, cfg: TRBConfig):
        self.cfg = cfg
        p = self._prepare(df)
        self.analysis_date = (as_date(cfg.analysis_date) if cfg.analysis_date
                              else p["ts"].max().date())
        self._adate_ts = pd.Timestamp(self.analysis_date)
        p = p[p["ts"] <= self._adate_ts].reset_index(drop=True)
        if p.empty:
            raise ValueError("no transactions on/before the analysis date")
        self._p = p

        origin = (as_date(cfg.launch_date) if cfg.launch_date
                  else self._first_brand_date(p))
        if origin is None:
            raise ValueError("cannot determine launch origin: set launch_date "
                             "or ensure there is at least one brand purchase")
        self.origin = origin
        self._origin_ts = pd.Timestamp(origin)
        self._build_calendar_axis()
        self.trials = self._build_trials()
        self._trier_cards = set(self.trials["card"])
        cat = p[p["is_cat"] & (p["ts"] >= self._origin_ts)]
        self.n_category_triers = int(cat["card"].nunique())

    # -- preparation -------------------------------------------------------- #
    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.cfg
        ts = pd.to_datetime(df[c.date_column]).dt.normalize()
        is_brand = df[c.brand_column].astype(bool).to_numpy()
        is_cat = df[c.category_column].astype(bool).to_numpy()
        if c.treat_brand_as_category:
            is_cat = is_cat | is_brand
        out = pd.DataFrame({
            "card": df[c.card_column].astype(str).to_numpy(),
            "ts": ts.to_numpy(),
            "is_brand": is_brand,
            "is_cat": is_cat,
            "qty": df[c.measure].astype(float).to_numpy(),
        })
        if c.bucket_column is not None:
            out["bucket"] = df[c.bucket_column].astype(str).to_numpy()
        return out

    @staticmethod
    def _first_brand_date(p: pd.DataFrame) -> Optional[date]:
        b = p.loc[p["is_brand"], "ts"]
        return b.min().date() if len(b) else None

    def _fdays(self, ts, ref) -> np.ndarray:
        """Float number of days between datetime64 `ts` and `ref` (Timestamp)."""
        return (np.asarray(ts, dtype="datetime64[ns]") - np.datetime64(ref)) / _DAY

    def _build_trials(self) -> pd.DataFrame:
        c = self.cfg
        brand = self._p[self._p["is_brand"]]
        if c.launch_date and not c.include_prelaunch_cohort:
            brand = brand[brand["ts"] >= self._origin_ts]
        idx = brand.groupby("card")["ts"].idxmin()
        keep = ["card", "ts"] + (["bucket"] if c.bucket_column is not None else [])
        trial = (brand.loc[idx, keep].reset_index(drop=True)
                      .rename(columns={"ts": "trial_ts"}))
        days = self._fdays(trial["trial_ts"], self._origin_ts)
        # entry_week is the WEEKLY stage-of-entry used for Parfitt Table 2
        # cohorts, regardless of the calendar-axis period_unit.
        trial["entry_week"] = np.floor(days / 7).astype(int) + 1
        # entry_period is the trial's index on the calendar axis (penetration).
        trial["entry_period"] = self._periods_for(
            trial["trial_ts"], trial.get("bucket"))
        trial["trial_date"] = trial["trial_ts"].dt.date
        trial["cohort"] = [cohort_label(int(w), c.cohort_boundaries_weeks,
                                        c.include_prelaunch_cohort)
                           for w in trial["entry_week"]]
        trial["max_interval"] = self._max_interval(trial["trial_ts"])
        return trial

    # -- interval maths (exact day-windows vs calendar buckets) ------------- #
    def _interval(self, ts, ref) -> np.ndarray:
        """1-based interval index of `ts` relative to reference dates `ref`."""
        c = self.cfg
        if c.rbr_interval_mode == "exact":
            d = (np.asarray(ts, "datetime64[ns]") - np.asarray(ref, "datetime64[ns]")) / _DAY
            return np.ceil(d / c.period_length_days).astype(int)
        if c.rbr_bucket_unit == "week":
            wk = lambda x: ((np.asarray(x, "datetime64[ns]")
                             - np.datetime64(_MONDAY_EPOCH)) / _DAY).astype(int) // 7
            return wk(ts) - wk(ref)
        # month buckets
        def midx(x):
            xx = pd.to_datetime(pd.Series(np.asarray(x, "datetime64[ns]")))
            return (xx.dt.year * 12 + xx.dt.month - 1).to_numpy()
        return midx(ts) - midx(ref)

    def _max_interval(self, trial_ts) -> np.ndarray:
        c = self.cfg
        a = np.datetime64(self._adate_ts)
        tt = np.asarray(trial_ts, "datetime64[ns]")
        if c.rbr_interval_mode == "exact":
            return np.floor((a - tt) / _DAY / c.period_length_days).astype(int)
        if c.rbr_bucket_unit == "week":
            wk = lambda x: ((np.asarray(x, "datetime64[ns]")
                             - np.datetime64(_MONDAY_EPOCH)) / _DAY).astype(int) // 7
            return wk(a) - wk(tt)
        ami = self.analysis_date.year * 12 + self.analysis_date.month - 1
        ttp = pd.to_datetime(pd.Series(tt))
        return ami - (ttp.dt.year * 12 + ttp.dt.month - 1).to_numpy()

    # -- calendar axis (penetration / share / per-period buying index) ------ #
    def _build_calendar_axis(self) -> None:
        """Prepare the calendar-time period axis shared by penetration, realised
        share and the per-period buying index. Cohorts and RBR are NOT on it.

        'week'/'month': periods derived from each row's date vs. the origin.
        bucket_column : a DENSE chronological ordinal over the observed labels
                        (rows on/after the origin), ordered by each label's first
                        date -- so weekly/monthly feeds and cross-year labels
                        (…-W52 -> …-W01) become consecutive periods 1..N."""
        c = self.cfg
        self.period_unit = "bucket" if c.bucket_column is not None else c.period_unit
        self._origin_month = month_index(self.origin)
        self._bucket_to_period: dict = {}
        if c.bucket_column is None:
            return
        obs = self._p[self._p["ts"] >= self._origin_ts]
        self._bucket_to_period = _build_bucket_map(obs.groupby("bucket")["ts"].min())

    def _periods_for(self, ts, bucket=None) -> np.ndarray:
        """1-based calendar period ordinal for timestamps `ts` (and matching
        `bucket` labels when bucket_column is set)."""
        return _periods_pandas(ts, bucket, self.cfg, self.origin,
                               self._origin_month, self._bucket_to_period)

    def period_labels(self, periods) -> dict:
        """Period ordinal -> calendar label (YYYY-MM / YYYY-Www / bucket label)."""
        return _period_labels(periods, self.cfg, self.origin, self._bucket_to_period)

    # -- tables ------------------------------------------------------------- #
    def entrants(self) -> pd.DataFrame:
        bw = (self.trials.loc[self.trials["entry_period"] >= 1, "entry_period"]
              .value_counts().rename_axis("period").rename("n_brand_new"))
        cat = self._p[self._p["is_cat"] & (self._p["ts"] >= self._origin_ts)]
        idx = cat.groupby("card")["ts"].idxmin()
        cat_first = cat.loc[idx]
        cat_period = self._periods_for(cat_first["ts"], cat_first.get("bucket"))
        cw = (pd.Series(cat_period).pipe(lambda s: s[s >= 1]).value_counts()
              .rename_axis("period").rename("n_cat_new"))
        out = (pd.concat([bw, cw], axis=1).fillna(0).astype(int)
               .reset_index().sort_values("period"))
        return out[["period", "n_brand_new", "n_cat_new"]]

    def _joined_trialists(self) -> pd.DataFrame:
        tr = self.trials[["card", "trial_ts", "cohort", "max_interval"]]
        j = self._p.merge(tr, on="card", how="inner")
        d = (j["ts"].to_numpy("datetime64[ns]")
             - j["trial_ts"].to_numpy("datetime64[ns]")) / _DAY
        j = j[d > 0].copy()
        j["interval"] = self._interval(j["ts"].to_numpy(), j["trial_ts"].to_numpy())
        j = j[(j["interval"] >= 1) & (j["interval"] <= j["max_interval"])]
        j["bq"] = np.where(j["is_brand"], j["qty"], 0.0)
        j["cq"] = np.where(j["is_cat"], j["qty"], 0.0)
        return j

    def _upper_interval(self) -> int:
        max_t = int(self.trials["max_interval"].max()) if len(self.trials) else 0
        if self.cfg.max_interval is not None:
            return min(self.cfg.max_interval, max_t)
        return max_t

    def rbr_pooled(self) -> pd.DataFrame:
        j = self._joined_trialists()
        upper = self._upper_interval()
        agg = (j.groupby("interval").agg(brand_qty=("bq", "sum"),
                                         cat_qty=("cq", "sum")))
        # n_eligible(t) = #cards with max_interval >= t
        mt = self.trials["max_interval"].to_numpy()
        rows = []
        for t in range(1, upper + 1):
            b = float(agg["brand_qty"].get(t, 0.0))
            c = float(agg["cat_qty"].get(t, 0.0))
            rows.append((t, b, c, int((mt >= t).sum())))
        return pd.DataFrame(rows, columns=["interval", "brand_qty", "cat_qty", "n_eligible"])

    def rbr_cohort(self) -> pd.DataFrame:
        j = self._joined_trialists()
        if j.empty:
            return pd.DataFrame(columns=["cohort", "interval", "brand_qty", "cat_qty"])
        agg = (j.groupby(["cohort", "interval"])
                .agg(brand_qty=("bq", "sum"), cat_qty=("cq", "sum"))
                .reset_index())
        return agg[["cohort", "interval", "brand_qty", "cat_qty"]]

    def _cat_by_card_window(self) -> pd.Series:
        wd = self.cfg.buying_index_window_days
        cat = self._p[self._p["is_cat"]]
        if wd is not None:
            start = self._adate_ts - pd.Timedelta(days=wd)
            cat = cat[cat["ts"] > start]
        return cat.groupby("card")["qty"].sum()

    def buying_scopes(self) -> pd.DataFrame:
        cbc = self._cat_by_card_window()
        brand_counts = self._p[self._p["is_brand"]].groupby("card").size()
        repeater_cards = set(brand_counts[brand_counts >= self.cfg.repeater_min_purchases].index)
        cohort_map = dict(zip(self.trials["card"], self.trials["cohort"]))

        def stat(cards):
            s = cbc[cbc.index.isin(cards)]
            return float(s.sum()), int(s.size)

        rows = [("__all__", float(cbc.sum()), int(cbc.size)),
                ("__triers__", *stat(self._trier_cards)),
                ("__repeaters__", *stat(repeater_cards))]
        for label in cohort_order(self.cfg.cohort_boundaries_weeks,
                                  self.cfg.include_prelaunch_cohort):
            cards = {c for c, l in cohort_map.items() if l == label}
            rows.append((label, *stat(cards)))
        return pd.DataFrame(rows, columns=["scope", "sum_cat", "n_buyers"])

    def buying_series(self) -> pd.DataFrame:
        cat = self._p[self._p["is_cat"] & (self._p["ts"] >= self._origin_ts)].copy()
        if cat.empty:
            return pd.DataFrame(columns=["period", "sel_sum", "sel_n", "all_sum", "all_n"])
        cat["period"] = self._periods_for(cat["ts"], cat.get("bucket"))
        cat["is_trier"] = cat["card"].isin(self._trier_cards)
        g_all = cat.groupby("period").agg(all_sum=("qty", "sum"),
                                          all_n=("card", "nunique"))
        g_sel = (cat[cat["is_trier"]].groupby("period")
                 .agg(sel_sum=("qty", "sum"), sel_n=("card", "nunique")))
        out = g_all.join(g_sel, how="left").fillna(0.0).reset_index()
        return out[["period", "sel_sum", "sel_n", "all_sum", "all_n"]].sort_values("period")

    def share_long(self) -> pd.DataFrame:
        return _share_long_pandas(self._p[self._p["ts"] >= self._origin_ts],
                                  self.cfg, self.origin, self._origin_month,
                                  self._bucket_to_period)


# --------------------------------------------------------------------------- #
# Spark backend (mirror of the pandas reductions; runs only where pyspark+Java
# are available -- prod). Logic is intentionally identical so the parity test
# can assert the two agree on tiny data.
# --------------------------------------------------------------------------- #
class SparkAggregator(Aggregator):
    def __init__(self, sdf, cfg: TRBConfig):
        from pyspark.sql import functions as F  # noqa: F401  (import guard)
        self.cfg = cfg
        self._F = F
        p = self._prepare(sdf)
        adate = (cfg.analysis_date if cfg.analysis_date
                 else p.agg(F.max("ts").alias("m")).collect()[0]["m"])
        self.analysis_date = as_date(adate)
        self._adate = self.analysis_date.isoformat()
        p = p.filter(F.col("ts") <= F.lit(self._adate).cast("date")).cache()
        self._p = p
        if cfg.launch_date:
            origin = as_date(cfg.launch_date)
        else:
            o = (p.filter(F.col("is_brand")).agg(F.min("ts").alias("o"))
                 .collect()[0]["o"])
            origin = as_date(o) if o is not None else None
        if origin is None:
            raise ValueError("cannot determine launch origin")
        self.origin = origin
        self._origin = origin.isoformat()
        self._build_calendar_axis()
        self.trials = self._build_trials()       # pandas (small)
        self._trier_cards = set(self.trials["card"])
        self.n_category_triers = int(self.trials.shape[0] and self._n_cat_triers())

    def _build_calendar_axis(self) -> None:
        """Spark mirror of the pandas calendar axis (see PandasAggregator)."""
        F, c = self._F, self.cfg
        self.period_unit = "bucket" if c.bucket_column is not None else c.period_unit
        self._origin_month = month_index(self.origin)
        self._bucket_to_period: dict = {}
        if c.bucket_column is None:
            return
        first = (self._p.filter(F.col("ts") >= F.lit(self._origin).cast("date"))
                 .groupBy("bucket").agg(F.min("ts").alias("first")).toPandas()
                 .set_index("bucket")["first"])
        self._bucket_to_period = _build_bucket_map(first)

    def period_labels(self, periods) -> dict:
        """Period ordinal -> calendar label (YYYY-MM / YYYY-Www / bucket label)."""
        return _period_labels(periods, self.cfg, self.origin, self._bucket_to_period)

    # All heavy groupBys run in Spark; results are tiny and returned as pandas.
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
        return out.select(*cols)

    def _build_trials(self) -> pd.DataFrame:
        F, c = self._F, self.cfg
        brand = self._p.filter(F.col("is_brand"))
        if c.launch_date and not c.include_prelaunch_cohort:
            brand = brand.filter(F.col("ts") >= F.lit(self._origin).cast("date"))
        if c.bucket_column is not None:
            # F.min(struct(ts, bucket)) keeps the bucket of the earliest line.
            m = brand.groupBy("card").agg(F.min(F.struct("ts", "bucket")).alias("m"))
            tr = m.select("card", F.col("m.ts").alias("trial_date"),
                          F.col("m.bucket").alias("bucket")).toPandas()
        else:
            tr = brand.groupBy("card").agg(F.min("ts").alias("trial_date")).toPandas()
            tr["bucket"] = None
        tr["trial_ts"] = pd.to_datetime(tr["trial_date"])
        origin_ts = pd.Timestamp(self.origin)
        days = (tr["trial_ts"] - origin_ts) / _DAY
        tr["entry_week"] = np.floor(days / 7).astype(int) + 1
        tr["entry_period"] = _periods_pandas(tr["trial_ts"], tr["bucket"], c,
                                             self.origin, self._origin_month,
                                             self._bucket_to_period)
        tr["cohort"] = [cohort_label(int(w), c.cohort_boundaries_weeks,
                                     c.include_prelaunch_cohort) for w in tr["entry_week"]]
        # max_interval reuses the pandas maths for identical semantics
        tr["max_interval"] = _max_interval_pandas(tr["trial_ts"], self.analysis_date, c)
        tr["trial_date"] = tr["trial_ts"].dt.date
        return tr[["card", "trial_date", "trial_ts", "entry_week", "entry_period",
                   "cohort", "max_interval"]]

    def _n_cat_triers(self) -> int:
        F = self._F
        return (self._p.filter(F.col("is_cat") & (F.col("ts") >= F.lit(self._origin).cast("date")))
                .select("card").distinct().count())

    # The remaining tables: do the Spark groupBy, hand the small result to the
    # SAME pandas finishing code via a tiny shared helper.
    def entrants(self) -> pd.DataFrame:
        F, c = self._F, self.cfg
        bw = (self.trials.loc[self.trials["entry_period"] >= 1, "entry_period"]
              .value_counts().rename_axis("period").rename("n_brand_new"))
        cat = self._p.filter(F.col("is_cat") & (F.col("ts") >= F.lit(self._origin).cast("date")))
        if c.bucket_column is not None:
            m = cat.groupBy("card").agg(F.min(F.struct("ts", "bucket")).alias("m"))
            cat_first = m.select(F.col("m.ts").alias("ts"),
                                 F.col("m.bucket").alias("bucket")).toPandas()
        else:
            cat_first = cat.groupBy("card").agg(F.min("ts").alias("ts")).toPandas()
            cat_first["bucket"] = None
        period = _periods_pandas(cat_first["ts"], cat_first["bucket"], c, self.origin,
                                 self._origin_month, self._bucket_to_period)
        cw = (pd.Series(period).pipe(lambda s: s[s >= 1]).value_counts()
              .rename_axis("period").rename("n_cat_new"))
        out = pd.concat([bw, cw], axis=1).fillna(0).astype(int).reset_index().sort_values("period")
        return out[["period", "n_brand_new", "n_cat_new"]]

    def _joined(self) -> pd.DataFrame:
        """Pull trialists' post-trial lines to pandas (small per analysis) and
        finish with the shared pandas interval maths."""
        F = self._F
        tr = self._p.sparkSession.createDataFrame(
            self.trials[["card", "trial_date"]].assign(
                trial_date=self.trials["trial_date"].astype(str)))
        j = (self._p.join(tr, on="card", how="inner")
             .withColumn("d", F.datediff("ts", F.to_date("trial_date")))
             .filter(F.col("d") > 0)).toPandas()
        return j

    def rbr_pooled(self) -> pd.DataFrame:
        return _rbr_pooled_from_joined(self._joined(), self.trials, self.cfg,
                                       self.analysis_date, self.origin,
                                       self._upper_interval())

    def rbr_cohort(self) -> pd.DataFrame:
        return _rbr_cohort_from_joined(self._joined(), self.trials, self.cfg,
                                       self.analysis_date, self.origin)

    def _upper_interval(self) -> int:
        mt = int(self.trials["max_interval"].max()) if len(self.trials) else 0
        return min(self.cfg.max_interval, mt) if self.cfg.max_interval is not None else mt

    def buying_scopes(self) -> pd.DataFrame:
        F = self._F
        cat = self._p.filter(F.col("is_cat"))
        if self.cfg.buying_index_window_days is not None:
            start = (self.analysis_date - pd.Timedelta(days=self.cfg.buying_index_window_days)).date() \
                if isinstance(self.analysis_date, pd.Timestamp) else \
                (pd.Timestamp(self.analysis_date) - pd.Timedelta(days=self.cfg.buying_index_window_days)).date()
            cat = cat.filter(F.col("ts") > F.lit(start.isoformat()).cast("date"))
        cbc = cat.groupBy("card").agg(F.sum("qty").alias("v")).toPandas().set_index("card")["v"]
        brand_counts = (self._p.filter(F.col("is_brand")).groupBy("card").count()
                        .toPandas().set_index("card")["count"])
        repeater_cards = set(brand_counts[brand_counts >= self.cfg.repeater_min_purchases].index)
        return _buying_scopes_from_cbc(cbc, self._trier_cards, repeater_cards,
                                       dict(zip(self.trials["card"], self.trials["cohort"])),
                                       self.cfg)

    def buying_series(self) -> pd.DataFrame:
        F, c = self._F, self.cfg
        cols = ["card", "ts", "qty"] + (["bucket"] if c.bucket_column else [])
        cat = (self._p.filter(F.col("is_cat") & (F.col("ts") >= F.lit(self._origin).cast("date")))
               .select(*cols).toPandas())
        return _buying_series_from_cat(cat, self._trier_cards, self.cfg, self.origin,
                                       self._origin_month, self._bucket_to_period)

    def share_long(self) -> pd.DataFrame:
        # Collect the post-origin lines and finish with the SAME pandas reduction
        # the local backend uses, so the calendar axis cannot drift between the
        # two (mirrors how buying_series / entrants already collect pre-aggregation).
        F, c = self._F, self.cfg
        cols = ["ts", "is_brand", "is_cat", "qty"] + (["bucket"] if c.bucket_column else [])
        d = (self._p.filter(F.col("ts") >= F.lit(self._origin).cast("date"))
             .select(*cols).toPandas())
        return _share_long_pandas(d, c, self.origin, self._origin_month, self._bucket_to_period)


# --------------------------------------------------------------------------- #
# Shared finishing helpers (used by the Spark path to reuse the pandas maths so
# the two backends cannot drift on the interval logic).
# --------------------------------------------------------------------------- #
def _max_interval_pandas(trial_ts: pd.Series, analysis_date, cfg: TRBConfig) -> np.ndarray:
    a = pd.Timestamp(analysis_date)
    tt = pd.to_datetime(trial_ts)
    if cfg.rbr_interval_mode == "exact":
        return np.floor((a - tt) / _DAY / cfg.period_length_days).astype(int).to_numpy()
    if cfg.rbr_bucket_unit == "week":
        awk = (a - pd.Timestamp(_MONDAY_EPOCH)).days // 7
        twk = ((tt - pd.Timestamp(_MONDAY_EPOCH)) / _DAY).astype(int) // 7
        return (awk - twk).to_numpy()
    ami = a.year * 12 + a.month - 1
    return (ami - (tt.dt.year * 12 + tt.dt.month - 1)).to_numpy()


def _interval_pandas(ts: pd.Series, ref: pd.Series, cfg: TRBConfig) -> np.ndarray:
    ts = pd.to_datetime(ts); ref = pd.to_datetime(ref)
    if cfg.rbr_interval_mode == "exact":
        d = (ts.to_numpy("datetime64[ns]") - ref.to_numpy("datetime64[ns]")) / _DAY
        return np.ceil(d / cfg.period_length_days).astype(int)
    if cfg.rbr_bucket_unit == "week":
        wk = lambda x: ((x.to_numpy("datetime64[ns]") - np.datetime64(_MONDAY_EPOCH)) / _DAY).astype(int) // 7
        return wk(ts) - wk(ref)
    return (ts.dt.year * 12 + ts.dt.month - 1).to_numpy() - (ref.dt.year * 12 + ref.dt.month - 1).to_numpy()


def _rbr_pooled_from_joined(j: pd.DataFrame, trials: pd.DataFrame, cfg, adate, origin,
                            upper: int) -> pd.DataFrame:
    tt = trials.set_index("card")["trial_date"]
    j = j.copy()
    j["trial_ref"] = pd.to_datetime(j["card"].map(tt))
    j["interval"] = _interval_pandas(j["ts"], j["trial_ref"], cfg)
    j["max_interval"] = j["card"].map(trials.set_index("card")["max_interval"])
    j = j[(j["interval"] >= 1) & (j["interval"] <= j["max_interval"])]
    j["bq"] = np.where(j["is_brand"], j["qty"], 0.0)
    j["cq"] = np.where(j["is_cat"], j["qty"], 0.0)
    agg = j.groupby("interval").agg(brand_qty=("bq", "sum"), cat_qty=("cq", "sum"))
    mt = trials["max_interval"].to_numpy()
    rows = [(t, float(agg["brand_qty"].get(t, 0.0)), float(agg["cat_qty"].get(t, 0.0)),
             int((mt >= t).sum())) for t in range(1, upper + 1)]
    return pd.DataFrame(rows, columns=["interval", "brand_qty", "cat_qty", "n_eligible"])


def _rbr_cohort_from_joined(j: pd.DataFrame, trials: pd.DataFrame, cfg, adate, origin) -> pd.DataFrame:
    if j.empty:
        return pd.DataFrame(columns=["cohort", "interval", "brand_qty", "cat_qty"])
    tt = trials.set_index("card")
    j = j.copy()
    j["trial_ref"] = pd.to_datetime(j["card"].map(tt["trial_date"]))
    j["cohort"] = j["card"].map(tt["cohort"])
    j["interval"] = _interval_pandas(j["ts"], j["trial_ref"], cfg)
    j["max_interval"] = j["card"].map(tt["max_interval"])
    j = j[(j["interval"] >= 1) & (j["interval"] <= j["max_interval"])]
    j["bq"] = np.where(j["is_brand"], j["qty"], 0.0)
    j["cq"] = np.where(j["is_cat"], j["qty"], 0.0)
    return (j.groupby(["cohort", "interval"]).agg(brand_qty=("bq", "sum"), cat_qty=("cq", "sum"))
            .reset_index()[["cohort", "interval", "brand_qty", "cat_qty"]])


def _buying_scopes_from_cbc(cbc: pd.Series, trier_cards, repeater_cards,
                            cohort_map, cfg: TRBConfig) -> pd.DataFrame:
    def stat(cards):
        s = cbc[cbc.index.isin(cards)]
        return float(s.sum()), int(s.size)
    rows = [("__all__", float(cbc.sum()), int(cbc.size)),
            ("__triers__", *stat(trier_cards)),
            ("__repeaters__", *stat(repeater_cards))]
    for label in cohort_order(cfg.cohort_boundaries_weeks, cfg.include_prelaunch_cohort):
        cards = {c for c, l in cohort_map.items() if l == label}
        rows.append((label, *stat(cards)))
    return pd.DataFrame(rows, columns=["scope", "sum_cat", "n_buyers"])


def _share_long_pandas(d: pd.DataFrame, cfg: TRBConfig, origin, origin_month: int,
                       bucket_to_period: dict) -> pd.DataFrame:
    """Realised brand/category volume per calendar-axis period. Shared by both
    backends (pandas operates on `_p`; Spark on the collected post-origin lines)."""
    d = d.copy()
    d["period"] = _periods_pandas(d["ts"], d.get("bucket"), cfg, origin,
                                  origin_month, bucket_to_period)
    d["bq"] = np.where(d["is_brand"], d["qty"], 0.0)
    d["cq"] = np.where(d["is_cat"], d["qty"], 0.0)
    g = (d.groupby("period").agg(brand_qty=("bq", "sum"), cat_qty=("cq", "sum"))
         .reset_index().sort_values("period"))
    return g[["period", "brand_qty", "cat_qty"]]


def _buying_series_from_cat(cat: pd.DataFrame, trier_cards, cfg: TRBConfig, origin,
                            origin_month: int, bucket_to_period: dict) -> pd.DataFrame:
    if cat.empty:
        return pd.DataFrame(columns=["period", "sel_sum", "sel_n", "all_sum", "all_n"])
    cat = cat.copy()
    cat["period"] = _periods_pandas(cat["ts"], cat.get("bucket"), cfg, origin,
                                    origin_month, bucket_to_period)
    cat["is_trier"] = cat["card"].isin(trier_cards)
    g_all = cat.groupby("period").agg(all_sum=("qty", "sum"), all_n=("card", "nunique"))
    g_sel = cat[cat["is_trier"]].groupby("period").agg(sel_sum=("qty", "sum"), sel_n=("card", "nunique"))
    out = g_all.join(g_sel, how="left").fillna(0.0).reset_index()
    return out[["period", "sel_sum", "sel_n", "all_sum", "all_n"]].sort_values("period")
