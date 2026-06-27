"""
TRB / Parfitt-Collins brand-share prediction in PySpark.

Implements the brand-share prediction method of Parfitt & Collins (1968),
"Use of Consumer Panels for Brand-Share Prediction", JMR 5(2):131-145

    Market Share = Trial Index x RBR(stable) x Buying Index

Design notes (read these, they encode decisions we settled, not assumptions):

* Faithful-to-Parfitt choices
  - RBR intervals are anchored to each shopper's EXACT trial date and are
    rolling windows of fixed length `period_length_days` (Parfitt: "calculation
    begins for each buyer from his date of entry ... not an expression of
    calendar time"). They are NOT calendar-month buckets.
  - RBR(t) is a SINGLE interval (the t-th window after trial), pooled across
    shoppers as a ratio of summed VOLUMEs (sum brand / sum category), with
    lapsed buyers included. It is not cumulative over 1..t.
  - Eligibility is INHERENT, not an extra filter: a shopper enters RBR(t) only
    if their entire t-th window has elapsed by `analysis_date`
    (trial_date + t*P <= analysis_date). This is the only way to avoid
    deflating long-horizon RBR; it is a consequence of the definition.
  - Trial Index is the ratio brand-penetration / category-penetration. Because
    it is a ratio over the same population base, the cardholder base cancels and
    equals (distinct brand triers / distinct category triers) -- Parfitt's
    "penetration as a percentage of category buyers". It is NOT computed
    separately: it is the end-of-window snapshot of the single penetration
    series (see Penetration), so the two can never disagree by construction.
  - The category-buyer denominator can be DYNAMIC (default; Parfitt appendix
    P(t)=ΣN(r)/ΣF(r), cumulative category triers entered by t) or STATIC (a
    fixed total, valid when the category base is saturated). Both converge to
    the same end-of-window snapshot. A category trier is a shopper whose first
    category purchase falls on/after the launch date (no pre-launch baseline).

* Deliberate deviation from Parfitt (settled earlier)
  - The trial term fed to `predict_share()` is OBSERVED penetration at
    `analysis_date` (the snapshot), NOT the projected ultimate level. Parfitt
    feeds the PROJECTED ultimate penetration K. `predict_share_projected()`
    offers the faithful K-based variant when the curve is projectable.

* Not automated
  - The "stable" RBR is NOT selected automatically. `rbr_series` holds the
    full series; the caller passes the chosen RBR AMOUNT to `predict_share()`.

The module is structured for testability (SOLID): each component
(`TrialIdentifier`, `RBRCalculator`, `PenetrationCalculator`,
`BuyingIndexCalculator`, `ShareSeriesCalculator`) has a single responsibility;
`TRBModel` orchestrates them and takes the transactions DataFrame directly.

Expected transaction schema (one row per purchase line):
    COMPANYCARD      : string/int  -- loyalty card id (single retailer per run)
    DATE_KEY        : date/string -- purchase date (the RBR anchor); the column
                                     name is configurable via `date_column`
                                     (e.g. DATE_KEY). RBR intervals are measured
                                     from each shopper's exact trial date.
    is_new_product  : boolean     -- True if the line is the new product/brand
    is_category     : boolean     -- True if the line is in the reference category
                                     (brand lines are in the category too)
    VOLUME, PIECES, AMOUNT : double -- quantity measures; one selected by config

Optional pre-computed calendar-bucket labels (e.g. ISOWEEKYEAR, YEARMONTH): when
`bucket_column` names one of them, the CALENDAR axes (penetration and share) are
bucketed by that label instead of by fixed day-windows, while RBR stays anchored
to the exact trial date. Bucket labels are mapped to a dense chronological
ordinal (1..N, ordered by each bucket's first observed date), so non-contiguous
labels across a year boundary (202352 -> 202401) become consecutive periods.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional, List, Dict, Tuple

import numpy as np
from pyspark.sql import DataFrame, SparkSession, Window, functions as F, types as T


# --------------------------------------------------------------------------- #
# Plotting is an OPTIONAL concern: matplotlib is imported lazily so the core
# model has no hard dependency on it. The chart methods live on TRBResult (the
# holder of the computed series) and reproduce the paper's figure styles.
# --------------------------------------------------------------------------- #
def _require_mpl():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        return plt
    except ImportError as e:               # pragma: no cover
        raise ImportError(
            "plotting requires matplotlib (pip install matplotlib)"
        ) from e


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TRBConfig:
    """All tunables live here, so nothing is a hidden constant."""
    measure: str = "VOLUME"                 # 'VOLUME' | 'PIECES' | 'AMOUNT'
    period_length_days: int = 30            # length P of one RBR interval
    repeater_min_purchases: int = 2         # >= this many brand buys => repeater
    launch_date: Optional[str] = None       # if set, trial must be on/after this
    analysis_date: Optional[str] = None     # "as of" date; default = max DATE_KEY
    max_interval: Optional[int] = None      # cap on t; default = max feasible
    treat_brand_as_category: bool = True    # OR brand into the category mask
    buying_index_base: str = "repeaters"    # 'repeaters' (Charan) | 'triers' (Parfitt)
    penetration_method: str = "discounted"  # 'discounted' (Gilchrist) | 'ols'
    discount_weight: float = 0.6            # Gilchrist lambda (paper uses w=0.6)
    penetration_denominator: str = "dynamic"  # 'dynamic' (Parfitt appendix) | 'static'
    date_column: str = "DATE_KEY"           # name of the purchase-date column (e.g. DATE_KEY)
    bucket_column: Optional[str] = None      # label column for calendar axes (e.g. YEARMONTH); None = day-windows

    def __post_init__(self) -> None:
        if self.measure not in ("VOLUME", "PIECES", "AMOUNT"):
            raise ValueError(f"measure must be VOLUME/PIECES/AMOUNT, got {self.measure!r}")
        if self.period_length_days <= 0:
            raise ValueError("period_length_days must be positive")
        if self.repeater_min_purchases < 2:
            raise ValueError("a repeater needs at least 2 brand purchases")
        if self.buying_index_base not in ("repeaters", "triers"):
            raise ValueError(f"buying_index_base must be repeaters/triers, got {self.buying_index_base!r}")
        if self.penetration_method not in ("discounted", "ols"):
            raise ValueError(f"penetration_method must be discounted/ols, got {self.penetration_method!r}")
        if not 0.0 < self.discount_weight <= 1.0:
            raise ValueError("discount_weight must be in (0, 1]")
        if self.penetration_denominator not in ("dynamic", "static"):
            raise ValueError(f"penetration_denominator must be dynamic/static, got {self.penetration_denominator!r}")


# --------------------------------------------------------------------------- #
# Shared preparation helpers
# --------------------------------------------------------------------------- #
def _origin_date(prepared: DataFrame, trials: DataFrame,
                 cfg: TRBConfig) -> Optional[date]:
    """Calendar origin for penetration / share: the launch date if given, else
    the first observed trial date. None when neither is available."""
    if cfg.launch_date is not None:
        return date.fromisoformat(cfg.launch_date)
    return trials.agg(F.min("trial_date").alias("o")).collect()[0]["o"]


def _period_expr(date_col, origin: date, P: int):
    """1-based day-window period index of `date_col` relative to `origin`:
    floor(days_since_origin / P) + 1. Used when no calendar bucket is chosen."""
    o = F.lit(origin.isoformat()).cast("date")
    return (F.floor(F.datediff(date_col, o) / F.lit(P)) + 1).cast("int")


def _with_period(prepared: DataFrame, origin: date, cfg: TRBConfig) -> DataFrame:
    """Keep rows on/after `origin` and attach a 1-based `period` column shared by
    penetration and share (so the calendar-axis rule lives in one place):

      * bucket mode (cfg.bucket_column set): a DENSE chronological ordinal over
        the observed bucket labels -- buckets ordered by their first observed
        date, numbered 1..N via row_number(). This makes non-contiguous labels
        (e.g. 202352 -> 202401) consecutive, exactly as intended.
      * day-window mode (default): floor(days_since_origin / P) + 1.

    RBR does NOT use this; it stays anchored to each shopper's exact trial date.
    """
    o = F.lit(origin.isoformat()).cast("date")
    df = prepared.filter(F.col("DATE_KEY") >= o)
    if cfg.bucket_column is None:
        return df.withColumn(
            "period", _period_expr(F.col("DATE_KEY"), origin, cfg.period_length_days))
    order = df.groupBy("bucket").agg(F.min("DATE_KEY").alias("_first"))
    bmap = (order.withColumn("period", F.row_number().over(Window.orderBy("_first")))
                 .select("bucket", "period"))
    return df.join(F.broadcast(bmap), on="bucket", how="inner")


def _prepare(df: DataFrame, cfg: TRBConfig) -> DataFrame:
    """Normalise types, select the active measure as `qty`, derive masks, clip to
    analysis_date. Returns: COMPANYCARD, DATE_KEY, is_brand, is_cat, qty
    (+ `bucket` when cfg.bucket_column is set)."""
    out = (
        df.withColumn("DATE_KEY", F.to_date(F.col(cfg.date_column)))
          .withColumn("is_brand", F.col("is_new_product").cast("boolean"))
    )
    cat_mask = F.col("is_category").cast("boolean")
    if cfg.treat_brand_as_category:
        cat_mask = cat_mask | F.col("is_brand")
    out = out.withColumn("is_cat", cat_mask)
    out = out.withColumn("qty", F.col(cfg.measure).cast("double"))
    cols = ["COMPANYCARD", "DATE_KEY", "is_brand", "is_cat", "qty"]
    if cfg.bucket_column is not None:
        out = out.withColumn("bucket", F.col(cfg.bucket_column))
        cols.append("bucket")

    if cfg.analysis_date is not None:
        out = out.filter(F.col("DATE_KEY") <= F.lit(cfg.analysis_date).cast("date"))
    return out.select(*cols)


def _resolve_analysis_date(prepared: DataFrame, cfg: TRBConfig) -> date:
    if cfg.analysis_date is not None:
        return date.fromisoformat(cfg.analysis_date)
    return prepared.agg(F.max("DATE_KEY").alias("d")).collect()[0]["d"]


# --------------------------------------------------------------------------- #
# Component 1: identify each shopper's trial date
# --------------------------------------------------------------------------- #
class TrialIdentifier:
    """trial_date = first brand purchase per shopper (on/after launch_date)."""
    def __init__(self, cfg: TRBConfig):
        self._cfg = cfg

    def identify(self, prepared: DataFrame) -> DataFrame:
		# identifico il primo acquisto per ogni carta (se conosco la data di lancio impongo che sia oltre, così evito casini SPE)
        brand = prepared.filter(F.col("is_brand"))
        if self._cfg.launch_date is not None:
            brand = brand.filter(F.col("DATE_KEY") >= F.lit(self._cfg.launch_date).cast("date"))
        return (brand.groupBy("COMPANYCARD")
                     .agg(F.min("DATE_KEY").alias("trial_date")))


# --------------------------------------------------------------------------- #
# Component 2: repeat-buying rate series RBR(t)
# --------------------------------------------------------------------------- #
@dataclass
class RBRPoint:
    interval: int
    rbr: Optional[float]      # None if no observed category VOLUME yet
    brand_qty: float
    category_qty: float
    n_eligible: int           # shoppers whose t-th window has fully elapsed


class RBRCalculator:
    """Pooled RBR(t) over rolling windows anchored to each shopper's trial date.

    Window t = (trial + (t-1)*P, trial + t*P]  (days). The trial purchase
    itself (day 0) is excluded. A shopper is eligible for interval t iff the
    whole window has elapsed: trial + t*P <= analysis_date.
    """
    def __init__(self, cfg: TRBConfig):
        self._cfg = cfg

    def compute(self, prepared: DataFrame, trials: DataFrame,
                analysis_date: date) -> List[RBRPoint]:
        P = self._cfg.period_length_days
        adate = F.lit(analysis_date.isoformat()).cast("date")

        # Per shopper, the highest fully-elapsed interval (eligibility ceiling).
        trials = trials.withColumn(
            "max_t", F.floor(F.datediff(adate, F.col("trial_date")) / F.lit(P)).cast("int")
        )

        # Transactions of trialists, days since trial, interval index t.
        joined = (
            prepared.join(trials, on="COMPANYCARD", how="inner")
                    .withColumn("d", F.datediff("DATE_KEY", "trial_date"))
                    .filter(F.col("d") > 0)                       # strictly after trial
                    .withColumn("t", F.ceil(F.col("d") / F.lit(P)).cast("int"))
                    .filter(F.col("t") <= F.col("max_t"))         # eligibility (inherent)
        )

        agg = (
            joined.groupBy("t").agg(
                F.sum(F.when(F.col("is_brand"), F.col("qty")).otherwise(0.0)).alias("brand_qty"),
                F.sum(F.when(F.col("is_cat"), F.col("qty")).otherwise(0.0)).alias("category_qty"),
            )
        )

        # True cohort size at t = shoppers with max_t >= t (reached the window,
        # whether or not they purchased in it).
        max_t_overall = trials.agg(F.max("max_t").alias("m")).collect()[0]["m"] or 0
        upper = self._cfg.max_interval or max_t_overall
        upper = min(upper, max_t_overall)

        cohort = {r["t"]: (r["brand_qty"], r["category_qty"])
                  for r in agg.collect()}
        # n_eligible per t via cumulative count of max_t.
        elig_rows = (trials.groupBy("max_t").count()
                           .collect())
        # n_eligible(t) = sum of counts where max_t >= t
        max_t_counts: Dict[int, int] = {r["max_t"]: r["count"] for r in elig_rows}
        series: List[RBRPoint] = []
        for t in range(1, int(upper) + 1):
            b, c = cohort.get(t, (0.0, 0.0))
            n_elig = sum(cnt for mt, cnt in max_t_counts.items() if mt is not None and mt >= t)
            rbr = (b / c) if c and c > 0 else None
            series.append(RBRPoint(interval=t, rbr=rbr, brand_qty=float(b),
                                   category_qty=float(c), n_eligible=int(n_elig)))
        return series


# --------------------------------------------------------------------------- #
# Component 3: Penetration
#
# Model form:
#     P(t) = K * (1 - exp(-a*t))                 [modified exponential]
# stochastic difference-equation form the paper prints:
#     dP(t) = a*(K - P(t)) + eps,   dP(t) = (P(t+1) - P(t-1)) / 2
# K = ultimate penetration (the projected "trial" term in Parfitt's formula).
#
# DENOMINATOR (Parfitt appendix P(t) = ΣN(r)/ΣF(r)):
#   N(r) = brand triers entering at r ; F(r) = category triers entering at r,
#   where "entering" = first brand/category purchase on/after the launch origin.
#   'dynamic' (default): denominator = cumulative F up to t (grows with the
#       category, faithful to the appendix and to Figure 10's changing market).
#   'static': denominator = total F over the window (valid when the category is
#       saturated; this is what the figure footnotes "total = 100%" assume).
#   Both converge at the last period to N_total/F_total = the trial-index snapshot.
#
# FIDELITY CAVEAT on the fit: the paper does NOT publish its exact estimation
# algorithm (defers to a thesis, calls Anscombe's MLE "arduous"). We regress the
# printed difference equation dP(t) on P(t) (slope=-a, intercept=a*K) by
# discounted least squares (Gilchrist [4], w default 0.6) or plain OLS.
# --------------------------------------------------------------------------- #
@dataclass
class Penetration:
    denominator: str                        # 'dynamic' | 'static'
    origin: date
    series: List[Tuple[int, float]]         # [(period, P(period)), ...] (chosen denominator)
    n_brand_triers: int                     # N_total
    n_category_triers: int                  # F_total (triers on/after launch)
    ultimate_penetration: Optional[float] = None   # K; None if not fitted/estimable; dopo 1/a periodi sei al 63.2% della penetrazione finale attesa, dopo 3/a periodi sei al 95%, dopo 4.6/a sei al 99%
    growth_rate: Optional[float] = None            # a
    note: str = ""

    @property
    def snapshot(self) -> float:
        """End-of-window penetration N_total/F_total (== series[-1]); this is
        the observed Trial Index."""
        return self.n_brand_triers / self.n_category_triers

    def fitted(self, t: float) -> Optional[float]:
        """Theoretical P(t) = K*(1 - e^{-a t}). None when not fitted."""
        if self.ultimate_penetration is None or self.growth_rate is None:
            return None
        import math
        return self.ultimate_penetration * (1.0 - math.exp(-self.growth_rate * t))


class PenetrationCalculator:
    """Builds the cumulative penetration curve once and (optionally) fits K/a.
    Everything penetration-related downstream reads from the returned object:
    the Trial Index is `snapshot`, the charts use `series`, the projected share
    uses `ultimate_penetration`."""
    def __init__(self, cfg: TRBConfig):
        self._cfg = cfg

    def compute(self, prepared: DataFrame, trials: DataFrame,
                *, fit: bool) -> Optional[Penetration]:
        origin = _origin_date(prepared, trials, self._cfg)
        if origin is None:
            return None
        # one period column (bucket ordinal or day-window), shared with share
        dfp = _with_period(prepared, origin, self._cfg)  # periodi quantizzati, in modo da sapere se l'atto è stato fatto nel periodo 1, 2, ecc.

        # entry period of a shopper = earliest period in which they bought
        # the brand / the category (first purchase on/after the launch origin)
        brand_new = self._entries(dfp, F.col("is_brand"))
        cat_new = self._entries(dfp, F.col("is_cat"))

        n_brand = sum(brand_new.values())
        n_cat = sum(cat_new.values())
        if n_cat == 0:
            raise ValueError("no category triers on/after the launch date")

		# calcolo cumulata
        T = max([*brand_new.keys(), *cat_new.keys()])
        dynamic = self._cfg.penetration_denominator == "dynamic"
        series: List[Tuple[int, float]] = []
        cb = cc = 0
        for t in range(1, T + 1):
            cb += brand_new.get(t, 0)
            cc += cat_new.get(t, 0)
            if cc == 0:                       # no one in the market yet
                continue
            series.append((t, cb / cc if dynamic else cb / n_cat))

        pen = Penetration(denominator=self._cfg.penetration_denominator,
                          origin=origin, series=series,
                          n_brand_triers=n_brand, n_category_triers=n_cat)
        if fit:
            self._fit(pen)
        return pen

    @staticmethod
    def _entries(dfp: DataFrame, mask) -> Dict[int, int]:
        """New entrants per period = count of shoppers whose first period under
        `mask` is that period."""
        entry = (dfp.filter(mask).groupBy("COMPANYCARD")
                    .agg(F.min("period").alias("p")))
        rows = entry.groupBy("p").agg(F.count(F.lit(1)).alias("n")).collect()
        return {int(r["p"]): r["n"] for r in rows if r["p"] is not None}

    def _fit(self, pen: Penetration) -> None:
        series = pen.series
        if len(series) < 4:
            pen.note = "need >=4 periods to fit; too few observed"
            return
        ps = np.array([p for _, p in series], dtype=float)
        dP = (ps[2:] - ps[:-2]) / 2.0          # centred difference, interior points
        x = ps[1:-1]
        if len(x) < 2:
            pen.note = "too few interior points to fit"
            return
        # Regress dP(t) on P(t): slope=-a, intercept=a*K. Discounted least squares
        # (Gilchrist [4]) weights recent points more (np.polyfit minimises
        # sum((w*resid)^2), so w_i = sqrt(lambda^age), age=0 for the latest point).
        # A degree-1 fit on few/collinear points can raise RankWarning; it does
        # not affect the slope/intercept we use, so we silence it locally.
        import warnings
        with warnings.catch_warnings():
            # warnings.simplefilter("ignore", np.exceptions.RankWarning)
			_rw = (getattr(getattr(np, 'exceptions', None), 'RankWarning', None)
					or getattr(np, 'RankWarning', Warning))
			warnings.simplefilter("ignore", _rw)
            if self._cfg.penetration_method == "discounted":
                lam = self._cfg.discount_weight
                ages = np.arange(len(x) - 1, -1, -1, dtype=float)
                w = np.sqrt(lam ** ages)
                slope, intercept = np.polyfit(x, dP, 1, w=w)
            else:
                slope, intercept = np.polyfit(x, dP, 1)
        a = -slope
        if a <= 0:
            pen.note = ("rate of increase not yet declining (a<=0): cannot project "
                        "ultimate penetration -- wait for the curve to decelerate")
            return
        pen.ultimate_penetration = float(intercept / a)
        pen.growth_rate = float(a)
        if pen.ultimate_penetration < max(ps):
            pen.note = ("estimated K below the latest observed penetration; fit "
                        "unstable (e.g. a dynamic denominator still growing) -- caution")


# --------------------------------------------------------------------------- #
# Component 3c: Realised brand share over CALENDAR time
#
# S(tau) = brand qty / category qty, bucketed into fixed calendar periods of
# length P from launch (or first trial). This is the directly-OBSERVED market-
# share trajectory -- the realistic analogue of Charan's Exhibit 11.17 bottom
# panel. It is NOT the multiplicative Parfitt prediction (that yields a single
# equilibrium number, the tau->infinity limit of this curve), and NOT on the
# RBR relative-time axis. RBR stays anchored to each shopper's trial; share is
# calendar-time, which is the correct axis for "what is my share this period".
#
# SCOPE CAVEAT: on single-retailer loyalty data this is the brand's share of
# category VOLUME among CARDED purchases at that retailer -- a proxy for market
# share within that panel, not the total-market share.
# --------------------------------------------------------------------------- #
class ShareSeriesCalculator:
    def __init__(self, cfg: TRBConfig):
        self._cfg = cfg

    def compute(self, prepared: DataFrame,
                trials: DataFrame) -> List[Tuple[int, Optional[float]]]:
        origin = _origin_date(prepared, trials, self._cfg)
        if origin is None:
            return []
        dfp = _with_period(prepared, origin, self._cfg)
        agg = (dfp.groupBy("period").agg(
                   F.sum(F.when(F.col("is_brand"), F.col("qty")).otherwise(0.0)).alias("b"),
                   F.sum(F.when(F.col("is_cat"), F.col("qty")).otherwise(0.0)).alias("c"))
                 .orderBy("period").collect())
        out: List[Tuple[int, Optional[float]]] = []
        for r in agg:
            if r["period"] is None:
                continue
            c = r["c"]
            out.append((int(r["period"]), (r["b"] / c) if c and c > 0 else None))
        return out


# --------------------------------------------------------------------------- #
# Ultimate RBR (NOT a Parfitt model -- operationalises "level off")
#
# The appendix gives NO parametric RBR projection; it defines the ultimate RBR
# as the limit of the levelling-off curve and the body says there is no rule /
# equilibrium for repeat purchasing. This helper therefore only DETECTS the
# plateau: the value where successive RBR(t) move by less than `tol` for `k`
# consecutive intervals. It is an operationalisation, not Parfitt's formula.
# --------------------------------------------------------------------------- #
def estimate_ultimate_rbr_plateau(series: List["RBRPoint"],
                                  tol: float = 0.005,
                                  k: int = 3) -> Optional[Tuple[int, float]]:
    """Return (interval, rbr) at the start of the first stable run, or None."""
    pts = [(p.interval, p.rbr) for p in series if p.rbr is not None]
    pts.sort()
    for i in range(len(pts) - k + 1):
        window = pts[i:i + k]
        vals = [v for _, v in window]
        if max(vals) - min(vals) <= tol:
            return window[0]
    return None


# --------------------------------------------------------------------------- #
# Component 4: Buying Index (on REPEAT buyers, per user's request)
# --------------------------------------------------------------------------- #
class BuyingIndexCalculator:
    """Buying Index = avg category VOLUME per selected buyer
                      / avg category VOLUME per category buyer.

    The selected-buyer base is set by cfg.buying_index_base:
      'repeaters' (Charan's TRB restatement): shoppers with
          >= repeater_min_purchases brand purchase lines.
      'triers'    (Parfitt's original article): all brand triers
          (>= 1 brand purchase line). This is the faithful definition.
    """
    def __init__(self, cfg: TRBConfig):
        self._cfg = cfg

    def compute(self, prepared: DataFrame) -> float:
        # brand purchase count per shopper
        brand_counts = (prepared.filter(F.col("is_brand"))
                                .groupBy("COMPANYCARD")
                                .agg(F.count(F.lit(1)).alias("n_brand")))
        threshold = (1 if self._cfg.buying_index_base == "triers"
                     else self._cfg.repeater_min_purchases)
        selected = brand_counts.filter(
            F.col("n_brand") >= threshold
        ).select("COMPANYCARD")

        # category VOLUME per shopper (category buyers only)
        cat_vol = (prepared.filter(F.col("is_cat"))
                           .groupBy("COMPANYCARD")
                           .agg(F.sum("qty").alias("cat_vol")))

        all_stats = cat_vol.agg(
            F.sum("cat_vol").alias("v"), F.count(F.lit(1)).alias("n")
        ).collect()[0]
        if not all_stats["n"]:
            raise ValueError("no category buyers found")
        avg_all = all_stats["v"] / all_stats["n"]

        sel_stats = (cat_vol.join(selected, on="COMPANYCARD", how="inner")
                            .agg(F.sum("cat_vol").alias("v"),
                                 F.count(F.lit(1)).alias("n"))
                            .collect()[0])
        if not sel_stats["n"]:
            raise ValueError(f"no {self._cfg.buying_index_base} found "
                             "(cannot compute buying index)")
        avg_sel = sel_stats["v"] / sel_stats["n"]
        return avg_sel / avg_all


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
@dataclass
class TRBResult:
    trial_index: float
    buying_index: float
    rbr_series: List[RBRPoint]
    analysis_date: date
    penetration: Optional[Penetration] = None
    share_series: List[Tuple[int, Optional[float]]] = field(default_factory=list)

    def rbr_at(self, interval: int) -> Optional[float]:
        for p in self.rbr_series:
            if p.interval == interval:
                return p.rbr
        return None

    def predict_share(self, rbr_value: float) -> float:
        """OBSERVED-trial share = Trial Index x RBR x Buying Index.
        Trial Index here is observed penetration relative to the category."""
        return self.trial_index * rbr_value * self.buying_index

    def predict_share_projected(self, rbr_value: float) -> Optional[float]:
        """PARFITT share = projected ultimate penetration (K) x RBR x Buying.
        Returns None if penetration could not be projected."""
        if self.penetration is None or self.penetration.ultimate_penetration is None:
            return None
        return self.penetration.ultimate_penetration * rbr_value * self.buying_index

    def ultimate_rbr(self, tol: float = 0.005, k: int = 3) -> Optional[Tuple[int, float]]:
        """Operationalised 'level off' value (NOT a Parfitt formula)."""
        return estimate_ultimate_rbr_plateau(self.rbr_series, tol=tol, k=k)

    # ----------------------------------------------------------------------- #
    # Charts (Parfitt figure styles). Each takes an optional matplotlib Axes
    # and RETURNS it, so callers can compose, restyle, or save. matplotlib is
    # imported lazily; these do nothing to the computation.
    # ----------------------------------------------------------------------- #
    def plot_penetration(self, ax=None, project_to: Optional[int] = None,
                         actual_overlay: Optional[List[Tuple[int, float]]] = None,
                         as_percent: bool = True,
                         title: str = "Cumulative penetration"):
        """Figure 1 / 3 style. Solid = observed (raw data); dashed = the
        PROJECTED ('expected') portion from the fitted K(1-e^{-a t}) curve,
        with a dotted line at the estimated ultimate level K.

        `actual_overlay` (list of (period, P_fraction)) draws a SECOND solid
        line of real penetration measured later, so you can see the gap between
        the theoretical projection and reality (Parfitt's Figures 12-13/18,
        marketing-disturbance attribution). Pass actuals you collected after the
        analysis date the projection was fitted on.
        """
        plt = _require_mpl()
        if self.penetration is None or not self.penetration.series:
            raise ValueError("no penetration series to plot "
                             "(run the model with a launch_date)")
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4.5))
        scale = 100.0 if as_percent else 1.0
        unit = "%" if as_percent else ""

        obs = sorted(self.penetration.series)
        ts = [t for t, _ in obs]
        ps = [p * scale for _, p in obs]
        last_obs = ts[-1]
        ax.plot(ts, ps, "-o", color="tab:blue", ms=4, label="Observed (raw data)")

        K = self.penetration.ultimate_penetration
        a = self.penetration.growth_rate
        if K is not None and a is not None:
            if project_to is None:
                # extend until ~99% of K, at least 6 periods past last observed
                t = last_obs
                while (t < last_obs + 60
                       and self.penetration.fitted(t) is not None
                       and self.penetration.fitted(t) < 0.99 * K):
                    t += 1
                project_to = max(t, last_obs + 6)
            tp = list(range(last_obs, project_to + 1))
            pp = [self.penetration.fitted(t) * scale for t in tp]
            ax.plot(tp, pp, "--", color="tab:blue", label="Projection (expected)")
            ax.axhline(K * scale, ls=":", color="grey", lw=1)
            ax.annotate(f"Estimated ultimate level {K * scale:.1f}{unit}",
                        xy=(project_to, K * scale), ha="right", va="bottom",
                        fontsize=8, color="grey")

        if actual_overlay:
            ao = sorted(actual_overlay)
            ax.plot([t for t, _ in ao], [p * scale for _, p in ao],
                    "-s", color="tab:red", ms=4, label="Actual (overlay)")

        ax.set_xlabel("Periods after launch")
        ax.set_ylabel(f"Penetration{(' (' + unit + ')') if as_percent else ''}")
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.margins(x=0.02)
        ax.legend(fontsize=8)
        return ax

    def plot_rbr(self, ax=None, mark_plateau: bool = True,
                 as_percent: bool = True, title: str = "Repeat-buying rate"):
        """Figure 2 style. RBR(t) vs interval, with a dotted line at the
        operationalised plateau (NOT a Parfitt formula; see ultimate_rbr)."""
        plt = _require_mpl()
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4.5))
        scale = 100.0 if as_percent else 1.0
        unit = "%" if as_percent else ""

        pts = sorted((p.interval, p.rbr) for p in self.rbr_series if p.rbr is not None)
        if not pts:
            raise ValueError("no RBR points to plot")
        xs = [t for t, _ in pts]
        ys = [v * scale for _, v in pts]
        ax.plot(xs, ys, "-o", color="tab:green", ms=4, label="RBR(t)")

        if mark_plateau:
            plat = self.ultimate_rbr()
            if plat is not None:
                t0, v0 = plat
                ax.axhline(v0 * scale, ls=":", color="grey", lw=1)
                ax.axvline(t0, ls=":", color="grey", lw=1, alpha=0.6)
                ax.annotate(f"Estimated ultimate level {v0 * scale:.1f}{unit}",
                            xy=(xs[-1], v0 * scale), ha="right", va="bottom",
                            fontsize=8, color="grey")

        ax.set_xlabel("Interval after first purchase (window t)")
        ax.set_ylabel(f"RBR{(' (' + unit + ')') if as_percent else ''}")
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.margins(x=0.02)
        ax.legend(fontsize=8)
        return ax

    def plot_predicted_share(self, ax=None, base: str = "observed",
                             as_percent: bool = True,
                             title: str = "Predicted brand share by RBR maturity"):
        """Predicted share = Trial x RBR(t) x Buying, plotted across the RBR
        interval t used. This shows the PREDICTION stabilising as RBR matures.

        NOTE: this is NOT the paper's Figure 5 (actual four-week shares over
        calendar time) -- the model does not compute realised share history.
        `base='observed'` uses the observed trial index; `base='projected'`
        uses the projected ultimate penetration K (Parfitt's original term)."""
        plt = _require_mpl()
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4.5))
        scale = 100.0 if as_percent else 1.0
        unit = "%" if as_percent else ""

        if base == "projected":
            trial = (self.penetration.ultimate_penetration
                     if self.penetration else None)
            if trial is None:
                raise ValueError("base='projected' needs an estimable ultimate penetration")
            label = "Predicted share (projected K)"
        elif base == "observed":
            trial = self.trial_index
            label = "Predicted share (observed trial)"
        else:
            raise ValueError("base must be 'observed' or 'projected'")

        pts = sorted((p.interval, p.rbr) for p in self.rbr_series if p.rbr is not None)
        if not pts:
            raise ValueError("no RBR points to plot")
        xs = [t for t, _ in pts]
        ys = [trial * v * self.buying_index * scale for _, v in pts]
        ax.plot(xs, ys, "-o", color="tab:purple", ms=4, label=label)

        ax.set_xlabel("RBR interval used (t)")
        ax.set_ylabel(f"Predicted share{(' (' + unit + ')') if as_percent else ''}")
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.margins(x=0.02)
        ax.legend(fontsize=8)
        return ax

    def plot_dashboard(self, as_percent: bool = True, share_base: str = "observed"):
        """Convenience: penetration | RBR | predicted share, side by side.
        Returns the matplotlib Figure."""
        plt = _require_mpl()
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        try:
            self.plot_penetration(ax=axes[0], as_percent=as_percent)
        except ValueError as e:
            axes[0].set_title("penetration n/a")
            axes[0].text(0.5, 0.5, str(e), ha="center", va="center",
                         fontsize=8, wrap=True, transform=axes[0].transAxes)
        self.plot_rbr(ax=axes[1], as_percent=as_percent)
        self.plot_predicted_share(ax=axes[2], base=share_base, as_percent=as_percent)
        fig.tight_layout()
        return fig

    def plot_share_over_time(self, ax=None, as_percent: bool = True,
                             show_equilibrium: bool = True,
                             title: str = "Realised brand share over time"):
        """Charan Exhibit 11.17 (bottom panel) analogue: realised share by
        CALENDAR period, measured directly from transactions (no modelling
        assumptions). An overshoot-then-decline here is the early-entrant
        high-RBR effect (Parfitt's stage-of-entry refinement) -- i.e. the
        year-1-peak / year-2-decline pattern.

        If `show_equilibrium` and the ultimate penetration + RBR plateau are
        estimable, a dotted line marks the Parfitt equilibrium share (the
        tau->infinity limit), so you can see the trajectory relative to it."""
        plt = _require_mpl()
        if not self.share_series:
            raise ValueError("no share series (run the model first)")
        pts = [(t, s) for t, s in self.share_series if s is not None]
        if not pts:
            raise ValueError("share series has no non-empty periods")
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4.5))
        scale = 100.0 if as_percent else 1.0
        unit = "%" if as_percent else ""

        xs = [t for t, _ in pts]
        ys = [s * scale for _, s in pts]
        ax.plot(xs, ys, "-o", color="tab:orange", ms=4, label="Realised share")

        if show_equilibrium:
            plat = self.ultimate_rbr()
            eq = (self.predict_share_projected(plat[1])
                  if plat is not None else None)
            if eq is not None:
                ax.axhline(eq * scale, ls=":", color="grey", lw=1)
                ax.annotate(f"Parfitt equilibrium {eq * scale:.1f}{unit}",
                            xy=(xs[-1], eq * scale), ha="right", va="bottom",
                            fontsize=8, color="grey")

        ax.set_xlabel("Period after launch (calendar time)")
        ax.set_ylabel(f"Share{(' (' + unit + ')') if as_percent else ''}")
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.margins(x=0.02)
        ax.legend(fontsize=8)
        return ax


class TRBModel:
    """Composes the components. Takes the transactions DataFrame and a TRBConfig."""
    def __init__(self, df: DataFrame, cfg: TRBConfig = TRBConfig(),
                 project_penetration: bool = True):
        self._df = df
        self._cfg = cfg
        self._project = project_penetration
        self._trial_id = TrialIdentifier(cfg)
        self._rbr = RBRCalculator(cfg)
        self._buying_idx = BuyingIndexCalculator(cfg)
        self._pen = PenetrationCalculator(cfg)
        self._share = ShareSeriesCalculator(cfg)

    def run(self) -> TRBResult:
        prepared = _prepare(self._df, self._cfg).cache()  # df transazionale ridotto: COMPANYCARD, DATE_KEY, is_brand, is_category, qty
        analysis_date = _resolve_analysis_date(prepared, self._cfg)  # datetime.date: assumi di avere i dati solo fino a questa data
        trials = self._trial_id.identify(prepared).cache()  # COMPANYCARD, trial_date

        # Penetration is computed ONCE; the trial index is its end-of-window
        # snapshot, so the two cannot diverge.
        penetration = self._pen.compute(prepared, trials, fit=self._project)
        if penetration is None:
            prepared.unpersist()
            trials.unpersist()
            raise ValueError("cannot determine the launch origin: set launch_date "
                             "or ensure there is at least one trial")
        trial_index = penetration.snapshot
        buying_index = self._buying_idx.compute(prepared)
        rbr_series = self._rbr.compute(prepared, trials, analysis_date)
        share_series = self._share.compute(prepared, trials)

        prepared.unpersist()
        trials.unpersist()
        return TRBResult(
            trial_index=trial_index,
            buying_index=buying_index,
            rbr_series=rbr_series,
            analysis_date=analysis_date,
            penetration=penetration,
            share_series=share_series,
        )