"""Orchestrator: compose the aggregation backend with the modelling core.

`run_trb(transactions, cfg, backend=...)` is the single entry point. It returns a
:class:`TRBResult` holding every series the analyst needs plus convenience
predictors. The old single-method classes (TrialIdentifier, RBRCalculator, ...)
are gone -- their logic now lives as plain functions in ``core``/``aggregation``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .aggregation import Aggregator, PandasAggregator, SparkAggregator
from .cohorts import cohort_order
from .config import TRBConfig
from .core import (
    Cohort, Penetration, RBRPoint, blended_rbr, build_cohorts, build_penetration,
    build_rbr, buying_index_from_scopes, buying_index_series, detect_plateau,
    fit_penetration, last_available_rbr, segmented_share, share_series,
)


@dataclass
class TRBResult:
    trial_index: float                 # observed penetration snapshot N_tot/F_tot
    buying_index: float                # overall B on the analysis window (base=cfg)
    rbr_series: List[RBRPoint]
    analysis_date: date
    origin: Optional[date] = None
    penetration: Optional[Penetration] = None
    cohorts: List[Cohort] = field(default_factory=list)
    buying_index_series: List[Tuple[int, Optional[float]]] = field(default_factory=list)
    share_series: List[Tuple[int, Optional[float], float, float]] = field(default_factory=list)
    config: Optional[TRBConfig] = None
    # calendar axis the period-indexed series above live on: 'week' | 'month' |
    # 'bucket', plus the period-ordinal -> calendar-label map for display.
    period_unit: str = "week"
    period_labels: Dict[int, str] = field(default_factory=dict)

    def label(self, period: int) -> str:
        """Calendar label for a period ordinal (falls back to the ordinal)."""
        return self.period_labels.get(int(period), str(int(period)))

    # -- RBR accessors ------------------------------------------------------ #
    def rbr_at(self, interval: int) -> Optional[float]:
        for p in self.rbr_series:
            if p.interval == interval:
                return p.rbr
        return None

    def ultimate_rbr(self) -> Optional[Tuple[int, float]]:
        """Furthest-available RBR (the r→∞ proxy used by the share formula)."""
        return last_available_rbr(self.rbr_series)

    def detect_plateau(self, tol: float = 0.005, k: int = 3):
        """DIAGNOSTIC: where the RBR curve levels off (not used by the share)."""
        return detect_plateau(self.rbr_series, tol=tol, k=k)

    # -- share predictors --------------------------------------------------- #
    def predict_share(self, rbr_value: float) -> float:
        """Trial Index × RBR × Buying Index (observed-trial multiplier).
        Reproduces the paper's worked example 34% × 25% × 1.00 = 8.5%."""
        return self.trial_index * rbr_value * self.buying_index

    def predict_share_projected(self, rbr_value: Optional[float] = None) -> Optional[float]:
        """FAITHFUL prediction = projected ultimate K × R(last) × B(analysis)
        (point 2). `rbr_value` defaults to the furthest-available RBR."""
        if self.penetration is None or self.penetration.ultimate_penetration is None:
            return None
        if rbr_value is None:
            ur = self.ultimate_rbr()
            if ur is None:
                return None
            rbr_value = ur[1]
        return self.penetration.ultimate_penetration * rbr_value * self.buying_index

    def segmented_share(self) -> float:
        """Σ Pᵢ × Rᵢ × Bᵢ over entry cohorts (Table 2). The headline number."""
        return segmented_share(self.cohorts)

    def blended_rbr(self) -> Optional[float]:
        """Penetration-weighted average RBR across cohorts (display only)."""
        return blended_rbr(self.cohorts)

    # -- display tables ----------------------------------------------------- #
    def cohort_table(self) -> pd.DataFrame:
        """Table 2 as a DataFrame: penetration, RBR, buying index, contribution."""
        rows = [{
            "cohort": c.label, "penetration": c.penetration, "rbr": c.rbr,
            "buying_index": c.buying_index, "contribution": c.contribution,
            "n_triers": c.n_triers, "future": c.is_future,
        } for c in self.cohorts]
        df = pd.DataFrame(rows)
        if not df.empty:
            total = {
                "cohort": "TOTAL", "penetration": df["penetration"].sum(),
                "rbr": self.blended_rbr(), "buying_index": None,
                "contribution": df["contribution"].sum(),
                "n_triers": int(df["n_triers"].sum()), "future": False,
            }
            df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)
        return df

    def share_ratio_series(self) -> List[Tuple[int, Optional[float]]]:
        """(period, brand-share ratio) -- the realised calendar-week share."""
        return [(p, r) for p, r, _, _ in self.share_series]


def run_trb(transactions, cfg: TRBConfig = TRBConfig(), *,
            backend: str = "pandas", project_penetration: bool = True) -> TRBResult:
    """End-to-end run. `transactions` is a pandas DataFrame (backend='pandas')
    or a Spark DataFrame (backend='spark')."""
    if backend == "pandas":
        agg: Aggregator = PandasAggregator(transactions, cfg)
    elif backend == "spark":
        agg = SparkAggregator(transactions, cfg)
    else:
        raise ValueError("backend must be 'pandas' or 'spark'")
    return _assemble(agg, cfg, project_penetration)


def _assemble(agg: Aggregator, cfg: TRBConfig, project: bool) -> TRBResult:
    pen = build_penetration(agg.entrants(), agg.origin, cfg.penetration_denominator)
    if project:
        fit_penetration(pen, method=cfg.penetration_method,
                        discount_weight=cfg.discount_weight)

    scopes = agg.buying_scopes()
    base_scope = "__triers__" if cfg.buying_index_base == "triers" else "__repeaters__"
    bi = buying_index_from_scopes(scopes, base_scope)
    bi = 1.0 if bi is None else bi

    order = cohort_order(cfg.cohort_boundaries_weeks, cfg.include_prelaunch_cohort)
    cohorts = build_cohorts(agg.trials, agg.rbr_cohort(), scopes,
                            agg.n_category_triers, order,
                            ultimate_penetration=pen.ultimate_penetration)

    share = share_series(agg.share_long())
    bidx = buying_index_series(agg.buying_series())
    all_periods = ({p for p, _ in pen.series}
                   | {p for p, *_ in share} | {p for p, _ in bidx})
    return TRBResult(
        trial_index=pen.snapshot,
        buying_index=bi,
        rbr_series=build_rbr(agg.rbr_pooled()),
        analysis_date=agg.analysis_date,
        origin=agg.origin,
        penetration=pen,
        cohorts=cohorts,
        buying_index_series=bidx,
        share_series=share,
        config=cfg,
        period_unit=getattr(agg, "period_unit", "week"),
        period_labels=agg.period_labels(all_periods),
    )
