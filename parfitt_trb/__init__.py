"""Parfitt-Collins (TRB) brand-share prediction.

    Market Share = Trial (penetration) × Repeat (RBR) × Buying Index

Faithful to Parfitt & Collins (1968), JMR 5(2):131-145. Dual-backend: a pandas
core for local work and a Spark aggregation mirror for production scale.
"""
from .config import TRBConfig
from .core import (
    Cohort, Penetration, PenetrationPromo, RBRPoint, blended_rbr,
    penetration_vs_actual, pwsd, segmented_share,
)
from .model import TRBResult, run_trb

__all__ = [
    "TRBConfig", "run_trb", "TRBResult",
    "Penetration", "PenetrationPromo", "RBRPoint", "Cohort",
    "pwsd", "penetration_vs_actual", "segmented_share", "blended_rbr",
]
