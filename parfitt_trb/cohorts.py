"""Entry-cohort labelling (Table 2). Pure, engine-free logic.

A cohort is defined by how many weeks after launch a shopper first tried the
brand. Boundaries ``(6, 12, 24)`` yield ``1-6w / 7-12w / 13-24w / 25+w`` -- the
exact grouping of Parfitt's Table 2 (first 6 weeks, second 6 weeks, 13-24
weeks, then later). ``entry_week`` is the 1-based weekly period of the trial
counted from the launch origin (<= 0 means a pre-launch buyer).
"""
from __future__ import annotations

from typing import List, Sequence

PRELAUNCH = "pre-launch"


def cohort_order(boundaries: Sequence[int], include_prelaunch: bool = False) -> List[str]:
    """Ordered cohort labels, earliest entrants first."""
    labels: List[str] = [PRELAUNCH] if include_prelaunch else []
    prev = 0
    for b in boundaries:
        labels.append(f"{prev + 1}-{b}w")
        prev = b
    labels.append(f"{prev + 1}+w")
    return labels


def cohort_label(entry_week: int, boundaries: Sequence[int],
                 include_prelaunch: bool = False) -> str:
    """Map a trial's entry week to its cohort label."""
    if entry_week <= 0:
        return PRELAUNCH if include_prelaunch else f"1-{boundaries[0]}w"
    prev = 0
    for b in boundaries:
        if entry_week <= b:
            return f"{prev + 1}-{b}w"
        prev = b
    return f"{prev + 1}+w"
