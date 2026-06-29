"""Backend aggregation layer (Spark only).

This is the ONLY place the DataFrame engine appears. It turns a raw transaction
log into a handful of small, card-collapsed tables with a fixed schema;
everything downstream (``core``, ``model``, ``plots``) is engine-free and works
on those small pandas tables.

Organised by theme:
  * :mod:`.calendar`   — the date dimension, the anchored bucket-label
    expressions, and the (pandas) axis descriptors.
  * :mod:`._expr`      — the Spark column expressions (period / RBR interval /
    cohort) that keep every reduction inside the engine.
  * :mod:`.aggregator` — :class:`SparkAggregator`, the single backend.
"""
from __future__ import annotations

from .aggregator import SparkAggregator

__all__ = ["SparkAggregator"]
