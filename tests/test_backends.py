"""Pandas-vs-Spark parity. Skipped automatically where pyspark/Java are absent
(this dev machine); runs on the production Spark box to guarantee the two
backends agree on identical data."""
from __future__ import annotations

import importlib.util

import pytest

from parfitt_trb import TRBConfig, run_trb
from tests.helpers import approx, make_df, row

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pyspark") is None, reason="pyspark not installed")


def _rows():
    return [
        row("r1", "2023-01-05", True, True, 25),
        row("r1", "2023-02-05", True, True, 25),
        row("r1", "2023-02-20", False, True, 50),
        row("r2", "2023-01-10", True, True, 20),
        row("r2", "2023-02-10", True, True, 20),
        row("r2", "2023-03-10", False, True, 80),
        row("n1", "2023-01-15", True, True, 30),
        row("n2", "2023-01-25", False, True, 40),
    ]


def test_pandas_spark_parity():
    from pyspark.sql import SparkSession

    spark = (SparkSession.builder.master("local[1]").appName("trb-parity")
             .config("spark.ui.enabled", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    try:
        pdf = make_df(_rows())
        sdf = spark.createDataFrame(pdf)
        cfg = TRBConfig(period_length_days=14, analysis_date="2023-12-31")
        rp = run_trb(pdf, cfg, backend="pandas")
        rs = run_trb(sdf, cfg, backend="spark")
        assert approx(rp.trial_index, rs.trial_index)
        assert approx(rp.buying_index, rs.buying_index)
        for p in rp.rbr_series:
            assert approx(p.rbr, rs.rbr_at(p.interval)) or (p.rbr is None and rs.rbr_at(p.interval) is None)
    finally:
        spark.stop()
