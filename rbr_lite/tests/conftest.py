"""Session-scoped local SparkSession for the lite test suite (standalone --
no import from the parent library)."""
from __future__ import annotations

import os
import sys

import pytest


@pytest.fixture(scope="session")
def spark():
    # Point both ends of the Py4J bridge at this interpreter and pin networking
    # to loopback -- what PySpark needs to start workers on Windows.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

    from pyspark.sql import SparkSession

    s = (SparkSession.builder.master("local[2]")
         .config("spark.driver.memory", "1g")
         .config("spark.executor.memory", "1g")
         .config("spark.sql.shuffle.partitions", "4")
         .config("spark.ui.enabled", "false")
         .config("spark.python.use.daemon", "false")
         .config("spark.driver.bindAddress", "127.0.0.1")
         .appName("rbr_lite_tests")
         .getOrCreate())
    s.sparkContext.setLogLevel("ERROR")
    yield s
