"""Resource-limited local SparkSession for **tests and the examples**.

Production runs on a real cluster and builds its own session — this helper is a
convenience for a small/under-powered dev machine, so it caps the engine to two
local cores and 1 GB driver/executor and disables the UI. It also applies the
Windows fixes that PySpark needs to let its Python workers connect back to the
JVM (explicit interpreter, loopback bind address, no worker daemon); without
them ``createDataFrame`` times out on Windows.
"""
from __future__ import annotations

import os
import sys


def build_local_spark(app_name: str = "test"):
    """Return a small local SparkSession (or the active one if already created)."""
    # Point both ends of the Py4J bridge at this interpreter and pin networking to
    # loopback — the combination PySpark needs to start workers on Windows.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[2]")
        .config("spark.driver.memory", "1g")
        .config("spark.executor.memory", "1g")
        # Tiny inputs: a handful of shuffle partitions avoids per-task overhead.
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        # Windows: the worker daemon's connect-back socket is unreliable; spawning
        # a fresh worker per task is slower but actually connects.
        .config("spark.python.use.daemon", "false")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .appName(app_name)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
