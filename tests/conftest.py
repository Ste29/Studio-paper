"""Shared pytest fixtures. A single, session-scoped SparkSession (the JVM starts
once for the whole suite — important on an under-powered machine) built with the
resource-limited local helper."""
from __future__ import annotations

import pytest

from parfitt_trb.local_spark import build_local_spark


@pytest.fixture(scope="session")
def spark():
    s = build_local_spark("trb-tests")
    yield s
    s.stop()
