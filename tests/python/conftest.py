"""Pytest configuration and fixtures for WayyDB tests."""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    path = tempfile.mkdtemp(prefix="wayy_test_")
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def sample_trades():
    """Sample trades data for testing."""
    return {
        "timestamp": np.array([1000, 2000, 3000, 4000, 5000], dtype=np.int64),
        "symbol": np.array([0, 1, 0, 1, 0], dtype=np.uint32),  # AAPL, MSFT alternating
        "price": np.array([150.0, 380.0, 151.0, 381.0, 152.0], dtype=np.float64),
        "size": np.array([100, 200, 150, 250, 100], dtype=np.int64),
    }


@pytest.fixture
def sample_quotes():
    """Sample quotes data for testing."""
    return {
        "timestamp": np.array([500, 900, 1500, 2500, 3500], dtype=np.int64),
        "symbol": np.array([0, 1, 0, 1, 0], dtype=np.uint32),
        "bid": np.array([149.5, 379.5, 150.5, 380.5, 151.5], dtype=np.float64),
        "ask": np.array([150.0, 380.0, 151.0, 381.0, 152.0], dtype=np.float64),
    }
