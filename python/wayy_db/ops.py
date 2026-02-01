"""
WayyDB Operations

High-performance operations for time-series analysis:
- Temporal joins (aj, wj)
- SIMD aggregations (sum, avg, min, max, std)
- Window functions (mavg, msum, mstd, ema, etc.)
"""

from wayy_db._core import ops as _ops

# Re-export all operations from C++ module
from wayy_db._core.ops import (
    # Aggregations
    sum,
    avg,
    min,
    max,
    std,
    # Joins
    aj,
    wj,
    # Window functions
    mavg,
    msum,
    mstd,
    mmin,
    mmax,
    ema,
    diff,
    pct_change,
    shift,
)

__all__ = [
    # Aggregations
    "sum",
    "avg",
    "min",
    "max",
    "std",
    # Joins
    "aj",
    "wj",
    # Window functions
    "mavg",
    "msum",
    "mstd",
    "mmin",
    "mmax",
    "ema",
    "diff",
    "pct_change",
    "shift",
]
