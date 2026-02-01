"""
WayyDB: High-performance columnar time-series database

A kdb+-like database with Python-first API, featuring:
- As-of joins (aj) and window joins (wj)
- Zero-copy numpy interop via memory mapping
- SIMD-accelerated aggregations
- Columnar storage with sorted indices
"""

from wayy_db._core import (
    # Core classes
    Database,
    Table,
    Column,
    # Types
    DType,
    # Exceptions
    WayyException,
    ColumnNotFound,
    TypeMismatch,
    InvalidOperation,
    # Version
    __version__,
)

# Operations module
from wayy_db import ops

__all__ = [
    # Core classes
    "Database",
    "Table",
    "Column",
    # Types
    "DType",
    # Exceptions
    "WayyException",
    "ColumnNotFound",
    "TypeMismatch",
    "InvalidOperation",
    # Submodules
    "ops",
    # Version
    "__version__",
]


def from_dict(data: dict, name: str = "", sorted_by: str | None = None) -> Table:
    """Create a Table from a dictionary of numpy arrays.

    Args:
        data: Dictionary mapping column names to numpy arrays
        name: Optional table name
        sorted_by: Optional column name to mark as sorted index

    Returns:
        Table with the provided data
    """
    import numpy as np

    table = Table(name)

    dtype_map = {
        np.dtype("int64"): DType.Int64,
        np.dtype("float64"): DType.Float64,
        np.dtype("uint32"): DType.Symbol,
        np.dtype("uint8"): DType.Bool,
    }

    for col_name, arr in data.items():
        arr = np.asarray(arr)
        if arr.dtype not in dtype_map:
            # Try to convert
            if np.issubdtype(arr.dtype, np.integer):
                arr = arr.astype(np.int64)
            elif np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float64)
            else:
                raise ValueError(f"Unsupported dtype {arr.dtype} for column {col_name}")

        dtype = dtype_map[arr.dtype]
        table.add_column_from_numpy(col_name, arr, dtype)

    if sorted_by is not None:
        table.set_sorted_by(sorted_by)

    return table


def from_pandas(df, name: str = "", sorted_by: str | None = None) -> Table:
    """Create a Table from a pandas DataFrame.

    Args:
        df: pandas DataFrame
        name: Optional table name
        sorted_by: Optional column name to mark as sorted index

    Returns:
        Table with the DataFrame data
    """
    data = {col: df[col].values for col in df.columns}
    return from_dict(data, name=name, sorted_by=sorted_by)


def from_polars(df, name: str = "", sorted_by: str | None = None) -> Table:
    """Create a Table from a polars DataFrame.

    Args:
        df: polars DataFrame
        name: Optional table name
        sorted_by: Optional column name to mark as sorted index

    Returns:
        Table with the DataFrame data
    """
    data = {col: df[col].to_numpy() for col in df.columns}
    return from_dict(data, name=name, sorted_by=sorted_by)
