"""
WayyDB REST API - High-performance columnar time-series database service
"""
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import wayyDB
import wayy_db as wdb

# Thread pool for running CPU-bound wayyDB operations
executor = ThreadPoolExecutor(max_workers=4)

# Global database instance
db: Optional[wdb.Database] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    global db
    data_path = os.environ.get("WAYY_DATA_PATH", "/data/wayydb")
    os.makedirs(data_path, exist_ok=True)
    db = wdb.Database(data_path)
    yield
    # Cleanup
    if db:
        db.save()


app = FastAPI(
    title="WayyDB API",
    description="High-performance columnar time-series database with kdb+-like functionality",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models ---

class TableCreate(BaseModel):
    name: str
    sorted_by: Optional[str] = None


class ColumnData(BaseModel):
    name: str
    dtype: str  # "int64", "float64", "timestamp", "symbol", "bool"
    data: list


class TableData(BaseModel):
    name: str
    columns: list[ColumnData]
    sorted_by: Optional[str] = None


class AggregationResult(BaseModel):
    column: str
    operation: str
    result: float


class JoinRequest(BaseModel):
    left_table: str
    right_table: str
    on: list[str]
    as_of: str
    window_before: Optional[int] = None  # For window join
    window_after: Optional[int] = None


class WindowRequest(BaseModel):
    table: str
    column: str
    operation: str  # mavg, msum, mstd, mmin, mmax, ema
    window: Optional[int] = None
    alpha: Optional[float] = None  # For EMA


# --- Helper Functions ---

def dtype_from_string(s: str) -> wdb.DType:
    mapping = {
        "int64": wdb.DType.Int64,
        "float64": wdb.DType.Float64,
        "timestamp": wdb.DType.Timestamp,
        "symbol": wdb.DType.Symbol,
        "bool": wdb.DType.Bool,
    }
    if s.lower() not in mapping:
        raise ValueError(f"Unknown dtype: {s}")
    return mapping[s.lower()]


def numpy_dtype_for(dtype: wdb.DType):
    mapping = {
        wdb.DType.Int64: np.int64,
        wdb.DType.Float64: np.float64,
        wdb.DType.Timestamp: np.int64,
        wdb.DType.Symbol: np.uint32,
        wdb.DType.Bool: np.uint8,
    }
    return mapping[dtype]


async def run_in_executor(func, *args):
    """Run blocking wayyDB operations in thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args)


# --- Routes ---

@app.get("/")
async def root():
    return {
        "service": "WayyDB API",
        "version": "0.1.0",
        "status": "healthy",
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "tables": len(db.tables()) if db else 0}


# --- Table Operations ---

@app.get("/tables")
async def list_tables():
    """List all tables in the database."""
    return {"tables": db.tables()}


@app.post("/tables")
async def create_table(table: TableCreate):
    """Create a new empty table."""
    if db.has_table(table.name):
        raise HTTPException(400, f"Table '{table.name}' already exists")

    t = db.create_table(table.name)
    if table.sorted_by:
        t.set_sorted_by(table.sorted_by)
    db.save()
    return {"created": table.name}


@app.post("/tables/upload")
async def upload_table(table_data: TableData):
    """Upload a complete table with data."""
    if db.has_table(table_data.name):
        raise HTTPException(400, f"Table '{table_data.name}' already exists")

    t = wdb.Table(table_data.name)

    for col in table_data.columns:
        dtype = dtype_from_string(col.dtype)
        np_dtype = numpy_dtype_for(dtype)
        arr = np.array(col.data, dtype=np_dtype)
        t.add_column_from_numpy(col.name, arr, dtype)

    if table_data.sorted_by:
        t.set_sorted_by(table_data.sorted_by)

    db.add_table(t)
    db.save()

    return {
        "created": table_data.name,
        "rows": t.num_rows,
        "columns": t.column_names(),
    }


@app.get("/tables/{name}")
async def get_table_info(name: str):
    """Get table metadata."""
    if not db.has_table(name):
        raise HTTPException(404, f"Table '{name}' not found")

    t = db[name]
    return {
        "name": t.name,
        "num_rows": t.num_rows,
        "num_columns": t.num_columns,
        "columns": t.column_names(),
        "sorted_by": t.sorted_by,
    }


@app.get("/tables/{name}/data")
async def get_table_data(
    name: str,
    limit: int = Query(default=100, le=10000),
    offset: int = Query(default=0, ge=0),
):
    """Get table data as JSON."""
    if not db.has_table(name):
        raise HTTPException(404, f"Table '{name}' not found")

    t = db[name]
    end = min(offset + limit, t.num_rows)

    result = {}
    for col_name in t.column_names():
        col = t[col_name]
        arr = col.to_numpy()[offset:end]
        result[col_name] = arr.tolist()

    return {
        "table": name,
        "offset": offset,
        "limit": limit,
        "total_rows": t.num_rows,
        "data": result,
    }


@app.delete("/tables/{name}")
async def delete_table(name: str):
    """Delete a table."""
    if not db.has_table(name):
        raise HTTPException(404, f"Table '{name}' not found")

    db.drop_table(name)
    return {"deleted": name}


# --- Aggregations ---

@app.get("/tables/{name}/agg/{column}/{operation}")
async def aggregate(name: str, column: str, operation: str):
    """
    Run aggregation on a column.
    Operations: sum, avg, min, max, std
    """
    if not db.has_table(name):
        raise HTTPException(404, f"Table '{name}' not found")

    t = db[name]
    if not t.has_column(column):
        raise HTTPException(404, f"Column '{column}' not found")

    col = t[column]

    ops_map = {
        "sum": wdb.ops.sum,
        "avg": wdb.ops.avg,
        "min": wdb.ops.min,
        "max": wdb.ops.max,
        "std": wdb.ops.std,
    }

    if operation not in ops_map:
        raise HTTPException(400, f"Unknown operation: {operation}")

    # Run in thread pool for concurrency
    result = await run_in_executor(ops_map[operation], col)

    return AggregationResult(column=column, operation=operation, result=result)


# --- Joins ---

@app.post("/join/aj")
async def as_of_join(req: JoinRequest):
    """
    As-of join: find most recent right row for each left row.
    Both tables must be sorted by the as_of column.
    """
    if not db.has_table(req.left_table):
        raise HTTPException(404, f"Table '{req.left_table}' not found")
    if not db.has_table(req.right_table):
        raise HTTPException(404, f"Table '{req.right_table}' not found")

    left = db[req.left_table]
    right = db[req.right_table]

    def do_join():
        return wdb.ops.aj(left, right, req.on, req.as_of)

    result = await run_in_executor(do_join)

    # Return as dict
    data = {}
    for col_name in result.column_names():
        data[col_name] = result[col_name].to_numpy().tolist()

    return {
        "join_type": "as_of",
        "rows": result.num_rows,
        "columns": result.column_names(),
        "data": data,
    }


@app.post("/join/wj")
async def window_join(req: JoinRequest):
    """
    Window join: find all right rows within time window.
    """
    if not db.has_table(req.left_table):
        raise HTTPException(404, f"Table '{req.left_table}' not found")
    if not db.has_table(req.right_table):
        raise HTTPException(404, f"Table '{req.right_table}' not found")

    if req.window_before is None or req.window_after is None:
        raise HTTPException(400, "window_before and window_after required for window join")

    left = db[req.left_table]
    right = db[req.right_table]

    def do_join():
        return wdb.ops.wj(left, right, req.on, req.as_of,
                          req.window_before, req.window_after)

    result = await run_in_executor(do_join)

    data = {}
    for col_name in result.column_names():
        data[col_name] = result[col_name].to_numpy().tolist()

    return {
        "join_type": "window",
        "rows": result.num_rows,
        "columns": result.column_names(),
        "data": data,
    }


# --- Window Functions ---

@app.post("/window")
async def window_function(req: WindowRequest):
    """
    Apply window function to a column.
    Operations: mavg, msum, mstd, mmin, mmax, ema, diff, pct_change
    """
    if not db.has_table(req.table):
        raise HTTPException(404, f"Table '{req.table}' not found")

    t = db[req.table]
    if not t.has_column(req.column):
        raise HTTPException(404, f"Column '{req.column}' not found")

    col = t[req.column]

    def do_window():
        if req.operation == "mavg":
            return wdb.ops.mavg(col, req.window)
        elif req.operation == "msum":
            return wdb.ops.msum(col, req.window)
        elif req.operation == "mstd":
            return wdb.ops.mstd(col, req.window)
        elif req.operation == "mmin":
            return wdb.ops.mmin(col, req.window)
        elif req.operation == "mmax":
            return wdb.ops.mmax(col, req.window)
        elif req.operation == "ema":
            return wdb.ops.ema(col, req.alpha)
        elif req.operation == "diff":
            return wdb.ops.diff(col, req.window or 1)
        elif req.operation == "pct_change":
            return wdb.ops.pct_change(col, req.window or 1)
        else:
            raise ValueError(f"Unknown operation: {req.operation}")

    result = await run_in_executor(do_window)

    return {
        "table": req.table,
        "column": req.column,
        "operation": req.operation,
        "result": result.tolist(),
    }
