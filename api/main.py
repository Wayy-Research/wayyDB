"""
WayyDB REST API - High-performance columnar time-series database service

Features:
- REST API for table operations, aggregations, joins, window functions
- WebSocket streaming ingestion for real-time tick data
- WebSocket pub/sub for streaming updates to clients
- Efficient batching and append operations
"""
import os
import re
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional, List

import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

# Import wayyDB
import wayy_db as wdb

# Import streaming module
from api.streaming import (
    get_streaming_manager,
    start_streaming,
    stop_streaming,
    StreamingManager,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread pool for running CPU-bound wayyDB operations
executor = ThreadPoolExecutor(max_workers=4)

# Global database instance
db: Optional[wdb.Database] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and streaming on startup."""
    global db
    data_path = os.environ.get("WAYY_DATA_PATH", "/data/wayydb")
    os.makedirs(data_path, exist_ok=True)
    db = wdb.Database(data_path)

    # Initialize streaming manager with database reference
    streaming = get_streaming_manager()
    streaming.set_database(db)
    await start_streaming()

    logger.info(f"WayyDB started with data path: {data_path}")

    yield

    # Cleanup
    await stop_streaming()
    if db:
        db.save()
    logger.info("WayyDB shutdown complete")


app = FastAPI(
    title="WayyDB API",
    description="High-performance columnar time-series database with kdb+-like functionality",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS - configurable via CORS_ORIGINS env var
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
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


class AppendData(BaseModel):
    """Data to append to an existing table."""
    columns: list[ColumnData]


class IngestTick(BaseModel):
    """A single tick for streaming ingestion."""
    symbol: str
    price: float
    timestamp: Optional[int] = None  # Nanoseconds since epoch
    volume: Optional[float] = 0.0
    bid: Optional[float] = None
    ask: Optional[float] = None


class IngestBatch(BaseModel):
    """Batch of ticks for streaming ingestion."""
    ticks: list[IngestTick]


class SubscribeRequest(BaseModel):
    """Subscription filter for WebSocket."""
    symbols: Optional[list[str]] = None  # None = all symbols


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


TABLE_NAME_RE = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]{0,63}$')


def validate_table_name(name: str) -> str:
    if not TABLE_NAME_RE.match(name):
        raise HTTPException(400, f"Invalid table name: {name}")
    return name


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


# --- Append API ---

@app.post("/tables/{name}/append")
async def append_to_table(name: str, data: AppendData):
    """
    Append rows to an existing table.

    This is more efficient than re-uploading the entire table.
    The new data must have the same columns as the existing table.
    """
    if not db.has_table(name):
        raise HTTPException(404, f"Table '{name}' not found")

    existing = db[name]
    existing_cols = set(existing.column_names())

    # Validate columns match
    new_cols = {col.name for col in data.columns}
    if existing_cols != new_cols:
        raise HTTPException(
            400,
            f"Column mismatch. Expected: {sorted(existing_cols)}, got: {sorted(new_cols)}"
        )

    # Get existing data
    existing_data = {}
    for col_name in existing.column_names():
        existing_data[col_name] = existing[col_name].to_numpy()

    # Prepare new data
    new_data = {}
    for col in data.columns:
        dtype = dtype_from_string(col.dtype)
        np_dtype = numpy_dtype_for(dtype)
        new_data[col.name] = np.array(col.data, dtype=np_dtype)

    # Concatenate
    combined = {}
    for col_name in existing_cols:
        combined[col_name] = np.concatenate([existing_data[col_name], new_data[col_name]])

    # Get sorted_by before dropping
    sorted_by = existing.sorted_by

    # Drop and recreate
    db.drop_table(name)
    new_table = wdb.from_dict(combined, name=name, sorted_by=sorted_by)
    db.add_table(new_table)
    db.save()

    return {
        "appended": name,
        "new_rows": len(data.columns[0].data) if data.columns else 0,
        "total_rows": new_table.num_rows,
    }


# --- Streaming Ingestion API ---

@app.post("/ingest/{table}")
async def ingest_tick(table: str, tick: IngestTick):
    """
    Ingest a single tick via REST.

    For high-throughput, use the WebSocket endpoint instead.
    """
    validate_table_name(table)
    streaming = get_streaming_manager()
    await streaming.ingest_tick(
        table=table,
        symbol=tick.symbol,
        price=tick.price,
        timestamp=tick.timestamp,
        volume=tick.volume or 0.0,
        bid=tick.bid or tick.price,
        ask=tick.ask or tick.price,
    )
    return {"ingested": 1, "table": table}


@app.post("/ingest/{table}/batch")
async def ingest_batch(table: str, batch: IngestBatch):
    """
    Ingest a batch of ticks via REST.

    For high-throughput, use the WebSocket endpoint instead.
    """
    validate_table_name(table)
    streaming = get_streaming_manager()
    ticks = [
        {
            "symbol": t.symbol,
            "price": t.price,
            "timestamp": t.timestamp,
            "volume": t.volume or 0.0,
            "bid": t.bid or t.price,
            "ask": t.ask or t.price,
        }
        for t in batch.ticks
    ]
    await streaming.ingest_batch(table=table, ticks=ticks)
    return {"ingested": len(ticks), "table": table}


# --- WebSocket Endpoints ---

@app.websocket("/ws/ingest/{table}")
async def ws_ingest(websocket: WebSocket, table: str):
    """
    WebSocket endpoint for streaming tick ingestion.

    Send JSON messages with tick data:
    {
        "symbol": "BTC-USD",
        "price": 42150.50,
        "timestamp": 1704067200000000000,  // Optional, nanoseconds
        "volume": 1.5,                      // Optional
        "bid": 42150.00,                    // Optional
        "ask": 42151.00                     // Optional
    }

    Or batches:
    {
        "batch": [
            {"symbol": "BTC-USD", "price": 42150.50, ...},
            {"symbol": "ETH-USD", "price": 2250.25, ...}
        ]
    }
    """
    await websocket.accept()
    streaming = get_streaming_manager()

    logger.info(f"Ingestion WebSocket connected for table: {table}")

    try:
        while True:
            data = await websocket.receive_json()

            if "batch" in data:
                # Batch ingestion
                ticks = data["batch"]
                await streaming.ingest_batch(table=table, ticks=ticks)
                await websocket.send_json({"ack": len(ticks)})
            else:
                # Single tick
                await streaming.ingest_tick(
                    table=table,
                    symbol=data["symbol"],
                    price=data["price"],
                    timestamp=data.get("timestamp"),
                    volume=data.get("volume", 0.0),
                    bid=data.get("bid", data["price"]),
                    ask=data.get("ask", data["price"]),
                )
                await websocket.send_json({"ack": 1})

    except WebSocketDisconnect:
        logger.info(f"Ingestion WebSocket disconnected for table: {table}")
    except Exception as e:
        logger.error(f"Ingestion WebSocket error: {e}")
        await websocket.close(code=1011, reason=str(e))


@app.websocket("/ws/subscribe/{table}")
async def ws_subscribe(websocket: WebSocket, table: str):
    """
    WebSocket endpoint for subscribing to real-time updates.

    Optionally send a filter message after connecting:
    {"symbols": ["BTC-USD", "ETH-USD"]}

    Receives updates as:
    {
        "symbol": "BTC-USD",
        "price": 42150.50,
        "bid": 42150.00,
        "ask": 42151.00,
        "volume": 1.5,
        "timestamp": 1704067200000000000,
        "table": "ticks"
    }

    Or batches during high-throughput:
    {"batch": [...]}
    """
    await websocket.accept()
    streaming = get_streaming_manager()

    # Default: subscribe to all symbols
    symbols = None

    # Check for initial filter message (non-blocking)
    try:
        # Wait briefly for filter message
        data = await asyncio.wait_for(websocket.receive_json(), timeout=0.5)
        if "symbols" in data:
            symbols = data["symbols"]
            logger.info(f"Subscription filter: {symbols}")
    except asyncio.TimeoutError:
        pass
    except Exception:
        pass

    subscriber = await streaming.subscribe(websocket, table, symbols)
    logger.info(f"Subscription WebSocket connected for table: {table}, symbols: {symbols or 'all'}")

    try:
        # Keep connection alive, handle any incoming messages
        while True:
            try:
                data = await websocket.receive_json()
                # Handle filter updates
                if "symbols" in data:
                    subscriber.symbols = set(data["symbols"]) if data["symbols"] else set()
                    await websocket.send_json({"filter_updated": list(subscriber.symbols) or "all"})
            except WebSocketDisconnect:
                raise
            except Exception:
                pass

    except WebSocketDisconnect:
        logger.info(f"Subscription WebSocket disconnected for table: {table}")
    finally:
        await streaming.unsubscribe(websocket, table)


# --- Streaming Stats ---

@app.get("/streaming/stats")
async def streaming_stats():
    """Get streaming ingestion and pub/sub statistics."""
    streaming = get_streaming_manager()
    return streaming.get_stats()


@app.get("/streaming/quote/{table}/{symbol}")
async def get_quote(table: str, symbol: str):
    """Get the latest quote for a symbol from the streaming cache."""
    streaming = get_streaming_manager()
    quote = streaming.get_latest_quote(table, symbol)
    if not quote:
        raise HTTPException(404, f"No quote for {symbol} in {table}")
    return quote


@app.get("/streaming/quotes/{table}")
async def get_all_quotes(table: str):
    """Get all latest quotes for a table from the streaming cache."""
    streaming = get_streaming_manager()
    return streaming.get_all_quotes(table)


@app.get("/streaming/pubsub")
async def pubsub_stats():
    """Get pub/sub backend statistics (channels, sequences, backend type)."""
    streaming = get_streaming_manager()
    stats = streaming.get_stats()
    return stats.get("pubsub", {"backend": "none", "info": "PubSub not configured"})
