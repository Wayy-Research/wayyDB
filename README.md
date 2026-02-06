---
title: WayyDB API
emoji: ⚡
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

<p align="center">
  <h1 align="center">WayyDB</h1>
  <p align="center">
    <strong>High-performance columnar time-series database for quantitative finance</strong>
  </p>
  <p align="center">
    kdb+ functionality &bull; Pythonic API &bull; Zero-copy NumPy &bull; SIMD-accelerated
  </p>
</p>

---

WayyDB is a C++ time-series database with Python bindings, designed for quantitative research and trading systems. It provides **kdb+-like temporal join operations** with a modern, accessible API—no q language required.

## Why WayyDB?

| Challenge | WayyDB Solution |
|-----------|-----------------|
| kdb+ costs $100K+/year | Open source, free forever |
| q language learning curve | Pythonic API you already know |
| Pandas/Polars lack temporal joins | Native `aj()` and `wj()` primitives |
| Memory copies kill performance | Zero-copy NumPy via mmap |
| Slow aggregations | AVX2/AVX-512 SIMD acceleration |

## Features

- **As-of Join (aj)** — For each trade, find the most recent quote. O(n log m) via binary search on sorted indices
- **Window Join (wj)** — Get all quotes within a time window around each trade
- **Zero-copy NumPy** — Columns are memory-mapped; `to_numpy()` returns views, not copies
- **SIMD Aggregations** — Sum, avg, min, max accelerated with AVX2 intrinsics
- **Window Functions** — Moving average, EMA, rolling std with O(n) complexity
- **Persistent Storage** — Tables saved as memory-mapped files for instant loading
- **Streaming API** — FastAPI REST + WebSocket endpoints for real-time tick ingestion and subscription
- **Pluggable Pub/Sub** — InMemory (default) or Redis backend for distributed deployments

## Installation

```bash
pip install wayy-db
```

Or build from source:

```bash
git clone https://github.com/wayy-research/wayydb.git
cd wayydb
pip install -e .
```

## Quick Start

### Create Tables from NumPy Arrays

```python
import wayy_db as wdb
import numpy as np

# Create trades table
trades = wdb.from_dict({
    "timestamp": np.array([1000, 2000, 3000, 4000, 5000], dtype=np.int64),
    "symbol": np.array([0, 1, 0, 1, 0], dtype=np.uint32),  # AAPL=0, MSFT=1
    "price": np.array([150.25, 380.50, 151.00, 381.25, 152.00]),
    "size": np.array([100, 200, 150, 250, 100], dtype=np.int64),
}, name="trades", sorted_by="timestamp")

# Create quotes table
quotes = wdb.from_dict({
    "timestamp": np.array([500, 900, 1500, 2500, 3500], dtype=np.int64),
    "symbol": np.array([0, 1, 0, 1, 0], dtype=np.uint32),
    "bid": np.array([149.50, 379.50, 150.50, 380.50, 151.50]),
    "ask": np.array([150.00, 380.00, 151.00, 381.00, 152.00]),
}, name="quotes", sorted_by="timestamp")
```

### As-of Join: Match Trades to Quotes

```python
# For each trade, get the most recent quote for that symbol
result = wdb.ops.aj(trades, quotes, on=["symbol"], as_of="timestamp")

# Result contains trade columns + quote columns (bid, ask)
print(result["bid"].to_numpy())  # [149.5, 379.5, 150.5, 380.5, 151.5]
```

### Aggregations and Window Functions

```python
# SIMD-accelerated aggregations
total_volume = wdb.ops.sum(trades["size"])
avg_price = wdb.ops.avg(trades["price"])
price_std = wdb.ops.std(trades["price"])

# Window functions
mavg_20 = wdb.ops.mavg(trades["price"], window=20)
ema = wdb.ops.ema(trades["price"], alpha=0.1)
rolling_std = wdb.ops.mstd(trades["price"], window=10)

# Returns and changes
returns = wdb.ops.pct_change(trades["price"])
price_diff = wdb.ops.diff(trades["price"])
```

### Persistent Database

```python
# Create persistent database
db = wdb.Database("/data/markets")

# Add table (automatically saved)
db.add_table(trades)

# Later: reload with zero-copy mmap
db2 = wdb.Database("/data/markets")
trades = db2["trades"]  # Instant load via memory mapping

# Access data without copying
prices = trades["price"].to_numpy()  # Zero-copy view into mmap'd file
```

### Pandas/Polars Interop

```python
import pandas as pd
import polars as pl

# From pandas
df = pd.DataFrame({"timestamp": [...], "price": [...]})
table = wdb.from_pandas(df, name="from_pandas", sorted_by="timestamp")

# From polars
df = pl.DataFrame({"timestamp": [...], "price": [...]})
table = wdb.from_polars(df, name="from_polars", sorted_by="timestamp")

# To dict (for conversion back)
data = table.to_dict()  # {"timestamp": np.array, "price": np.array, ...}
```

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `Database(path="")` | Container for tables. Empty path = in-memory |
| `Table(name="")` | Columnar table with optional sorted index |
| `Column` | Typed column with zero-copy NumPy access |

### Table Methods

```python
table.num_rows          # Number of rows
table.num_columns       # Number of columns
table.column_names()    # List of column names
table.sorted_by         # Column used for temporal ordering (or None)
table["col"]            # Get column by name
table.to_dict()         # Export as {name: np.array} dict
table.save(path)        # Save to directory
Table.load(path)        # Load from directory (copies data)
Table.mmap(path)        # Memory-map from directory (zero-copy)
```

### Operations (wayy_db.ops)

#### Aggregations
| Function | Description |
|----------|-------------|
| `sum(col)` | Sum of values (SIMD) |
| `avg(col)` | Mean of values |
| `min(col)` | Minimum value |
| `max(col)` | Maximum value |
| `std(col)` | Standard deviation |

#### Temporal Joins
| Function | Description |
|----------|-------------|
| `aj(left, right, on, as_of)` | As-of join: most recent right row for each left row |
| `wj(left, right, on, as_of, before, after)` | Window join: all right rows within time window |

#### Window Functions
| Function | Description |
|----------|-------------|
| `mavg(col, window)` | Moving average |
| `msum(col, window)` | Moving sum |
| `mstd(col, window)` | Moving standard deviation |
| `mmin(col, window)` | Moving minimum (O(n) via monotonic deque) |
| `mmax(col, window)` | Moving maximum (O(n) via monotonic deque) |
| `ema(col, alpha)` | Exponential moving average |
| `diff(col, periods=1)` | Difference from n periods ago |
| `pct_change(col, periods=1)` | Percent change from n periods ago |
| `shift(col, n)` | Shift values by n positions |

## Type System

| Type | Python | C++ | Size | Use Case |
|------|--------|-----|------|----------|
| Int64 | `np.int64` | `int64_t` | 8B | Quantities, IDs |
| Float64 | `np.float64` | `double` | 8B | Prices, returns |
| Timestamp | `np.int64` | `int64_t` | 8B | Nanoseconds since epoch |
| Symbol | `np.uint32` | `uint32_t` | 4B | Interned strings (tickers) |
| Bool | `np.uint8` | `uint8_t` | 1B | Flags |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Python Interface                        │
│         wayy_db.Database | Table | Column | ops              │
├─────────────────────────────────────────────────────────────┤
│                    pybind11 Bindings                         │
│         Zero-copy NumPy arrays via buffer protocol           │
├─────────────────────────────────────────────────────────────┤
│                       C++ Core Engine                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Storage   │  │   Compute   │  │      Joins          │  │
│  │  • mmap I/O │  │  • AVX2 agg │  │  • O(n log m) aj    │  │
│  │  • columnar │  │  • windows  │  │  • O(n) wj          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                  Memory-Mapped File Storage                  │
│              Zero-copy | Lazy loading | Shared               │
└─────────────────────────────────────────────────────────────┘
```

## Performance

### Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| As-of join | O(n log(m/k)) | n=left rows, m=right rows, k=unique keys |
| Window join | O(n log m + matches) | Plus output size |
| Aggregations | O(n) | SIMD 4x speedup for sum |
| Window functions | O(n) | Single pass with O(1) update |
| Point lookup | O(log n) | Binary search on sorted index |
| Load from disk | O(1) | Memory mapping, no deserialization |

### Design Targets

| Metric | Target |
|--------|--------|
| As-of join (1M x 1M rows) | < 150ms |
| Simple aggregation (1B rows) | < 80ms |
| Binary size | < 5 MB |
| Memory overhead | < 1% beyond data |

## Building from Source

### Requirements

- CMake >= 3.20
- C++20 compiler (GCC 11+, Clang 14+, MSVC 2022+)
- Python >= 3.9

### Build

```bash
git clone https://github.com/wayy-research/wayydb.git
cd wayydb

# Option 1: pip install (recommended)
pip install -e .

# Option 2: CMake directly
mkdir build && cd build
cmake .. -DWAYY_BUILD_PYTHON=ON -DWAYY_BUILD_TESTS=ON
make -j$(nproc)
```

### Run Tests

```bash
# C++ tests (31 tests)
cd build && ctest --output-on-failure

# Python tests (81 tests)
pytest tests/python -v
```

## Comparison with Alternatives

| Feature | WayyDB | kdb+ | DuckDB | Polars |
|---------|--------|------|--------|--------|
| As-of join | Native | Native | Extension | None |
| Window join | Native | Native | None | None |
| Zero-copy Python | Yes | No | No | Limited |
| Sorted index optimization | Yes | Yes | No | No |
| License | MIT | Commercial | MIT | MIT |
| Learning curve | Low | High (q) | Low | Low |
| Persistence | mmap | Native | Native | None |

## Roadmap

- [x] Streaming ingestion API (WebSocket + REST)
- [x] Pluggable pub/sub (InMemory + Redis)
- [x] Multi-deployment Docker (Fly.io, Render, HF Spaces)
- [ ] String column type with dictionary encoding
- [ ] LZ4 compression for columns
- [ ] Parallel aggregations
- [ ] More join types (inner, left, full)
- [ ] Query optimizer

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read our contributing guidelines and submit PRs to the `develop` branch.

---

<p align="center">
  Built with C++20 and Python by <a href="https://wayy.io">Wayy Research</a>
</p>
