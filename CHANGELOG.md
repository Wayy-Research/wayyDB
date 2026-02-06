# Changelog

## [0.1.0] - 2026-02-06

### Core Engine
- Columnar storage with 5 data types (Int64, Float64, Timestamp, Symbol, Bool)
- SIMD-accelerated aggregations (AVX2): sum, avg, min, max, std_dev
- Memory-mapped I/O for zero-copy persistence
- Temporal joins: as-of join (aj), window join (wj)
- Window functions: mavg, msum, mstd, ema, diff, pct_change, shift
- Thread-safe Database with shared_mutex for concurrent reads

### Python Bindings
- Complete pybind11 bindings with GIL release for compute
- Zero-copy NumPy interop
- Free-threaded Python 3.13 support

### Streaming API
- FastAPI REST + WebSocket endpoints
- Tick ingestion (single, batch, WebSocket)
- Real-time subscription with symbol filtering
- Pluggable pub/sub (InMemory + Redis backends)
- Atomic table swap with backpressure handling

### Deployment
- Docker with multi-target support (Fly.io, Render, HuggingFace Spaces)
- cibuildwheel configured for Python 3.9-3.13
