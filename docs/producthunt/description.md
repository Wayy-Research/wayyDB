wayyDB is a high-performance columnar time-series database built in C++20 with
Python bindings. It's designed for quantitative finance workflows where speed matters.

**What makes it different:**
- **SIMD Aggregations**: AVX2-accelerated sum, avg, min, max, std
- **Zero-Copy NumPy**: Memory-mapped columns share directly with NumPy arrays
- **As-of Joins**: O(n log m) temporal joins -- the operation quant finance lives on
- **Window Join**: Get all quotes within a time window around each trade -- no other Python library has this
- **Streaming API**: FastAPI REST + WebSocket, 1M ticks/sec ingestion
- **Free forever**: MIT licensed. No cloud lock-in. No enterprise pricing.

**Benchmarks** (1M rows, median of 5 runs):
- As-of join: 58x faster than pandas, 4x faster than Polars
- Aggregation: 20x faster than pandas (AVX2 SIMD)
- Table creation: 12x faster than pandas
- Load from disk: mmap = near-instant (vs 62ms for parquet)

**Get started:**
```
pip install wayy-db
```

Built by a quant with 9 years of institutional experience.
The tools I wished existed, now open source.
