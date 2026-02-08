# Why We Wrote a Time-Series Database in C++20 Instead of Using Rust

*The story of wayyDB: from "pandas is too slow" to a pip-installable
C++ database that does as-of joins 50x faster.*

---

## The Problem

If you've worked in quantitative finance, you know the pain. Python is the language
of choice for research -- pandas, numpy, scikit-learn, the whole ecosystem. But the
moment you need to do anything temporal at scale, you hit a wall.

The canonical operation is the **as-of join**: for each trade, find the most recent
quote for that symbol. This is the join that every trading system, every backtester,
every risk engine needs. In kdb+/q, it's a single character (`aj`). In pandas, it's
`pd.merge_asof()`, and it's slow. At 1M rows, you're waiting seconds. At 10M rows,
you're going to lunch.

The obvious answer is kdb+. It's the industry standard for time-series databases,
it's fast, and it has first-class temporal joins. The problem is it costs $100K/year
and uses q, a language that looks like someone's cat walked across the keyboard.

There's a gap in the market: Python developers who need kdb+-class performance on
temporal operations, without the kdb+ price tag or learning curve.

## Why Not Just Use DuckDB or Polars?

This is a fair question, and I want to answer it honestly.

**DuckDB** is excellent. If your workload is analytical SQL -- aggregations, group-bys,
window functions over parquet files -- DuckDB is probably the right tool. It recently
added ASOF JOIN syntax, which is great. But DuckDB is a general-purpose analytical
database. Its as-of join goes through the SQL query planner, which adds overhead for
this specific operation. DuckDB also doesn't have a streaming ingestion API or
memory-mapped storage that you can share zero-copy with NumPy.

**Polars** is also excellent. It's the best DataFrame library for batch processing,
period. It has `join_asof`, and it's faster than pandas. But Polars is a DataFrame
library, not a database. It doesn't have persistent storage, it doesn't have a
streaming API, and its as-of join -- while good -- isn't optimized for the specific
access pattern that quant finance uses (sorted indices with multi-key lookups).

Neither DuckDB nor Polars has **window joins** (`wj`): "give me all quotes within
500 microseconds of each trade." This is a kdb+ primitive that doesn't exist
anywhere else in the Python ecosystem. wayyDB has it.

The point isn't that DuckDB and Polars are bad. They're great tools. The point is
that they're general-purpose tools solving general-purpose problems. wayyDB is a
special-purpose tool solving a specific problem: fast temporal operations on
columnar data with zero-copy Python interop.

## Why C++20 Over Rust

This is the question I get the most. In 2024-2025, writing a new systems project
in C++ instead of Rust feels like a contrarian choice. Here's why I made it:

### 1. SIMD Intrinsics Are Mature and Well-Documented

wayyDB uses AVX2 intrinsics for vectorized aggregations. Here's the actual sum
implementation from our codebase:

```cpp
#ifdef WAYY_USE_AVX2
double sum_simd(const ColumnView<double>& col) {
    const double* data = col.data();
    size_t n = col.size();

    __m256d vsum = _mm256_setzero_pd();

    // Process 4 doubles per iteration
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d v = _mm256_loadu_pd(data + i);
        vsum = _mm256_add_pd(vsum, v);
    }

    // Horizontal reduction
    __m128d vlow = _mm256_castpd256_pd128(vsum);
    __m128d vhigh = _mm256_extractf128_pd(vsum, 1);
    vlow = _mm_add_pd(vlow, vhigh);
    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    double result = _mm_cvtsd_f64(_mm_add_sd(vlow, high64));

    // Handle remainder
    for (; i < n; ++i) {
        result += data[i];
    }
    return result;
}
#endif
```

This is 10 lines of code that processes 4 doubles per clock cycle. GCC and Clang
compile this to tight AVX2 instructions. The equivalent in Rust using
`std::arch::x86_64` works fine, but the documentation, StackOverflow answers, and
Intel Intrinsics Guide all target C/C++. When you're debugging SIMD, you want the
ecosystem that has 20 years of answers.

### 2. pybind11 Buffer Protocol = Zero-Copy NumPy

This is the biggest reason. pybind11's buffer protocol integration lets us expose
C++ memory directly as NumPy arrays with **zero copies**. When you call
`table["price"].to_numpy()`, you get a NumPy array that points to the same memory
as the C++ column. No memcpy. No serialization. The NumPy array literally is the
C++ data.

This works because pybind11 implements Python's buffer protocol, which NumPy
natively understands. The C++ column owns the memory, pybind11 exposes it as a
buffer, and NumPy wraps it as an ndarray. Three layers, zero copies.

PyO3 (Rust's Python binding library) is catching up here. `numpy` crate for Rust
has gotten better. But pybind11's buffer protocol is a solved problem with years of
production use. When I started wayyDB, PyO3's zero-copy NumPy story had edge cases
that I didn't want to debug.

### 3. Memory-Mapped I/O Is a Solved Problem in C++

wayyDB's persistence layer is built on `mmap()`. Tables are saved as raw columnar
files. Loading a table means calling `mmap()` on each column file, which returns
a pointer to the kernel's page cache. No deserialization, no parsing, no allocation.
Loading 10 million rows takes microseconds because it's just setting up virtual
memory mappings.

This is old technology. The POSIX mmap interface has been stable since the 1980s.
C++ gives us direct access to it with no abstraction overhead. Rust's `memmap2`
crate works fine, but mmap is inherently unsafe -- you're telling the OS "give me
a pointer to this file and let me read it as typed data." In Rust, every mmap
access is inside an `unsafe` block. In C++, it's just a pointer cast. For a
database engine where performance is the whole point, the C++ approach is more
natural.

## The Architecture

wayyDB's architecture is simple by design. Three layers:

```
Python Interface (pybind11)
    |
C++ Core Engine
    |-- Storage: columnar mmap files
    |-- Compute: SIMD aggregations, O(n) window functions
    |-- Joins: sorted index + binary search (O(n log m))
    |
Memory-Mapped File Storage
```

### Columnar Storage

Each table is a collection of typed columns. Each column is a contiguous array
of fixed-size values (int64, float64, uint32 for symbols, uint8 for bools).
No nulls, no variable-length data (yet), no compression (yet). This keeps the
hot path -- read a column, aggregate it -- as fast as possible.

### Sorted Indices for Temporal Joins

The key insight behind fast as-of joins: if both tables are sorted by timestamp,
you can use binary search instead of hash lookups. wayyDB's `aj()` implementation:

1. Group the right table rows by join key (symbol)
2. For each left row, look up the key group
3. Binary search within the group for the largest timestamp <= the left timestamp

This gives O(n log(m/k)) complexity where n = left rows, m = right rows, k = unique
keys. For a typical as-of join with 10 symbols, the binary search runs on ~100K
elements instead of 1M. The actual implementation uses `std::upper_bound` on
the group indices:

```cpp
// Binary search for largest timestamp <= ts
auto it = std::upper_bound(group.begin(), group.end(), ts,
    [&right_ts](int64_t t, size_t idx) { return t < right_ts[idx]; });
if (it != group.begin()) {
    --it;
    right_indices.push_back(*it);
}
```

### Why This Beats pandas

pandas' `merge_asof` sorts the data, then does a linear scan with a two-pointer
approach. This is O(n + m) per key group, but pandas also has Python-level overhead
for each row. wayyDB's approach has slightly higher algorithmic complexity (log factor)
but eliminates the Python overhead entirely -- the binary search runs in compiled
C++ with cache-friendly memory access patterns.

For 1M trades x 1M quotes with 10 symbols, wayyDB finishes in ~150ms. pandas
takes ~8 seconds. That's a 50x+ difference.

## The Benchmarks

We built a comprehensive benchmark suite that compares wayyDB against pandas,
polars, duckdb, and numpy. You can reproduce these yourself:

```bash
pip install wayy-db[bench]
python -m benchmarks.benchmark --compare pandas,polars,duckdb
```

Key results on our test hardware:

| Operation | wayyDB | pandas | Polars | DuckDB |
|-----------|--------|--------|--------|--------|
| As-of Join (1M x 1M) | 142ms | 8,234ms (58x) | 568ms (4x) | 345ms (2.4x) |
| Aggregation (5 ops, 1M) | 0.8ms | 16.2ms (20x) | 4.1ms (5x) | 5.6ms (7x) |
| Create Table (1M, 7 cols) | 12ms | 145ms (12x) | 35ms (3x) | 89ms (7x) |
| Load from Disk (1M) | 0.05ms | 62ms (1240x) | 18ms (360x) | 32ms (640x) |

The persistence numbers look absurd, but they're real. wayyDB's "load" is an mmap
call, which sets up virtual memory mappings without reading any data from disk.
The actual data gets paged in on first access by the OS kernel. Parquet readers
(used by pandas, polars, and duckdb) have to decompress and deserialize the data
upfront.

### Where wayyDB Doesn't Win

Honesty matters for credibility, so here's where other tools are better:

- **General-purpose analytics**: DuckDB is better for complex SQL queries with
  multiple joins, subqueries, and GROUP BY operations. wayyDB doesn't have a
  query optimizer.
- **DataFrame operations**: Polars is better for filter/select/groupby workflows.
  wayyDB is not a DataFrame library.
- **Ecosystem**: pandas has 15 years of ecosystem. wayyDB is brand new. If you
  need to read CSV files, plot with matplotlib, or interop with scikit-learn,
  pandas has the integrations.

wayyDB is a special-purpose tool. It's the right choice when your bottleneck is
temporal joins, SIMD aggregations, or zero-copy NumPy interop. For everything
else, use the right tool for the job.

## What's Next

The roadmap for wayyDB:

- **String columns** with dictionary encoding (intern strings to uint32 IDs)
- **LZ4 compression** for columns on disk (decompress on mmap page fault)
- **Parallel aggregations** using C++20 `std::execution::par`
- **More join types**: inner, left, full
- **Query optimizer** for multi-operation pipelines

## Try It

```bash
pip install wayy-db
```

```python
import wayy_db as wdb
import numpy as np

trades = wdb.from_dict({
    "timestamp": np.array([1000, 2000, 3000], dtype=np.int64),
    "symbol": np.array([0, 1, 0], dtype=np.uint32),
    "price": np.array([150.25, 380.50, 151.00]),
}, name="trades", sorted_by="timestamp")

# As-of join in one line
result = wdb.ops.aj(trades, quotes, on=["symbol"], as_of="timestamp")

# Zero-copy NumPy
prices = trades["price"].to_numpy()  # No copy!
```

- **GitHub**: [github.com/Wayy-Research/wayyDB](https://github.com/Wayy-Research/wayyDB)
- **PyPI**: [pypi.org/project/wayy-db](https://pypi.org/project/wayy-db/)
- **Benchmarks**: `pip install wayy-db[bench] && python -m benchmarks.benchmark`

---

*Built by [Wayy Research](https://wayy.io) in Buffalo, NY.
9 years of institutional experience, pip-installable.*
