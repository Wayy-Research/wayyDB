#!/usr/bin/env python3
"""
WayyDB Read/Write Performance Benchmarks
"""
import time
import sys
import numpy as np

# Add local build to path
sys.path.insert(0, "/home/rcgalbo/wayy-research/wf/wayyDB/build")

import wayy_db as wdb

def format_rate(rows: int, elapsed: float) -> str:
    rate = rows / elapsed
    if rate >= 1e6:
        return f"{rate/1e6:.2f}M rows/sec"
    elif rate >= 1e3:
        return f"{rate/1e3:.2f}K rows/sec"
    else:
        return f"{rate:.2f} rows/sec"

def format_bytes(bytes_count: int, elapsed: float) -> str:
    rate = bytes_count / elapsed
    if rate >= 1e9:
        return f"{rate/1e9:.2f} GB/sec"
    elif rate >= 1e6:
        return f"{rate/1e6:.2f} MB/sec"
    else:
        return f"{rate/1e3:.2f} KB/sec"

def bench_write(sizes: list[int]):
    """Benchmark table creation/write performance."""
    print("\n=== WRITE PERFORMANCE ===\n")
    print(f"{'Rows':<12} {'Time (ms)':<12} {'Rate':<20} {'Throughput':<15}")
    print("-" * 60)

    for n in sizes:
        # Generate data
        timestamps = np.arange(n, dtype=np.int64)
        prices = np.random.uniform(100, 200, n).astype(np.float64)
        volumes = np.random.randint(100, 10000, n).astype(np.int64)
        symbols = np.random.randint(0, 100, n).astype(np.uint32)

        # Time table creation
        start = time.perf_counter()
        table = wdb.Table(f"bench_{n}")
        table.add_column_from_numpy("timestamp", timestamps, wdb.DType.Timestamp)
        table.add_column_from_numpy("price", prices, wdb.DType.Float64)
        table.add_column_from_numpy("volume", volumes, wdb.DType.Int64)
        table.add_column_from_numpy("symbol", symbols, wdb.DType.Symbol)
        table.set_sorted_by("timestamp")
        elapsed = time.perf_counter() - start

        # Calculate bytes (8+8+8+4 = 28 bytes per row)
        bytes_per_row = 28
        total_bytes = n * bytes_per_row

        print(f"{n:<12,} {elapsed*1000:<12.2f} {format_rate(n, elapsed):<20} {format_bytes(total_bytes, elapsed):<15}")

    return table  # Return last table for further tests

def bench_read(table: wdb.Table):
    """Benchmark read/access performance."""
    print("\n=== READ PERFORMANCE ===\n")
    n = table.num_rows

    # Column access
    start = time.perf_counter()
    for _ in range(100):
        col = table["price"]
    elapsed = time.perf_counter() - start
    print(f"Column lookup (100x):     {elapsed*1000:.3f}ms ({elapsed*10:.3f}ms per lookup)")

    # Zero-copy numpy access
    start = time.perf_counter()
    for _ in range(100):
        arr = table["price"].to_numpy()
    elapsed = time.perf_counter() - start
    print(f"to_numpy() (100x):        {elapsed*1000:.3f}ms ({elapsed*10:.3f}ms per call)")

    # Full table scan (sum all values)
    col = table["price"]
    start = time.perf_counter()
    for _ in range(10):
        total = wdb.ops.sum(col)
    elapsed = time.perf_counter() - start
    print(f"Full scan sum (10x):      {elapsed*1000:.3f}ms ({elapsed*100:.3f}ms per scan)")
    print(f"  -> {format_rate(n * 10, elapsed)}")

def bench_aggregations(table: wdb.Table):
    """Benchmark aggregation operations."""
    print("\n=== AGGREGATION PERFORMANCE ===\n")
    n = table.num_rows
    col = table["price"]

    ops = [
        ("sum", wdb.ops.sum),
        ("avg", wdb.ops.avg),
        ("min", wdb.ops.min),
        ("max", wdb.ops.max),
        ("std", wdb.ops.std),
    ]

    print(f"{'Operation':<12} {'Time (ms)':<12} {'Rate':<20}")
    print("-" * 45)

    for name, func in ops:
        # Warm up
        func(col)

        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            result = func(col)
        elapsed = time.perf_counter() - start

        print(f"{name:<12} {elapsed*10:.3f}        {format_rate(n * 100, elapsed)}")

def bench_window_functions(table: wdb.Table):
    """Benchmark window functions."""
    print("\n=== WINDOW FUNCTION PERFORMANCE ===\n")
    n = table.num_rows
    col = table["price"]

    ops = [
        ("mavg(20)", lambda c: wdb.ops.mavg(c, 20)),
        ("msum(20)", lambda c: wdb.ops.msum(c, 20)),
        ("mstd(20)", lambda c: wdb.ops.mstd(c, 20)),
        ("ema(0.1)", lambda c: wdb.ops.ema(c, 0.1)),
        ("diff(1)", lambda c: wdb.ops.diff(c, 1)),
        ("pct_change", lambda c: wdb.ops.pct_change(c, 1)),
    ]

    print(f"{'Operation':<15} {'Time (ms)':<12} {'Rate':<20}")
    print("-" * 50)

    for name, func in ops:
        # Warm up
        func(col)

        # Benchmark
        start = time.perf_counter()
        for _ in range(10):
            result = func(col)
        elapsed = time.perf_counter() - start

        print(f"{name:<15} {elapsed*100:.3f}        {format_rate(n * 10, elapsed)}")

def bench_joins():
    """Benchmark temporal join operations."""
    print("\n=== JOIN PERFORMANCE ===\n")

    sizes = [(10_000, 10_000), (100_000, 100_000), (1_000_000, 1_000_000)]

    print(f"{'Left x Right':<20} {'aj (ms)':<12} {'Rate':<20}")
    print("-" * 55)

    for left_n, right_n in sizes:
        # Create left table (trades)
        left = wdb.Table("trades")
        left.add_column_from_numpy("timestamp",
            np.sort(np.random.randint(0, left_n * 10, left_n)).astype(np.int64),
            wdb.DType.Timestamp)
        left.add_column_from_numpy("symbol",
            np.random.randint(0, 10, left_n).astype(np.uint32),
            wdb.DType.Symbol)
        left.add_column_from_numpy("price",
            np.random.uniform(100, 200, left_n).astype(np.float64),
            wdb.DType.Float64)
        left.set_sorted_by("timestamp")

        # Create right table (quotes)
        right = wdb.Table("quotes")
        right.add_column_from_numpy("timestamp",
            np.sort(np.random.randint(0, right_n * 10, right_n)).astype(np.int64),
            wdb.DType.Timestamp)
        right.add_column_from_numpy("symbol",
            np.random.randint(0, 10, right_n).astype(np.uint32),
            wdb.DType.Symbol)
        right.add_column_from_numpy("bid",
            np.random.uniform(99, 199, right_n).astype(np.float64),
            wdb.DType.Float64)
        right.add_column_from_numpy("ask",
            np.random.uniform(101, 201, right_n).astype(np.float64),
            wdb.DType.Float64)
        right.set_sorted_by("timestamp")

        # Warm up
        if left_n <= 100_000:
            wdb.ops.aj(left, right, ["symbol"], "timestamp")

        # Benchmark as-of join
        start = time.perf_counter()
        result = wdb.ops.aj(left, right, ["symbol"], "timestamp")
        elapsed = time.perf_counter() - start

        size_str = f"{left_n//1000}K x {right_n//1000}K"
        print(f"{size_str:<20} {elapsed*1000:<12.2f} {format_rate(left_n, elapsed)}")

def bench_persistence(n: int = 1_000_000):
    """Benchmark save/load/mmap performance."""
    print("\n=== PERSISTENCE PERFORMANCE ===\n")

    import tempfile
    import os

    # Create table
    table = wdb.Table("persist_test")
    table.add_column_from_numpy("timestamp",
        np.arange(n, dtype=np.int64), wdb.DType.Timestamp)
    table.add_column_from_numpy("price",
        np.random.uniform(100, 200, n).astype(np.float64), wdb.DType.Float64)
    table.add_column_from_numpy("volume",
        np.random.randint(100, 10000, n).astype(np.int64), wdb.DType.Int64)
    table.set_sorted_by("timestamp")

    bytes_total = n * (8 + 8 + 8)  # 24 bytes per row

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_table")

        # Benchmark save
        start = time.perf_counter()
        table.save(path)
        save_elapsed = time.perf_counter() - start
        print(f"Save {n:,} rows:    {save_elapsed*1000:.2f}ms  ({format_bytes(bytes_total, save_elapsed)})")

        # Benchmark load (copies data)
        start = time.perf_counter()
        loaded = wdb.Table.load(path)
        load_elapsed = time.perf_counter() - start
        print(f"Load {n:,} rows:    {load_elapsed*1000:.2f}ms  ({format_bytes(bytes_total, load_elapsed)})")

        # Benchmark mmap (zero-copy)
        start = time.perf_counter()
        mmapped = wdb.Table.mmap(path)
        mmap_elapsed = time.perf_counter() - start
        print(f"Mmap {n:,} rows:    {mmap_elapsed*1000:.2f}ms  ({format_bytes(bytes_total, mmap_elapsed)})")
        print(f"  -> mmap is {load_elapsed/mmap_elapsed:.0f}x faster than load")

def bench_concurrent():
    """Benchmark concurrent read performance."""
    print("\n=== CONCURRENT READ PERFORMANCE ===\n")

    import threading

    n = 1_000_000
    table = wdb.Table("concurrent_test")
    table.add_column_from_numpy("price",
        np.random.uniform(100, 200, n).astype(np.float64), wdb.DType.Float64)
    col = table["price"]

    def worker(results, idx):
        for _ in range(10):
            results[idx] = wdb.ops.sum(col)

    for num_threads in [1, 2, 4, 8]:
        results = [0.0] * num_threads
        threads = [threading.Thread(target=worker, args=(results, i))
                   for i in range(num_threads)]

        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - start

        ops_per_sec = (num_threads * 10) / elapsed
        print(f"{num_threads} threads:  {elapsed*1000:.2f}ms  ({ops_per_sec:.1f} ops/sec, {format_rate(n * num_threads * 10, elapsed)})")


if __name__ == "__main__":
    print("=" * 60)
    print("  WayyDB Performance Benchmarks")
    print("=" * 60)

    # Write benchmarks with increasing sizes
    sizes = [10_000, 100_000, 1_000_000, 10_000_000]
    table = bench_write(sizes)

    # Use 1M row table for read tests
    table_1m = wdb.Table("bench_1m")
    n = 1_000_000
    table_1m.add_column_from_numpy("timestamp", np.arange(n, dtype=np.int64), wdb.DType.Timestamp)
    table_1m.add_column_from_numpy("price", np.random.uniform(100, 200, n).astype(np.float64), wdb.DType.Float64)
    table_1m.add_column_from_numpy("volume", np.random.randint(100, 10000, n).astype(np.int64), wdb.DType.Int64)
    table_1m.add_column_from_numpy("symbol", np.random.randint(0, 100, n).astype(np.uint32), wdb.DType.Symbol)
    table_1m.set_sorted_by("timestamp")

    bench_read(table_1m)
    bench_aggregations(table_1m)
    bench_window_functions(table_1m)
    bench_joins()
    bench_persistence()
    bench_concurrent()

    print("\n" + "=" * 60)
    print("  Benchmarks Complete")
    print("=" * 60)
