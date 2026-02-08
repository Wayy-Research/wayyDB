#!/usr/bin/env python3
"""
wayyDB Benchmark Suite v0.1.0
=============================

Comprehensive, publication-quality benchmarks comparing wayyDB against
pandas, polars, duckdb, and numpy for common time-series operations.

Usage:
    python -m benchmarks.benchmark                    # Full suite
    python -m benchmarks.benchmark --quick            # Smaller datasets
    python -m benchmarks.benchmark --only asof_join   # Single benchmark
    python -m benchmarks.benchmark --compare pandas,polars,duckdb
    python -m benchmarks.benchmark --output results.json

All timings use time.perf_counter_ns() with nanosecond precision.
Each benchmark runs 5 iterations (median reported) with 1 warmup run.
"""

import argparse
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Import wayyDB
# ---------------------------------------------------------------------------
try:
    import wayy_db as wdb
    WAYY_AVAILABLE = True
except ImportError:
    WAYY_AVAILABLE = False
    print("ERROR: wayy_db not installed. Run: pip install wayy-db")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Optional competitor imports
# ---------------------------------------------------------------------------
PANDAS_AVAILABLE = False
POLARS_AVAILABLE = False
DUCKDB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pass

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    pass

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_RUNS = 5          # Number of timed iterations (median reported)
N_WARMUP = 1        # Warmup runs (excluded from timing)
BAR_MAX_WIDTH = 40  # Max width for bar chart in terminal
NUM_SYMBOLS = 10    # Number of unique symbols for join benchmarks

# Default row counts for each benchmark category
SIZES_FULL = [1_000, 10_000, 100_000, 1_000_000]
SIZES_QUICK = [1_000, 10_000, 100_000]
JOIN_SIZES_FULL = [(10_000, 10_000), (100_000, 100_000), (1_000_000, 1_000_000)]
JOIN_SIZES_QUICK = [(1_000, 1_000), (10_000, 10_000), (100_000, 100_000)]


# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkResult:
    library: str
    operation: str
    n_rows: int
    median_ns: int          # Median time in nanoseconds
    all_runs_ns: List[int]  # All run times
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def median_ms(self) -> float:
        return self.median_ns / 1_000_000

    @property
    def median_us(self) -> float:
        return self.median_ns / 1_000


@dataclass
class BenchmarkGroup:
    name: str
    description: str
    n_rows: int
    results: List[BenchmarkResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------
def time_fn(fn: Callable, n_runs: int = N_RUNS, n_warmup: int = N_WARMUP) -> Tuple[int, List[int]]:
    """Run fn n_warmup + n_runs times. Return (median_ns, all_run_ns)."""
    for _ in range(n_warmup):
        fn()

    times = []
    for _ in range(n_runs):
        start = time.perf_counter_ns()
        fn()
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed)

    median = int(statistics.median(times))
    return median, times


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------
def get_system_info() -> Dict[str, str]:
    """Collect hardware and software info for reproducibility."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": str(os.cpu_count() or "unknown"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "wayy_db": wdb.__version__,
    }

    if PANDAS_AVAILABLE:
        info["pandas"] = pd.__version__
    if POLARS_AVAILABLE:
        info["polars"] = pl.__version__
    if DUCKDB_AVAILABLE:
        info["duckdb"] = duckdb.__version__

    # Try to get CPU model
    try:
        import psutil
        info["ram_gb"] = f"{psutil.virtual_memory().total / (1024**3):.1f}"
    except ImportError:
        pass

    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":")[1].strip()
                    break
    except (FileNotFoundError, PermissionError):
        pass

    return info


def print_header(info: Dict[str, str]) -> None:
    """Print the benchmark header with system info."""
    print()
    print("=" * 70)
    print("  wayyDB Benchmark Suite v0.1.0")
    print("=" * 70)
    print()
    cpu = info.get("cpu_model", info.get("processor", "unknown"))
    ram = info.get("ram_gb", "?")
    print(f"  Hardware: {cpu}")
    print(f"  RAM: {ram} GB | CPUs: {info['cpu_count']}")
    print(f"  OS: {info['platform']}")
    print()
    libs = [f"Python {info['python']}", f"NumPy {info['numpy']}"]
    if "pandas" in info:
        libs.append(f"pandas {info['pandas']}")
    if "polars" in info:
        libs.append(f"polars {info['polars']}")
    if "duckdb" in info:
        libs.append(f"duckdb {info['duckdb']}")
    print(f"  {' | '.join(libs)}")
    print()


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------
def generate_ohlcv(n: int) -> Dict[str, np.ndarray]:
    """Generate OHLCV-style data with n rows."""
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    high = close + rng.uniform(0, 2, n)
    low = close - rng.uniform(0, 2, n)
    open_ = close + rng.standard_normal(n) * 0.3

    return {
        "timestamp": np.arange(n, dtype=np.int64),
        "open": open_.astype(np.float64),
        "high": high.astype(np.float64),
        "low": low.astype(np.float64),
        "close": close.astype(np.float64),
        "volume": rng.integers(100, 100_000, n).astype(np.int64),
        "symbol": rng.integers(0, NUM_SYMBOLS, n).astype(np.uint32),
    }


def generate_trades_quotes(n_trades: int, n_quotes: int) -> Tuple[Dict, Dict]:
    """Generate matching trades and quotes data for join benchmarks."""
    rng = np.random.default_rng(42)

    trades = {
        "timestamp": np.sort(rng.integers(0, n_trades * 10, n_trades)).astype(np.int64),
        "symbol": rng.integers(0, NUM_SYMBOLS, n_trades).astype(np.uint32),
        "price": (100.0 + rng.standard_normal(n_trades) * 10).astype(np.float64),
        "size": rng.integers(1, 1000, n_trades).astype(np.int64),
    }

    quotes = {
        "timestamp": np.sort(rng.integers(0, n_quotes * 10, n_quotes)).astype(np.int64),
        "symbol": rng.integers(0, NUM_SYMBOLS, n_quotes).astype(np.uint32),
        "bid": (99.0 + rng.standard_normal(n_quotes) * 10).astype(np.float64),
        "ask": (101.0 + rng.standard_normal(n_quotes) * 10).astype(np.float64),
    }

    return trades, quotes


# ---------------------------------------------------------------------------
# Benchmark implementations
# ---------------------------------------------------------------------------
def bench_create_table(n: int, compare: List[str]) -> BenchmarkGroup:
    """Benchmark creating a table from arrays."""
    group = BenchmarkGroup(
        name="CREATE TABLE",
        description=f"Create table with {n:,} rows, 7 columns (OHLCV + symbol)",
        n_rows=n,
    )
    data = generate_ohlcv(n)

    # wayyDB
    def wayy_create():
        wdb.from_dict(data, name="bench", sorted_by="timestamp")

    med, runs = time_fn(wayy_create)
    group.results.append(BenchmarkResult("wayydb", "create_table", n, med, runs))

    # pandas
    if PANDAS_AVAILABLE and "pandas" in compare:
        pdf = {k: v for k, v in data.items()}

        def pd_create():
            pd.DataFrame(pdf)

        med, runs = time_fn(pd_create)
        group.results.append(BenchmarkResult("pandas", "create_table", n, med, runs))

    # polars
    if POLARS_AVAILABLE and "polars" in compare:
        plf = {k: v for k, v in data.items()}

        def pl_create():
            pl.DataFrame(plf)

        med, runs = time_fn(pl_create)
        group.results.append(BenchmarkResult("polars", "create_table", n, med, runs))

    # duckdb
    if DUCKDB_AVAILABLE and "duckdb" in compare:
        # DuckDB: create a relation from dict
        def duck_create():
            conn = duckdb.connect()
            conn.execute("CREATE TABLE bench AS SELECT * FROM ?", [pd.DataFrame(data) if PANDAS_AVAILABLE else None])
            conn.close()

        if PANDAS_AVAILABLE:
            med, runs = time_fn(duck_create)
            group.results.append(BenchmarkResult("duckdb", "create_table", n, med, runs))

    return group


def bench_column_access(n: int, compare: List[str]) -> BenchmarkGroup:
    """Benchmark column access and conversion to numpy."""
    group = BenchmarkGroup(
        name="COLUMN ACCESS",
        description=f"Access 'close' column and get numpy array ({n:,} rows)",
        n_rows=n,
    )
    data = generate_ohlcv(n)

    # wayyDB
    tbl = wdb.from_dict(data, name="bench_access", sorted_by="timestamp")

    def wayy_access():
        arr = tbl["close"].to_numpy()
        return arr

    med, runs = time_fn(wayy_access)
    group.results.append(BenchmarkResult("wayydb", "column_access", n, med, runs,
                                          extra={"note": "zero-copy"}))

    # pandas
    if PANDAS_AVAILABLE and "pandas" in compare:
        pdf = pd.DataFrame(data)

        def pd_access():
            arr = pdf["close"].values
            return arr

        med, runs = time_fn(pd_access)
        group.results.append(BenchmarkResult("pandas", "column_access", n, med, runs))

    # polars
    if POLARS_AVAILABLE and "polars" in compare:
        plf = pl.DataFrame(data)

        def pl_access():
            arr = plf["close"].to_numpy()
            return arr

        med, runs = time_fn(pl_access)
        group.results.append(BenchmarkResult("polars", "column_access", n, med, runs))

    return group


def bench_aggregations(n: int, compare: List[str]) -> BenchmarkGroup:
    """Benchmark sum, mean, min, max, std on a single column."""
    group = BenchmarkGroup(
        name="AGGREGATIONS",
        description=f"sum + mean + min + max + std on 'close' ({n:,} rows)",
        n_rows=n,
    )
    data = generate_ohlcv(n)

    # wayyDB
    tbl = wdb.from_dict(data, name="bench_agg", sorted_by="timestamp")
    col = tbl["close"]

    def wayy_agg():
        wdb.ops.sum(col)
        wdb.ops.avg(col)
        wdb.ops.min(col)
        wdb.ops.max(col)
        wdb.ops.std(col)

    med, runs = time_fn(wayy_agg)
    group.results.append(BenchmarkResult("wayydb", "aggregations", n, med, runs,
                                          extra={"note": "SIMD AVX2"}))

    # numpy (baseline for SIMD comparison)
    close_np = data["close"]

    def np_agg():
        np.sum(close_np)
        np.mean(close_np)
        np.min(close_np)
        np.max(close_np)
        np.std(close_np)

    med, runs = time_fn(np_agg)
    group.results.append(BenchmarkResult("numpy", "aggregations", n, med, runs))

    # pandas
    if PANDAS_AVAILABLE and "pandas" in compare:
        pdf = pd.DataFrame(data)
        s = pdf["close"]

        def pd_agg():
            s.sum()
            s.mean()
            s.min()
            s.max()
            s.std()

        med, runs = time_fn(pd_agg)
        group.results.append(BenchmarkResult("pandas", "aggregations", n, med, runs))

    # polars
    if POLARS_AVAILABLE and "polars" in compare:
        plf = pl.DataFrame(data)

        def pl_agg():
            plf["close"].sum()
            plf["close"].mean()
            plf["close"].min()
            plf["close"].max()
            plf["close"].std()

        med, runs = time_fn(pl_agg)
        group.results.append(BenchmarkResult("polars", "aggregations", n, med, runs))

    # duckdb
    if DUCKDB_AVAILABLE and "duckdb" in compare:
        conn = duckdb.connect()
        if PANDAS_AVAILABLE:
            pdf_duck = pd.DataFrame(data)
            conn.execute("CREATE TABLE bench_agg AS SELECT * FROM pdf_duck")
        else:
            # Use numpy arrays directly via duckdb
            conn.execute(
                "CREATE TABLE bench_agg (close DOUBLE)"
            )
            conn.executemany("INSERT INTO bench_agg VALUES (?)", [(float(x),) for x in data["close"][:min(n, 100_000)]])

        def duck_agg():
            conn.execute("SELECT SUM(close), AVG(close), MIN(close), MAX(close), STDDEV(close) FROM bench_agg")
            conn.fetchone()

        med, runs = time_fn(duck_agg)
        group.results.append(BenchmarkResult("duckdb", "aggregations", n, med, runs))
        conn.close()

    return group


def bench_asof_join(n_trades: int, n_quotes: int, compare: List[str]) -> BenchmarkGroup:
    """Benchmark as-of join -- THE killer benchmark for time-series DBs."""
    group = BenchmarkGroup(
        name="AS-OF JOIN",
        description=f"As-of join: {n_trades:,} trades x {n_quotes:,} quotes, {NUM_SYMBOLS} symbols",
        n_rows=n_trades,
    )
    trades_data, quotes_data = generate_trades_quotes(n_trades, n_quotes)

    # wayyDB
    trades_wdb = wdb.from_dict(trades_data, name="trades", sorted_by="timestamp")
    quotes_wdb = wdb.from_dict(quotes_data, name="quotes", sorted_by="timestamp")

    def wayy_aj():
        wdb.ops.aj(trades_wdb, quotes_wdb, ["symbol"], "timestamp")

    med, runs = time_fn(wayy_aj)
    group.results.append(BenchmarkResult("wayydb", "asof_join", n_trades, med, runs,
                                          extra={"n_quotes": n_quotes}))

    # pandas
    if PANDAS_AVAILABLE and "pandas" in compare:
        trades_pdf = pd.DataFrame(trades_data)
        quotes_pdf = pd.DataFrame(quotes_data)
        # pandas merge_asof requires sorted data and uses "by" for grouping
        trades_pdf = trades_pdf.sort_values("timestamp")
        quotes_pdf = quotes_pdf.sort_values("timestamp")

        def pd_aj():
            pd.merge_asof(
                trades_pdf, quotes_pdf,
                on="timestamp", by="symbol",
                direction="backward"
            )

        med, runs = time_fn(pd_aj)
        group.results.append(BenchmarkResult("pandas", "asof_join", n_trades, med, runs,
                                              extra={"n_quotes": n_quotes}))

    # polars
    if POLARS_AVAILABLE and "polars" in compare:
        trades_plf = pl.DataFrame(trades_data).sort("timestamp")
        quotes_plf = pl.DataFrame(quotes_data).sort("timestamp")

        def pl_aj():
            trades_plf.join_asof(
                quotes_plf,
                on="timestamp",
                by="symbol",
                strategy="backward"
            )

        med, runs = time_fn(pl_aj)
        group.results.append(BenchmarkResult("polars", "asof_join", n_trades, med, runs,
                                              extra={"n_quotes": n_quotes}))

    # duckdb
    if DUCKDB_AVAILABLE and "duckdb" in compare and PANDAS_AVAILABLE:
        trades_pdf_duck = pd.DataFrame(trades_data).sort_values("timestamp")
        quotes_pdf_duck = pd.DataFrame(quotes_data).sort_values("timestamp")
        conn = duckdb.connect()
        conn.execute("CREATE TABLE trades AS SELECT * FROM trades_pdf_duck")
        conn.execute("CREATE TABLE quotes AS SELECT * FROM quotes_pdf_duck")

        def duck_aj():
            conn.execute("""
                SELECT t.*, q.bid, q.ask
                FROM trades t
                ASOF JOIN quotes q
                ON t.symbol = q.symbol AND t.timestamp >= q.timestamp
            """)
            conn.fetchall()

        med, runs = time_fn(duck_aj)
        group.results.append(BenchmarkResult("duckdb", "asof_join", n_trades, med, runs,
                                              extra={"n_quotes": n_quotes}))
        conn.close()

    return group


def bench_window_functions(n: int, compare: List[str]) -> BenchmarkGroup:
    """Benchmark moving average (20-period), EMA, rolling std."""
    group = BenchmarkGroup(
        name="WINDOW FUNCTIONS",
        description=f"mavg(20) + ema(0.1) + mstd(20) on 'close' ({n:,} rows)",
        n_rows=n,
    )
    data = generate_ohlcv(n)

    # wayyDB
    tbl = wdb.from_dict(data, name="bench_window", sorted_by="timestamp")
    col = tbl["close"]

    def wayy_window():
        wdb.ops.mavg(col, 20)
        wdb.ops.ema(col, 0.1)
        wdb.ops.mstd(col, 20)

    med, runs = time_fn(wayy_window)
    group.results.append(BenchmarkResult("wayydb", "window_functions", n, med, runs))

    # pandas
    if PANDAS_AVAILABLE and "pandas" in compare:
        pdf = pd.DataFrame(data)
        s = pdf["close"]

        def pd_window():
            s.rolling(20).mean()
            s.ewm(alpha=0.1, adjust=False).mean()
            s.rolling(20).std()

        med, runs = time_fn(pd_window)
        group.results.append(BenchmarkResult("pandas", "window_functions", n, med, runs))

    # polars
    if POLARS_AVAILABLE and "polars" in compare:
        plf = pl.DataFrame(data)

        def pl_window():
            plf["close"].rolling_mean(window_size=20)
            plf["close"].ewm_mean(alpha=0.1, adjust=False)
            plf["close"].rolling_std(window_size=20)

        med, runs = time_fn(pl_window)
        group.results.append(BenchmarkResult("polars", "window_functions", n, med, runs))

    return group


def bench_persistence(n: int, compare: List[str]) -> BenchmarkGroup:
    """Benchmark save/load persistence."""
    group = BenchmarkGroup(
        name="PERSISTENCE",
        description=f"Save to disk, then load ({n:,} rows, 7 columns)",
        n_rows=n,
    )
    data = generate_ohlcv(n)

    with tempfile.TemporaryDirectory() as tmpdir:
        # --- wayyDB save ---
        tbl = wdb.from_dict(data, name="bench_persist", sorted_by="timestamp")
        save_path = os.path.join(tmpdir, "wayy_save")

        def wayy_save():
            tbl.save(save_path)

        med_save, runs_save = time_fn(wayy_save)

        # --- wayyDB mmap load ---
        tbl.save(save_path)  # ensure file exists

        def wayy_mmap():
            wdb.Table.mmap(save_path)

        med_load, runs_load = time_fn(wayy_mmap)
        group.results.append(BenchmarkResult("wayydb", "persistence_save", n, med_save, runs_save))
        group.results.append(BenchmarkResult("wayydb", "persistence_load", n, med_load, runs_load,
                                              extra={"note": "mmap zero-copy"}))

        # --- pandas parquet ---
        if PANDAS_AVAILABLE and "pandas" in compare:
            pdf = pd.DataFrame(data)
            pq_path = os.path.join(tmpdir, "pandas.parquet")

            def pd_save():
                pdf.to_parquet(pq_path)

            med_save, runs_save = time_fn(pd_save)

            pdf.to_parquet(pq_path)

            def pd_load():
                pd.read_parquet(pq_path)

            med_load, runs_load = time_fn(pd_load)
            group.results.append(BenchmarkResult("pandas", "persistence_save", n, med_save, runs_save))
            group.results.append(BenchmarkResult("pandas", "persistence_load", n, med_load, runs_load))

        # --- polars parquet ---
        if POLARS_AVAILABLE and "polars" in compare:
            plf = pl.DataFrame(data)
            pq_path_pl = os.path.join(tmpdir, "polars.parquet")

            def pl_save():
                plf.write_parquet(pq_path_pl)

            med_save, runs_save = time_fn(pl_save)

            plf.write_parquet(pq_path_pl)

            def pl_load():
                pl.read_parquet(pq_path_pl)

            med_load, runs_load = time_fn(pl_load)
            group.results.append(BenchmarkResult("polars", "persistence_save", n, med_save, runs_save))
            group.results.append(BenchmarkResult("polars", "persistence_load", n, med_load, runs_load))

        # --- duckdb parquet ---
        if DUCKDB_AVAILABLE and "duckdb" in compare and PANDAS_AVAILABLE:
            conn = duckdb.connect()
            pdf_duck = pd.DataFrame(data)
            conn.execute("CREATE TABLE bench_persist AS SELECT * FROM pdf_duck")
            pq_path_duck = os.path.join(tmpdir, "duckdb.parquet")

            def duck_save():
                conn.execute(f"COPY bench_persist TO '{pq_path_duck}' (FORMAT PARQUET)")

            med_save, runs_save = time_fn(duck_save)

            conn.execute(f"COPY bench_persist TO '{pq_path_duck}' (FORMAT PARQUET)")

            def duck_load():
                conn.execute(f"SELECT * FROM read_parquet('{pq_path_duck}')")
                conn.fetchall()

            med_load, runs_load = time_fn(duck_load)
            group.results.append(BenchmarkResult("duckdb", "persistence_save", n, med_save, runs_save))
            group.results.append(BenchmarkResult("duckdb", "persistence_load", n, med_load, runs_load))
            conn.close()

    return group


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def format_time(ns: int) -> str:
    """Format nanoseconds to human-readable string."""
    if ns < 1_000:
        return f"{ns} ns"
    elif ns < 1_000_000:
        return f"{ns / 1_000:.1f} us"
    elif ns < 1_000_000_000:
        return f"{ns / 1_000_000:.1f} ms"
    else:
        return f"{ns / 1_000_000_000:.2f} s"


def print_group(group: BenchmarkGroup) -> None:
    """Print a benchmark group with bar chart."""
    print()
    print(f"{group.name} ({group.description})")
    print("-" * 70)

    if not group.results:
        print("  (no results)")
        return

    # Find the minimum time (baseline) for this group
    # Group results by operation to handle persistence (save vs load)
    ops = {}
    for r in group.results:
        ops.setdefault(r.operation, []).append(r)

    for op_name, results in ops.items():
        if len(ops) > 1:
            # Sub-header for persistence save vs load
            label = op_name.replace("persistence_", "").upper()
            print(f"  [{label}]")

        baseline = min(r.median_ns for r in results)

        for r in sorted(results, key=lambda x: x.median_ns):
            ratio = r.median_ns / baseline if baseline > 0 else 1.0
            bar_len = max(1, int(BAR_MAX_WIDTH * (r.median_ns / max(rr.median_ns for rr in results))))
            bar = "#" * bar_len

            time_str = format_time(r.median_ns)
            name = r.library.ljust(10)

            if ratio <= 1.01:
                ratio_str = "1.0x (baseline)"
            else:
                ratio_str = f"{ratio:.1f}x slower"

            note = ""
            if r.extra.get("note"):
                note = f"  [{r.extra['note']}]"

            print(f"  {name} {time_str:>12}   {bar:<{BAR_MAX_WIDTH}}  {ratio_str}{note}")

        if len(ops) > 1:
            print()


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_benchmarks(
    compare: List[str],
    only: Optional[str] = None,
    quick: bool = False,
    output: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the benchmark suite and return results."""

    sizes = SIZES_QUICK if quick else SIZES_FULL
    join_sizes = JOIN_SIZES_QUICK if quick else JOIN_SIZES_FULL
    # Use the largest size for single-size benchmarks
    n = sizes[-1]

    info = get_system_info()
    print_header(info)

    all_groups: List[BenchmarkGroup] = []

    benchmarks = {
        "create_table": lambda: bench_create_table(n, compare),
        "column_access": lambda: bench_column_access(n, compare),
        "aggregations": lambda: bench_aggregations(n, compare),
        "asof_join": lambda: [bench_asof_join(t, q, compare) for t, q in join_sizes],
        "window_functions": lambda: bench_window_functions(n, compare),
        "persistence": lambda: bench_persistence(n, compare),
    }

    if only:
        if only not in benchmarks:
            print(f"Unknown benchmark: {only}")
            print(f"Available: {', '.join(benchmarks.keys())}")
            sys.exit(1)
        to_run = {only: benchmarks[only]}
    else:
        to_run = benchmarks

    for name, fn in to_run.items():
        print(f"\n>>> Running: {name} ...")
        result = fn()
        if isinstance(result, list):
            for g in result:
                print_group(g)
                all_groups.append(g)
        else:
            print_group(result)
            all_groups.append(result)

    # Summary
    print()
    print("=" * 70)
    print("  Benchmark Suite Complete")
    print("=" * 70)

    # Compile JSON output
    output_data = {
        "version": "0.1.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "system": info,
        "benchmarks": [],
    }

    for g in all_groups:
        bench_entry = {
            "name": g.name,
            "description": g.description,
            "n_rows": g.n_rows,
            "results": [],
        }
        for r in g.results:
            bench_entry["results"].append({
                "library": r.library,
                "operation": r.operation,
                "n_rows": r.n_rows,
                "median_ms": round(r.median_ms, 3),
                "median_ns": r.median_ns,
                "all_runs_ns": r.all_runs_ns,
                "extra": r.extra,
            })
        output_data["benchmarks"].append(bench_entry)

    # Save results
    if output:
        output_path = output
    else:
        output_path = os.path.join(os.path.dirname(__file__), "results.json")

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n  Results saved to: {output_path}")

    return output_data


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="wayyDB Benchmark Suite - Compare performance against pandas, polars, duckdb",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m benchmarks.benchmark                          # Full suite
  python -m benchmarks.benchmark --quick                  # Smaller datasets
  python -m benchmarks.benchmark --only asof_join         # Single benchmark
  python -m benchmarks.benchmark --compare pandas,polars  # Specific libraries
  python -m benchmarks.benchmark --output results.json    # Custom output path
        """,
    )
    parser.add_argument(
        "--compare",
        type=str,
        default="pandas,polars,duckdb",
        help="Comma-separated list of libraries to compare (default: pandas,polars,duckdb)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        choices=["create_table", "column_access", "aggregations", "asof_join",
                 "window_functions", "persistence"],
        help="Run only a specific benchmark",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller datasets for faster runs (100K max instead of 1M)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results JSON (default: benchmarks/results.json)",
    )

    args = parser.parse_args()

    compare = [lib.strip().lower() for lib in args.compare.split(",")]

    # Check availability
    for lib in compare:
        if lib == "pandas" and not PANDAS_AVAILABLE:
            print(f"WARNING: pandas not installed, skipping. Install: pip install pandas")
        elif lib == "polars" and not POLARS_AVAILABLE:
            print(f"WARNING: polars not installed, skipping. Install: pip install polars")
        elif lib == "duckdb" and not DUCKDB_AVAILABLE:
            print(f"WARNING: duckdb not installed, skipping. Install: pip install duckdb")

    run_benchmarks(
        compare=compare,
        only=args.only,
        quick=args.quick,
        output=args.output,
    )


if __name__ == "__main__":
    main()
