Hey Product Hunt!

I'm Rick, a former institutional quant who spent 9 years building trading systems
at hedge funds. The #1 pain point in quant finance is that Python is too slow for
time-series operations at scale, but kdb+ costs $100K/year.

wayyDB is my answer: a C++20 engine with Python bindings that makes quant-specific
operations (as-of joins, temporal aggregations, window functions) fast enough for
production -- while being pip-installable and MIT licensed.

**Why C++ instead of Rust?** Mature SIMD intrinsics, proven memory-mapped I/O
patterns, and pybind11 gives us zero-copy NumPy interop that Rust bindings
can't match yet.

**The architecture:**
- Columnar storage with sorted indices for binary search joins
- AVX2 SIMD for vectorized aggregations (8 doubles at a time)
- Memory-mapped files for instant loading (10M rows in microseconds)
- pybind11 buffer protocol for zero-copy NumPy arrays

Happy to answer any technical questions about the implementation!
