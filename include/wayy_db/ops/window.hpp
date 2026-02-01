#pragma once

#include "wayy_db/column_view.hpp"

#include <vector>

namespace wayy_db::ops {

/// Moving average over a sliding window
/// @param col Input column
/// @param window Window size
/// @return Vector of moving averages (first window-1 values are partial averages)
std::vector<double> mavg(const ColumnView<double>& col, size_t window);
std::vector<double> mavg(const ColumnView<int64_t>& col, size_t window);

/// Moving sum over a sliding window
std::vector<double> msum(const ColumnView<double>& col, size_t window);
std::vector<int64_t> msum(const ColumnView<int64_t>& col, size_t window);

/// Moving standard deviation over a sliding window
std::vector<double> mstd(const ColumnView<double>& col, size_t window);
std::vector<double> mstd(const ColumnView<int64_t>& col, size_t window);

/// Moving minimum over a sliding window (O(n) using monotonic deque)
std::vector<double> mmin(const ColumnView<double>& col, size_t window);
std::vector<int64_t> mmin(const ColumnView<int64_t>& col, size_t window);

/// Moving maximum over a sliding window (O(n) using monotonic deque)
std::vector<double> mmax(const ColumnView<double>& col, size_t window);
std::vector<int64_t> mmax(const ColumnView<int64_t>& col, size_t window);

/// Exponential moving average
/// @param col Input column
/// @param alpha Smoothing factor (0 < alpha <= 1)
/// @return Vector of EMA values
std::vector<double> ema(const ColumnView<double>& col, double alpha);
std::vector<double> ema(const ColumnView<int64_t>& col, double alpha);

/// Exponential moving average with span
/// alpha = 2 / (span + 1)
std::vector<double> ema_span(const ColumnView<double>& col, size_t span);

/// Diff: difference between consecutive values
std::vector<double> diff(const ColumnView<double>& col, size_t periods = 1);
std::vector<int64_t> diff(const ColumnView<int64_t>& col, size_t periods = 1);

/// Percent change between consecutive values
std::vector<double> pct_change(const ColumnView<double>& col, size_t periods = 1);

/// Shift values by n positions (positive = forward, negative = backward)
std::vector<double> shift(const ColumnView<double>& col, int64_t n);
std::vector<int64_t> shift(const ColumnView<int64_t>& col, int64_t n);

}  // namespace wayy_db::ops
