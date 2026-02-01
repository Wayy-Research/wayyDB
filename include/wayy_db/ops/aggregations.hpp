#pragma once

#include "wayy_db/column_view.hpp"
#include "wayy_db/column.hpp"

#include <cmath>
#include <limits>

namespace wayy_db::ops {

/// Sum of all values in a column
template<typename T>
T sum(const ColumnView<T>& col);

/// SIMD-optimized sum for float64
double sum_simd(const ColumnView<double>& col);
int64_t sum_simd(const ColumnView<int64_t>& col);

/// Mean (average) of all values
template<typename T>
double avg(const ColumnView<T>& col) {
    if (col.empty()) return std::numeric_limits<double>::quiet_NaN();
    return static_cast<double>(sum(col)) / static_cast<double>(col.size());
}

/// Minimum value
template<typename T>
T min(const ColumnView<T>& col);

/// Maximum value
template<typename T>
T max(const ColumnView<T>& col);

/// Standard deviation (population)
template<typename T>
double std_dev(const ColumnView<T>& col);

/// Variance (population)
template<typename T>
double variance(const ColumnView<T>& col);

/// Count non-null values (for future nullable support)
template<typename T>
size_t count(const ColumnView<T>& col) {
    return col.size();
}

/// First value
template<typename T>
T first(const ColumnView<T>& col) {
    if (col.empty()) throw InvalidOperation("first() on empty column");
    return col.front();
}

/// Last value
template<typename T>
T last(const ColumnView<T>& col) {
    if (col.empty()) throw InvalidOperation("last() on empty column");
    return col.back();
}

// Type-erased aggregations on Column objects
double sum(const Column& col);
double avg(const Column& col);
double min_val(const Column& col);
double max_val(const Column& col);
double std_dev(const Column& col);

}  // namespace wayy_db::ops
