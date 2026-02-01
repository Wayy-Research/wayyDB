#include "wayy_db/ops/aggregations.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#ifdef WAYY_USE_AVX2
#include <immintrin.h>
#endif

namespace wayy_db::ops {

// Scalar implementations

template<typename T>
T sum(const ColumnView<T>& col) {
    return std::accumulate(col.begin(), col.end(), T{0});
}

template int64_t sum(const ColumnView<int64_t>&);
template double sum(const ColumnView<double>&);

template<typename T>
T min(const ColumnView<T>& col) {
    if (col.empty()) {
        throw InvalidOperation("min() on empty column");
    }
    return *std::min_element(col.begin(), col.end());
}

template int64_t min(const ColumnView<int64_t>&);
template double min(const ColumnView<double>&);

template<typename T>
T max(const ColumnView<T>& col) {
    if (col.empty()) {
        throw InvalidOperation("max() on empty column");
    }
    return *std::max_element(col.begin(), col.end());
}

template int64_t max(const ColumnView<int64_t>&);
template double max(const ColumnView<double>&);

template<typename T>
double variance(const ColumnView<T>& col) {
    if (col.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double mean = avg(col);
    double sum_sq = 0.0;

    for (const auto& val : col) {
        double diff = static_cast<double>(val) - mean;
        sum_sq += diff * diff;
    }

    return sum_sq / static_cast<double>(col.size());
}

template double variance(const ColumnView<int64_t>&);
template double variance(const ColumnView<double>&);

template<typename T>
double std_dev(const ColumnView<T>& col) {
    return std::sqrt(variance(col));
}

template double std_dev(const ColumnView<int64_t>&);
template double std_dev(const ColumnView<double>&);

// SIMD implementations

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

int64_t sum_simd(const ColumnView<int64_t>& col) {
    const int64_t* data = col.data();
    size_t n = col.size();

    __m256i vsum = _mm256_setzero_si256();

    // Process 4 int64s per iteration
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
        vsum = _mm256_add_epi64(vsum, v);
    }

    // Horizontal reduction
    alignas(32) int64_t temp[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(temp), vsum);
    int64_t result = temp[0] + temp[1] + temp[2] + temp[3];

    // Handle remainder
    for (; i < n; ++i) {
        result += data[i];
    }

    return result;
}

#else

double sum_simd(const ColumnView<double>& col) {
    return sum(col);
}

int64_t sum_simd(const ColumnView<int64_t>& col) {
    return sum(col);
}

#endif

// Type-erased implementations

double sum(const Column& col) {
    switch (col.dtype()) {
        case DType::Int64:
        case DType::Timestamp:
            return static_cast<double>(sum_simd(const_cast<Column&>(col).as_int64()));
        case DType::Float64:
            return sum_simd(const_cast<Column&>(col).as_float64());
        default:
            throw InvalidOperation("sum() not supported for this type");
    }
}

double avg(const Column& col) {
    if (col.size() == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return sum(col) / static_cast<double>(col.size());
}

double min_val(const Column& col) {
    switch (col.dtype()) {
        case DType::Int64:
        case DType::Timestamp:
            return static_cast<double>(min(const_cast<Column&>(col).as_int64()));
        case DType::Float64:
            return min(const_cast<Column&>(col).as_float64());
        default:
            throw InvalidOperation("min() not supported for this type");
    }
}

double max_val(const Column& col) {
    switch (col.dtype()) {
        case DType::Int64:
        case DType::Timestamp:
            return static_cast<double>(max(const_cast<Column&>(col).as_int64()));
        case DType::Float64:
            return max(const_cast<Column&>(col).as_float64());
        default:
            throw InvalidOperation("max() not supported for this type");
    }
}

double std_dev(const Column& col) {
    switch (col.dtype()) {
        case DType::Int64:
        case DType::Timestamp:
            return std_dev(const_cast<Column&>(col).as_int64());
        case DType::Float64:
            return std_dev(const_cast<Column&>(col).as_float64());
        default:
            throw InvalidOperation("std_dev() not supported for this type");
    }
}

}  // namespace wayy_db::ops
