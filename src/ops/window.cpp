#include "wayy_db/ops/window.hpp"

#include <deque>
#include <cmath>
#include <numeric>

namespace wayy_db::ops {

// Moving average

std::vector<double> mavg(const ColumnView<double>& col, size_t window) {
    if (col.empty() || window == 0) return {};

    std::vector<double> result(col.size());
    double sum = 0.0;

    for (size_t i = 0; i < col.size(); ++i) {
        sum += col[i];
        if (i >= window) {
            sum -= col[i - window];
            result[i] = sum / static_cast<double>(window);
        } else {
            result[i] = sum / static_cast<double>(i + 1);
        }
    }

    return result;
}

std::vector<double> mavg(const ColumnView<int64_t>& col, size_t window) {
    if (col.empty() || window == 0) return {};

    std::vector<double> result(col.size());
    int64_t sum = 0;

    for (size_t i = 0; i < col.size(); ++i) {
        sum += col[i];
        if (i >= window) {
            sum -= col[i - window];
            result[i] = static_cast<double>(sum) / static_cast<double>(window);
        } else {
            result[i] = static_cast<double>(sum) / static_cast<double>(i + 1);
        }
    }

    return result;
}

// Moving sum

std::vector<double> msum(const ColumnView<double>& col, size_t window) {
    if (col.empty() || window == 0) return {};

    std::vector<double> result(col.size());
    double sum = 0.0;

    for (size_t i = 0; i < col.size(); ++i) {
        sum += col[i];
        if (i >= window) {
            sum -= col[i - window];
        }
        result[i] = sum;
    }

    return result;
}

std::vector<int64_t> msum(const ColumnView<int64_t>& col, size_t window) {
    if (col.empty() || window == 0) return {};

    std::vector<int64_t> result(col.size());
    int64_t sum = 0;

    for (size_t i = 0; i < col.size(); ++i) {
        sum += col[i];
        if (i >= window) {
            sum -= col[i - window];
        }
        result[i] = sum;
    }

    return result;
}

// Moving standard deviation (Welford's online algorithm)

std::vector<double> mstd(const ColumnView<double>& col, size_t window) {
    if (col.empty() || window == 0) return {};

    std::vector<double> result(col.size());

    for (size_t i = 0; i < col.size(); ++i) {
        size_t start = (i >= window) ? i - window + 1 : 0;
        size_t count = i - start + 1;

        double mean = 0.0;
        double m2 = 0.0;
        size_t n = 0;

        for (size_t j = start; j <= i; ++j) {
            ++n;
            double delta = col[j] - mean;
            mean += delta / static_cast<double>(n);
            double delta2 = col[j] - mean;
            m2 += delta * delta2;
        }

        result[i] = (n > 1) ? std::sqrt(m2 / static_cast<double>(n)) : 0.0;
    }

    return result;
}

std::vector<double> mstd(const ColumnView<int64_t>& col, size_t window) {
    if (col.empty() || window == 0) return {};

    std::vector<double> result(col.size());

    for (size_t i = 0; i < col.size(); ++i) {
        size_t start = (i >= window) ? i - window + 1 : 0;

        double mean = 0.0;
        double m2 = 0.0;
        size_t n = 0;

        for (size_t j = start; j <= i; ++j) {
            ++n;
            double val = static_cast<double>(col[j]);
            double delta = val - mean;
            mean += delta / static_cast<double>(n);
            double delta2 = val - mean;
            m2 += delta * delta2;
        }

        result[i] = (n > 1) ? std::sqrt(m2 / static_cast<double>(n)) : 0.0;
    }

    return result;
}

// Moving min/max using monotonic deque for O(n) complexity

template<typename T, typename Compare>
std::vector<T> monotonic_window(const ColumnView<T>& col, size_t window, Compare cmp) {
    if (col.empty() || window == 0) return {};

    std::vector<T> result(col.size());
    std::deque<size_t> dq;  // Indices

    for (size_t i = 0; i < col.size(); ++i) {
        // Remove elements outside window
        while (!dq.empty() && dq.front() + window <= i) {
            dq.pop_front();
        }

        // Remove elements that won't be min/max
        while (!dq.empty() && cmp(col[i], col[dq.back()])) {
            dq.pop_back();
        }

        dq.push_back(i);
        result[i] = col[dq.front()];
    }

    return result;
}

std::vector<double> mmin(const ColumnView<double>& col, size_t window) {
    return monotonic_window(col, window, std::less<double>{});
}

std::vector<int64_t> mmin(const ColumnView<int64_t>& col, size_t window) {
    return monotonic_window(col, window, std::less<int64_t>{});
}

std::vector<double> mmax(const ColumnView<double>& col, size_t window) {
    return monotonic_window(col, window, std::greater<double>{});
}

std::vector<int64_t> mmax(const ColumnView<int64_t>& col, size_t window) {
    return monotonic_window(col, window, std::greater<int64_t>{});
}

// Exponential moving average

std::vector<double> ema(const ColumnView<double>& col, double alpha) {
    if (col.empty()) return {};
    if (alpha <= 0.0 || alpha > 1.0) {
        throw std::invalid_argument("EMA alpha must be in (0, 1]");
    }

    std::vector<double> result(col.size());
    result[0] = col[0];

    for (size_t i = 1; i < col.size(); ++i) {
        result[i] = alpha * col[i] + (1.0 - alpha) * result[i - 1];
    }

    return result;
}

std::vector<double> ema(const ColumnView<int64_t>& col, double alpha) {
    if (col.empty()) return {};
    if (alpha <= 0.0 || alpha > 1.0) {
        throw std::invalid_argument("EMA alpha must be in (0, 1]");
    }

    std::vector<double> result(col.size());
    result[0] = static_cast<double>(col[0]);

    for (size_t i = 1; i < col.size(); ++i) {
        result[i] = alpha * static_cast<double>(col[i]) + (1.0 - alpha) * result[i - 1];
    }

    return result;
}

std::vector<double> ema_span(const ColumnView<double>& col, size_t span) {
    double alpha = 2.0 / (static_cast<double>(span) + 1.0);
    return ema(col, alpha);
}

// Diff

std::vector<double> diff(const ColumnView<double>& col, size_t periods) {
    if (col.empty() || periods >= col.size()) return std::vector<double>(col.size(), 0.0);

    std::vector<double> result(col.size());
    for (size_t i = 0; i < periods; ++i) {
        result[i] = std::numeric_limits<double>::quiet_NaN();
    }
    for (size_t i = periods; i < col.size(); ++i) {
        result[i] = col[i] - col[i - periods];
    }

    return result;
}

std::vector<int64_t> diff(const ColumnView<int64_t>& col, size_t periods) {
    if (col.empty() || periods >= col.size()) return std::vector<int64_t>(col.size(), 0);

    std::vector<int64_t> result(col.size(), 0);
    for (size_t i = periods; i < col.size(); ++i) {
        result[i] = col[i] - col[i - periods];
    }

    return result;
}

// Percent change

std::vector<double> pct_change(const ColumnView<double>& col, size_t periods) {
    if (col.empty() || periods >= col.size()) {
        return std::vector<double>(col.size(), std::numeric_limits<double>::quiet_NaN());
    }

    std::vector<double> result(col.size());
    for (size_t i = 0; i < periods; ++i) {
        result[i] = std::numeric_limits<double>::quiet_NaN();
    }
    for (size_t i = periods; i < col.size(); ++i) {
        if (col[i - periods] != 0.0) {
            result[i] = (col[i] - col[i - periods]) / col[i - periods];
        } else {
            result[i] = std::numeric_limits<double>::quiet_NaN();
        }
    }

    return result;
}

// Shift

std::vector<double> shift(const ColumnView<double>& col, int64_t n) {
    if (col.empty()) return {};

    std::vector<double> result(col.size(), std::numeric_limits<double>::quiet_NaN());

    if (n >= 0) {
        size_t offset = static_cast<size_t>(n);
        for (size_t i = offset; i < col.size(); ++i) {
            result[i] = col[i - offset];
        }
    } else {
        size_t offset = static_cast<size_t>(-n);
        for (size_t i = 0; i + offset < col.size(); ++i) {
            result[i] = col[i + offset];
        }
    }

    return result;
}

std::vector<int64_t> shift(const ColumnView<int64_t>& col, int64_t n) {
    if (col.empty()) return {};

    std::vector<int64_t> result(col.size(), 0);

    if (n >= 0) {
        size_t offset = static_cast<size_t>(n);
        for (size_t i = offset; i < col.size(); ++i) {
            result[i] = col[i - offset];
        }
    } else {
        size_t offset = static_cast<size_t>(-n);
        for (size_t i = 0; i + offset < col.size(); ++i) {
            result[i] = col[i + offset];
        }
    }

    return result;
}

}  // namespace wayy_db::ops
