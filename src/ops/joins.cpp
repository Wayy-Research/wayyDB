#include "wayy_db/ops/joins.hpp"

#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace wayy_db::ops {

namespace {

// Hash combine for multi-key joins
struct KeyHash {
    size_t operator()(const std::vector<int64_t>& key) const {
        size_t hash = 0;
        for (auto val : key) {
            hash ^= std::hash<int64_t>{}(val) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

// Extract join key values from a row
std::vector<int64_t> extract_key(const Table& table,
                                  const std::vector<std::string>& on,
                                  size_t row) {
    std::vector<int64_t> key;
    key.reserve(on.size());

    for (const auto& col_name : on) {
        const Column& col = table.column(col_name);
        switch (col.dtype()) {
            case DType::Int64:
            case DType::Timestamp:
                key.push_back(const_cast<Column&>(col).as_int64()[row]);
                break;
            case DType::Symbol:
                key.push_back(const_cast<Column&>(col).as_symbol()[row]);
                break;
            default:
                throw InvalidOperation("Join key column must be Int64, Timestamp, or Symbol");
        }
    }

    return key;
}

// Group row indices by key values
std::unordered_map<std::vector<int64_t>, std::vector<size_t>, KeyHash>
group_by_key(const Table& table, const std::vector<std::string>& on) {
    std::unordered_map<std::vector<int64_t>, std::vector<size_t>, KeyHash> groups;

    for (size_t i = 0; i < table.num_rows(); ++i) {
        auto key = extract_key(table, on, i);
        groups[key].push_back(i);
    }

    return groups;
}

}  // namespace

Table aj(const Table& left, const Table& right,
         const std::vector<std::string>& on,
         const std::string& as_of) {

    // Validate inputs
    if (!left.is_sorted() || left.sorted_by() != as_of) {
        throw InvalidOperation("Left table must be sorted by " + as_of);
    }
    if (!right.is_sorted() || right.sorted_by() != as_of) {
        throw InvalidOperation("Right table must be sorted by " + as_of);
    }

    // Group right table by join keys
    auto right_groups = group_by_key(right, on);

    // Get timestamp columns
    auto left_ts = const_cast<Table&>(left).column(as_of).as_int64();
    auto right_ts = const_cast<Table&>(right).column(as_of).as_int64();

    // Result builders - collect matching indices
    std::vector<size_t> left_indices;
    std::vector<size_t> right_indices;  // -1 means no match
    left_indices.reserve(left.num_rows());
    right_indices.reserve(left.num_rows());

    // For each left row, find the most recent right row
    for (size_t i = 0; i < left.num_rows(); ++i) {
        auto key = extract_key(left, on, i);
        int64_t ts = left_ts[i];

        auto group_it = right_groups.find(key);
        if (group_it == right_groups.end()) {
            // No matching key in right table
            left_indices.push_back(i);
            right_indices.push_back(static_cast<size_t>(-1));
            continue;
        }

        const auto& group = group_it->second;

        // Binary search for largest timestamp <= ts
        auto it = std::upper_bound(group.begin(), group.end(), ts,
            [&right_ts](int64_t t, size_t idx) { return t < right_ts[idx]; });

        if (it != group.begin()) {
            --it;
            left_indices.push_back(i);
            right_indices.push_back(*it);
        } else {
            // No timestamp <= ts
            left_indices.push_back(i);
            right_indices.push_back(static_cast<size_t>(-1));
        }
    }

    // Build result table
    Table result("aj_result");

    // Add left columns
    for (const auto& col_name : left.column_names()) {
        const Column& src = left.column(col_name);
        size_t elem_size = dtype_size(src.dtype());
        std::vector<uint8_t> data(left_indices.size() * elem_size);

        const uint8_t* src_data = static_cast<const uint8_t*>(src.data());
        for (size_t i = 0; i < left_indices.size(); ++i) {
            std::memcpy(data.data() + i * elem_size,
                       src_data + left_indices[i] * elem_size,
                       elem_size);
        }

        result.add_column(Column(col_name, src.dtype(), std::move(data)));
    }

    // Add right columns (excluding join keys and as_of)
    for (const auto& col_name : right.column_names()) {
        // Skip if already in left or is a join key
        if (result.has_column(col_name)) continue;
        if (std::find(on.begin(), on.end(), col_name) != on.end()) continue;

        const Column& src = right.column(col_name);
        size_t elem_size = dtype_size(src.dtype());
        std::vector<uint8_t> data(right_indices.size() * elem_size, 0);

        const uint8_t* src_data = static_cast<const uint8_t*>(src.data());
        for (size_t i = 0; i < right_indices.size(); ++i) {
            if (right_indices[i] != static_cast<size_t>(-1)) {
                std::memcpy(data.data() + i * elem_size,
                           src_data + right_indices[i] * elem_size,
                           elem_size);
            }
            // else: leave as zero (null representation)
        }

        result.add_column(Column(col_name, src.dtype(), std::move(data)));
    }

    result.set_sorted_by(as_of);
    return result;
}

Table wj(const Table& left, const Table& right,
         const std::vector<std::string>& on,
         const std::string& as_of,
         int64_t window_before,
         int64_t window_after) {

    // Validate inputs
    if (!left.is_sorted() || left.sorted_by() != as_of) {
        throw InvalidOperation("Left table must be sorted by " + as_of);
    }
    if (!right.is_sorted() || right.sorted_by() != as_of) {
        throw InvalidOperation("Right table must be sorted by " + as_of);
    }

    // Group right table by join keys
    auto right_groups = group_by_key(right, on);

    // Get timestamp columns
    auto left_ts = const_cast<Table&>(left).column(as_of).as_int64();
    auto right_ts = const_cast<Table&>(right).column(as_of).as_int64();

    // Result builders
    std::vector<size_t> left_indices;
    std::vector<size_t> right_indices;

    // For each left row, find all right rows in window
    for (size_t i = 0; i < left.num_rows(); ++i) {
        auto key = extract_key(left, on, i);
        int64_t ts = left_ts[i];
        int64_t ts_min = ts - window_before;
        int64_t ts_max = ts + window_after;

        auto group_it = right_groups.find(key);
        if (group_it == right_groups.end()) {
            continue;  // No matching key
        }

        const auto& group = group_it->second;

        // Find range [ts_min, ts_max]
        auto lower = std::lower_bound(group.begin(), group.end(), ts_min,
            [&right_ts](size_t idx, int64_t t) { return right_ts[idx] < t; });
        auto upper = std::upper_bound(group.begin(), group.end(), ts_max,
            [&right_ts](int64_t t, size_t idx) { return t < right_ts[idx]; });

        for (auto it = lower; it != upper; ++it) {
            left_indices.push_back(i);
            right_indices.push_back(*it);
        }
    }

    // Build result table (similar to aj)
    Table result("wj_result");

    // Add left columns
    for (const auto& col_name : left.column_names()) {
        const Column& src = left.column(col_name);
        size_t elem_size = dtype_size(src.dtype());
        std::vector<uint8_t> data(left_indices.size() * elem_size);

        const uint8_t* src_data = static_cast<const uint8_t*>(src.data());
        for (size_t i = 0; i < left_indices.size(); ++i) {
            std::memcpy(data.data() + i * elem_size,
                       src_data + left_indices[i] * elem_size,
                       elem_size);
        }

        result.add_column(Column(col_name, src.dtype(), std::move(data)));
    }

    // Add right columns (excluding join keys)
    for (const auto& col_name : right.column_names()) {
        if (result.has_column(col_name)) continue;
        if (std::find(on.begin(), on.end(), col_name) != on.end()) continue;

        const Column& src = right.column(col_name);
        size_t elem_size = dtype_size(src.dtype());
        std::vector<uint8_t> data(right_indices.size() * elem_size);

        const uint8_t* src_data = static_cast<const uint8_t*>(src.data());
        for (size_t i = 0; i < right_indices.size(); ++i) {
            std::memcpy(data.data() + i * elem_size,
                       src_data + right_indices[i] * elem_size,
                       elem_size);
        }

        result.add_column(Column(col_name, src.dtype(), std::move(data)));
    }

    if (!result.column_names().empty()) {
        result.set_sorted_by(as_of);
    }
    return result;
}

Table inner_join(const Table& left, const Table& right,
                 const std::vector<std::string>& on) {
    // TODO: Implement inner join
    throw InvalidOperation("inner_join not yet implemented");
}

Table left_join(const Table& left, const Table& right,
                const std::vector<std::string>& on) {
    // TODO: Implement left join
    throw InvalidOperation("left_join not yet implemented");
}

}  // namespace wayy_db::ops
