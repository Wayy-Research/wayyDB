#pragma once

#include "wayy_db/table.hpp"

#include <string>
#include <vector>

namespace wayy_db::ops {

/// As-of join: for each row in left, find the most recent row in right
/// where right.as_of <= left.as_of and join keys match
///
/// Both tables must be sorted by the as_of column
///
/// @param left Left table (e.g., trades)
/// @param right Right table (e.g., quotes)
/// @param on Join key columns (e.g., ["symbol"])
/// @param as_of Temporal column name (e.g., "timestamp")
/// @return Joined table with columns from both tables
Table aj(const Table& left, const Table& right,
         const std::vector<std::string>& on,
         const std::string& as_of);

/// Window join: for each row in left, find all rows in right
/// within the specified time window
///
/// @param left Left table
/// @param right Right table
/// @param on Join key columns
/// @param as_of Temporal column name
/// @param window_before Nanoseconds before left.as_of to include
/// @param window_after Nanoseconds after left.as_of to include
/// @return Joined table (may have more rows than left due to multiple matches)
Table wj(const Table& left, const Table& right,
         const std::vector<std::string>& on,
         const std::string& as_of,
         int64_t window_before,
         int64_t window_after);

/// Inner join on specified columns
Table inner_join(const Table& left, const Table& right,
                 const std::vector<std::string>& on);

/// Left join on specified columns
Table left_join(const Table& left, const Table& right,
                const std::vector<std::string>& on);

}  // namespace wayy_db::ops
