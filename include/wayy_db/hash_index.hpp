#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace wayy_db {

// Forward declarations
class Table;

/// Hash-based primary key index supporting both int64 and string keys.
class HashIndex {
public:
    HashIndex() = default;

    /// Build index from table column
    void build_int(const Table& table, const std::string& col_name);
    void build_str(const Table& table, const std::string& col_name);

    /// Lookup
    std::optional<size_t> find_int(int64_t key) const;
    std::optional<size_t> find_str(std::string_view key) const;

    /// Insert
    void insert_int(int64_t key, size_t row);
    void insert_str(std::string_view key, size_t row);

    /// Remove
    void remove_int(int64_t key);
    void remove_str(std::string_view key);

    /// Clear
    void clear();

    /// Size
    size_t size() const { return int_map_.size() + str_map_.size(); }

private:
    std::unordered_map<int64_t, size_t> int_map_;
    std::unordered_map<std::string, size_t> str_map_;
};

}  // namespace wayy_db
