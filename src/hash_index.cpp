#include "wayy_db/hash_index.hpp"
#include "wayy_db/table.hpp"
#include "wayy_db/column.hpp"
#include "wayy_db/string_column.hpp"

namespace wayy_db {

void HashIndex::build_int(const Table& table, const std::string& col_name) {
    clear();
    const Column& col = table.column(col_name);
    auto view = col.as<const int64_t>();
    for (size_t i = 0; i < view.size(); ++i) {
        if (col.is_valid(i)) {
            int_map_[view[i]] = i;
        }
    }
}

void HashIndex::build_str(const Table& table, const std::string& col_name) {
    clear();
    const StringColumn& col = table.string_column(col_name);
    for (size_t i = 0; i < col.size(); ++i) {
        if (col.is_valid(i)) {
            str_map_[std::string(col.get(i))] = i;
        }
    }
}

std::optional<size_t> HashIndex::find_int(int64_t key) const {
    auto it = int_map_.find(key);
    if (it != int_map_.end()) return it->second;
    return std::nullopt;
}

std::optional<size_t> HashIndex::find_str(std::string_view key) const {
    auto it = str_map_.find(std::string(key));
    if (it != str_map_.end()) return it->second;
    return std::nullopt;
}

void HashIndex::insert_int(int64_t key, size_t row) {
    int_map_[key] = row;
}

void HashIndex::insert_str(std::string_view key, size_t row) {
    str_map_[std::string(key)] = row;
}

void HashIndex::remove_int(int64_t key) {
    int_map_.erase(key);
}

void HashIndex::remove_str(std::string_view key) {
    str_map_.erase(std::string(key));
}

void HashIndex::clear() {
    int_map_.clear();
    str_map_.clear();
}

}  // namespace wayy_db
