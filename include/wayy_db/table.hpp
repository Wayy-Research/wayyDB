#pragma once

#include "wayy_db/types.hpp"
#include "wayy_db/column.hpp"
#include "wayy_db/mmap_file.hpp"

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace wayy_db {

/// Columnar table with optional sorted index
class Table {
public:
    /// Construct an empty table
    explicit Table(std::string name = "");

    /// Move-only semantics
    Table(Table&&) = default;
    Table& operator=(Table&&) = default;
    Table(const Table&) = delete;
    Table& operator=(const Table&) = delete;

    /// Table metadata
    const std::string& name() const { return name_; }
    size_t num_rows() const { return num_rows_; }
    size_t num_columns() const { return columns_.size(); }

    /// Column management
    void add_column(Column column);
    void add_column(const std::string& name, DType dtype, void* data, size_t size);

    bool has_column(const std::string& name) const;
    Column& column(const std::string& name);
    const Column& column(const std::string& name) const;
    Column& operator[](const std::string& name) { return column(name); }
    const Column& operator[](const std::string& name) const { return column(name); }

    std::vector<std::string> column_names() const;

    /// Sorted index (critical for temporal joins)
    void set_sorted_by(const std::string& col);
    std::optional<std::string> sorted_by() const { return sorted_by_; }
    bool is_sorted() const { return sorted_by_.has_value(); }

    /// Persistence
    void save(const std::string& dir_path) const;
    static Table load(const std::string& dir_path);

    /// Create from memory-mapped directory (zero-copy)
    static Table mmap(const std::string& dir_path);

private:
    std::string name_;
    size_t num_rows_ = 0;
    std::vector<Column> columns_;
    std::unordered_map<std::string, size_t> column_index_;
    std::optional<std::string> sorted_by_;

    // For mmap'd tables, keep file handles alive
    std::vector<MmapFile> mmap_files_;

    /// Write metadata JSON
    void write_metadata(const std::string& dir_path) const;

    /// Read metadata JSON and return column info
    static std::tuple<std::string, size_t, std::optional<std::string>,
                      std::vector<std::pair<std::string, DType>>>
    read_metadata(const std::string& dir_path);
};

}  // namespace wayy_db
