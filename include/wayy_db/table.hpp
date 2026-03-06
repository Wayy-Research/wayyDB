#pragma once

#include "wayy_db/types.hpp"
#include "wayy_db/column.hpp"
#include "wayy_db/string_column.hpp"
#include "wayy_db/mmap_file.hpp"

#include <any>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace wayy_db {

// Forward declarations
class HashIndex;

/// Columnar table with optional sorted index, OLTP capabilities,
/// and per-table reader-writer locking.
class Table {
public:
    /// Construct an empty table
    explicit Table(std::string name = "");

    /// Move-only semantics (shared_mutex is non-movable, so custom move ctor)
    Table(Table&& other) noexcept;
    Table& operator=(Table&& other) noexcept;
    Table(const Table&) = delete;
    Table& operator=(const Table&) = delete;
    ~Table();

    /// Table metadata
    const std::string& name() const { return name_; }
    size_t num_rows() const { return num_rows_; }
    size_t num_columns() const { return columns_.size() + string_columns_.size(); }

    /// Per-table reader-writer lock
    auto read_lock() const { return std::shared_lock(mu_); }
    auto write_lock() { return std::unique_lock(mu_); }

    /// Column management (fixed-width columns)
    void add_column(Column column);
    void add_column(const std::string& name, DType dtype, void* data, size_t size);

    /// String column management
    void add_string_column(StringColumn col);
    bool has_string_column(const std::string& name) const;
    StringColumn& string_column(const std::string& name);
    const StringColumn& string_column(const std::string& name) const;

    bool has_column(const std::string& name) const;
    Column& column(const std::string& name);
    const Column& column(const std::string& name) const;
    Column& operator[](const std::string& name) { return column(name); }
    const Column& operator[](const std::string& name) const { return column(name); }

    /// Get the DType of any column (fixed or string)
    DType column_dtype(const std::string& name) const;

    std::vector<std::string> column_names() const;

    /// Sorted index (critical for temporal joins)
    void set_sorted_by(const std::string& col);
    std::optional<std::string> sorted_by() const { return sorted_by_; }
    bool is_sorted() const { return sorted_by_.has_value(); }

    /// Primary key + hash index
    void set_primary_key(const std::string& col_name);
    const std::optional<std::string>& primary_key() const { return primary_key_; }
    std::optional<size_t> find_row(int64_t key) const;
    std::optional<size_t> find_row(std::string_view key) const;
    void rebuild_index();

    /// CRUD operations
    size_t append_row(const std::unordered_map<std::string, std::any>& values);
    bool update_row(int64_t pk, const std::unordered_map<std::string, std::any>& values);
    bool update_row(std::string_view pk, const std::unordered_map<std::string, std::any>& values);
    bool delete_row(int64_t pk);
    bool delete_row(std::string_view pk);

    /// Filter: returns vector of row indices matching predicate
    std::vector<size_t> where_eq(const std::string& col, int64_t val) const;
    std::vector<size_t> where_eq(const std::string& col, std::string_view val) const;

    /// Compaction: physically remove deleted rows, rebuild index
    void compact();

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

    // String columns (separate storage)
    std::vector<StringColumn> string_columns_;
    std::unordered_map<std::string, size_t> string_column_index_;

    // Primary key + hash index
    std::optional<std::string> primary_key_;
    std::unique_ptr<HashIndex> pk_index_;

    // Per-table reader-writer lock
    mutable std::shared_mutex mu_;

    // For mmap'd tables, keep file handles alive
    std::vector<MmapFile> mmap_files_;

    /// Write metadata JSON
    void write_metadata(const std::string& dir_path) const;

    /// Read metadata JSON and return column info
    static std::tuple<std::string, size_t, std::optional<std::string>,
                      std::optional<std::string>,
                      std::vector<std::pair<std::string, DType>>>
    read_metadata(const std::string& dir_path);

    /// Internal row update by row index (no PK lookup)
    bool update_row_at(size_t row_idx, const std::unordered_map<std::string, std::any>& values);
};

}  // namespace wayy_db
