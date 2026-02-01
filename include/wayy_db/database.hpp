#pragma once

#include "wayy_db/table.hpp"

#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace wayy_db {

/// High-level database interface managing multiple tables
class Database {
public:
    /// Create an in-memory database
    Database();

    /// Create or open a persistent database at the given path
    explicit Database(const std::string& path);

    /// Move-only semantics
    Database(Database&&) = default;
    Database& operator=(Database&&) = default;
    Database(const Database&) = delete;
    Database& operator=(const Database&) = delete;

    ~Database() = default;

    /// Database path (empty for in-memory)
    const std::string& path() const { return path_; }

    /// Check if database is persistent
    bool is_persistent() const { return !path_.empty(); }

    /// List all table names
    std::vector<std::string> tables() const;

    /// Check if a table exists
    bool has_table(const std::string& name) const;

    /// Get a table by name (loads from disk if persistent and not cached)
    Table& table(const std::string& name);
    Table& operator[](const std::string& name) { return table(name); }

    /// Create a new table
    Table& create_table(const std::string& name);

    /// Add an existing table to the database
    void add_table(Table table);

    /// Drop a table (removes from disk if persistent)
    void drop_table(const std::string& name);

    /// Save all modified tables to disk (no-op for in-memory)
    void save();

    /// Reload table list from disk
    void refresh();

private:
    std::string path_;
    std::unordered_map<std::string, Table> tables_;
    std::unordered_map<std::string, bool> loaded_;  // Track which tables are loaded

    // Mutex for thread-safe access (mutable allows const methods to lock)
    // Uses shared_mutex for concurrent reads, exclusive writes
    mutable std::shared_mutex mutex_;

    /// Get the directory path for a table
    std::string table_path(const std::string& name) const;

    /// Scan directory for existing tables
    void scan_tables();
};

}  // namespace wayy_db
