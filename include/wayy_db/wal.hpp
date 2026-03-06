#pragma once

#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace wayy_db {

// Forward declaration
class Database;

/// WAL operation types
enum class WalOp : uint8_t {
    Insert = 1,
    Update = 2,
    Delete = 3,
};

/// WAL magic number
constexpr uint32_t WAL_MAGIC = 0x57414C01;  // "WAL\x01"

/// Binary WAL entry format:
///   [4B magic][1B op_type][4B table_name_len][table_name]
///   [8B row_id][4B payload_len][payload][4B CRC32]
///
/// For Insert: payload = serialized row (col_name:type:data pairs)
/// For Update: payload = serialized partial row (only changed columns)
/// For Delete: payload = empty

class WriteAheadLog {
public:
    /// Create or open a WAL at the given directory
    explicit WriteAheadLog(const std::string& db_path);

    ~WriteAheadLog();

    /// Log an insert operation
    void log_insert(const std::string& table, size_t row,
                    const std::vector<uint8_t>& data);

    /// Log an update operation
    void log_update(const std::string& table, size_t row,
                    const std::string& col, const std::vector<uint8_t>& data);

    /// Log a delete operation
    void log_delete(const std::string& table, size_t row);

    /// Checkpoint: flush WAL, save all tables, truncate WAL
    void checkpoint(Database& db);

    /// Replay WAL entries to recover state after crash
    void replay(Database& db);

    /// Check if WAL has unprocessed entries
    bool has_entries() const;

    /// Get WAL file path
    const std::string& path() const { return path_; }

private:
    std::string path_;
    std::ofstream file_;
    mutable std::mutex mu_;

    /// Write a raw entry to the WAL file
    void write_entry(WalOp op, const std::string& table, size_t row,
                     const std::vector<uint8_t>& payload);

    /// Compute CRC32 over buffer
    static uint32_t crc32(const uint8_t* data, size_t len);

    /// Open WAL file for appending
    void open_for_append();
};

}  // namespace wayy_db
