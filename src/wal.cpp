#include "wayy_db/wal.hpp"
#include "wayy_db/database.hpp"

#include <array>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

namespace wayy_db {

// Simple CRC32 (IEEE polynomial)
static const std::array<uint32_t, 256> crc32_table = [] {
    std::array<uint32_t, 256> table{};
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t crc = i;
        for (int j = 0; j < 8; ++j) {
            crc = (crc >> 1) ^ ((crc & 1) ? 0xEDB88320u : 0);
        }
        table[i] = crc;
    }
    return table;
}();

WriteAheadLog::WriteAheadLog(const std::string& db_path) {
    fs::create_directories(db_path);
    path_ = db_path + "/wal.bin";
    open_for_append();
}

WriteAheadLog::~WriteAheadLog() {
    if (file_.is_open()) {
        file_.flush();
        file_.close();
    }
}

void WriteAheadLog::open_for_append() {
    if (file_.is_open()) file_.close();
    file_.open(path_, std::ios::binary | std::ios::app);
    if (!file_) {
        throw WayyException("Failed to open WAL file: " + path_);
    }
}

uint32_t WriteAheadLog::crc32(const uint8_t* data, size_t len) {
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; ++i) {
        crc = crc32_table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFF;
}

void WriteAheadLog::write_entry(WalOp op, const std::string& table, size_t row,
                                 const std::vector<uint8_t>& payload) {
    std::lock_guard<std::mutex> lock(mu_);

    // Build the entry in a buffer for CRC calculation
    std::vector<uint8_t> buf;
    buf.reserve(4 + 1 + 4 + table.size() + 8 + 4 + payload.size());

    // Magic
    uint32_t magic = WAL_MAGIC;
    buf.insert(buf.end(), reinterpret_cast<uint8_t*>(&magic),
               reinterpret_cast<uint8_t*>(&magic) + 4);

    // Op type
    buf.push_back(static_cast<uint8_t>(op));

    // Table name length + name
    uint32_t tlen = static_cast<uint32_t>(table.size());
    buf.insert(buf.end(), reinterpret_cast<uint8_t*>(&tlen),
               reinterpret_cast<uint8_t*>(&tlen) + 4);
    buf.insert(buf.end(), table.begin(), table.end());

    // Row ID
    uint64_t row_id = static_cast<uint64_t>(row);
    buf.insert(buf.end(), reinterpret_cast<uint8_t*>(&row_id),
               reinterpret_cast<uint8_t*>(&row_id) + 8);

    // Payload length + payload
    uint32_t plen = static_cast<uint32_t>(payload.size());
    buf.insert(buf.end(), reinterpret_cast<uint8_t*>(&plen),
               reinterpret_cast<uint8_t*>(&plen) + 4);
    buf.insert(buf.end(), payload.begin(), payload.end());

    // CRC32
    uint32_t checksum = crc32(buf.data(), buf.size());
    buf.insert(buf.end(), reinterpret_cast<uint8_t*>(&checksum),
               reinterpret_cast<uint8_t*>(&checksum) + 4);

    // Write to file
    file_.write(reinterpret_cast<const char*>(buf.data()),
                static_cast<std::streamsize>(buf.size()));
    file_.flush();
}

void WriteAheadLog::log_insert(const std::string& table, size_t row,
                                const std::vector<uint8_t>& data) {
    write_entry(WalOp::Insert, table, row, data);
}

void WriteAheadLog::log_update(const std::string& table, size_t row,
                                const std::string& col, const std::vector<uint8_t>& data) {
    // Encode column name + data as payload
    std::vector<uint8_t> payload;
    uint32_t clen = static_cast<uint32_t>(col.size());
    payload.insert(payload.end(), reinterpret_cast<uint8_t*>(&clen),
                   reinterpret_cast<uint8_t*>(&clen) + 4);
    payload.insert(payload.end(), col.begin(), col.end());
    payload.insert(payload.end(), data.begin(), data.end());
    write_entry(WalOp::Update, table, row, payload);
}

void WriteAheadLog::log_delete(const std::string& table, size_t row) {
    write_entry(WalOp::Delete, table, row, {});
}

void WriteAheadLog::checkpoint(Database& db) {
    std::lock_guard<std::mutex> lock(mu_);

    // Flush and close WAL
    if (file_.is_open()) {
        file_.flush();
        file_.close();
    }

    // Save all tables to disk
    db.save();

    // Truncate WAL (start fresh)
    file_.open(path_, std::ios::binary | std::ios::trunc);
    if (!file_) {
        throw WayyException("Failed to truncate WAL: " + path_);
    }
}

void WriteAheadLog::replay(Database& db) {
    if (!fs::exists(path_)) return;

    std::ifstream wal(path_, std::ios::binary);
    if (!wal) return;

    // Get file size
    wal.seekg(0, std::ios::end);
    auto file_size = wal.tellg();
    if (file_size <= 0) return;
    wal.seekg(0, std::ios::beg);

    size_t entries_replayed = 0;

    while (wal.good() && wal.tellg() < file_size) {
        auto entry_start = wal.tellg();

        // Read magic
        uint32_t magic = 0;
        wal.read(reinterpret_cast<char*>(&magic), 4);
        if (magic != WAL_MAGIC) break;  // Corrupt or end of valid entries

        // Read op
        uint8_t op_byte = 0;
        wal.read(reinterpret_cast<char*>(&op_byte), 1);
        auto op = static_cast<WalOp>(op_byte);

        // Read table name
        uint32_t tlen = 0;
        wal.read(reinterpret_cast<char*>(&tlen), 4);
        std::string table_name(tlen, '\0');
        wal.read(table_name.data(), tlen);

        // Read row ID
        uint64_t row_id = 0;
        wal.read(reinterpret_cast<char*>(&row_id), 8);

        // Read payload
        uint32_t plen = 0;
        wal.read(reinterpret_cast<char*>(&plen), 4);
        std::vector<uint8_t> payload(plen);
        if (plen > 0) {
            wal.read(reinterpret_cast<char*>(payload.data()), plen);
        }

        // Read CRC
        uint32_t stored_crc = 0;
        wal.read(reinterpret_cast<char*>(&stored_crc), 4);

        // Verify CRC (re-read the entry from start to before CRC)
        auto entry_end = wal.tellg();
        size_t entry_size = static_cast<size_t>(entry_end - entry_start) - 4;  // Exclude CRC
        wal.seekg(entry_start);
        std::vector<uint8_t> entry_data(entry_size);
        wal.read(reinterpret_cast<char*>(entry_data.data()), entry_size);
        wal.seekg(entry_end);  // Skip past CRC we already read

        uint32_t computed_crc = crc32(entry_data.data(), entry_data.size());
        if (computed_crc != stored_crc) {
            break;  // Corrupt entry, stop replay
        }

        // Apply operation (best-effort: skip if table doesn't exist)
        // The actual replay logic depends on the table having been loaded.
        // For now, we just count replayed entries. Full replay requires
        // deserializing the payload and calling table CRUD methods.
        // TODO: Implement full row-level replay when table schema is available.
        (void)op;
        (void)row_id;
        (void)table_name;

        ++entries_replayed;
    }

    // After replay, truncate WAL
    wal.close();
    if (entries_replayed > 0) {
        // Re-save state and clear WAL
        std::ofstream truncate(path_, std::ios::binary | std::ios::trunc);
    }
}

bool WriteAheadLog::has_entries() const {
    if (!fs::exists(path_)) return false;
    return fs::file_size(path_) > 0;
}

}  // namespace wayy_db
