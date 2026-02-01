#include "wayy_db/table.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace wayy_db {

Table::Table(std::string name) : name_(std::move(name)) {}

void Table::add_column(Column column) {
    if (columns_.empty()) {
        num_rows_ = column.size();
    } else if (column.size() != num_rows_) {
        throw InvalidOperation(
            "Column size mismatch: expected " + std::to_string(num_rows_) +
            ", got " + std::to_string(column.size()));
    }

    const std::string& col_name = column.name();
    if (column_index_.count(col_name)) {
        throw InvalidOperation("Column already exists: " + col_name);
    }

    column_index_[col_name] = columns_.size();
    columns_.push_back(std::move(column));
}

void Table::add_column(const std::string& name, DType dtype, void* data, size_t size) {
    add_column(Column(name, dtype, data, size, true));
}

bool Table::has_column(const std::string& name) const {
    return column_index_.count(name) > 0;
}

Column& Table::column(const std::string& name) {
    auto it = column_index_.find(name);
    if (it == column_index_.end()) {
        throw ColumnNotFound(name);
    }
    return columns_[it->second];
}

const Column& Table::column(const std::string& name) const {
    auto it = column_index_.find(name);
    if (it == column_index_.end()) {
        throw ColumnNotFound(name);
    }
    return columns_[it->second];
}

std::vector<std::string> Table::column_names() const {
    std::vector<std::string> names;
    names.reserve(columns_.size());
    for (const auto& col : columns_) {
        names.push_back(col.name());
    }
    return names;
}

void Table::set_sorted_by(const std::string& col) {
    if (!has_column(col)) {
        throw ColumnNotFound(col);
    }
    sorted_by_ = col;
}

void Table::save(const std::string& dir_path) const {
    fs::create_directories(dir_path);

    // Write metadata
    write_metadata(dir_path);

    // Write each column
    for (const auto& col : columns_) {
        std::string col_path = dir_path + "/" + col.name() + ".col";
        std::ofstream file(col_path, std::ios::binary);

        if (!file) {
            throw WayyException("Failed to create column file: " + col_path);
        }

        // Write header
        ColumnHeader header{};
        header.magic = WAYY_MAGIC;
        header.version = WAYY_VERSION;
        header.dtype = col.dtype();
        header.row_count = col.size();
        header.compression = 0;
        header.data_offset = sizeof(ColumnHeader);

        file.write(reinterpret_cast<const char*>(&header), sizeof(header));

        // Write data
        file.write(static_cast<const char*>(col.data()), col.byte_size());
    }
}

void Table::write_metadata(const std::string& dir_path) const {
    std::string meta_path = dir_path + "/_meta.json";
    std::ofstream file(meta_path);

    if (!file) {
        throw WayyException("Failed to create metadata file: " + meta_path);
    }

    // Simple JSON serialization (no external dependency)
    file << "{\n";
    file << "  \"version\": " << WAYY_VERSION << ",\n";
    file << "  \"name\": \"" << name_ << "\",\n";
    file << "  \"num_rows\": " << num_rows_ << ",\n";

    if (sorted_by_) {
        file << "  \"sorted_by\": \"" << *sorted_by_ << "\",\n";
    } else {
        file << "  \"sorted_by\": null,\n";
    }

    file << "  \"columns\": [\n";
    for (size_t i = 0; i < columns_.size(); ++i) {
        const auto& col = columns_[i];
        file << "    {\"name\": \"" << col.name()
             << "\", \"dtype\": \"" << dtype_to_string(col.dtype()) << "\"}";
        if (i + 1 < columns_.size()) file << ",";
        file << "\n";
    }
    file << "  ]\n";
    file << "}\n";
}

Table Table::load(const std::string& dir_path) {
    auto [name, num_rows, sorted_by, col_info] = read_metadata(dir_path);

    Table table(name);

    for (const auto& [col_name, dtype] : col_info) {
        std::string col_path = dir_path + "/" + col_name + ".col";
        std::ifstream file(col_path, std::ios::binary);

        if (!file) {
            throw WayyException("Failed to open column file: " + col_path);
        }

        // Read header
        ColumnHeader header;
        file.read(reinterpret_cast<char*>(&header), sizeof(header));

        if (header.magic != WAYY_MAGIC) {
            throw WayyException("Invalid column file magic: " + col_path);
        }

        // Read data
        size_t byte_size = header.row_count * dtype_size(header.dtype);
        std::vector<uint8_t> data(byte_size);
        file.read(reinterpret_cast<char*>(data.data()), byte_size);

        table.add_column(Column(col_name, header.dtype, std::move(data)));
    }

    if (sorted_by) {
        table.set_sorted_by(*sorted_by);
    }

    return table;
}

Table Table::mmap(const std::string& dir_path) {
    auto [name, num_rows, sorted_by, col_info] = read_metadata(dir_path);

    Table table(name);

    for (const auto& [col_name, dtype] : col_info) {
        std::string col_path = dir_path + "/" + col_name + ".col";

        MmapFile mmap_file(col_path, MmapFile::Mode::ReadOnly);

        // Validate header
        auto* header = static_cast<const ColumnHeader*>(mmap_file.data());
        if (header->magic != WAYY_MAGIC) {
            throw WayyException("Invalid column file magic: " + col_path);
        }

        // Create column pointing to mmap'd data
        void* data_ptr = static_cast<uint8_t*>(mmap_file.data()) + header->data_offset;
        table.add_column(Column(col_name, header->dtype, data_ptr, header->row_count, false));

        // Keep mmap file alive
        table.mmap_files_.push_back(std::move(mmap_file));
    }

    if (sorted_by) {
        table.set_sorted_by(*sorted_by);
    }

    return table;
}

std::tuple<std::string, size_t, std::optional<std::string>,
           std::vector<std::pair<std::string, DType>>>
Table::read_metadata(const std::string& dir_path) {
    std::string meta_path = dir_path + "/_meta.json";
    std::ifstream file(meta_path);

    if (!file) {
        throw WayyException("Failed to open metadata file: " + meta_path);
    }

    // Simple JSON parsing (minimal implementation)
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json = buffer.str();

    // Extract fields using simple string parsing
    // (In production, use a proper JSON library)
    auto extract_string = [&json](const std::string& key) -> std::string {
        std::string pattern = "\"" + key + "\": \"";
        auto pos = json.find(pattern);
        if (pos == std::string::npos) return "";
        pos += pattern.size();
        auto end = json.find("\"", pos);
        return json.substr(pos, end - pos);
    };

    auto extract_int = [&json](const std::string& key) -> size_t {
        std::string pattern = "\"" + key + "\": ";
        auto pos = json.find(pattern);
        if (pos == std::string::npos) return 0;
        pos += pattern.size();
        return std::stoull(json.substr(pos));
    };

    std::string name = extract_string("name");
    size_t num_rows = extract_int("num_rows");

    std::optional<std::string> sorted_by;
    std::string sorted_str = extract_string("sorted_by");
    if (!sorted_str.empty()) {
        sorted_by = sorted_str;
    }

    // Parse columns array
    std::vector<std::pair<std::string, DType>> columns;
    auto cols_start = json.find("\"columns\":");
    if (cols_start != std::string::npos) {
        auto arr_start = json.find("[", cols_start);
        auto arr_end = json.find("]", arr_start);
        std::string arr = json.substr(arr_start, arr_end - arr_start + 1);

        size_t pos = 0;
        while ((pos = arr.find("{", pos)) != std::string::npos) {
            auto obj_end = arr.find("}", pos);
            std::string obj = arr.substr(pos, obj_end - pos + 1);

            // Extract name and dtype from object
            auto name_pos = obj.find("\"name\": \"");
            if (name_pos != std::string::npos) {
                name_pos += 9;
                auto name_end = obj.find("\"", name_pos);
                std::string col_name = obj.substr(name_pos, name_end - name_pos);

                auto dtype_pos = obj.find("\"dtype\": \"");
                dtype_pos += 10;
                auto dtype_end = obj.find("\"", dtype_pos);
                std::string dtype_str = obj.substr(dtype_pos, dtype_end - dtype_pos);

                columns.emplace_back(col_name, dtype_from_string(dtype_str));
            }

            pos = obj_end + 1;
        }
    }

    return {name, num_rows, sorted_by, columns};
}

}  // namespace wayy_db
