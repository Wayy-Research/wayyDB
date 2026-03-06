#include "wayy_db/table.hpp"
#include "wayy_db/hash_index.hpp"

#include <algorithm>
#include <any>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace wayy_db {

Table::Table(std::string name) : name_(std::move(name)) {}

Table::~Table() = default;

Table::Table(Table&& other) noexcept
    : name_(std::move(other.name_)),
      num_rows_(other.num_rows_),
      columns_(std::move(other.columns_)),
      column_index_(std::move(other.column_index_)),
      sorted_by_(std::move(other.sorted_by_)),
      string_columns_(std::move(other.string_columns_)),
      string_column_index_(std::move(other.string_column_index_)),
      primary_key_(std::move(other.primary_key_)),
      pk_index_(std::move(other.pk_index_)),
      mmap_files_(std::move(other.mmap_files_)) {
    other.num_rows_ = 0;
}

Table& Table::operator=(Table&& other) noexcept {
    if (this != &other) {
        name_ = std::move(other.name_);
        num_rows_ = other.num_rows_;
        columns_ = std::move(other.columns_);
        column_index_ = std::move(other.column_index_);
        sorted_by_ = std::move(other.sorted_by_);
        string_columns_ = std::move(other.string_columns_);
        string_column_index_ = std::move(other.string_column_index_);
        primary_key_ = std::move(other.primary_key_);
        pk_index_ = std::move(other.pk_index_);
        mmap_files_ = std::move(other.mmap_files_);
        other.num_rows_ = 0;
    }
    return *this;
}

// --- Fixed-width column management ---

void Table::add_column(Column column) {
    if (columns_.empty() && string_columns_.empty()) {
        num_rows_ = column.size();
    } else if (column.size() != num_rows_) {
        throw InvalidOperation(
            "Column size mismatch: expected " + std::to_string(num_rows_) +
            ", got " + std::to_string(column.size()));
    }

    const std::string& col_name = column.name();
    if (column_index_.count(col_name) || string_column_index_.count(col_name)) {
        throw InvalidOperation("Column already exists: " + col_name);
    }

    column_index_[col_name] = columns_.size();
    columns_.push_back(std::move(column));
}

void Table::add_column(const std::string& name, DType dtype, void* data, size_t size) {
    add_column(Column(name, dtype, data, size, true));
}

// --- String column management ---

void Table::add_string_column(StringColumn col) {
    if (columns_.empty() && string_columns_.empty()) {
        num_rows_ = col.size();
    } else if (col.size() != num_rows_) {
        throw InvalidOperation(
            "StringColumn size mismatch: expected " + std::to_string(num_rows_) +
            ", got " + std::to_string(col.size()));
    }

    const std::string& col_name = col.name();
    if (column_index_.count(col_name) || string_column_index_.count(col_name)) {
        throw InvalidOperation("Column already exists: " + col_name);
    }

    string_column_index_[col_name] = string_columns_.size();
    string_columns_.push_back(std::move(col));
}

bool Table::has_string_column(const std::string& name) const {
    return string_column_index_.count(name) > 0;
}

StringColumn& Table::string_column(const std::string& name) {
    auto it = string_column_index_.find(name);
    if (it == string_column_index_.end()) {
        throw ColumnNotFound(name);
    }
    return string_columns_[it->second];
}

const StringColumn& Table::string_column(const std::string& name) const {
    auto it = string_column_index_.find(name);
    if (it == string_column_index_.end()) {
        throw ColumnNotFound(name);
    }
    return string_columns_[it->second];
}

// --- General column queries ---

bool Table::has_column(const std::string& name) const {
    return column_index_.count(name) > 0 || string_column_index_.count(name) > 0;
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

DType Table::column_dtype(const std::string& name) const {
    auto it = column_index_.find(name);
    if (it != column_index_.end()) {
        return columns_[it->second].dtype();
    }
    auto sit = string_column_index_.find(name);
    if (sit != string_column_index_.end()) {
        return DType::String;
    }
    throw ColumnNotFound(name);
}

std::vector<std::string> Table::column_names() const {
    std::vector<std::string> names;
    names.reserve(columns_.size() + string_columns_.size());
    for (const auto& col : columns_) {
        names.push_back(col.name());
    }
    for (const auto& col : string_columns_) {
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

// --- Primary key + hash index ---

void Table::set_primary_key(const std::string& col_name) {
    if (!has_column(col_name)) {
        throw ColumnNotFound(col_name);
    }
    primary_key_ = col_name;
    rebuild_index();
}

void Table::rebuild_index() {
    if (!primary_key_) return;

    pk_index_ = std::make_unique<HashIndex>();
    DType pk_dtype = column_dtype(*primary_key_);

    if (pk_dtype == DType::String) {
        pk_index_->build_str(*this, *primary_key_);
    } else if (pk_dtype == DType::Int64 || pk_dtype == DType::Timestamp || pk_dtype == DType::Decimal6) {
        pk_index_->build_int(*this, *primary_key_);
    } else {
        throw InvalidOperation("Primary key must be String, Int64, Timestamp, or Decimal6");
    }
}

std::optional<size_t> Table::find_row(int64_t key) const {
    if (!pk_index_) return std::nullopt;
    auto row = pk_index_->find_int(key);
    if (row && !columns_.empty() && columns_[0].has_validity()) {
        // Check validity of any fixed column
        if (!columns_[0].is_valid(*row)) return std::nullopt;
    }
    return row;
}

std::optional<size_t> Table::find_row(std::string_view key) const {
    if (!pk_index_) return std::nullopt;
    auto row = pk_index_->find_str(key);
    if (row) {
        // Check validity via the PK string column itself
        const auto& pk_col = string_column(*primary_key_);
        if (pk_col.has_validity() && !pk_col.is_valid(*row)) return std::nullopt;
    }
    return row;
}

// --- CRUD operations ---

size_t Table::append_row(const std::unordered_map<std::string, std::any>& values) {
    size_t row_idx = num_rows_;

    // Append to each fixed-width column
    for (auto& col : columns_) {
        auto it = values.find(col.name());
        if (it == values.end()) {
            // Append default (zero) value
            uint8_t zeros[8] = {};
            col.append(zeros, dtype_size(col.dtype()));
            col.ensure_validity();
            col.set_valid(row_idx, false);  // Mark as null
        } else {
            const auto& val = it->second;
            DType dt = col.dtype();

            if (dt == DType::Int64 || dt == DType::Timestamp || dt == DType::Decimal6) {
                int64_t v = std::any_cast<int64_t>(val);
                col.append(&v, sizeof(v));
            } else if (dt == DType::Float64) {
                double v = std::any_cast<double>(val);
                col.append(&v, sizeof(v));
            } else if (dt == DType::Symbol) {
                uint32_t v = std::any_cast<uint32_t>(val);
                col.append(&v, sizeof(v));
            } else if (dt == DType::Bool) {
                uint8_t v = std::any_cast<uint8_t>(val);
                col.append(&v, sizeof(v));
            }
        }
    }

    // Append to each string column
    for (auto& scol : string_columns_) {
        auto it = values.find(scol.name());
        if (it == values.end()) {
            scol.append_null();
        } else {
            auto sv = std::any_cast<std::string>(it->second);
            scol.append(sv);
        }
    }

    ++num_rows_;

    // Update index
    if (pk_index_ && primary_key_) {
        DType pk_dtype = column_dtype(*primary_key_);
        auto it = values.find(*primary_key_);
        if (it != values.end()) {
            if (pk_dtype == DType::String) {
                pk_index_->insert_str(std::any_cast<std::string>(it->second), row_idx);
            } else {
                pk_index_->insert_int(std::any_cast<int64_t>(it->second), row_idx);
            }
        }
    }

    return row_idx;
}

bool Table::update_row(int64_t pk, const std::unordered_map<std::string, std::any>& values) {
    auto row = find_row(pk);
    if (!row) return false;
    return update_row_at(*row, values);
}

bool Table::update_row(std::string_view pk, const std::unordered_map<std::string, std::any>& values) {
    auto row = find_row(pk);
    if (!row) return false;
    return update_row_at(*row, values);
}

bool Table::update_row_at(size_t row_idx, const std::unordered_map<std::string, std::any>& values) {
    if (row_idx >= num_rows_) return false;

    for (const auto& [col_name, val] : values) {
        // Check if it's a string column
        auto sit = string_column_index_.find(col_name);
        if (sit != string_column_index_.end()) {
            auto sv = std::any_cast<std::string>(val);
            string_columns_[sit->second].set(row_idx, sv);
            continue;
        }

        // Fixed-width column
        auto it = column_index_.find(col_name);
        if (it == column_index_.end()) continue;  // Skip unknown columns

        Column& col = columns_[it->second];
        DType dt = col.dtype();

        if (dt == DType::Int64 || dt == DType::Timestamp || dt == DType::Decimal6) {
            int64_t v = std::any_cast<int64_t>(val);
            col.set(row_idx, &v, sizeof(v));
        } else if (dt == DType::Float64) {
            double v = std::any_cast<double>(val);
            col.set(row_idx, &v, sizeof(v));
        } else if (dt == DType::Symbol) {
            uint32_t v = std::any_cast<uint32_t>(val);
            col.set(row_idx, &v, sizeof(v));
        } else if (dt == DType::Bool) {
            uint8_t v = std::any_cast<uint8_t>(val);
            col.set(row_idx, &v, sizeof(v));
        }
    }

    return true;
}

bool Table::delete_row(int64_t pk) {
    auto row = find_row(pk);
    if (!row) return false;

    // Soft delete: set validity bit to 0 on all columns
    for (auto& col : columns_) {
        col.ensure_validity();
        col.set_valid(*row, false);
    }
    for (auto& scol : string_columns_) {
        scol.set_valid(*row, false);
    }

    // Remove from index
    if (pk_index_) {
        pk_index_->remove_int(pk);
    }

    return true;
}

bool Table::delete_row(std::string_view pk) {
    auto row = find_row(pk);
    if (!row) return false;

    for (auto& col : columns_) {
        col.ensure_validity();
        col.set_valid(*row, false);
    }
    for (auto& scol : string_columns_) {
        scol.set_valid(*row, false);
    }

    if (pk_index_) {
        pk_index_->remove_str(pk);
    }

    return true;
}

// --- Filter ---

std::vector<size_t> Table::where_eq(const std::string& col_name, int64_t val) const {
    std::vector<size_t> result;
    auto it = column_index_.find(col_name);
    if (it == column_index_.end()) throw ColumnNotFound(col_name);

    const Column& col = columns_[it->second];
    auto view = col.as<const int64_t>();
    for (size_t i = 0; i < view.size(); ++i) {
        if (col.is_valid(i) && view[i] == val) {
            result.push_back(i);
        }
    }
    return result;
}

std::vector<size_t> Table::where_eq(const std::string& col_name, std::string_view val) const {
    std::vector<size_t> result;
    auto sit = string_column_index_.find(col_name);
    if (sit == string_column_index_.end()) throw ColumnNotFound(col_name);

    const StringColumn& scol = string_columns_[sit->second];
    for (size_t i = 0; i < scol.size(); ++i) {
        if (scol.is_valid(i) && scol.get(i) == val) {
            result.push_back(i);
        }
    }
    return result;
}

// --- Compaction ---

void Table::compact() {
    // Determine which rows are valid (check first available column)
    std::vector<bool> keep(num_rows_, true);
    bool any_deleted = false;

    // Check fixed columns for validity
    for (const auto& col : columns_) {
        if (col.has_validity()) {
            for (size_t i = 0; i < num_rows_; ++i) {
                if (!col.is_valid(i)) {
                    keep[i] = false;
                    any_deleted = true;
                }
            }
            break;  // Only need to check one column
        }
    }

    // Also check string columns
    if (!any_deleted) {
        for (const auto& scol : string_columns_) {
            if (scol.has_validity()) {
                for (size_t i = 0; i < scol.size(); ++i) {
                    if (!scol.is_valid(i)) {
                        keep[i] = false;
                        any_deleted = true;
                    }
                }
                break;
            }
        }
    }

    if (!any_deleted) return;  // Nothing to compact

    // Count new rows
    size_t new_rows = 0;
    for (bool k : keep) {
        if (k) ++new_rows;
    }

    // Compact fixed columns
    for (size_t ci = 0; ci < columns_.size(); ++ci) {
        Column& col = columns_[ci];
        size_t elem_size = dtype_size(col.dtype());
        std::vector<uint8_t> new_data;
        new_data.reserve(new_rows * elem_size);

        const uint8_t* src = static_cast<const uint8_t*>(col.data());
        for (size_t i = 0; i < num_rows_; ++i) {
            if (keep[i]) {
                new_data.insert(new_data.end(), src + i * elem_size, src + (i + 1) * elem_size);
            }
        }

        // Replace column
        std::string cname = col.name();
        DType cdtype = col.dtype();
        columns_[ci] = Column(std::move(cname), cdtype, std::move(new_data));
    }

    // Compact string columns
    for (size_t si = 0; si < string_columns_.size(); ++si) {
        StringColumn& scol = string_columns_[si];
        StringColumn new_scol(scol.name());
        for (size_t i = 0; i < scol.size(); ++i) {
            if (keep[i]) {
                if (scol.is_valid(i)) {
                    new_scol.append(scol.get(i));
                } else {
                    new_scol.append_null();
                }
            }
        }
        string_columns_[si] = std::move(new_scol);
    }

    num_rows_ = new_rows;

    // Rebuild index
    rebuild_index();
}

// --- Persistence ---

void Table::save(const std::string& dir_path) const {
    fs::create_directories(dir_path);

    // Write metadata
    write_metadata(dir_path);

    // Write each fixed-width column
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

        // Write validity bitmap if present
        if (col.has_validity()) {
            std::string vpath = dir_path + "/" + col.name() + ".validity";
            std::ofstream vf(vpath, std::ios::binary);
            if (vf) {
                const auto& bmap = col.validity_bitmap();
                uint64_t sz = bmap.size();
                vf.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
                vf.write(reinterpret_cast<const char*>(bmap.data()),
                         static_cast<std::streamsize>(sz));
            }
        }
    }

    // Write each string column
    for (const auto& scol : string_columns_) {
        scol.save(dir_path, scol.name());
    }
}

void Table::write_metadata(const std::string& dir_path) const {
    std::string meta_path = dir_path + "/_meta.json";
    std::ofstream file(meta_path);

    if (!file) {
        throw WayyException("Failed to create metadata file: " + meta_path);
    }

    file << "{\n";
    file << "  \"version\": " << WAYY_VERSION << ",\n";
    file << "  \"name\": \"" << name_ << "\",\n";
    file << "  \"num_rows\": " << num_rows_ << ",\n";

    if (sorted_by_) {
        file << "  \"sorted_by\": \"" << *sorted_by_ << "\",\n";
    } else {
        file << "  \"sorted_by\": null,\n";
    }

    if (primary_key_) {
        file << "  \"primary_key\": \"" << *primary_key_ << "\",\n";
    } else {
        file << "  \"primary_key\": null,\n";
    }

    file << "  \"columns\": [\n";
    size_t total_cols = columns_.size() + string_columns_.size();
    size_t idx = 0;
    for (const auto& col : columns_) {
        file << "    {\"name\": \"" << col.name()
             << "\", \"dtype\": \"" << dtype_to_string(col.dtype()) << "\"}";
        if (++idx < total_cols) file << ",";
        file << "\n";
    }
    for (const auto& scol : string_columns_) {
        file << "    {\"name\": \"" << scol.name()
             << "\", \"dtype\": \"string\"}";
        if (++idx < total_cols) file << ",";
        file << "\n";
    }
    file << "  ]\n";
    file << "}\n";
}

Table Table::load(const std::string& dir_path) {
    auto [name, num_rows, sorted_by, primary_key, col_info] = read_metadata(dir_path);

    Table table(name);

    for (const auto& [col_name, dtype] : col_info) {
        if (dtype == DType::String) {
            // Load string column
            table.add_string_column(StringColumn::load(dir_path, col_name));
        } else {
            // Load fixed-width column
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

            Column col(col_name, header.dtype, std::move(data));

            // Load validity bitmap if present
            std::string vpath = dir_path + "/" + col_name + ".validity";
            if (fs::exists(vpath)) {
                std::ifstream vf(vpath, std::ios::binary);
                if (vf) {
                    uint64_t sz = 0;
                    vf.read(reinterpret_cast<char*>(&sz), sizeof(sz));
                    std::vector<uint8_t> bitmap(sz);
                    vf.read(reinterpret_cast<char*>(bitmap.data()),
                            static_cast<std::streamsize>(sz));
                    col.set_validity_bitmap(std::move(bitmap));
                }
            }

            table.add_column(std::move(col));
        }
    }

    if (sorted_by) {
        table.set_sorted_by(*sorted_by);
    }

    if (primary_key) {
        table.set_primary_key(*primary_key);
    }

    return table;
}

Table Table::mmap(const std::string& dir_path) {
    auto [name, num_rows, sorted_by, primary_key, col_info] = read_metadata(dir_path);

    Table table(name);

    for (const auto& [col_name, dtype] : col_info) {
        if (dtype == DType::String) {
            // String columns are loaded (not mmap'd) since they have complex structure
            table.add_string_column(StringColumn::load(dir_path, col_name));
        } else {
            std::string col_path = dir_path + "/" + col_name + ".col";

            MmapFile mmap_file(col_path, MmapFile::Mode::ReadOnly);

            // Validate header
            auto* header = static_cast<const ColumnHeader*>(mmap_file.data());
            if (header->magic != WAYY_MAGIC) {
                throw WayyException("Invalid column file magic: " + col_path);
            }

            // Create column pointing to mmap'd data
            void* data_ptr = static_cast<uint8_t*>(mmap_file.data()) + header->data_offset;
            Column col(col_name, header->dtype, data_ptr, header->row_count, false);

            // Load validity bitmap (always into memory, small)
            std::string vpath = dir_path + "/" + col_name + ".validity";
            if (fs::exists(vpath)) {
                std::ifstream vf(vpath, std::ios::binary);
                if (vf) {
                    uint64_t sz = 0;
                    vf.read(reinterpret_cast<char*>(&sz), sizeof(sz));
                    std::vector<uint8_t> bitmap(sz);
                    vf.read(reinterpret_cast<char*>(bitmap.data()),
                            static_cast<std::streamsize>(sz));
                    col.set_validity_bitmap(std::move(bitmap));
                }
            }

            table.add_column(std::move(col));

            // Keep mmap file alive
            table.mmap_files_.push_back(std::move(mmap_file));
        }
    }

    if (sorted_by) {
        table.set_sorted_by(*sorted_by);
    }

    if (primary_key) {
        table.set_primary_key(*primary_key);
    }

    return table;
}

std::tuple<std::string, size_t, std::optional<std::string>,
           std::optional<std::string>,
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
    size_t num_rows_val = extract_int("num_rows");

    std::optional<std::string> sorted_by;
    std::string sorted_str = extract_string("sorted_by");
    if (!sorted_str.empty()) {
        sorted_by = sorted_str;
    }

    std::optional<std::string> primary_key;
    std::string pk_str = extract_string("primary_key");
    if (!pk_str.empty()) {
        primary_key = pk_str;
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

    return {name, num_rows_val, sorted_by, primary_key, columns};
}

}  // namespace wayy_db
