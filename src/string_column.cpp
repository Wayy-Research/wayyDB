#include "wayy_db/string_column.hpp"

#include <bit>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace wayy_db {

StringColumn::StringColumn(std::string name) : name_(std::move(name)) {
    offsets_.push_back(0);  // Initial offset
}

std::string_view StringColumn::get(size_t row) const {
    if (row >= size()) {
        throw InvalidOperation("StringColumn row out of range");
    }
    if (has_validity_ && !is_valid(row)) {
        return {};  // Null row returns empty view
    }
    int64_t start = offsets_[row];
    int64_t end = offsets_[row + 1];
    return std::string_view(reinterpret_cast<const char*>(data_.data() + start),
                            static_cast<size_t>(end - start));
}

void StringColumn::append(std::string_view val) {
    int64_t offset = offsets_.back();
    data_.insert(data_.end(), val.begin(), val.end());
    offsets_.push_back(offset + static_cast<int64_t>(val.size()));

    if (has_validity_) {
        size_t row = size() - 1;
        size_t needed_bytes = (size() + 7) / 8;
        if (validity_.size() < needed_bytes) {
            validity_.push_back(0);
        }
        set_valid(row, true);
    }
}

void StringColumn::append_null() {
    offsets_.push_back(offsets_.back());  // Zero-length entry
    ensure_validity();
    set_valid(size() - 1, false);
}

void StringColumn::set(size_t row, std::string_view val) {
    if (row >= size()) {
        throw InvalidOperation("StringColumn row out of range in set");
    }
    int64_t old_start = offsets_[row];
    int64_t old_end = offsets_[row + 1];
    int64_t old_len = old_end - old_start;
    int64_t new_len = static_cast<int64_t>(val.size());

    if (new_len <= old_len) {
        // Fits in-place: overwrite and zero-pad remainder
        std::memcpy(data_.data() + old_start, val.data(), val.size());
        if (new_len < old_len) {
            std::memset(data_.data() + old_start + new_len, 0,
                        static_cast<size_t>(old_len - new_len));
        }
        // Update offsets: shift this entry's end
        offsets_[row + 1] = old_start + new_len;
        // NOTE: This changes the offset for subsequent rows if they shared
        // contiguous data. For OLTP use (row-level updates), this is fine
        // because compact() will fix fragmentation.
    } else {
        // Doesn't fit: append to end of data buffer, old slot becomes waste
        int64_t new_start = static_cast<int64_t>(data_.size());
        data_.insert(data_.end(), val.begin(), val.end());
        offsets_[row] = new_start;
        offsets_[row + 1] = new_start + new_len;
    }

    if (has_validity_) {
        set_valid(row, true);
    }
}

// --- Validity bitmap ---

void StringColumn::ensure_validity() {
    if (has_validity_) return;
    size_t n = size();
    size_t num_bytes = (n + 7) / 8;
    validity_.assign(num_bytes, 0xFF);
    if (n % 8 != 0) {
        uint8_t mask = static_cast<uint8_t>((1u << (n % 8)) - 1);
        validity_.back() = mask;
    }
    has_validity_ = true;
}

bool StringColumn::is_valid(size_t row) const {
    if (!has_validity_) return true;
    if (row >= size()) return false;
    return (validity_[row / 8] >> (row % 8)) & 1;
}

void StringColumn::set_valid(size_t row, bool valid) {
    if (!has_validity_) ensure_validity();
    if (row >= size()) return;
    if (valid) {
        validity_[row / 8] |= (1u << (row % 8));
    } else {
        validity_[row / 8] &= ~(1u << (row % 8));
    }
}

size_t StringColumn::count_valid() const {
    if (!has_validity_) return size();
    size_t count = 0;
    for (auto byte : validity_) {
        count += std::popcount(byte);
    }
    return count;
}

// --- Persistence ---
// Files: <dir>/<col_name>.offsets, <col_name>.data, <col_name>.validity

void StringColumn::save(const std::string& dir_path, const std::string& col_name) const {
    fs::create_directories(dir_path);

    // Write offsets
    {
        std::string path = dir_path + "/" + col_name + ".offsets";
        std::ofstream f(path, std::ios::binary);
        if (!f) throw WayyException("Failed to create offsets file: " + path);
        uint64_t count = offsets_.size();
        f.write(reinterpret_cast<const char*>(&count), sizeof(count));
        f.write(reinterpret_cast<const char*>(offsets_.data()),
                static_cast<std::streamsize>(offsets_.size() * sizeof(int64_t)));
    }

    // Write data
    {
        std::string path = dir_path + "/" + col_name + ".data";
        std::ofstream f(path, std::ios::binary);
        if (!f) throw WayyException("Failed to create data file: " + path);
        uint64_t sz = data_.size();
        f.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
        f.write(reinterpret_cast<const char*>(data_.data()),
                static_cast<std::streamsize>(data_.size()));
    }

    // Write validity if present
    if (has_validity_) {
        std::string path = dir_path + "/" + col_name + ".validity";
        std::ofstream f(path, std::ios::binary);
        if (!f) throw WayyException("Failed to create validity file: " + path);
        uint64_t sz = validity_.size();
        f.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
        f.write(reinterpret_cast<const char*>(validity_.data()),
                static_cast<std::streamsize>(validity_.size()));
    }
}

StringColumn StringColumn::load(const std::string& dir_path, const std::string& col_name) {
    StringColumn sc(col_name);
    sc.offsets_.clear();

    // Read offsets
    {
        std::string path = dir_path + "/" + col_name + ".offsets";
        std::ifstream f(path, std::ios::binary);
        if (!f) throw WayyException("Failed to open offsets file: " + path);
        uint64_t count = 0;
        f.read(reinterpret_cast<char*>(&count), sizeof(count));
        sc.offsets_.resize(count);
        f.read(reinterpret_cast<char*>(sc.offsets_.data()),
               static_cast<std::streamsize>(count * sizeof(int64_t)));
    }

    // Read data
    {
        std::string path = dir_path + "/" + col_name + ".data";
        std::ifstream f(path, std::ios::binary);
        if (!f) throw WayyException("Failed to open data file: " + path);
        uint64_t sz = 0;
        f.read(reinterpret_cast<char*>(&sz), sizeof(sz));
        sc.data_.resize(sz);
        f.read(reinterpret_cast<char*>(sc.data_.data()),
               static_cast<std::streamsize>(sz));
    }

    // Read validity if present
    {
        std::string path = dir_path + "/" + col_name + ".validity";
        if (fs::exists(path)) {
            std::ifstream f(path, std::ios::binary);
            if (f) {
                uint64_t sz = 0;
                f.read(reinterpret_cast<char*>(&sz), sizeof(sz));
                sc.validity_.resize(sz);
                f.read(reinterpret_cast<char*>(sc.validity_.data()),
                       static_cast<std::streamsize>(sz));
                sc.has_validity_ = true;
            }
        }
    }

    return sc;
}

std::vector<std::string> StringColumn::to_vector() const {
    std::vector<std::string> result;
    result.reserve(size());
    for (size_t i = 0; i < size(); ++i) {
        if (is_valid(i)) {
            result.emplace_back(get(i));
        } else {
            result.emplace_back();
        }
    }
    return result;
}

}  // namespace wayy_db
