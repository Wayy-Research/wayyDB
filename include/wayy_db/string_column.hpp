#pragma once

#include "wayy_db/types.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace wayy_db {

/// Arrow-style variable-length string column.
/// Storage layout:
///   offsets_: int64_t[N+1] — byte offsets into data_
///   data_:    uint8_t[]    — concatenated UTF-8 bytes
///   validity_: uint8_t[]   — 1 bit per row (bit=1 valid, bit=0 null)
///
/// String at row i = data_[offsets_[i] .. offsets_[i+1]]
class StringColumn {
public:
    /// Construct an empty string column
    explicit StringColumn(std::string name = "");

    /// Move-only semantics
    StringColumn(StringColumn&&) = default;
    StringColumn& operator=(StringColumn&&) = default;
    StringColumn(const StringColumn&) = delete;
    StringColumn& operator=(const StringColumn&) = delete;

    /// Column metadata
    const std::string& name() const { return name_; }
    DType dtype() const { return DType::String; }
    size_t size() const { return offsets_.empty() ? 0 : offsets_.size() - 1; }
    size_t data_bytes() const { return data_.size(); }

    /// Read a string at the given row
    std::string_view get(size_t row) const;

    /// Append a new string
    void append(std::string_view val);

    /// Append a null value
    void append_null();

    /// Overwrite the string at a given row.
    /// If the new string fits in the existing slot, it's written in-place.
    /// Otherwise, old slot is wasted and the new value is appended to data_.
    void set(size_t row, std::string_view val);

    /// Validity bitmap
    bool has_validity() const { return has_validity_; }
    bool is_valid(size_t row) const;
    void set_valid(size_t row, bool valid);
    size_t count_valid() const;

    /// Persistence
    void save(const std::string& dir_path, const std::string& col_name) const;
    static StringColumn load(const std::string& dir_path, const std::string& col_name);

    /// Direct access for bulk operations
    const std::vector<int64_t>& offsets() const { return offsets_; }
    const std::vector<uint8_t>& data_buf() const { return data_; }
    const std::vector<uint8_t>& validity_bitmap() const { return validity_; }

    /// Collect all strings as a vector (copy)
    std::vector<std::string> to_vector() const;

private:
    std::string name_;
    std::vector<int64_t> offsets_;   // N+1 offsets
    std::vector<uint8_t> data_;      // Concatenated UTF-8 bytes
    std::vector<uint8_t> validity_;  // Null bitmap
    bool has_validity_ = false;

    void ensure_validity();
};

}  // namespace wayy_db
