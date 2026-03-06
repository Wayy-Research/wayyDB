#pragma once

#include "wayy_db/types.hpp"
#include "wayy_db/column_view.hpp"

#include <bit>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace wayy_db {

/// Type-erased column that owns its data or references mmap'd memory
class Column {
public:
    /// Construct an empty column
    Column() = default;

    /// Construct a column with owned data
    Column(std::string name, DType dtype, std::vector<uint8_t> data);

    /// Construct a column referencing external memory (e.g., mmap)
    Column(std::string name, DType dtype, void* data, size_t size, bool owns_data = false);

    /// Move-only semantics
    Column(Column&&) = default;
    Column& operator=(Column&&) = default;
    Column(const Column&) = delete;
    Column& operator=(const Column&) = delete;

    /// Column metadata
    const std::string& name() const { return name_; }
    DType dtype() const { return dtype_; }
    size_t size() const { return size_; }
    size_t byte_size() const { return size_ * dtype_size(dtype_); }

    /// Raw data access
    void* data() { return data_; }
    const void* data() const { return data_; }

    /// Typed view access (throws TypeMismatch if wrong type)
    template<typename T>
    ColumnView<T> as();

    template<typename T>
    ColumnView<const T> as() const;

    /// Convenience accessors
    Int64View as_int64() { return as<int64_t>(); }
    Float64View as_float64() { return as<double>(); }
    TimestampView as_timestamp() { return as<int64_t>(); }
    SymbolView as_symbol() { return as<uint32_t>(); }
    BoolView as_bool() { return as<uint8_t>(); }

    /// Decimal6 accessor (underlying int64, but tagged as Decimal6)
    Int64View as_decimal6() {
        if (dtype_ != DType::Decimal6) throw TypeMismatch(DType::Decimal6, dtype_);
        return ColumnView<int64_t>(static_cast<int64_t*>(data_), size_);
    }

    /// Validity bitmap (null/deleted tracking)
    bool has_validity() const { return has_validity_; }
    void ensure_validity();                     // Allocate bitmap, mark all valid
    bool is_valid(size_t row) const;
    void set_valid(size_t row, bool valid);
    size_t count_valid() const;                 // popcount over bitmap

    /// Direct access to validity bitmap bytes (for persistence)
    const std::vector<uint8_t>& validity_bitmap() const { return validity_; }
    void set_validity_bitmap(std::vector<uint8_t> bitmap);

    /// Append a single element (column must own its data)
    void append(const void* value, size_t value_size);

    /// Overwrite element at row index (column must own its data)
    void set(size_t row, const void* value, size_t value_size);

private:
    std::string name_;
    DType dtype_ = DType::Int64;
    void* data_ = nullptr;
    size_t size_ = 0;
    bool owns_data_ = false;
    std::vector<uint8_t> owned_data_;  // Storage when we own the data

    // Validity bitmap: 1 bit per row (bit=1 means valid, bit=0 means null/deleted)
    std::vector<uint8_t> validity_;
    bool has_validity_ = false;

    /// Check that the requested type matches the column's dtype
    template<typename T>
    void check_type() const;
};

// Template implementations

template<typename T>
ColumnView<T> Column::as() {
    check_type<T>();
    return ColumnView<T>(static_cast<T*>(data_), size_);
}

template<typename T>
ColumnView<const T> Column::as() const {
    check_type<T>();
    return ColumnView<const T>(static_cast<const T*>(data_), size_);
}

template<typename T>
void Column::check_type() const {
    using U = std::remove_cv_t<T>;
    DType expected;
    if constexpr (std::is_same_v<U, int64_t>) {
        // Could be Int64, Timestamp, or Decimal6 (all stored as int64_t)
        if (dtype_ != DType::Int64 && dtype_ != DType::Timestamp && dtype_ != DType::Decimal6) {
            throw TypeMismatch(DType::Int64, dtype_);
        }
        return;
    } else if constexpr (std::is_same_v<U, double>) {
        expected = DType::Float64;
    } else if constexpr (std::is_same_v<U, uint32_t>) {
        expected = DType::Symbol;
    } else if constexpr (std::is_same_v<U, uint8_t>) {
        expected = DType::Bool;
    } else {
        static_assert(sizeof(U) == 0, "Unsupported column type");
    }

    if (dtype_ != expected) {
        throw TypeMismatch(expected, dtype_);
    }
}

}  // namespace wayy_db
