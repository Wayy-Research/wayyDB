#pragma once

#include "wayy_db/types.hpp"
#include "wayy_db/column_view.hpp"

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

private:
    std::string name_;
    DType dtype_ = DType::Int64;
    void* data_ = nullptr;
    size_t size_ = 0;
    bool owns_data_ = false;
    std::vector<uint8_t> owned_data_;  // Storage when we own the data

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
    DType expected;
    if constexpr (std::is_same_v<T, int64_t>) {
        // Could be Int64 or Timestamp
        if (dtype_ != DType::Int64 && dtype_ != DType::Timestamp) {
            throw TypeMismatch(DType::Int64, dtype_);
        }
        return;
    } else if constexpr (std::is_same_v<T, double>) {
        expected = DType::Float64;
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        expected = DType::Symbol;
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        expected = DType::Bool;
    } else {
        static_assert(sizeof(T) == 0, "Unsupported column type");
    }

    if (dtype_ != expected) {
        throw TypeMismatch(expected, dtype_);
    }
}

}  // namespace wayy_db
