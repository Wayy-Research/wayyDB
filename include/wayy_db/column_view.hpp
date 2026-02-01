#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <iterator>

namespace wayy_db {

/// Non-owning typed view over contiguous column data
/// Provides zero-copy access for SIMD operations and Python bindings
template<typename T>
class ColumnView {
public:
    using value_type = T;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using iterator = T*;
    using const_iterator = const T*;

    /// Construct an empty view
    ColumnView() : data_(nullptr), size_(0) {}

    /// Construct a view over existing data
    ColumnView(T* data, size_t size) : data_(data), size_(size) {}

    /// Construct from std::span
    explicit ColumnView(std::span<T> span) : data_(span.data()), size_(span.size()) {}

    // Element access
    reference operator[](size_t i) { return data_[i]; }
    const_reference operator[](size_t i) const { return data_[i]; }

    reference at(size_t i) {
        if (i >= size_) throw std::out_of_range("ColumnView index out of range");
        return data_[i];
    }
    const_reference at(size_t i) const {
        if (i >= size_) throw std::out_of_range("ColumnView index out of range");
        return data_[i];
    }

    reference front() { return data_[0]; }
    const_reference front() const { return data_[0]; }

    reference back() { return data_[size_ - 1]; }
    const_reference back() const { return data_[size_ - 1]; }

    // Iterators
    iterator begin() { return data_; }
    iterator end() { return data_ + size_; }
    const_iterator begin() const { return data_; }
    const_iterator end() const { return data_ + size_; }
    const_iterator cbegin() const { return data_; }
    const_iterator cend() const { return data_ + size_; }

    // Capacity
    bool empty() const { return size_ == 0; }
    size_t size() const { return size_; }

    // Data access (for Python buffer protocol and SIMD)
    T* data() { return data_; }
    const T* data() const { return data_; }

    /// Get as std::span for modern C++ APIs
    std::span<T> span() { return {data_, size_}; }
    std::span<const T> span() const { return {data_, size_}; }

    /// Create a subview
    ColumnView subview(size_t offset, size_t count) const {
        if (offset + count > size_) {
            throw std::out_of_range("ColumnView subview out of range");
        }
        return ColumnView(const_cast<T*>(data_) + offset, count);
    }

private:
    T* data_;
    size_t size_;
};

// Common type aliases
using Int64View = ColumnView<int64_t>;
using Float64View = ColumnView<double>;
using TimestampView = ColumnView<int64_t>;
using SymbolView = ColumnView<uint32_t>;
using BoolView = ColumnView<uint8_t>;

}  // namespace wayy_db
