#include "wayy_db/column.hpp"

#include <cstring>

namespace wayy_db {

Column::Column(std::string name, DType dtype, std::vector<uint8_t> data)
    : name_(std::move(name))
    , dtype_(dtype)
    , size_(dtype_size(dtype) > 0 ? data.size() / dtype_size(dtype) : 0)
    , owns_data_(true)
    , owned_data_(std::move(data)) {
    data_ = owned_data_.data();
}

Column::Column(std::string name, DType dtype, void* data, size_t size, bool owns_data)
    : name_(std::move(name))
    , dtype_(dtype)
    , data_(data)
    , size_(size)
    , owns_data_(owns_data) {
    if (owns_data && data != nullptr && dtype_size(dtype) > 0) {
        // Copy data into owned buffer
        size_t byte_size = size * dtype_size(dtype);
        owned_data_.resize(byte_size);
        std::memcpy(owned_data_.data(), data, byte_size);
        data_ = owned_data_.data();
    }
}

// --- Validity bitmap ---

void Column::ensure_validity() {
    if (has_validity_) return;
    size_t num_bytes = (size_ + 7) / 8;
    validity_.assign(num_bytes, 0xFF);  // All bits set = all valid
    // Handle trailing bits in last byte
    if (size_ % 8 != 0) {
        uint8_t mask = static_cast<uint8_t>((1u << (size_ % 8)) - 1);
        validity_.back() = mask;
    }
    has_validity_ = true;
}

bool Column::is_valid(size_t row) const {
    if (!has_validity_) return true;  // No bitmap = all valid
    if (row >= size_) return false;
    return (validity_[row / 8] >> (row % 8)) & 1;
}

void Column::set_valid(size_t row, bool valid) {
    if (!has_validity_) ensure_validity();
    if (row >= size_) return;
    if (valid) {
        validity_[row / 8] |= (1u << (row % 8));
    } else {
        validity_[row / 8] &= ~(1u << (row % 8));
    }
}

size_t Column::count_valid() const {
    if (!has_validity_) return size_;  // All valid
    size_t count = 0;
    for (size_t i = 0; i < validity_.size(); ++i) {
        count += std::popcount(validity_[i]);
    }
    return count;
}

void Column::set_validity_bitmap(std::vector<uint8_t> bitmap) {
    validity_ = std::move(bitmap);
    has_validity_ = !validity_.empty();
}

void Column::append(const void* value, size_t value_size) {
    if (!owns_data_) {
        throw InvalidOperation("Cannot append to non-owned column");
    }
    size_t elem_size = dtype_size(dtype_);
    if (elem_size == 0) {
        throw InvalidOperation("Cannot append to variable-length column via Column::append");
    }
    if (value_size != elem_size) {
        throw InvalidOperation("Value size mismatch in append");
    }

    size_t old_byte_size = owned_data_.size();
    owned_data_.resize(old_byte_size + elem_size);
    std::memcpy(owned_data_.data() + old_byte_size, value, elem_size);
    data_ = owned_data_.data();
    ++size_;

    // Extend validity bitmap if present
    if (has_validity_) {
        size_t needed_bytes = (size_ + 7) / 8;
        if (validity_.size() < needed_bytes) {
            validity_.push_back(0);
        }
        set_valid(size_ - 1, true);
    }
}

void Column::set(size_t row, const void* value, size_t value_size) {
    if (!owns_data_) {
        throw InvalidOperation("Cannot set on non-owned column");
    }
    if (row >= size_) {
        throw InvalidOperation("Row index out of range in set");
    }
    size_t elem_size = dtype_size(dtype_);
    if (elem_size == 0) {
        throw InvalidOperation("Cannot set on variable-length column via Column::set");
    }
    if (value_size != elem_size) {
        throw InvalidOperation("Value size mismatch in set");
    }

    std::memcpy(owned_data_.data() + row * elem_size, value, elem_size);
}

}  // namespace wayy_db
