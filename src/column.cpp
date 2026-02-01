#include "wayy_db/column.hpp"

#include <cstring>

namespace wayy_db {

Column::Column(std::string name, DType dtype, std::vector<uint8_t> data)
    : name_(std::move(name))
    , dtype_(dtype)
    , size_(data.size() / dtype_size(dtype))
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
    if (owns_data && data != nullptr) {
        // Copy data into owned buffer
        size_t byte_size = size * dtype_size(dtype);
        owned_data_.resize(byte_size);
        std::memcpy(owned_data_.data(), data, byte_size);
        data_ = owned_data_.data();
    }
}

}  // namespace wayy_db
