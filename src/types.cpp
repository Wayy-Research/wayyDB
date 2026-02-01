#include "wayy_db/types.hpp"

#include <unordered_map>

namespace wayy_db {

DType dtype_from_string(std::string_view s) {
    static const std::unordered_map<std::string_view, DType> map = {
        {"int64", DType::Int64},
        {"float64", DType::Float64},
        {"timestamp", DType::Timestamp},
        {"symbol", DType::Symbol},
        {"bool", DType::Bool},
    };

    auto it = map.find(s);
    if (it == map.end()) {
        throw WayyException("Unknown dtype: " + std::string(s));
    }
    return it->second;
}

}  // namespace wayy_db
