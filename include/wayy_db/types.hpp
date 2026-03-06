#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <stdexcept>

namespace wayy_db {

/// Supported data types for columns
enum class DType : uint8_t {
    Int64 = 0,
    Float64 = 1,
    Timestamp = 2,  // Nanoseconds since Unix epoch
    Symbol = 3,     // Interned string index
    Bool = 4,
    String = 5,     // Arrow-style variable-length UTF-8 (offsets + data)
    Decimal6 = 6,   // Int64 with implied 6 decimal places (max ±9.2T)
};

/// Get the size in bytes for a given type (0 for variable-length types)
constexpr size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Int64:     return sizeof(int64_t);
        case DType::Float64:   return sizeof(double);
        case DType::Timestamp: return sizeof(int64_t);
        case DType::Symbol:    return sizeof(uint32_t);
        case DType::Bool:      return sizeof(uint8_t);
        case DType::String:    return 0;  // Variable-length, use StringColumn
        case DType::Decimal6:  return sizeof(int64_t);  // Stored as int64
    }
    return 0;  // Unreachable
}

/// Check if a dtype is fixed-width
constexpr bool dtype_is_fixed(DType dtype) {
    return dtype != DType::String;
}

/// Convert DType to string representation
constexpr std::string_view dtype_to_string(DType dtype) {
    switch (dtype) {
        case DType::Int64:     return "int64";
        case DType::Float64:   return "float64";
        case DType::Timestamp: return "timestamp";
        case DType::Symbol:    return "symbol";
        case DType::Bool:      return "bool";
        case DType::String:    return "string";
        case DType::Decimal6:  return "decimal6";
    }
    return "unknown";
}

/// Parse DType from string
DType dtype_from_string(std::string_view s);

/// Magic number for WayyDB files: "WAYYDB\x00\x01"
constexpr uint64_t WAYY_MAGIC = 0x57415959'44420001ULL;

/// Current file format version
constexpr uint32_t WAYY_VERSION = 1;

/// Column file header (64 bytes)
struct ColumnHeader {
    uint64_t magic;           // WAYY_MAGIC
    uint32_t version;         // WAYY_VERSION
    DType dtype;              // Data type
    uint8_t reserved1[3];     // Padding
    uint64_t row_count;       // Number of rows
    uint64_t compression;     // 0 = none, 1 = LZ4
    uint8_t reserved2[24];    // Reserved for future use
    uint64_t data_offset;     // Offset to data (typically 64)
};

static_assert(sizeof(ColumnHeader) == 64, "ColumnHeader must be 64 bytes");

/// Exception types
class WayyException : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

class ColumnNotFound : public WayyException {
public:
    explicit ColumnNotFound(const std::string& name)
        : WayyException("Column not found: " + name) {}
};

class TypeMismatch : public WayyException {
public:
    TypeMismatch(DType expected, DType actual)
        : WayyException("Type mismatch: expected " +
                       std::string(dtype_to_string(expected)) +
                       ", got " + std::string(dtype_to_string(actual))) {}
};

class InvalidOperation : public WayyException {
    using WayyException::WayyException;
};

}  // namespace wayy_db
