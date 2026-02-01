#include <gtest/gtest.h>
#include "wayy_db/types.hpp"

using namespace wayy_db;

TEST(TypesTest, DTypeSizes) {
    EXPECT_EQ(dtype_size(DType::Int64), 8);
    EXPECT_EQ(dtype_size(DType::Float64), 8);
    EXPECT_EQ(dtype_size(DType::Timestamp), 8);
    EXPECT_EQ(dtype_size(DType::Symbol), 4);
    EXPECT_EQ(dtype_size(DType::Bool), 1);
}

TEST(TypesTest, DTypeToString) {
    EXPECT_EQ(dtype_to_string(DType::Int64), "int64");
    EXPECT_EQ(dtype_to_string(DType::Float64), "float64");
    EXPECT_EQ(dtype_to_string(DType::Timestamp), "timestamp");
    EXPECT_EQ(dtype_to_string(DType::Symbol), "symbol");
    EXPECT_EQ(dtype_to_string(DType::Bool), "bool");
}

TEST(TypesTest, DTypeFromString) {
    EXPECT_EQ(dtype_from_string("int64"), DType::Int64);
    EXPECT_EQ(dtype_from_string("float64"), DType::Float64);
    EXPECT_EQ(dtype_from_string("timestamp"), DType::Timestamp);
    EXPECT_EQ(dtype_from_string("symbol"), DType::Symbol);
    EXPECT_EQ(dtype_from_string("bool"), DType::Bool);
}

TEST(TypesTest, DTypeFromStringInvalid) {
    EXPECT_THROW(dtype_from_string("invalid"), WayyException);
}

TEST(TypesTest, ColumnHeaderSize) {
    EXPECT_EQ(sizeof(ColumnHeader), 64);
}

TEST(TypesTest, MagicNumber) {
    EXPECT_EQ(WAYY_MAGIC, 0x5741595944420001ULL);
}
