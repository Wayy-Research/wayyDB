#include <gtest/gtest.h>
#include "wayy_db/column.hpp"

using namespace wayy_db;

TEST(ColumnViewTest, BasicOperations) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    ColumnView<double> view(data.data(), data.size());

    EXPECT_EQ(view.size(), 5);
    EXPECT_FALSE(view.empty());
    EXPECT_EQ(view[0], 1.0);
    EXPECT_EQ(view[4], 5.0);
    EXPECT_EQ(view.front(), 1.0);
    EXPECT_EQ(view.back(), 5.0);
}

TEST(ColumnViewTest, Iteration) {
    std::vector<int64_t> data = {10, 20, 30};
    ColumnView<int64_t> view(data.data(), data.size());

    int64_t sum = 0;
    for (auto val : view) {
        sum += val;
    }
    EXPECT_EQ(sum, 60);
}

TEST(ColumnViewTest, Subview) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    ColumnView<double> view(data.data(), data.size());

    auto sub = view.subview(1, 3);
    EXPECT_EQ(sub.size(), 3);
    EXPECT_EQ(sub[0], 2.0);
    EXPECT_EQ(sub[2], 4.0);
}

TEST(ColumnViewTest, OutOfRange) {
    std::vector<double> data = {1.0, 2.0};
    ColumnView<double> view(data.data(), data.size());

    EXPECT_THROW(view.at(5), std::out_of_range);
    EXPECT_THROW(view.subview(1, 5), std::out_of_range);
}

TEST(ColumnTest, ConstructWithOwnedData) {
    std::vector<uint8_t> data(40);  // 5 doubles
    auto* ptr = reinterpret_cast<double*>(data.data());
    for (int i = 0; i < 5; ++i) ptr[i] = static_cast<double>(i);

    Column col("test", DType::Float64, std::move(data));

    EXPECT_EQ(col.name(), "test");
    EXPECT_EQ(col.dtype(), DType::Float64);
    EXPECT_EQ(col.size(), 5);
    EXPECT_EQ(col.byte_size(), 40);
}

TEST(ColumnTest, TypedAccess) {
    std::vector<uint8_t> data(24);
    auto* ptr = reinterpret_cast<int64_t*>(data.data());
    ptr[0] = 100;
    ptr[1] = 200;
    ptr[2] = 300;

    Column col("ints", DType::Int64, std::move(data));
    auto view = col.as_int64();

    EXPECT_EQ(view.size(), 3);
    EXPECT_EQ(view[0], 100);
    EXPECT_EQ(view[2], 300);
}

TEST(ColumnTest, TypeMismatch) {
    std::vector<uint8_t> data(24);
    Column col("ints", DType::Int64, std::move(data));

    EXPECT_THROW(col.as_float64(), TypeMismatch);
}
