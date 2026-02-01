#include <gtest/gtest.h>
#include "wayy_db/table.hpp"

#include <filesystem>
#include <cstring>

using namespace wayy_db;
namespace fs = std::filesystem;

class TableTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir_ = "/tmp/wayy_test_" + std::to_string(getpid());
        fs::create_directories(test_dir_);
    }

    void TearDown() override {
        fs::remove_all(test_dir_);
    }

    std::string test_dir_;
};

TEST_F(TableTest, EmptyTable) {
    Table table("test");

    EXPECT_EQ(table.name(), "test");
    EXPECT_EQ(table.num_rows(), 0);
    EXPECT_EQ(table.num_columns(), 0);
    EXPECT_FALSE(table.is_sorted());
}

TEST_F(TableTest, AddColumn) {
    Table table("test");

    std::vector<double> prices = {100.0, 101.0, 102.0};
    table.add_column("price", DType::Float64, prices.data(), prices.size());

    EXPECT_EQ(table.num_rows(), 3);
    EXPECT_EQ(table.num_columns(), 1);
    EXPECT_TRUE(table.has_column("price"));
    EXPECT_FALSE(table.has_column("nonexistent"));
}

TEST_F(TableTest, MultipleColumns) {
    Table table("trades");

    std::vector<int64_t> timestamps = {1000, 2000, 3000};
    std::vector<double> prices = {100.0, 101.0, 102.0};
    std::vector<int64_t> sizes = {10, 20, 30};

    table.add_column("timestamp", DType::Timestamp, timestamps.data(), timestamps.size());
    table.add_column("price", DType::Float64, prices.data(), prices.size());
    table.add_column("size", DType::Int64, sizes.data(), sizes.size());

    EXPECT_EQ(table.num_columns(), 3);
    EXPECT_EQ(table.num_rows(), 3);

    auto names = table.column_names();
    EXPECT_EQ(names.size(), 3);
}

TEST_F(TableTest, ColumnSizeMismatch) {
    Table table("test");

    std::vector<double> col1 = {1.0, 2.0, 3.0};
    std::vector<double> col2 = {1.0, 2.0};  // Different size

    table.add_column("col1", DType::Float64, col1.data(), col1.size());
    EXPECT_THROW(
        table.add_column("col2", DType::Float64, col2.data(), col2.size()),
        InvalidOperation
    );
}

TEST_F(TableTest, SortedBy) {
    Table table("test");

    std::vector<int64_t> timestamps = {1000, 2000, 3000};
    table.add_column("timestamp", DType::Timestamp, timestamps.data(), timestamps.size());

    EXPECT_FALSE(table.is_sorted());

    table.set_sorted_by("timestamp");
    EXPECT_TRUE(table.is_sorted());
    EXPECT_EQ(table.sorted_by(), "timestamp");
}

TEST_F(TableTest, SortedByNonexistent) {
    Table table("test");

    std::vector<int64_t> data = {1, 2, 3};
    table.add_column("col", DType::Int64, data.data(), data.size());

    EXPECT_THROW(table.set_sorted_by("nonexistent"), ColumnNotFound);
}

TEST_F(TableTest, SaveAndLoad) {
    std::string table_dir = test_dir_ + "/trades";

    // Create and save
    {
        Table table("trades");

        std::vector<int64_t> timestamps = {1000, 2000, 3000};
        std::vector<double> prices = {100.0, 101.0, 102.0};

        table.add_column("timestamp", DType::Timestamp, timestamps.data(), timestamps.size());
        table.add_column("price", DType::Float64, prices.data(), prices.size());
        table.set_sorted_by("timestamp");

        table.save(table_dir);
    }

    // Load and verify
    {
        Table loaded = Table::load(table_dir);

        EXPECT_EQ(loaded.name(), "trades");
        EXPECT_EQ(loaded.num_rows(), 3);
        EXPECT_EQ(loaded.num_columns(), 2);
        EXPECT_EQ(loaded.sorted_by(), "timestamp");

        auto ts = loaded.column("timestamp").as_int64();
        EXPECT_EQ(ts[0], 1000);
        EXPECT_EQ(ts[2], 3000);

        auto prices = loaded.column("price").as_float64();
        EXPECT_DOUBLE_EQ(prices[1], 101.0);
    }
}

TEST_F(TableTest, Mmap) {
    std::string table_dir = test_dir_ + "/mmap_test";

    // Create and save
    {
        Table table("mmap_test");

        std::vector<int64_t> data = {10, 20, 30, 40, 50};
        table.add_column("values", DType::Int64, data.data(), data.size());
        table.save(table_dir);
    }

    // Memory-map and verify
    {
        Table mapped = Table::mmap(table_dir);

        EXPECT_EQ(mapped.num_rows(), 5);

        auto values = mapped.column("values").as_int64();
        EXPECT_EQ(values[0], 10);
        EXPECT_EQ(values[4], 50);
    }
}
