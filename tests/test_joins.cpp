#include <gtest/gtest.h>
#include "wayy_db/table.hpp"
#include "wayy_db/ops/joins.hpp"

using namespace wayy_db;

class JoinsTest : public ::testing::Test {
protected:
    Table create_trades() {
        Table trades("trades");

        // Trades at times 100, 200, 300 for symbols 0 (AAPL) and 1 (MSFT)
        std::vector<int64_t> timestamps = {100, 150, 200, 250, 300};
        std::vector<uint32_t> symbols = {0, 1, 0, 1, 0};  // AAPL, MSFT, AAPL, MSFT, AAPL
        std::vector<double> prices = {150.0, 380.0, 151.0, 381.0, 152.0};
        std::vector<int64_t> sizes = {100, 200, 150, 250, 100};

        trades.add_column("timestamp", DType::Timestamp, timestamps.data(), timestamps.size());
        trades.add_column("symbol", DType::Symbol, symbols.data(), symbols.size());
        trades.add_column("price", DType::Float64, prices.data(), prices.size());
        trades.add_column("size", DType::Int64, sizes.data(), sizes.size());
        trades.set_sorted_by("timestamp");

        return trades;
    }

    Table create_quotes() {
        Table quotes("quotes");

        // Quotes at times 50, 90, 140, 190, 280
        std::vector<int64_t> timestamps = {50, 90, 140, 190, 280};
        std::vector<uint32_t> symbols = {0, 1, 0, 1, 0};
        std::vector<double> bids = {149.5, 379.5, 150.5, 380.5, 151.5};
        std::vector<double> asks = {150.0, 380.0, 151.0, 381.0, 152.0};

        quotes.add_column("timestamp", DType::Timestamp, timestamps.data(), timestamps.size());
        quotes.add_column("symbol", DType::Symbol, symbols.data(), symbols.size());
        quotes.add_column("bid", DType::Float64, bids.data(), bids.size());
        quotes.add_column("ask", DType::Float64, asks.data(), asks.size());
        quotes.set_sorted_by("timestamp");

        return quotes;
    }
};

TEST_F(JoinsTest, AsOfJoinBasic) {
    auto trades = create_trades();
    auto quotes = create_quotes();

    auto result = ops::aj(trades, quotes, {"symbol"}, "timestamp");

    // Result should have same number of rows as trades
    EXPECT_EQ(result.num_rows(), 5);

    // Check that we have columns from both tables
    EXPECT_TRUE(result.has_column("timestamp"));
    EXPECT_TRUE(result.has_column("symbol"));
    EXPECT_TRUE(result.has_column("price"));
    EXPECT_TRUE(result.has_column("bid"));
    EXPECT_TRUE(result.has_column("ask"));

    // Verify as-of semantics:
    // Trade at t=100, symbol=AAPL should get quote at t=90... wait, that's MSFT
    // Trade at t=100, symbol=AAPL should get quote at t=50 (AAPL)
    auto bids = result.column("bid").as_float64();
    EXPECT_DOUBLE_EQ(bids[0], 149.5);  // AAPL trade at 100 -> AAPL quote at 50

    // Trade at t=150, symbol=MSFT should get quote at t=90 (MSFT)
    EXPECT_DOUBLE_EQ(bids[1], 379.5);

    // Trade at t=200, symbol=AAPL should get quote at t=140 (AAPL)
    EXPECT_DOUBLE_EQ(bids[2], 150.5);
}

TEST_F(JoinsTest, AsOfJoinRequiresSorted) {
    Table left("left");
    Table right("right");

    std::vector<int64_t> ts = {1, 2, 3};
    left.add_column("ts", DType::Timestamp, ts.data(), ts.size());
    right.add_column("ts", DType::Timestamp, ts.data(), ts.size());

    // Neither is sorted
    EXPECT_THROW(ops::aj(left, right, {}, "ts"), InvalidOperation);

    // Only left is sorted
    left.set_sorted_by("ts");
    EXPECT_THROW(ops::aj(left, right, {}, "ts"), InvalidOperation);
}

TEST_F(JoinsTest, WindowJoinBasic) {
    auto trades = create_trades();
    auto quotes = create_quotes();

    // Window: 60ns before, 0ns after
    auto result = ops::wj(trades, quotes, {"symbol"}, "timestamp", 60, 0);

    // Window join may have more rows than left table
    EXPECT_GT(result.num_rows(), 0);

    // Check columns exist
    EXPECT_TRUE(result.has_column("bid"));
    EXPECT_TRUE(result.has_column("price"));
}

TEST_F(JoinsTest, AsOfJoinNoMatches) {
    Table trades("trades");
    Table quotes("quotes");

    // Trades for symbol 0
    std::vector<int64_t> trade_ts = {100, 200};
    std::vector<uint32_t> trade_sym = {0, 0};
    std::vector<double> trade_px = {100.0, 101.0};

    trades.add_column("timestamp", DType::Timestamp, trade_ts.data(), trade_ts.size());
    trades.add_column("symbol", DType::Symbol, trade_sym.data(), trade_sym.size());
    trades.add_column("price", DType::Float64, trade_px.data(), trade_px.size());
    trades.set_sorted_by("timestamp");

    // Quotes for symbol 1 (different symbol)
    std::vector<int64_t> quote_ts = {50, 150};
    std::vector<uint32_t> quote_sym = {1, 1};
    std::vector<double> quote_bid = {99.0, 100.0};

    quotes.add_column("timestamp", DType::Timestamp, quote_ts.data(), quote_ts.size());
    quotes.add_column("symbol", DType::Symbol, quote_sym.data(), quote_sym.size());
    quotes.add_column("bid", DType::Float64, quote_bid.data(), quote_bid.size());
    quotes.set_sorted_by("timestamp");

    auto result = ops::aj(trades, quotes, {"symbol"}, "timestamp");

    // Should still have 2 rows, but bid should be 0 (null)
    EXPECT_EQ(result.num_rows(), 2);

    auto bids = result.column("bid").as_float64();
    EXPECT_DOUBLE_EQ(bids[0], 0.0);
    EXPECT_DOUBLE_EQ(bids[1], 0.0);
}
