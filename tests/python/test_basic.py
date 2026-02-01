"""Basic functionality tests for WayyDB Python bindings."""

import pytest
import numpy as np
import wayy_db as wdb


class TestTable:
    """Tests for Table class."""

    def test_create_empty_table(self):
        table = wdb.Table("test")
        assert table.name == "test"
        assert table.num_rows == 0
        assert table.num_columns == 0
        assert len(table) == 0

    def test_from_dict(self, sample_trades):
        table = wdb.from_dict(sample_trades, name="trades", sorted_by="timestamp")

        assert table.name == "trades"
        assert table.num_rows == 5
        assert table.num_columns == 4
        assert table.sorted_by == "timestamp"

    def test_column_access(self, sample_trades):
        table = wdb.from_dict(sample_trades, name="trades")

        assert table.has_column("price")
        assert not table.has_column("nonexistent")

        price_col = table["price"]
        assert price_col.name == "price"
        assert price_col.dtype == wdb.DType.Float64
        assert len(price_col) == 5

    def test_to_numpy_zero_copy(self, sample_trades):
        table = wdb.from_dict(sample_trades, name="trades")

        prices = table["price"].to_numpy()

        assert isinstance(prices, np.ndarray)
        assert prices.dtype == np.float64
        assert len(prices) == 5
        np.testing.assert_array_equal(prices, sample_trades["price"])

    def test_to_dict(self, sample_trades):
        table = wdb.from_dict(sample_trades, name="trades")

        result = table.to_dict()

        assert set(result.keys()) == {"timestamp", "symbol", "price", "size"}
        np.testing.assert_array_equal(result["price"], sample_trades["price"])

    def test_column_names(self, sample_trades):
        table = wdb.from_dict(sample_trades, name="trades")

        names = table.column_names()

        assert set(names) == {"timestamp", "symbol", "price", "size"}


class TestDatabase:
    """Tests for Database class."""

    def test_in_memory_database(self):
        db = wdb.Database()

        assert not db.is_persistent
        assert db.tables() == []

    def test_create_table(self):
        db = wdb.Database()
        table = db.create_table("trades")

        assert db.has_table("trades")
        assert "trades" in db.tables()

    def test_persistent_database(self, temp_dir, sample_trades):
        # Create and populate
        db = wdb.Database(temp_dir)
        table = db.create_table("trades")

        for name, data in sample_trades.items():
            dtype = {
                np.dtype("int64"): wdb.DType.Int64,
                np.dtype("float64"): wdb.DType.Float64,
                np.dtype("uint32"): wdb.DType.Symbol,
            }[data.dtype]
            table.add_column_from_numpy(name, data, dtype)

        table.set_sorted_by("timestamp")
        db.save()

        # Reload and verify
        db2 = wdb.Database(temp_dir)
        assert db2.has_table("trades")

        loaded = db2["trades"]
        assert loaded.num_rows == 5
        assert loaded.sorted_by == "timestamp"


class TestOperations:
    """Tests for operations module."""

    def test_aggregations(self, sample_trades):
        table = wdb.from_dict(sample_trades, name="trades")
        price_col = table["price"]

        assert wdb.ops.sum(price_col) == pytest.approx(1214.0)
        assert wdb.ops.avg(price_col) == pytest.approx(242.8)
        assert wdb.ops.min(price_col) == pytest.approx(150.0)
        assert wdb.ops.max(price_col) == pytest.approx(381.0)

    def test_window_functions(self, sample_trades):
        table = wdb.from_dict(sample_trades, name="trades")
        price_col = table["price"]

        mavg = wdb.ops.mavg(price_col, 2)
        assert len(mavg) == 5
        assert mavg[1] == pytest.approx((150.0 + 380.0) / 2)

        msum = wdb.ops.msum(price_col, 2)
        assert len(msum) == 5

    def test_ema(self, sample_trades):
        table = wdb.from_dict(sample_trades, name="trades")
        price_col = table["price"]

        ema = wdb.ops.ema(price_col, 0.5)
        assert len(ema) == 5
        assert ema[0] == pytest.approx(150.0)  # First value unchanged

    def test_diff(self, sample_trades):
        table = wdb.from_dict(sample_trades, name="trades")
        price_col = table["price"]

        diff = wdb.ops.diff(price_col, 1)
        assert len(diff) == 5
        assert diff[1] == pytest.approx(380.0 - 150.0)


class TestAsOfJoin:
    """Tests for as-of join operation."""

    def test_aj_basic(self, sample_trades, sample_quotes):
        trades = wdb.from_dict(sample_trades, name="trades", sorted_by="timestamp")
        quotes = wdb.from_dict(sample_quotes, name="quotes", sorted_by="timestamp")

        result = wdb.ops.aj(trades, quotes, on=["symbol"], as_of="timestamp")

        assert result.num_rows == 5
        assert result.has_column("bid")
        assert result.has_column("ask")
        assert result.has_column("price")

    def test_aj_requires_sorted(self, sample_trades, sample_quotes):
        trades = wdb.from_dict(sample_trades, name="trades")  # Not sorted
        quotes = wdb.from_dict(sample_quotes, name="quotes", sorted_by="timestamp")

        with pytest.raises(wdb.InvalidOperation):
            wdb.ops.aj(trades, quotes, on=["symbol"], as_of="timestamp")


class TestExceptions:
    """Tests for exception handling."""

    def test_column_not_found(self, sample_trades):
        table = wdb.from_dict(sample_trades, name="trades")

        with pytest.raises(wdb.ColumnNotFound):
            _ = table["nonexistent"]

    def test_invalid_operation(self, sample_trades):
        table = wdb.from_dict(sample_trades, name="trades")

        with pytest.raises(wdb.ColumnNotFound):
            table.set_sorted_by("nonexistent")
