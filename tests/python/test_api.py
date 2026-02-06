"""Tests for WayyDB REST API endpoints."""

import pytest
import asyncio
import numpy as np
from httpx import AsyncClient, ASGITransport
from fastapi.testclient import TestClient
import tempfile
import shutil
import os

# Set up test environment before importing app
_test_data_path = tempfile.mkdtemp(prefix="wayydb_test_")
os.environ["WAYY_DATA_PATH"] = _test_data_path


@pytest.fixture(scope="module")
def api_client():
    """Create a test client with lifespan managed."""
    from api.main import app
    with TestClient(app) as client:
        yield client


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root(self, api_client):
        """Test root endpoint."""
        response = api_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "WayyDB API"
        assert "version" in data
        assert data["status"] == "healthy"

    def test_health(self, api_client):
        """Test health endpoint."""
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "tables" in data


class TestTableOperations:
    """Tests for table CRUD operations."""

    def test_list_tables(self, api_client):
        """Test listing tables."""
        response = api_client.get("/tables")
        assert response.status_code == 200
        assert "tables" in response.json()

    def test_create_and_delete_table(self, api_client):
        """Test creating and deleting a table."""
        # Create
        response = api_client.post("/tables", json={"name": "test_table"})
        assert response.status_code == 200
        assert response.json()["created"] == "test_table"

        # Verify exists
        response = api_client.get("/tables/test_table")
        assert response.status_code == 200

        # Delete
        response = api_client.delete("/tables/test_table")
        assert response.status_code == 200

        # Verify deleted
        response = api_client.get("/tables/test_table")
        assert response.status_code == 404

    def test_upload_table(self, api_client):
        """Test uploading a table with data."""
        data = {
            "name": "uploaded_table",
            "columns": [
                {"name": "timestamp", "dtype": "int64", "data": [1000, 2000, 3000]},
                {"name": "price", "dtype": "float64", "data": [100.0, 101.0, 102.0]},
            ],
            "sorted_by": "timestamp",
        }

        response = api_client.post("/tables/upload", json=data)
        assert response.status_code == 200
        result = response.json()
        assert result["created"] == "uploaded_table"
        assert result["rows"] == 3

        # Get data back
        response = api_client.get("/tables/uploaded_table/data")
        assert response.status_code == 200
        result = response.json()
        assert result["total_rows"] == 3
        assert result["data"]["timestamp"] == [1000, 2000, 3000]
        assert result["data"]["price"] == [100.0, 101.0, 102.0]

        # Cleanup
        api_client.delete("/tables/uploaded_table")

    def test_get_table_info(self, api_client):
        """Test getting table metadata."""
        # Upload a table
        data = {
            "name": "info_test",
            "columns": [
                {"name": "ts", "dtype": "int64", "data": [1, 2, 3]},
                {"name": "val", "dtype": "float64", "data": [1.0, 2.0, 3.0]},
            ],
            "sorted_by": "ts",
        }
        api_client.post("/tables/upload", json=data)

        # Get info
        response = api_client.get("/tables/info_test")
        assert response.status_code == 200
        info = response.json()
        assert info["name"] == "info_test"
        assert info["num_rows"] == 3
        assert info["num_columns"] == 2
        assert "ts" in info["columns"]
        assert info["sorted_by"] == "ts"

        # Cleanup
        api_client.delete("/tables/info_test")


class TestAppendAPI:
    """Tests for the append endpoint."""

    def test_append_to_table(self, api_client):
        """Test appending rows to an existing table."""
        # Create initial table
        data = {
            "name": "append_test",
            "columns": [
                {"name": "timestamp", "dtype": "int64", "data": [1000, 2000]},
                {"name": "price", "dtype": "float64", "data": [100.0, 101.0]},
            ],
            "sorted_by": "timestamp",
        }
        api_client.post("/tables/upload", json=data)

        # Append more data
        append_data = {
            "columns": [
                {"name": "timestamp", "dtype": "int64", "data": [3000, 4000]},
                {"name": "price", "dtype": "float64", "data": [102.0, 103.0]},
            ]
        }
        response = api_client.post("/tables/append_test/append", json=append_data)
        assert response.status_code == 200
        result = response.json()
        assert result["new_rows"] == 2
        assert result["total_rows"] == 4

        # Verify data
        response = api_client.get("/tables/append_test/data")
        data = response.json()["data"]
        assert len(data["timestamp"]) == 4
        assert data["timestamp"] == [1000, 2000, 3000, 4000]
        assert data["price"] == [100.0, 101.0, 102.0, 103.0]

        # Cleanup
        api_client.delete("/tables/append_test")

    def test_append_column_mismatch(self, api_client):
        """Test that append fails with mismatched columns."""
        # Create table
        data = {
            "name": "mismatch_test",
            "columns": [
                {"name": "timestamp", "dtype": "int64", "data": [1000]},
                {"name": "price", "dtype": "float64", "data": [100.0]},
            ],
        }
        api_client.post("/tables/upload", json=data)

        # Try to append with wrong columns
        append_data = {
            "columns": [
                {"name": "timestamp", "dtype": "int64", "data": [2000]},
                {"name": "volume", "dtype": "float64", "data": [50.0]},  # Wrong column
            ]
        }
        response = api_client.post("/tables/mismatch_test/append", json=append_data)
        assert response.status_code == 400
        assert "mismatch" in response.json()["detail"].lower()

        # Cleanup
        api_client.delete("/tables/mismatch_test")


class TestIngestAPI:
    """Tests for streaming ingestion REST endpoints."""

    def test_ingest_single_tick(self, api_client):
        """Test ingesting a single tick via REST."""
        tick = {
            "symbol": "BTC-USD",
            "price": 42150.50,
            "volume": 1.5,
        }
        response = api_client.post("/ingest/test_ticks", json=tick)
        assert response.status_code == 200
        assert response.json()["ingested"] == 1

    def test_ingest_batch(self, api_client):
        """Test ingesting a batch of ticks via REST."""
        batch = {
            "ticks": [
                {"symbol": "BTC-USD", "price": 42150.0},
                {"symbol": "ETH-USD", "price": 2250.0},
                {"symbol": "SOL-USD", "price": 100.0},
            ]
        }
        response = api_client.post("/ingest/test_ticks/batch", json=batch)
        assert response.status_code == 200
        assert response.json()["ingested"] == 3


class TestStreamingStats:
    """Tests for streaming statistics endpoints."""

    def test_streaming_stats(self, api_client):
        """Test getting streaming statistics."""
        response = api_client.get("/streaming/stats")
        assert response.status_code == 200
        stats = response.json()
        assert "ticks_received" in stats
        assert "ticks_flushed" in stats
        assert "buffer_sizes" in stats
        assert "running" in stats

    def test_get_quote(self, api_client):
        """Test getting a specific quote."""
        # Ingest a tick first
        api_client.post("/ingest/ticks", json={"symbol": "BTC-USD", "price": 42000.0})

        # Get quote
        response = api_client.get("/streaming/quote/ticks/BTC-USD")
        assert response.status_code == 200
        quote = response.json()
        assert quote["symbol"] == "BTC-USD"
        assert quote["price"] == 42000.0

    def test_get_all_quotes(self, api_client):
        """Test getting all quotes for a table."""
        # Ingest multiple ticks
        for symbol, price in [("BTC-USD", 42000.0), ("ETH-USD", 2200.0)]:
            api_client.post("/ingest/quotes_test", json={"symbol": symbol, "price": price})

        # Get all quotes
        response = api_client.get("/streaming/quotes/quotes_test")
        assert response.status_code == 200
        quotes = response.json()
        assert "BTC-USD" in quotes
        assert "ETH-USD" in quotes


class TestAggregations:
    """Tests for aggregation endpoints."""

    @pytest.fixture
    def setup_table(self, api_client):
        """Set up a table for aggregation tests."""
        data = {
            "name": "agg_test",
            "columns": [
                {"name": "timestamp", "dtype": "int64", "data": [1, 2, 3, 4, 5]},
                {"name": "price", "dtype": "float64", "data": [10.0, 20.0, 30.0, 40.0, 50.0]},
            ],
        }
        api_client.post("/tables/upload", json=data)
        yield
        api_client.delete("/tables/agg_test")

    def test_sum(self, api_client, setup_table):
        """Test sum aggregation."""
        response = api_client.get("/tables/agg_test/agg/price/sum")
        assert response.status_code == 200
        assert response.json()["result"] == pytest.approx(150.0)

    def test_avg(self, api_client, setup_table):
        """Test average aggregation."""
        response = api_client.get("/tables/agg_test/agg/price/avg")
        assert response.status_code == 200
        assert response.json()["result"] == pytest.approx(30.0)

    def test_min_max(self, api_client, setup_table):
        """Test min/max aggregations."""
        response = api_client.get("/tables/agg_test/agg/price/min")
        assert response.json()["result"] == pytest.approx(10.0)

        response = api_client.get("/tables/agg_test/agg/price/max")
        assert response.json()["result"] == pytest.approx(50.0)


class TestWindowFunctions:
    """Tests for window function endpoint."""

    @pytest.fixture
    def setup_table(self, api_client):
        """Set up a table for window tests."""
        data = {
            "name": "window_test",
            "columns": [
                {"name": "timestamp", "dtype": "int64", "data": list(range(10))},
                {"name": "price", "dtype": "float64", "data": [float(i) for i in range(10)]},
            ],
        }
        api_client.post("/tables/upload", json=data)
        yield
        api_client.delete("/tables/window_test")

    def test_mavg(self, api_client, setup_table):
        """Test moving average."""
        response = api_client.post("/window", json={
            "table": "window_test",
            "column": "price",
            "operation": "mavg",
            "window": 3,
        })
        assert response.status_code == 200
        result = response.json()["result"]
        assert len(result) == 10
        # Third element should be avg of 0, 1, 2 = 1.0
        assert result[2] == pytest.approx(1.0)

    def test_ema(self, api_client, setup_table):
        """Test exponential moving average."""
        response = api_client.post("/window", json={
            "table": "window_test",
            "column": "price",
            "operation": "ema",
            "alpha": 0.5,
        })
        assert response.status_code == 200
        result = response.json()["result"]
        assert len(result) == 10
        assert result[0] == pytest.approx(0.0)  # First value unchanged


# Cleanup test directory after all tests
def pytest_sessionfinish(session, exitstatus):
    """Clean up test data directory."""
    if _test_data_path and os.path.exists(_test_data_path):
        shutil.rmtree(_test_data_path, ignore_errors=True)
