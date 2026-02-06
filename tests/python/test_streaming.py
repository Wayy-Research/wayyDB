"""Tests for WayyDB streaming functionality with PubSub integration."""

import asyncio
import time
from datetime import datetime, timezone

import numpy as np
import pytest

import wayy_db as wdb

import sys
sys.path.insert(0, str(__file__).replace("/tests/python/test_streaming.py", ""))
from api.pubsub import InMemoryPubSub
from api.streaming import StreamingManager, TickBuffer


class TestTickBuffer:
    """Tests for TickBuffer data structure."""

    def test_empty_buffer(self):
        buffer = TickBuffer()
        assert len(buffer) == 0

    def test_append_single_tick(self):
        buffer = TickBuffer()
        buffer.append(
            timestamp=1704067200000000000,
            symbol="BTC-USD",
            price=42150.50,
            volume=1.5,
            bid=42150.00,
            ask=42151.00,
        )
        assert len(buffer) == 1
        assert buffer.timestamps[0] == 1704067200000000000
        assert buffer.symbols[0] == "BTC-USD"
        assert buffer.prices[0] == 42150.50

    def test_append_multiple_ticks(self):
        buffer = TickBuffer()
        for i in range(100):
            buffer.append(
                timestamp=1704067200000000000 + i * 1000000,
                symbol=f"SYM-{i % 5}",
                price=100.0 + i * 0.1,
                volume=float(i),
            )
        assert len(buffer) == 100

    def test_clear(self):
        buffer = TickBuffer()
        for i in range(10):
            buffer.append(
                timestamp=i,
                symbol="BTC-USD",
                price=100.0,
            )
        assert len(buffer) == 10
        buffer.clear()
        assert len(buffer) == 0

    def test_to_columnar(self):
        buffer = TickBuffer()
        buffer.append(1000, "BTC-USD", 42150.0, 1.0, 42149.0, 42151.0)
        buffer.append(2000, "ETH-USD", 2250.0, 10.0, 2249.0, 2251.0)

        data = buffer.to_columnar()

        assert "timestamp" in data
        assert "symbol" in data
        assert "price" in data
        assert "volume" in data
        assert "bid" in data
        assert "ask" in data

        assert data["timestamp"].dtype == np.int64
        assert data["symbol"].dtype == np.uint32
        assert data["price"].dtype == np.float64

        assert len(data["timestamp"]) == 2
        np.testing.assert_array_equal(data["timestamp"], [1000, 2000])
        np.testing.assert_array_equal(data["price"], [42150.0, 2250.0])


class TestStreamingManager:
    """Tests for StreamingManager with PubSub integration."""

    @pytest.fixture
    def pubsub(self):
        return InMemoryPubSub(max_buffer_per_channel=10000)

    @pytest.fixture
    def manager(self, pubsub):
        """Create a streaming manager with in-memory pubsub."""
        manager = StreamingManager(
            flush_interval=0.1,
            max_buffer_size=100,
            batch_broadcast_interval=0.01,
            pubsub=pubsub,
        )
        return manager

    @pytest.fixture
    def temp_db(self, temp_dir):
        """Create a temporary database for testing."""
        return wdb.Database(temp_dir)

    @pytest.mark.asyncio
    async def test_ingest_single_tick(self, manager):
        """Test ingesting a single tick."""
        await manager.start()
        await manager.ingest_tick(
            table="ticks",
            symbol="BTC-USD",
            price=42150.50,
            volume=1.5,
        )

        stats = manager.get_stats()
        assert stats["ticks_received"] == 1
        assert stats["buffer_sizes"]["ticks"] == 1
        await manager.stop()

    @pytest.mark.asyncio
    async def test_ingest_publishes_to_pubsub(self, manager):
        """Test that ingestion publishes to PubSub channels."""
        await manager.start()

        received = []

        async def on_tick(msg):
            received.append(msg)

        await manager._pubsub.subscribe("ticks:BTC-USD", on_tick, "test")

        await manager.ingest_tick(
            table="ticks",
            symbol="BTC-USD",
            price=42150.50,
        )

        assert len(received) == 1
        assert received[0]["price"] == 42150.50
        assert received[0]["_channel"] == "ticks:BTC-USD"
        assert received[0]["_seq"] == 1
        await manager.stop()

    @pytest.mark.asyncio
    async def test_ingest_batch(self, manager):
        """Test ingesting a batch of ticks."""
        await manager.start()
        ticks = [
            {"symbol": "BTC-USD", "price": 42150.0, "volume": 1.0},
            {"symbol": "ETH-USD", "price": 2250.0, "volume": 10.0},
            {"symbol": "SOL-USD", "price": 100.0, "volume": 100.0},
        ]
        await manager.ingest_batch(table="ticks", ticks=ticks)

        stats = manager.get_stats()
        assert stats["ticks_received"] == 3
        assert stats["buffer_sizes"]["ticks"] == 3
        await manager.stop()

    @pytest.mark.asyncio
    async def test_batch_publishes_to_channels(self, manager):
        """Test that batch ingestion publishes to per-symbol channels."""
        await manager.start()

        btc_received = []
        eth_received = []

        async def on_btc(msg):
            btc_received.append(msg)

        async def on_eth(msg):
            eth_received.append(msg)

        await manager._pubsub.subscribe("ticks:BTC-USD", on_btc, "btc_sub")
        await manager._pubsub.subscribe("ticks:ETH-USD", on_eth, "eth_sub")

        ticks = [
            {"symbol": "BTC-USD", "price": 42150.0},
            {"symbol": "ETH-USD", "price": 2250.0},
            {"symbol": "BTC-USD", "price": 42160.0},
        ]
        await manager.ingest_batch(table="ticks", ticks=ticks)

        assert len(btc_received) == 2
        assert len(eth_received) == 1
        assert btc_received[0]["price"] == 42150.0
        assert btc_received[1]["price"] == 42160.0
        await manager.stop()

    @pytest.mark.asyncio
    async def test_latest_quotes(self, manager):
        """Test that latest quotes are cached."""
        await manager.start()
        await manager.ingest_tick(table="ticks", symbol="BTC-USD", price=42150.50)
        await manager.ingest_tick(table="ticks", symbol="BTC-USD", price=42200.00)

        quote = manager.get_latest_quote("ticks", "BTC-USD")
        assert quote is not None
        assert quote["price"] == 42200.00
        await manager.stop()

    @pytest.mark.asyncio
    async def test_get_all_quotes(self, manager):
        """Test getting all quotes for a table."""
        await manager.start()
        await manager.ingest_tick(table="ticks", symbol="BTC-USD", price=42150.0)
        await manager.ingest_tick(table="ticks", symbol="ETH-USD", price=2250.0)
        await manager.ingest_tick(table="ticks", symbol="SOL-USD", price=100.0)

        quotes = manager.get_all_quotes("ticks")
        assert len(quotes) == 3
        assert "BTC-USD" in quotes
        assert "ETH-USD" in quotes
        assert "SOL-USD" in quotes
        await manager.stop()

    @pytest.mark.asyncio
    async def test_flush_to_database(self, manager, temp_db):
        """Test flushing buffered data to database."""
        manager.set_database(temp_db)
        await manager.start()

        for i in range(10):
            await manager.ingest_tick(
                table="test_ticks",
                symbol="BTC-USD",
                price=42000.0 + i,
                timestamp=1704067200000000000 + i * 1000000000,
            )

        await manager._flush_table("test_ticks")

        assert temp_db.has_table("test_ticks")
        table = temp_db["test_ticks"]
        assert table.num_rows == 10

        prices = table["price"].to_numpy()
        assert prices[0] == pytest.approx(42000.0)
        assert prices[9] == pytest.approx(42009.0)
        await manager.stop()

    @pytest.mark.asyncio
    async def test_append_to_existing_table(self, manager, temp_db):
        """Test appending to an existing table."""
        manager.set_database(temp_db)
        await manager.start()

        for i in range(5):
            await manager.ingest_tick(
                table="test_ticks",
                symbol="BTC-USD",
                price=42000.0 + i,
                timestamp=1704067200000000000 + i * 1000000000,
            )
        await manager._flush_table("test_ticks")

        for i in range(5, 10):
            await manager.ingest_tick(
                table="test_ticks",
                symbol="BTC-USD",
                price=42000.0 + i,
                timestamp=1704067200000000000 + i * 1000000000,
            )
        await manager._flush_table("test_ticks")

        table = temp_db["test_ticks"]
        assert table.num_rows == 10

        prices = table["price"].to_numpy()
        for i in range(10):
            assert prices[i] == pytest.approx(42000.0 + i)
        await manager.stop()

    @pytest.mark.asyncio
    async def test_auto_flush_on_buffer_full(self, manager, temp_db):
        """Test that buffer auto-flushes when full."""
        manager.set_database(temp_db)
        manager.max_buffer_size = 50
        await manager.start()

        for i in range(60):
            await manager.ingest_tick(
                table="test_ticks",
                symbol="BTC-USD",
                price=42000.0 + i,
                timestamp=1704067200000000000 + i * 1000000000,
            )

        stats = manager.get_stats()
        assert stats["ticks_flushed"] >= 50
        await manager.stop()

    @pytest.mark.asyncio
    async def test_start_stop(self, manager, temp_db):
        """Test starting and stopping the streaming manager."""
        manager.set_database(temp_db)

        await manager.start()
        assert manager._running

        await manager.ingest_tick(table="ticks", symbol="BTC-USD", price=42150.0)

        await manager.stop()
        assert not manager._running

    @pytest.mark.asyncio
    async def test_stats_includes_pubsub(self, manager):
        """Test that stats include pubsub info."""
        await manager.start()

        for i in range(10):
            await manager.ingest_tick(
                table="ticks",
                symbol=f"SYM-{i % 3}",
                price=100.0 + i,
            )

        stats = manager.get_stats()

        assert stats["ticks_received"] == 10
        assert "buffer_sizes" in stats
        assert "subscriber_counts" in stats
        assert stats["latest_quotes"] == 3
        assert "pubsub" in stats
        assert stats["pubsub"]["backend"] == "in_memory"
        assert stats["pubsub"]["messages_published"] == 10
        await manager.stop()


class TestStreamingManagerNoPubSub:
    """Tests for StreamingManager without PubSub (backward compat)."""

    @pytest.fixture
    def manager(self):
        return StreamingManager(
            flush_interval=0.1,
            max_buffer_size=100,
            batch_broadcast_interval=0.01,
            pubsub=None,
        )

    @pytest.mark.asyncio
    async def test_works_without_pubsub(self, manager):
        """Test that streaming still works without a PubSub backend."""
        await manager.start()

        await manager.ingest_tick(table="ticks", symbol="BTC-USD", price=42150.0)

        stats = manager.get_stats()
        assert stats["ticks_received"] == 1
        assert "pubsub" not in stats

        await manager.stop()


class TestStreamingPerformance:
    """Performance tests for streaming with PubSub."""

    @pytest.mark.asyncio
    async def test_high_throughput_ingestion(self):
        """Test high-throughput tick ingestion."""
        pubsub = InMemoryPubSub(max_buffer_per_channel=100000)
        manager = StreamingManager(
            flush_interval=10.0,
            max_buffer_size=100000,
            pubsub=pubsub,
        )
        await manager.start()

        num_ticks = 10000
        start = time.time()

        for i in range(num_ticks):
            await manager.ingest_tick(
                table="benchmark",
                symbol=f"SYM-{i % 100}",
                price=100.0 + (i % 1000) * 0.01,
                timestamp=1704067200000000000 + i * 1000,
            )

        elapsed = time.time() - start
        ticks_per_second = num_ticks / elapsed

        print(f"\nIngestion w/ PubSub: {ticks_per_second:.0f} ticks/second")
        print(f"  Total ticks: {num_ticks}")
        print(f"  Elapsed: {elapsed:.3f}s")

        assert ticks_per_second > 5000

        stats = manager.get_stats()
        assert stats["ticks_received"] == num_ticks
        assert stats["pubsub"]["messages_published"] == num_ticks
        await manager.stop()

    @pytest.mark.asyncio
    async def test_batch_vs_single_ingestion(self):
        """Compare batch vs single tick ingestion performance."""
        pubsub = InMemoryPubSub(max_buffer_per_channel=100000)
        manager = StreamingManager(
            flush_interval=10.0,
            max_buffer_size=100000,
            pubsub=pubsub,
        )
        await manager.start()

        num_ticks = 5000

        # Single tick ingestion
        start = time.time()
        for i in range(num_ticks):
            await manager.ingest_tick(
                table="single",
                symbol="BTC-USD",
                price=42000.0 + i * 0.01,
            )
        single_elapsed = time.time() - start

        # Batch ingestion
        batch_size = 100
        batches = num_ticks // batch_size
        start = time.time()
        for b in range(batches):
            ticks = [
                {"symbol": "BTC-USD", "price": 42000.0 + (b * batch_size + i) * 0.01}
                for i in range(batch_size)
            ]
            await manager.ingest_batch(table="batch", ticks=ticks)
        batch_elapsed = time.time() - start

        print(f"\nSingle ingestion: {num_ticks / single_elapsed:.0f} ticks/second")
        print(f"Batch ingestion: {num_ticks / batch_elapsed:.0f} ticks/second")
        print(f"Batch speedup: {single_elapsed / batch_elapsed:.1f}x")

        assert batch_elapsed <= single_elapsed * 1.5
        await manager.stop()


class TestMultipleSymbols:
    """Tests for multi-symbol streaming."""

    @pytest.mark.asyncio
    async def test_multiple_symbols_separate_quotes(self):
        pubsub = InMemoryPubSub()
        manager = StreamingManager(pubsub=pubsub)
        await manager.start()

        symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOT-USD"]

        for i, symbol in enumerate(symbols):
            await manager.ingest_tick(
                table="ticks",
                symbol=symbol,
                price=1000.0 * (i + 1),
            )

        for i, symbol in enumerate(symbols):
            quote = manager.get_latest_quote("ticks", symbol)
            assert quote is not None
            assert quote["price"] == pytest.approx(1000.0 * (i + 1))

        await manager.stop()

    @pytest.mark.asyncio
    async def test_pubsub_channels_per_symbol(self):
        """Test that each symbol gets its own PubSub channel."""
        pubsub = InMemoryPubSub()
        manager = StreamingManager(pubsub=pubsub)
        await manager.start()

        btc_msgs = []
        eth_msgs = []

        async def on_btc(msg):
            btc_msgs.append(msg)

        async def on_eth(msg):
            eth_msgs.append(msg)

        await pubsub.subscribe("ticks:BTC-USD", on_btc, "btc")
        await pubsub.subscribe("ticks:ETH-USD", on_eth, "eth")

        await manager.ingest_tick(table="ticks", symbol="BTC-USD", price=42000.0)
        await manager.ingest_tick(table="ticks", symbol="ETH-USD", price=2200.0)
        await manager.ingest_tick(table="ticks", symbol="BTC-USD", price=42100.0)

        assert len(btc_msgs) == 2
        assert len(eth_msgs) == 1
        await manager.stop()


class TestTimestamps:
    """Tests for timestamp handling."""

    @pytest.mark.asyncio
    async def test_auto_timestamp(self):
        pubsub = InMemoryPubSub()
        manager = StreamingManager(pubsub=pubsub)
        await manager.start()

        before = int(datetime.now(timezone.utc).timestamp() * 1e9)
        await manager.ingest_tick(table="ticks", symbol="BTC-USD", price=42000.0)
        after = int(datetime.now(timezone.utc).timestamp() * 1e9)

        quote = manager.get_latest_quote("ticks", "BTC-USD")
        assert before <= quote["timestamp"] <= after
        await manager.stop()

    @pytest.mark.asyncio
    async def test_explicit_timestamp(self):
        pubsub = InMemoryPubSub()
        manager = StreamingManager(pubsub=pubsub)
        await manager.start()

        ts = 1704067200000000000
        await manager.ingest_tick(
            table="ticks",
            symbol="BTC-USD",
            price=42000.0,
            timestamp=ts,
        )

        quote = manager.get_latest_quote("ticks", "BTC-USD")
        assert quote["timestamp"] == ts
        await manager.stop()

    @pytest.mark.asyncio
    async def test_timestamp_ordering_in_flush(self, temp_dir):
        pubsub = InMemoryPubSub()
        manager = StreamingManager(pubsub=pubsub)
        db = wdb.Database(temp_dir)
        manager.set_database(db)
        await manager.start()

        timestamps = [3000, 1000, 4000, 2000, 5000]
        for ts in timestamps:
            await manager.ingest_tick(
                table="ticks",
                symbol="BTC-USD",
                price=42000.0,
                timestamp=ts,
            )

        await manager._flush_table("ticks")

        table = db["ticks"]
        stored_ts = table["timestamp"].to_numpy()
        np.testing.assert_array_equal(stored_ts, timestamps)
        await manager.stop()
