"""Tests for WayyDB PubSub abstraction layer."""

import asyncio
import time

import pytest

import sys
sys.path.insert(0, str(__file__).replace("/tests/python/test_pubsub.py", ""))
from api.pubsub import InMemoryPubSub, RedisPubSub, create_pubsub, Message


class TestInMemoryPubSub:
    """Tests for InMemoryPubSub backend."""

    @pytest.fixture
    def pubsub(self):
        return InMemoryPubSub(max_buffer_per_channel=100)

    @pytest.mark.asyncio
    async def test_start_stop(self, pubsub):
        await pubsub.start()
        assert pubsub._running
        await pubsub.stop()
        assert not pubsub._running

    @pytest.mark.asyncio
    async def test_publish_returns_sequence(self, pubsub):
        await pubsub.start()
        seq1 = await pubsub.publish("ticks:AAPL", {"price": 150.0})
        seq2 = await pubsub.publish("ticks:AAPL", {"price": 151.0})
        assert seq1 == 1
        assert seq2 == 2
        await pubsub.stop()

    @pytest.mark.asyncio
    async def test_sequence_per_channel(self, pubsub):
        await pubsub.start()
        seq_a = await pubsub.publish("ticks:AAPL", {"price": 150.0})
        seq_m = await pubsub.publish("ticks:MSFT", {"price": 380.0})
        assert seq_a == 1
        assert seq_m == 1  # Separate sequence per channel
        await pubsub.stop()

    @pytest.mark.asyncio
    async def test_subscribe_receives_messages(self, pubsub):
        await pubsub.start()
        received = []

        async def callback(msg):
            received.append(msg)

        await pubsub.subscribe("ticks:AAPL", callback, "test_sub")
        await pubsub.publish("ticks:AAPL", {"price": 150.0})
        await pubsub.publish("ticks:AAPL", {"price": 151.0})

        assert len(received) == 2
        assert received[0]["price"] == 150.0
        assert received[1]["price"] == 151.0
        # Messages include metadata
        assert received[0]["_seq"] == 1
        assert received[0]["_channel"] == "ticks:AAPL"
        await pubsub.stop()

    @pytest.mark.asyncio
    async def test_subscribe_wildcard(self, pubsub):
        await pubsub.start()
        received = []

        async def callback(msg):
            received.append(msg)

        await pubsub.subscribe("ticks:*", callback, "wildcard_sub")
        await pubsub.publish("ticks:AAPL", {"symbol": "AAPL"})
        await pubsub.publish("ticks:MSFT", {"symbol": "MSFT"})
        await pubsub.publish("quotes:AAPL", {"symbol": "AAPL"})  # Not ticks:*

        assert len(received) == 2
        assert received[0]["symbol"] == "AAPL"
        assert received[1]["symbol"] == "MSFT"
        await pubsub.stop()

    @pytest.mark.asyncio
    async def test_unsubscribe(self, pubsub):
        await pubsub.start()
        received = []

        async def callback(msg):
            received.append(msg)

        await pubsub.subscribe("ticks:AAPL", callback, "test_sub")
        await pubsub.publish("ticks:AAPL", {"price": 150.0})
        assert len(received) == 1

        await pubsub.unsubscribe("ticks:AAPL", "test_sub")
        await pubsub.publish("ticks:AAPL", {"price": 151.0})
        assert len(received) == 1  # No new message
        await pubsub.stop()

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, pubsub):
        await pubsub.start()
        received_a = []
        received_b = []

        async def cb_a(msg):
            received_a.append(msg)

        async def cb_b(msg):
            received_b.append(msg)

        await pubsub.subscribe("ticks:AAPL", cb_a, "sub_a")
        await pubsub.subscribe("ticks:AAPL", cb_b, "sub_b")
        await pubsub.publish("ticks:AAPL", {"price": 150.0})

        assert len(received_a) == 1
        assert len(received_b) == 1
        await pubsub.stop()

    @pytest.mark.asyncio
    async def test_backpressure_buffer_overflow(self):
        pubsub = InMemoryPubSub(max_buffer_per_channel=5)
        await pubsub.start()

        for i in range(10):
            await pubsub.publish("ticks:AAPL", {"price": float(i)})

        # Buffer should only keep last 5
        buf = pubsub.get_channel_buffer("ticks:AAPL")
        assert len(buf) == 5
        assert buf[0].data["price"] == 5.0  # Oldest kept
        assert buf[-1].data["price"] == 9.0  # Newest

        stats = pubsub.get_stats()
        assert stats["messages_dropped"] == 5
        await pubsub.stop()

    @pytest.mark.asyncio
    async def test_channel_buffer_replay(self, pubsub):
        await pubsub.start()

        for i in range(5):
            await pubsub.publish("ticks:AAPL", {"price": float(i)})

        # Replay from sequence 0 (all messages)
        all_msgs = pubsub.get_channel_buffer("ticks:AAPL", since_seq=0)
        assert len(all_msgs) == 5

        # Replay from sequence 3 (only seq 4 and 5)
        recent = pubsub.get_channel_buffer("ticks:AAPL", since_seq=3)
        assert len(recent) == 2
        assert recent[0].sequence == 4
        await pubsub.stop()

    @pytest.mark.asyncio
    async def test_dead_subscriber_removal(self, pubsub):
        await pubsub.start()

        async def bad_callback(msg):
            raise ConnectionError("WebSocket closed")

        received_good = []

        async def good_callback(msg):
            received_good.append(msg)

        await pubsub.subscribe("ticks:AAPL", bad_callback, "bad_sub")
        await pubsub.subscribe("ticks:AAPL", good_callback, "good_sub")

        await pubsub.publish("ticks:AAPL", {"price": 150.0})

        # Bad subscriber should be removed, good one still works
        assert len(received_good) == 1
        stats = pubsub.get_stats()
        assert stats["active_subscriptions"] == 1
        await pubsub.stop()

    @pytest.mark.asyncio
    async def test_publish_batch(self, pubsub):
        await pubsub.start()
        received = []

        async def callback(msg):
            received.append(msg)

        await pubsub.subscribe("ticks:AAPL", callback, "batch_sub")
        messages = [{"price": float(i)} for i in range(5)]
        last_seq = await pubsub.publish_batch("ticks:AAPL", messages)

        assert last_seq == 5
        assert len(received) == 5
        await pubsub.stop()

    @pytest.mark.asyncio
    async def test_stats(self, pubsub):
        await pubsub.start()

        async def noop(msg):
            pass

        await pubsub.subscribe("ticks:AAPL", noop, "s1")
        await pubsub.subscribe("ticks:MSFT", noop, "s2")
        await pubsub.publish("ticks:AAPL", {"price": 150.0})
        await pubsub.publish("ticks:MSFT", {"price": 380.0})

        stats = pubsub.get_stats()
        assert stats["backend"] == "in_memory"
        assert stats["messages_published"] == 2
        assert stats["messages_delivered"] == 2
        assert stats["active_subscriptions"] == 2
        assert stats["channels"] == 2
        await pubsub.stop()


class TestCreatePubSub:
    """Tests for the factory function."""

    def test_creates_inmemory_by_default(self):
        ps = create_pubsub(None)
        assert isinstance(ps, InMemoryPubSub)

    def test_creates_inmemory_for_empty_string(self):
        ps = create_pubsub("")
        assert isinstance(ps, InMemoryPubSub)

    def test_creates_redis_with_url(self):
        ps = create_pubsub("redis://localhost:6379")
        assert isinstance(ps, RedisPubSub)


class TestPubSubPerformance:
    """Performance tests for InMemoryPubSub."""

    @pytest.mark.asyncio
    async def test_publish_throughput(self):
        """Test raw publish throughput without subscribers."""
        pubsub = InMemoryPubSub(max_buffer_per_channel=100000)
        await pubsub.start()

        num_msgs = 50000
        start = time.time()

        for i in range(num_msgs):
            await pubsub.publish(f"ticks:SYM-{i % 100}", {"price": float(i)})

        elapsed = time.time() - start
        rate = num_msgs / elapsed

        print(f"\nPublish throughput (no subscribers): {rate:.0f} msgs/sec")
        assert rate > 50000  # Should handle at least 50K msgs/sec

        await pubsub.stop()

    @pytest.mark.asyncio
    async def test_publish_with_subscribers_throughput(self):
        """Test publish throughput with active subscribers."""
        pubsub = InMemoryPubSub(max_buffer_per_channel=100000)
        await pubsub.start()

        counter = {"count": 0}

        async def counting_callback(msg):
            counter["count"] += 1

        # Subscribe to 10 channels
        for i in range(10):
            await pubsub.subscribe(f"ticks:SYM-{i}", counting_callback, f"sub_{i}")

        num_msgs = 10000
        start = time.time()

        for i in range(num_msgs):
            await pubsub.publish(f"ticks:SYM-{i % 10}", {"price": float(i)})

        elapsed = time.time() - start
        rate = num_msgs / elapsed

        print(f"\nPublish throughput (10 subscribers): {rate:.0f} msgs/sec")
        print(f"Messages delivered: {counter['count']}")
        assert rate > 5000
        assert counter["count"] == num_msgs

        await pubsub.stop()

    @pytest.mark.asyncio
    async def test_many_channels(self):
        """Test performance with many channels."""
        pubsub = InMemoryPubSub(max_buffer_per_channel=1000)
        await pubsub.start()

        num_channels = 1000
        msgs_per_channel = 10

        start = time.time()
        for ch in range(num_channels):
            for m in range(msgs_per_channel):
                await pubsub.publish(f"ticks:SYM-{ch}", {"price": float(m)})

        elapsed = time.time() - start
        total = num_channels * msgs_per_channel
        rate = total / elapsed

        print(f"\nMany channels throughput: {rate:.0f} msgs/sec ({num_channels} channels)")

        stats = pubsub.get_stats()
        assert stats["channels"] == num_channels
        assert stats["messages_published"] == total

        await pubsub.stop()

    @pytest.mark.asyncio
    async def test_wildcard_subscriber_performance(self):
        """Test wildcard subscriber doesn't destroy performance."""
        pubsub = InMemoryPubSub(max_buffer_per_channel=100000)
        await pubsub.start()

        counter = {"count": 0}

        async def callback(msg):
            counter["count"] += 1

        # Wildcard subscriber for all ticks
        await pubsub.subscribe("ticks:*", callback, "wildcard")

        num_msgs = 10000
        start = time.time()

        for i in range(num_msgs):
            await pubsub.publish(f"ticks:SYM-{i % 50}", {"price": float(i)})

        elapsed = time.time() - start
        rate = num_msgs / elapsed

        print(f"\nWildcard subscriber throughput: {rate:.0f} msgs/sec")
        assert counter["count"] == num_msgs
        assert rate > 3000

        await pubsub.stop()


class TestPubSubStress:
    """Stress tests for PubSub under adverse conditions."""

    @pytest.mark.asyncio
    async def test_concurrent_publish_subscribe(self):
        """Test concurrent publishers and subscribers."""
        pubsub = InMemoryPubSub(max_buffer_per_channel=10000)
        await pubsub.start()

        results = {"total_received": 0}

        async def subscriber_cb(msg):
            results["total_received"] += 1

        # Register 5 subscribers on different channels
        for i in range(5):
            await pubsub.subscribe(f"ticks:SYM-{i}", subscriber_cb, f"sub_{i}")

        # Concurrent publishers
        async def publisher(channel_idx, count):
            for j in range(count):
                await pubsub.publish(f"ticks:SYM-{channel_idx}", {"price": float(j)})

        tasks = [publisher(i, 200) for i in range(5)]
        await asyncio.gather(*tasks)

        assert results["total_received"] == 1000  # 5 channels x 200 msgs
        await pubsub.stop()

    @pytest.mark.asyncio
    async def test_subscribe_unsubscribe_churn(self):
        """Test rapid subscribe/unsubscribe cycles."""
        pubsub = InMemoryPubSub()
        await pubsub.start()

        async def noop(msg):
            pass

        for cycle in range(100):
            sub_id = f"churn_{cycle}"
            await pubsub.subscribe("ticks:AAPL", noop, sub_id)
            await pubsub.publish("ticks:AAPL", {"cycle": cycle})
            await pubsub.unsubscribe("ticks:AAPL", sub_id)

        stats = pubsub.get_stats()
        assert stats["active_subscriptions"] == 0
        assert stats["messages_published"] == 100
        await pubsub.stop()

    @pytest.mark.asyncio
    async def test_buffer_overflow_under_load(self):
        """Test backpressure behavior under high load."""
        pubsub = InMemoryPubSub(max_buffer_per_channel=100)
        await pubsub.start()

        # Publish 1000 messages to a channel with buffer size 100
        for i in range(1000):
            await pubsub.publish("ticks:AAPL", {"price": float(i)})

        stats = pubsub.get_stats()
        buf = pubsub.get_channel_buffer("ticks:AAPL")

        assert len(buf) == 100  # Buffer capped
        assert stats["messages_published"] == 1000
        assert stats["messages_dropped"] == 900
        # Buffer should have the latest 100 messages
        assert buf[-1].data["price"] == 999.0
        await pubsub.stop()
