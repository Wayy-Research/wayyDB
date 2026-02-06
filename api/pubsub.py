"""
WayyDB PubSub Abstraction Layer

Provides a pluggable pub/sub transport for real-time tick distribution.
Two backends:
  - InMemoryPubSub: Default, zero dependencies, single-process
  - RedisPubSub: Optional, requires redis-py, multi-process capable

Configure via REDIS_URL environment variable:
  - Not set or empty: uses InMemoryPubSub
  - Set to redis://...: uses RedisPubSub

Channel naming convention:
  ticks:{symbol}         - Trade ticks for a symbol
  quotes:{symbol}        - Quote updates for a symbol
  ticks:*                - All trade ticks
  {table}:{symbol}       - Generic table:symbol pattern
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Type alias for async callback
AsyncCallback = Callable[[dict], Coroutine[Any, Any, None]]


@dataclass
class Message:
    """A pub/sub message with metadata."""
    channel: str
    data: dict
    sequence: int
    timestamp: float = field(default_factory=time.time)


class PubSubBackend(ABC):
    """Abstract pub/sub backend interface.

    Implementations must provide publish, subscribe, and unsubscribe.
    This abstraction allows swapping between in-memory, Redis, NATS, etc.
    """

    @abstractmethod
    async def publish(self, channel: str, data: dict) -> int:
        """Publish a message to a channel.

        Args:
            channel: Channel name (e.g., "ticks:AAPL")
            data: Message payload

        Returns:
            Sequence number of the published message
        """
        ...

    @abstractmethod
    async def subscribe(
        self,
        channel: str,
        callback: AsyncCallback,
        subscriber_id: str = "",
    ) -> None:
        """Subscribe to a channel with a callback.

        Args:
            channel: Channel name or pattern (e.g., "ticks:AAPL" or "ticks:*")
            callback: Async function called with each message dict
            subscriber_id: Unique identifier for this subscriber
        """
        ...

    @abstractmethod
    async def unsubscribe(self, channel: str, subscriber_id: str = "") -> None:
        """Unsubscribe from a channel.

        Args:
            channel: Channel name or pattern
            subscriber_id: The subscriber to remove
        """
        ...

    @abstractmethod
    async def publish_batch(self, channel: str, messages: List[dict]) -> int:
        """Publish a batch of messages to a channel.

        Args:
            channel: Channel name
            messages: List of message payloads

        Returns:
            Sequence number of the last message
        """
        ...

    @abstractmethod
    def get_stats(self) -> dict:
        """Get pub/sub statistics."""
        ...

    @abstractmethod
    async def start(self) -> None:
        """Start the backend (connect, initialize)."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the backend (disconnect, cleanup)."""
        ...


class InMemoryPubSub(PubSubBackend):
    """In-process pub/sub using asyncio.

    Features:
    - Channel-based routing with wildcard support
    - Per-channel sequence numbers
    - Ring buffer for backpressure (drops oldest on overflow)
    - Concurrent broadcast via asyncio.gather
    - Message replay from buffer
    """

    def __init__(
        self,
        max_buffer_per_channel: int = 10000,
        broadcast_timeout: float = 5.0,
    ):
        self._subscribers: Dict[str, Dict[str, AsyncCallback]] = defaultdict(dict)
        self._sequence: Dict[str, int] = defaultdict(int)
        self._buffers: Dict[str, deque] = {}
        self._max_buffer = max_buffer_per_channel
        self._broadcast_timeout = broadcast_timeout
        self._stats = {
            "messages_published": 0,
            "messages_delivered": 0,
            "messages_dropped": 0,
            "active_subscriptions": 0,
            "channels": 0,
        }
        self._running = False

    async def start(self) -> None:
        self._running = True
        logger.info("InMemoryPubSub started")

    async def stop(self) -> None:
        self._running = False
        self._subscribers.clear()
        self._buffers.clear()
        logger.info("InMemoryPubSub stopped")

    async def publish(self, channel: str, data: dict) -> int:
        self._sequence[channel] += 1
        seq = self._sequence[channel]

        msg = Message(channel=channel, data=data, sequence=seq)

        # Buffer the message
        if channel not in self._buffers:
            self._buffers[channel] = deque(maxlen=self._max_buffer)
        buf = self._buffers[channel]
        if len(buf) >= self._max_buffer:
            self._stats["messages_dropped"] += 1
        buf.append(msg)

        self._stats["messages_published"] += 1
        self._stats["channels"] = len(self._buffers)

        # Deliver to subscribers
        await self._deliver(channel, data, seq)

        return seq

    async def publish_batch(self, channel: str, messages: List[dict]) -> int:
        last_seq = 0
        for data in messages:
            last_seq = await self.publish(channel, data)
        return last_seq

    async def subscribe(
        self,
        channel: str,
        callback: AsyncCallback,
        subscriber_id: str = "",
    ) -> None:
        if not subscriber_id:
            subscriber_id = f"sub_{id(callback)}"

        self._subscribers[channel][subscriber_id] = callback
        self._stats["active_subscriptions"] = sum(
            len(subs) for subs in self._subscribers.values()
        )
        logger.debug(f"Subscribed {subscriber_id} to {channel}")

    async def unsubscribe(self, channel: str, subscriber_id: str = "") -> None:
        if channel in self._subscribers:
            if subscriber_id and subscriber_id in self._subscribers[channel]:
                del self._subscribers[channel][subscriber_id]
            elif not subscriber_id:
                self._subscribers[channel].clear()

            if not self._subscribers[channel]:
                del self._subscribers[channel]

        self._stats["active_subscriptions"] = sum(
            len(subs) for subs in self._subscribers.values()
        )

    async def _deliver(self, channel: str, data: dict, sequence: int) -> None:
        """Deliver message to all matching subscribers concurrently."""
        enriched = {**data, "_seq": sequence, "_channel": channel}

        # Collect all matching callbacks
        callbacks: List[AsyncCallback] = []

        # Exact match subscribers
        if channel in self._subscribers:
            callbacks.extend(self._subscribers[channel].values())

        # Wildcard subscribers (e.g., "ticks:*" matches "ticks:AAPL")
        for pattern, subs in self._subscribers.items():
            if pattern.endswith(":*"):
                prefix = pattern[:-1]  # "ticks:"
                if channel.startswith(prefix) and pattern != channel:
                    callbacks.extend(subs.values())

        if not callbacks:
            return

        # Concurrent delivery with timeout
        dead_callbacks: List[AsyncCallback] = []

        async def safe_deliver(cb: AsyncCallback) -> None:
            try:
                await asyncio.wait_for(cb(enriched), timeout=self._broadcast_timeout)
                self._stats["messages_delivered"] += 1
            except asyncio.TimeoutError:
                logger.warning(f"Subscriber timed out on {channel}")
                dead_callbacks.append(cb)
            except Exception:
                dead_callbacks.append(cb)

        await asyncio.gather(*(safe_deliver(cb) for cb in callbacks))

        # Remove dead subscribers
        for dead_cb in dead_callbacks:
            for pattern, subs in list(self._subscribers.items()):
                to_remove = [
                    sid for sid, cb in subs.items() if cb is dead_cb
                ]
                for sid in to_remove:
                    del subs[sid]
                    logger.debug(f"Removed dead subscriber {sid} from {pattern}")

        if dead_callbacks:
            self._stats["active_subscriptions"] = sum(
                len(subs) for subs in self._subscribers.values()
            )

    def get_channel_buffer(self, channel: str, since_seq: int = 0) -> List[Message]:
        """Get buffered messages for replay.

        Args:
            channel: Channel name
            since_seq: Only return messages with sequence > since_seq

        Returns:
            List of messages for replay
        """
        if channel not in self._buffers:
            return []
        return [m for m in self._buffers[channel] if m.sequence > since_seq]

    def get_stats(self) -> dict:
        return {
            "backend": "in_memory",
            **self._stats,
            "buffer_sizes": {ch: len(buf) for ch, buf in self._buffers.items()},
        }


class RedisPubSub(PubSubBackend):
    """Redis-backed pub/sub for multi-process deployments.

    Uses Redis pub/sub for real-time delivery and Redis Streams
    for message persistence and replay.

    Requires: pip install redis[hiredis]
    Configure via REDIS_URL environment variable.
    """

    def __init__(self, redis_url: str, max_stream_len: int = 100000):
        self._redis_url = redis_url
        self._max_stream_len = max_stream_len
        self._redis = None
        self._pubsub = None
        self._subscribers: Dict[str, Dict[str, AsyncCallback]] = defaultdict(dict)
        self._sequence: Dict[str, int] = defaultdict(int)
        self._listener_task: Optional[asyncio.Task] = None
        self._running = False
        self._stats = {
            "messages_published": 0,
            "messages_delivered": 0,
            "messages_dropped": 0,
            "active_subscriptions": 0,
            "channels": 0,
            "redis_connected": False,
        }

    async def start(self) -> None:
        try:
            import redis.asyncio as aioredis
        except ImportError:
            raise ImportError(
                "redis package required for RedisPubSub. "
                "Install with: pip install redis[hiredis]"
            )

        self._redis = aioredis.from_url(
            self._redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            retry_on_timeout=True,
        )

        # Test connection
        await self._redis.ping()
        self._stats["redis_connected"] = True

        self._pubsub = self._redis.pubsub()
        self._running = True
        self._listener_task = asyncio.create_task(self._listen_loop())

        logger.info(f"RedisPubSub connected to {self._redis_url}")

    async def stop(self) -> None:
        self._running = False

        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass

        if self._pubsub:
            await self._pubsub.unsubscribe()
            await self._pubsub.close()

        if self._redis:
            await self._redis.close()

        self._stats["redis_connected"] = False
        logger.info("RedisPubSub stopped")

    async def publish(self, channel: str, data: dict) -> int:
        import json

        self._sequence[channel] += 1
        seq = self._sequence[channel]

        enriched = {**data, "_seq": seq, "_ts": time.time()}
        payload = json.dumps(enriched)

        # Publish to Redis pub/sub channel
        await self._redis.publish(f"wayy:{channel}", payload)

        # Also write to Redis Stream for persistence/replay
        stream_key = f"wayy:stream:{channel}"
        await self._redis.xadd(
            stream_key,
            {"data": payload},
            maxlen=self._max_stream_len,
        )

        self._stats["messages_published"] += 1
        return seq

    async def publish_batch(self, channel: str, messages: List[dict]) -> int:
        import json

        pipe = self._redis.pipeline()
        last_seq = 0

        for data in messages:
            self._sequence[channel] += 1
            seq = self._sequence[channel]
            last_seq = seq

            enriched = {**data, "_seq": seq, "_ts": time.time()}
            payload = json.dumps(enriched)

            pipe.publish(f"wayy:{channel}", payload)

            stream_key = f"wayy:stream:{channel}"
            pipe.xadd(stream_key, {"data": payload}, maxlen=self._max_stream_len)

        await pipe.execute()
        self._stats["messages_published"] += len(messages)
        return last_seq

    async def subscribe(
        self,
        channel: str,
        callback: AsyncCallback,
        subscriber_id: str = "",
    ) -> None:
        if not subscriber_id:
            subscriber_id = f"sub_{id(callback)}"

        is_new_channel = channel not in self._subscribers or not self._subscribers[channel]
        self._subscribers[channel][subscriber_id] = callback

        if is_new_channel and self._pubsub:
            if channel.endswith(":*"):
                await self._pubsub.psubscribe(f"wayy:{channel}")
            else:
                await self._pubsub.subscribe(f"wayy:{channel}")

        self._stats["active_subscriptions"] = sum(
            len(subs) for subs in self._subscribers.values()
        )
        self._stats["channels"] = len(self._subscribers)

    async def unsubscribe(self, channel: str, subscriber_id: str = "") -> None:
        if channel in self._subscribers:
            if subscriber_id and subscriber_id in self._subscribers[channel]:
                del self._subscribers[channel][subscriber_id]
            elif not subscriber_id:
                self._subscribers[channel].clear()

            if not self._subscribers[channel]:
                del self._subscribers[channel]
                if self._pubsub:
                    if channel.endswith(":*"):
                        await self._pubsub.punsubscribe(f"wayy:{channel}")
                    else:
                        await self._pubsub.unsubscribe(f"wayy:{channel}")

        self._stats["active_subscriptions"] = sum(
            len(subs) for subs in self._subscribers.values()
        )

    async def _listen_loop(self) -> None:
        """Background task that listens for Redis pub/sub messages."""
        import json

        while self._running:
            try:
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=0.1
                )
                if message is None:
                    await asyncio.sleep(0.01)
                    continue

                if message["type"] not in ("message", "pmessage"):
                    continue

                raw_channel = message.get("channel", "")
                # Strip "wayy:" prefix
                if raw_channel.startswith("wayy:"):
                    channel = raw_channel[5:]
                else:
                    channel = raw_channel

                data = json.loads(message["data"])

                # Deliver to local subscribers
                await self._deliver_local(channel, data)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Redis listener error: {e}")
                await asyncio.sleep(1.0)

    async def _deliver_local(self, channel: str, data: dict) -> None:
        """Deliver a received message to local subscribers."""
        callbacks: List[AsyncCallback] = []

        if channel in self._subscribers:
            callbacks.extend(self._subscribers[channel].values())

        # Wildcard matching
        for pattern, subs in self._subscribers.items():
            if pattern.endswith(":*"):
                prefix = pattern[:-1]
                if channel.startswith(prefix) and pattern != channel:
                    callbacks.extend(subs.values())

        for cb in callbacks:
            try:
                await asyncio.wait_for(cb(data), timeout=5.0)
                self._stats["messages_delivered"] += 1
            except Exception:
                self._stats["messages_dropped"] += 1

    async def replay(
        self, channel: str, since_id: str = "0-0", count: int = 1000
    ) -> List[dict]:
        """Replay messages from Redis Stream.

        Args:
            channel: Channel name
            since_id: Redis Stream ID to start from
            count: Maximum messages to return

        Returns:
            List of message dicts
        """
        import json

        stream_key = f"wayy:stream:{channel}"
        messages = await self._redis.xrange(stream_key, min=since_id, count=count)

        return [json.loads(entry["data"]) for _id, entry in messages]

    def get_stats(self) -> dict:
        return {
            "backend": "redis",
            "redis_url": self._redis_url.split("@")[-1] if "@" in self._redis_url else self._redis_url,
            **self._stats,
        }


def create_pubsub(redis_url: Optional[str] = None) -> PubSubBackend:
    """Factory function to create the appropriate PubSub backend.

    Args:
        redis_url: Redis URL. If None/empty, uses InMemoryPubSub.

    Returns:
        PubSubBackend instance
    """
    if redis_url:
        logger.info(f"Using RedisPubSub backend")
        return RedisPubSub(redis_url=redis_url)
    else:
        logger.info("Using InMemoryPubSub backend (set REDIS_URL for Redis)")
        return InMemoryPubSub()
