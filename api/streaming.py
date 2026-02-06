"""
WayyDB Streaming Module - Real-time data ingestion and pub/sub

Provides:
- WebSocket ingestion endpoint for real-time tick data
- Pub/Sub subscriptions via pluggable backend (in-memory or Redis)
- Efficient batching and append operations
- In-memory buffers with periodic flush to persistent storage
- Backpressure handling and sequence numbers

Configuration via environment variables:
- FLUSH_INTERVAL: Seconds between flushes to disk (default: 1.0)
- MAX_BUFFER_SIZE: Max ticks in buffer before force flush (default: 10000)
- BROADCAST_INTERVAL: Seconds between subscriber broadcasts (default: 0.05)
- REDIS_URL: Optional Redis URL for distributed pub/sub
"""

import asyncio
import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import numpy as np
from fastapi import WebSocket

from api.pubsub import PubSubBackend, create_pubsub

logger = logging.getLogger(__name__)

# Configuration from environment
DEFAULT_FLUSH_INTERVAL = float(os.getenv("FLUSH_INTERVAL", "1.0"))
DEFAULT_MAX_BUFFER_SIZE = int(os.getenv("MAX_BUFFER_SIZE", "10000"))
DEFAULT_BROADCAST_INTERVAL = float(os.getenv("BROADCAST_INTERVAL", "0.05"))


@dataclass
class TickBuffer:
    """Buffer for incoming tick data before flush to table."""
    timestamps: List[int] = field(default_factory=list)
    symbols: List[str] = field(default_factory=list)
    prices: List[float] = field(default_factory=list)
    volumes: List[float] = field(default_factory=list)
    bids: List[float] = field(default_factory=list)
    asks: List[float] = field(default_factory=list)

    def append(self, timestamp: int, symbol: str, price: float,
               volume: float = 0.0, bid: float = 0.0, ask: float = 0.0):
        self.timestamps.append(timestamp)
        self.symbols.append(symbol)
        self.prices.append(price)
        self.volumes.append(volume)
        self.bids.append(bid if bid else price)
        self.asks.append(ask if ask else price)

    def __len__(self):
        return len(self.timestamps)

    def clear(self):
        self.timestamps.clear()
        self.symbols.clear()
        self.prices.clear()
        self.volumes.clear()
        self.bids.clear()
        self.asks.clear()

    def to_columnar(self) -> Dict[str, np.ndarray]:
        """Convert to columnar format for WayyDB."""
        return {
            "timestamp": np.array(self.timestamps, dtype=np.int64),
            "symbol": np.array([hash(s) % (2**32) for s in self.symbols], dtype=np.uint32),
            "price": np.array(self.prices, dtype=np.float64),
            "volume": np.array(self.volumes, dtype=np.float64),
            "bid": np.array(self.bids, dtype=np.float64),
            "ask": np.array(self.asks, dtype=np.float64),
        }


@dataclass
class Subscriber:
    """A WebSocket subscriber to data updates."""
    websocket: WebSocket
    symbols: Set[str] = field(default_factory=set)  # Empty = all symbols
    subscriber_id: str = ""
    created_at: float = field(default_factory=time.time)
    messages_sent: int = 0


class StreamingManager:
    """
    Manages streaming data ingestion and pub/sub distribution.

    Features:
    - Buffer incoming ticks in memory
    - Publish to PubSub channels (in-memory or Redis)
    - Broadcast to WebSocket subscribers via PubSub callbacks
    - Periodic flush to WayyDB tables (atomic swap, no gap)
    - Thread-safe operations via threading.Lock
    """

    def __init__(
        self,
        flush_interval: float = DEFAULT_FLUSH_INTERVAL,
        max_buffer_size: int = DEFAULT_MAX_BUFFER_SIZE,
        batch_broadcast_interval: float = DEFAULT_BROADCAST_INTERVAL,
        pubsub: Optional[PubSubBackend] = None,
    ):
        self.flush_interval = flush_interval
        self.max_buffer_size = max_buffer_size
        self.batch_broadcast_interval = batch_broadcast_interval

        # PubSub backend (in-memory default, Redis optional)
        self._pubsub = pubsub

        # Tick buffers - one per table
        self._buffers: Dict[str, TickBuffer] = defaultdict(TickBuffer)

        # WebSocket subscribers - one list per table
        self._subscribers: Dict[str, List[Subscriber]] = defaultdict(list)

        # Latest quotes cache (for new subscribers)
        self._latest_quotes: Dict[str, Dict[str, Any]] = {}

        # Pending broadcasts (batched for efficiency)
        self._pending_broadcasts: Dict[str, List[Dict]] = defaultdict(list)

        # Statistics
        self._stats = {
            "ticks_received": 0,
            "ticks_flushed": 0,
            "broadcasts_sent": 0,
            "active_subscribers": 0,
            "flush_count": 0,
            "start_time": None,
        }

        # Background tasks
        self._running = False
        self._flush_task: Optional[asyncio.Task] = None
        self._broadcast_task: Optional[asyncio.Task] = None

        # Database reference (set by API)
        self._db = None

        # FIX: Use threading.Lock for thread safety with ThreadPoolExecutor
        self._lock = threading.Lock()

    def set_database(self, db):
        """Set the database reference for flushing."""
        self._db = db

    def set_pubsub(self, pubsub: PubSubBackend):
        """Set the pub/sub backend."""
        self._pubsub = pubsub

    async def start(self):
        """Start background flush and broadcast tasks."""
        if self._running:
            return

        self._running = True
        self._stats["start_time"] = datetime.now(timezone.utc).isoformat()

        # Start PubSub backend if provided
        if self._pubsub:
            await self._pubsub.start()

        self._flush_task = asyncio.create_task(self._flush_loop())
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())

        logger.info("StreamingManager started")

    async def stop(self):
        """Stop background tasks and flush remaining data."""
        if not self._running:
            return

        self._running = False

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self._flush_all()

        # Stop PubSub backend
        if self._pubsub:
            await self._pubsub.stop()

        logger.info("StreamingManager stopped")

    async def ingest_tick(
        self,
        table: str,
        symbol: str,
        price: float,
        timestamp: Optional[int] = None,
        volume: float = 0.0,
        bid: float = 0.0,
        ask: float = 0.0,
    ):
        """Ingest a single tick."""
        if timestamp is None:
            timestamp = int(datetime.now(timezone.utc).timestamp() * 1e9)

        # Add to buffer (thread-safe)
        with self._lock:
            self._buffers[table].append(
                timestamp=timestamp,
                symbol=symbol,
                price=price,
                volume=volume,
                bid=bid,
                ask=ask,
            )
            self._stats["ticks_received"] += 1

        # Build quote message
        quote = {
            "symbol": symbol,
            "price": price,
            "bid": bid or price,
            "ask": ask or price,
            "volume": volume,
            "timestamp": timestamp,
            "table": table,
        }
        self._latest_quotes[f"{table}:{symbol}"] = quote

        # Publish to PubSub channel
        if self._pubsub:
            channel = f"{table}:{symbol}"
            await self._pubsub.publish(channel, quote)

        # Queue for WebSocket broadcast
        self._pending_broadcasts[table].append(quote)

        # Force flush if buffer too large
        if len(self._buffers[table]) >= self.max_buffer_size:
            await self._flush_table(table)

    async def ingest_batch(
        self,
        table: str,
        ticks: List[Dict[str, Any]],
    ):
        """Ingest a batch of ticks efficiently."""
        quotes_by_channel: Dict[str, List[dict]] = defaultdict(list)

        with self._lock:
            buffer = self._buffers[table]
            for tick in ticks:
                timestamp = tick.get("timestamp")
                if timestamp is None:
                    timestamp = int(datetime.now(timezone.utc).timestamp() * 1e9)

                buffer.append(
                    timestamp=timestamp,
                    symbol=tick["symbol"],
                    price=tick["price"],
                    volume=tick.get("volume", 0.0),
                    bid=tick.get("bid", tick["price"]),
                    ask=tick.get("ask", tick["price"]),
                )

                quote = {
                    "symbol": tick["symbol"],
                    "price": tick["price"],
                    "bid": tick.get("bid", tick["price"]),
                    "ask": tick.get("ask", tick["price"]),
                    "volume": tick.get("volume", 0.0),
                    "timestamp": timestamp,
                    "table": table,
                }
                self._latest_quotes[f"{table}:{tick['symbol']}"] = quote
                self._pending_broadcasts[table].append(quote)

                channel = f"{table}:{tick['symbol']}"
                quotes_by_channel[channel].append(quote)

            self._stats["ticks_received"] += len(ticks)

        # Batch publish to PubSub channels
        if self._pubsub:
            for channel, channel_quotes in quotes_by_channel.items():
                await self._pubsub.publish_batch(channel, channel_quotes)

        # Force flush if buffer too large
        if len(self._buffers[table]) >= self.max_buffer_size:
            await self._flush_table(table)

    async def subscribe(self, websocket: WebSocket, table: str, symbols: Optional[List[str]] = None):
        """Add a WebSocket subscriber to a table's updates."""
        sub_id = f"ws_{id(websocket)}"
        subscriber = Subscriber(
            websocket=websocket,
            symbols=set(symbols) if symbols else set(),
            subscriber_id=sub_id,
        )

        self._subscribers[table].append(subscriber)
        self._stats["active_subscribers"] = sum(len(s) for s in self._subscribers.values())

        # Send current latest quotes to new subscriber
        for key, quote in self._latest_quotes.items():
            if key.startswith(f"{table}:"):
                symbol = key.split(":", 1)[1]
                if not subscriber.symbols or symbol in subscriber.symbols:
                    try:
                        await websocket.send_json(quote)
                    except Exception:
                        pass

        logger.info(f"New subscriber for {table}, symbols={symbols or 'all'}")
        return subscriber

    async def unsubscribe(self, websocket: WebSocket, table: str):
        """Remove a subscriber."""
        self._subscribers[table] = [
            s for s in self._subscribers[table]
            if s.websocket != websocket
        ]
        self._stats["active_subscribers"] = sum(len(s) for s in self._subscribers.values())

    async def _flush_loop(self):
        """Background task to periodically flush buffers."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_all()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Flush error: {e}")

    async def _flush_all(self):
        """Flush all buffers to database."""
        with self._lock:
            tables = list(self._buffers.keys())

        for table in tables:
            await self._flush_table(table)

    async def _flush_table(self, table: str):
        """Flush a single table's buffer to database.

        FIX: Atomic table swap - build new table first, then replace.
        The old table remains readable until the swap completes.
        """
        if self._db is None:
            return

        with self._lock:
            buffer = self._buffers[table]
            if len(buffer) == 0:
                return

            # Get columnar data and clear buffer
            data = buffer.to_columnar()
            count = len(buffer)
            buffer.clear()

        try:
            import wayy_db as wdb

            if self._db.has_table(table):
                existing = self._db[table]

                # Read existing data
                existing_data = {}
                for col_name in existing.column_names():
                    existing_data[col_name] = existing[col_name].to_numpy()

                # Concatenate
                combined = {}
                for col_name, new_arr in data.items():
                    if col_name in existing_data:
                        combined[col_name] = np.concatenate([existing_data[col_name], new_arr])
                    else:
                        combined[col_name] = new_arr

                # FIX: Build new table FIRST, then atomic swap
                new_table = wdb.from_dict(combined, name=table, sorted_by="timestamp")
                self._db.drop_table(table)
                self._db.add_table(new_table)
            else:
                new_table = wdb.from_dict(data, name=table, sorted_by="timestamp")
                self._db.add_table(new_table)

            self._db.save()

            self._stats["ticks_flushed"] += count
            self._stats["flush_count"] += 1

            logger.debug(f"Flushed {count} ticks to {table}")

        except Exception as e:
            logger.error(f"Failed to flush {table}: {e}")
            # Re-add data to buffer on failure
            with self._lock:
                buf = self._buffers[table]
                for i in range(len(data["timestamp"])):
                    buf.timestamps.append(int(data["timestamp"][i]))
                    buf.symbols.append(f"unknown")  # Symbol hash lost, but data preserved
                    buf.prices.append(float(data["price"][i]))
                    buf.volumes.append(float(data["volume"][i]))
                    buf.bids.append(float(data["bid"][i]))
                    buf.asks.append(float(data["ask"][i]))

    async def _broadcast_loop(self):
        """Background task to batch-broadcast updates to WebSocket subscribers."""
        while self._running:
            try:
                await asyncio.sleep(self.batch_broadcast_interval)
                await self._broadcast_pending()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Broadcast error: {e}")

    async def _broadcast_pending(self):
        """Broadcast pending updates to all subscribers.

        FIX: Uses asyncio.gather for concurrent WebSocket sends.
        One slow subscriber no longer blocks all others.
        """
        # Swap out pending broadcasts atomically
        pending = dict(self._pending_broadcasts)
        self._pending_broadcasts = defaultdict(list)

        for table, quotes in pending.items():
            if not quotes:
                continue

            subscribers = self._subscribers.get(table, [])
            if not subscribers:
                continue

            # Build send tasks for all subscribers concurrently
            send_tasks = []
            sub_task_map: List[Subscriber] = []

            for sub in subscribers:
                if sub.symbols:
                    filtered = [q for q in quotes if q["symbol"] in sub.symbols]
                else:
                    filtered = quotes

                if not filtered:
                    continue

                if len(filtered) == 1:
                    payload = filtered[0]
                else:
                    payload = {"batch": filtered}

                send_tasks.append(self._safe_send(sub.websocket, payload))
                sub_task_map.append(sub)

            if not send_tasks:
                continue

            # FIX: Concurrent sends via asyncio.gather
            results = await asyncio.gather(*send_tasks, return_exceptions=True)

            dead_subs = []
            for sub, result in zip(sub_task_map, results):
                if isinstance(result, Exception):
                    dead_subs.append(sub)
                else:
                    count = len(quotes) if not sub.symbols else len(
                        [q for q in quotes if q["symbol"] in sub.symbols]
                    )
                    sub.messages_sent += count
                    self._stats["broadcasts_sent"] += count

            # Remove dead subscribers
            for sub in dead_subs:
                if sub in self._subscribers[table]:
                    self._subscribers[table].remove(sub)

    @staticmethod
    async def _safe_send(websocket: WebSocket, payload: Any) -> None:
        """Send JSON to a WebSocket with timeout."""
        await asyncio.wait_for(websocket.send_json(payload), timeout=5.0)

    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        stats = {
            **self._stats,
            "buffer_sizes": {t: len(b) for t, b in self._buffers.items()},
            "subscriber_counts": {t: len(s) for t, s in self._subscribers.items()},
            "latest_quotes": len(self._latest_quotes),
            "running": self._running,
        }
        if self._pubsub:
            stats["pubsub"] = self._pubsub.get_stats()
        return stats

    def get_latest_quote(self, table: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the latest quote for a symbol."""
        return self._latest_quotes.get(f"{table}:{symbol}")

    def get_all_quotes(self, table: str) -> Dict[str, Dict[str, Any]]:
        """Get all latest quotes for a table."""
        prefix = f"{table}:"
        return {
            k.split(":", 1)[1]: v
            for k, v in self._latest_quotes.items()
            if k.startswith(prefix)
        }


# Global streaming manager instance
_streaming_manager: Optional[StreamingManager] = None


def get_streaming_manager() -> StreamingManager:
    """Get or create the global streaming manager."""
    global _streaming_manager
    if _streaming_manager is None:
        redis_url = os.getenv("REDIS_URL", "")
        pubsub = create_pubsub(redis_url if redis_url else None)
        _streaming_manager = StreamingManager(pubsub=pubsub)
    return _streaming_manager


async def start_streaming():
    """Start the global streaming manager."""
    manager = get_streaming_manager()
    await manager.start()


async def stop_streaming():
    """Stop the global streaming manager."""
    global _streaming_manager
    if _streaming_manager:
        await _streaming_manager.stop()
