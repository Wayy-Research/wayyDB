"""
KV Store - In-memory key-value store with TTL for wayyDB.

Provides Redis-like KV semantics for future multi-process scaling.
Background eviction runs every 60 seconds.
"""

import asyncio
import logging
import time
from fnmatch import fnmatch
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class KVEntry:
    """A stored value with optional TTL."""
    __slots__ = ("value", "expires_at", "created_at")

    def __init__(self, value: Any, ttl: Optional[float] = None):
        now = time.time()
        self.value = value
        self.expires_at = now + ttl if ttl else float("inf")
        self.created_at = now

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    @property
    def ttl_remaining(self) -> Optional[float]:
        if self.expires_at == float("inf"):
            return None
        remaining = self.expires_at - time.time()
        return max(0, remaining)


class KVStore:
    """
    In-memory KV store with TTL and background eviction.

    Thread-safe for single-process async use (GIL + event loop).
    """

    def __init__(self) -> None:
        self._data: Dict[str, KVEntry] = {}
        self._eviction_task: Optional[asyncio.Task] = None
        self._sets: int = 0
        self._gets: int = 0
        self._deletes: int = 0
        self._evictions: int = 0

    async def start(self) -> None:
        """Start the background eviction task."""
        if self._eviction_task is None:
            self._eviction_task = asyncio.create_task(self._eviction_loop())
            logger.info("KVStore eviction task started")

    async def stop(self) -> None:
        """Stop the background eviction task."""
        if self._eviction_task:
            self._eviction_task.cancel()
            try:
                await self._eviction_task
            except asyncio.CancelledError:
                pass
            self._eviction_task = None

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set a key with optional TTL (seconds)."""
        self._data[key] = KVEntry(value, ttl)
        self._sets += 1

    def get(self, key: str) -> Optional[Any]:
        """Get a value by key. Returns None if missing or expired."""
        self._gets += 1
        entry = self._data.get(key)
        if entry is None:
            return None
        if entry.is_expired:
            del self._data[key]
            self._evictions += 1
            return None
        return entry.value

    def delete(self, key: str) -> bool:
        """Delete a key. Returns True if existed."""
        existed = key in self._data
        if existed:
            del self._data[key]
            self._deletes += 1
        return existed

    def keys(self, pattern: Optional[str] = None) -> List[str]:
        """List keys, optionally filtered by glob pattern."""
        now = time.time()
        result = []
        for k, v in self._data.items():
            if v.expires_at > now:
                if pattern is None or fnmatch(k, pattern):
                    result.append(k)
        return result

    def stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        now = time.time()
        active = sum(1 for v in self._data.values() if v.expires_at > now)
        return {
            "total_keys": len(self._data),
            "active_keys": active,
            "sets": self._sets,
            "gets": self._gets,
            "deletes": self._deletes,
            "evictions": self._evictions,
        }

    async def _eviction_loop(self) -> None:
        """Background loop to evict expired entries every 60s."""
        while True:
            try:
                await asyncio.sleep(60)
                count = self._evict_expired()
                if count > 0:
                    logger.debug(f"KVStore evicted {count} expired entries")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"KVStore eviction error: {e}")

    def _evict_expired(self) -> int:
        """Evict all expired entries. Returns count evicted."""
        now = time.time()
        expired = [k for k, v in self._data.items() if now > v.expires_at]
        for k in expired:
            del self._data[k]
        self._evictions += len(expired)
        return len(expired)


# Global singleton
_kv_store: Optional[KVStore] = None


def get_kv_store() -> KVStore:
    """Get the global KV store instance."""
    global _kv_store
    if _kv_store is None:
        _kv_store = KVStore()
    return _kv_store
