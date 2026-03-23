"""Redis cache backend for relay.

This module provides a :class:`RedisCache` backend that stores relay cache
entries in a Redis instance. It requires the ``redis`` package with async
support (``redis[asyncio]`` or ``redis>=4.2``).

If the ``redis`` package is not installed, all methods raise
:class:`ImportError` with a helpful installation message rather than failing
at import time. This allows the module to be imported safely regardless of
whether Redis support is desired.

Usage::

    from relay.cache.redis import RedisCache

    cache = RedisCache(url="redis://localhost:6379/0")
    async with cache:
        await cache.put("key", "anthropic", "claude-opus-4-5", response, 10, 5)
        result = await cache.get("key")
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_REDIS_INSTALL_MSG = (
    "The redis package is required to use RedisCache. "
    "Install it with: pip install redis[asyncio]"
)

# ---------------------------------------------------------------------------
# Optional dependency check (deferred to method calls, not import time)
# ---------------------------------------------------------------------------


def _get_redis_client(url: str, **kwargs: Any) -> Any:
    """Return an async Redis client for *url*.

    Args:
        url: Redis connection URL (e.g. ``"redis://localhost:6379/0"``).
        **kwargs: Additional keyword arguments forwarded to
            ``redis.asyncio.from_url``.

    Returns:
        An ``redis.asyncio.Redis`` client instance.

    Raises:
        ImportError: If the ``redis`` package is not installed.
    """
    try:
        from redis import asyncio as aioredis  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(_REDIS_INSTALL_MSG) from exc

    return aioredis.from_url(url, decode_responses=False, **kwargs)


_KEY_PREFIX = "relay:cache:"
_STATS_KEY = "relay:cache:__stats__"


class RedisCache:
    """Async Redis cache backend for relay.

    Cache entries are stored as Redis hash objects under keys of the form
    ``relay:cache:<cache_key>``. TTL is set directly on the Redis key via
    ``EXPIRE``.

    Response dicts are JSON-serialized and stored as raw bytes. Zstandard
    compression is applied when the ``zstandard`` package is available.

    Args:
        url: Redis connection URL. Defaults to ``"redis://localhost:6379/0"``.
        key_prefix: Prefix prepended to every Redis key managed by this
            backend. Useful for namespacing when sharing a Redis instance.
        default_ttl: Default time-to-live in seconds. ``None`` means entries
            never expire.
        max_connections: Maximum number of connections in the connection pool.

    Example:
        >>> cache = RedisCache(url="redis://localhost:6379/0")
        >>> async with cache:
        ...     await cache.put("k", "openai", "gpt-4o", {"text": "hi"}, 5, 3)
        ...     result = await cache.get("k")
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        key_prefix: str = _KEY_PREFIX,
        default_ttl: int | None = None,
        max_connections: int = 10,
    ) -> None:
        self._url = url
        self._key_prefix = key_prefix
        self._default_ttl = default_ttl
        self._max_connections = max_connections
        self._client: Any = None

        # In-process counters (reset on restart).
        self._hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _require_redis(self) -> None:
        """Raise :class:`ImportError` if ``redis`` is not installed.

        Raises:
            ImportError: If the ``redis`` package is not installed.
        """
        try:
            import redis  # noqa: F401  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(_REDIS_INSTALL_MSG) from exc

    async def _ensure_connected(self) -> Any:
        """Return an open Redis client, creating it if necessary.

        Returns:
            An ``redis.asyncio.Redis`` instance.

        Raises:
            ImportError: If ``redis`` is not installed.
        """
        if self._client is None:
            self._client = _get_redis_client(
                self._url,
                max_connections=self._max_connections,
            )
        return self._client

    async def close(self) -> None:
        """Close the Redis connection pool.

        Safe to call even if the connection was never opened.
        """
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _make_key(self, cache_key: str) -> str:
        return f"{self._key_prefix}{cache_key}"

    # ------------------------------------------------------------------
    # CacheBackend interface
    # ------------------------------------------------------------------

    async def get(self, cache_key: str) -> dict | None:
        """Return a cached response dict, or ``None`` on a miss.

        Args:
            cache_key: Opaque cache key string.

        Returns:
            Deserialized response dict on a hit, or ``None``.

        Raises:
            ImportError: If the ``redis`` package is not installed.
            relay.exceptions.CacheError: On unexpected storage errors.
        """
        from relay.exceptions import CacheError  # noqa: PLC0415

        self._require_redis()
        client = await self._ensure_connected()
        redis_key = self._make_key(cache_key)

        try:
            data = await client.hgetall(redis_key)
            if not data:
                self._misses += 1
                return None

            blob: bytes = data[b"response_json"]
            response = _decompress_response(blob)

            # Update LRU metadata (best-effort; ignore errors).
            now = time.time()
            try:
                pipe = client.pipeline(transaction=False)
                pipe.hset(redis_key, "last_hit_at", now)
                pipe.hincrby(redis_key, "hit_count", 1)
                await pipe.execute()
            except Exception:
                pass

            self._hits += 1
            return response

        except ImportError:
            raise
        except Exception as exc:
            raise CacheError(f"RedisCache.get failed: {exc}") from exc

    async def put(
        self,
        cache_key: str,
        provider: str,
        model: str,
        response: dict,
        input_tokens: int,
        output_tokens: int,
        ttl: int | None = None,
    ) -> None:
        """Store a response in the cache.

        Args:
            cache_key: Opaque cache key string.
            provider: Provider identifier.
            model: Model identifier.
            response: Response dict to cache.
            input_tokens: Input token count.
            output_tokens: Output token count.
            ttl: Time-to-live in seconds. Falls back to *default_ttl* if
                ``None``. A resolved ``None`` means no expiry.

        Raises:
            ImportError: If the ``redis`` package is not installed.
            relay.exceptions.CacheError: On write failure.
        """
        from relay.exceptions import CacheError  # noqa: PLC0415

        self._require_redis()
        client = await self._ensure_connected()
        redis_key = self._make_key(cache_key)

        now = time.time()
        effective_ttl = ttl if ttl is not None else self._default_ttl
        blob = _compress_response(response)

        mapping: dict[str | bytes, Any] = {
            "provider": provider,
            "model": model,
            "response_json": blob,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "created_at": now,
            "last_hit_at": now,
            "hit_count": 0,
            "size_bytes": len(blob),
        }
        if effective_ttl is not None:
            mapping["expires_at"] = now + effective_ttl

        try:
            pipe = client.pipeline()
            pipe.hset(redis_key, mapping=mapping)
            if effective_ttl is not None:
                pipe.expire(redis_key, effective_ttl)
            await pipe.execute()
        except ImportError:
            raise
        except Exception as exc:
            raise CacheError(f"RedisCache.put failed: {exc}") from exc

    async def invalidate(self, cache_key: str) -> None:
        """Remove a single entry from the cache.

        Args:
            cache_key: Opaque cache key string.

        Raises:
            ImportError: If the ``redis`` package is not installed.
            relay.exceptions.CacheError: On deletion failure.
        """
        from relay.exceptions import CacheError  # noqa: PLC0415

        self._require_redis()
        client = await self._ensure_connected()
        try:
            await client.delete(self._make_key(cache_key))
        except ImportError:
            raise
        except Exception as exc:
            raise CacheError(f"RedisCache.invalidate failed: {exc}") from exc

    async def invalidate_job(self, job_id: str) -> None:
        """Remove all cache entries associated with *job_id*.

        The Redis backend does not currently store job-level associations,
        so this method is a no-op. Job-level invalidation should be performed
        through the relay job-persistence layer.

        Args:
            job_id: The relay job identifier.

        Raises:
            ImportError: If the ``redis`` package is not installed.
        """
        self._require_redis()
        logger.debug(
            "RedisCache.invalidate_job(%r) called but this backend does not "
            "store job associations; no entries removed.",
            job_id,
        )

    async def stats(self) -> dict:
        """Return a snapshot of cache statistics.

        Scans all keys under the configured prefix to count entries and sum
        size bytes. This is an O(N) operation and should not be called in a
        tight loop on large caches.

        Returns:
            Dict with keys ``"entry_count"``, ``"size_bytes"``,
            ``"hit_rate"``, ``"hits"``, ``"misses"``,
            ``"compression"`` (``"zstd"`` or ``"none"``).

        Raises:
            ImportError: If the ``redis`` package is not installed.
            relay.exceptions.CacheError: On query failure.
        """
        from relay.exceptions import CacheError  # noqa: PLC0415

        self._require_redis()
        client = await self._ensure_connected()

        try:
            entry_count = 0
            size_bytes = 0
            pattern = f"{self._key_prefix}*"

            async for key in client.scan_iter(pattern):
                entry_count += 1
                sb = await client.hget(key, "size_bytes")
                if sb:
                    size_bytes += int(sb)

            total_calls = self._hits + self._misses
            hit_rate = self._hits / total_calls if total_calls > 0 else 0.0

            return {
                "entry_count": entry_count,
                "size_bytes": size_bytes,
                "hit_rate": hit_rate,
                "hits": self._hits,
                "misses": self._misses,
                "compression": "zstd" if _zstd_available() else "none",
            }
        except ImportError:
            raise
        except Exception as exc:
            raise CacheError(f"RedisCache.stats failed: {exc}") from exc

    async def vacuum(self) -> None:
        """Expire TTL-exceeded entries.

        Redis handles TTL expiry natively, so this method performs a
        lightweight scan to delete any entries whose ``expires_at`` field has
        passed but whose Redis key TTL was not set (e.g. entries inserted
        without a TTL that later need to be cleaned up manually).

        Raises:
            ImportError: If the ``redis`` package is not installed.
            relay.exceptions.CacheError: On maintenance failure.
        """
        from relay.exceptions import CacheError  # noqa: PLC0415

        self._require_redis()
        client = await self._ensure_connected()
        now = time.time()

        try:
            pattern = f"{self._key_prefix}*"
            keys_deleted = 0
            async for key in client.scan_iter(pattern):
                expires_at_raw = await client.hget(key, "expires_at")
                if expires_at_raw is not None:
                    try:
                        expires_at = float(expires_at_raw)
                        if expires_at < now:
                            await client.delete(key)
                            keys_deleted += 1
                    except (ValueError, TypeError):
                        pass
            if keys_deleted:
                logger.debug("RedisCache.vacuum: deleted %d expired entries.", keys_deleted)
        except ImportError:
            raise
        except Exception as exc:
            raise CacheError(f"RedisCache.vacuum failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "RedisCache":
        self._require_redis()
        await self._ensure_connected()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()


# ---------------------------------------------------------------------------
# Compression helpers (shared with sqlite backend logic)
# ---------------------------------------------------------------------------


def _zstd_available() -> bool:
    try:
        import zstandard  # noqa: F401  # type: ignore[import-not-found]
        return True
    except ImportError:
        return False


def _compress_response(response: dict) -> bytes:
    """Serialize and optionally Zstd-compress a response dict.

    Args:
        response: Response dict to compress.

    Returns:
        Compressed or raw UTF-8 JSON bytes.
    """
    raw: bytes = json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if _zstd_available():
        import zstandard as zstd  # type: ignore[import-not-found]
        return zstd.ZstdCompressor(level=3).compress(raw)
    return raw


def _decompress_response(blob: bytes) -> dict:
    """Decompress and deserialize a stored response blob.

    Args:
        blob: Raw bytes from the Redis hash field.

    Returns:
        Deserialized response dict.
    """
    if _zstd_available():
        import zstandard as zstd  # type: ignore[import-not-found]
        try:
            raw = zstd.ZstdDecompressor().decompress(blob)
            return json.loads(raw)
        except Exception:
            pass
    return json.loads(blob)
