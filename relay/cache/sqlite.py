"""SQLite-backed cache backend for relay.

Uses ``aiosqlite`` for fully async I/O and WAL journal mode for high
read-concurrency. Stored response blobs are compressed with Zstandard
(``zstd``) when the ``zstandard`` package is installed, otherwise they are
stored as raw UTF-8 JSON.

LRU eviction is triggered automatically on :meth:`SQLiteCache.put` when the
total on-disk size would exceed *max_size_gb*. TTL expiry is applied lazily
on :meth:`SQLiteCache.get` and eagerly during :meth:`SQLiteCache.vacuum`.

Schema
------
.. code-block:: sql

    CREATE TABLE cache_entries (
        cache_key     TEXT    PRIMARY KEY,
        provider      TEXT    NOT NULL,
        model         TEXT    NOT NULL,
        response_json BLOB    NOT NULL,
        input_tokens  INTEGER NOT NULL,
        output_tokens INTEGER NOT NULL,
        created_at    REAL    NOT NULL,
        last_hit_at   REAL    NOT NULL,
        hit_count     INTEGER NOT NULL DEFAULT 0,
        expires_at    REAL,
        size_bytes    INTEGER NOT NULL
    );
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: zstandard
# ---------------------------------------------------------------------------
try:
    import zstandard as zstd  # type: ignore[import-not-found]

    _ZSTD_AVAILABLE = True
    _zstd_compressor = zstd.ZstdCompressor(level=3)
    _zstd_decompressor = zstd.ZstdDecompressor()
except ImportError:
    _ZSTD_AVAILABLE = False
    _zstd_compressor = None  # type: ignore[assignment]
    _zstd_decompressor = None  # type: ignore[assignment]

_GB = 1024 ** 3

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS cache_entries (
    cache_key     TEXT    PRIMARY KEY,
    provider      TEXT    NOT NULL,
    model         TEXT    NOT NULL,
    response_json BLOB    NOT NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    created_at    REAL    NOT NULL,
    last_hit_at   REAL    NOT NULL,
    hit_count     INTEGER NOT NULL DEFAULT 0,
    expires_at    REAL,
    size_bytes    INTEGER NOT NULL
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_last_hit_at ON cache_entries (last_hit_at);
"""


class SQLiteCache:
    """Async SQLite cache backend with Zstd compression and LRU eviction.

    Args:
        db_path: Filesystem path to the SQLite database file. The parent
            directory must already exist (or be ``":memory:"`` for an
            in-memory database used in tests).
        max_size_gb: Maximum total size of cached response blobs in gigabytes.
            When exceeded, LRU eviction removes the oldest entries until the
            cache fits within the limit. ``None`` disables size-based eviction.
        default_ttl: Default time-to-live in seconds applied to entries that
            do not specify their own TTL. ``None`` means entries never expire.

    Example:
        >>> cache = SQLiteCache("~/.relay/cache.db", max_size_gb=10.0)
        >>> await cache.get("abc123")
        None
        >>> await cache.put(
        ...     "abc123", "anthropic", "claude-opus-4-5",
        ...     {"content": "Paris"}, input_tokens=10, output_tokens=5
        ... )
        >>> result = await cache.get("abc123")
        >>> result["content"]
        'Paris'
    """

    def __init__(
        self,
        db_path: str | Path = "~/.relay/cache.db",
        max_size_gb: float | None = 10.0,
        default_ttl: int | None = None,
    ) -> None:
        self._db_path = str(Path(db_path).expanduser())
        self._max_size_bytes: int | None = int(max_size_gb * _GB) if max_size_gb is not None else None
        self._default_ttl = default_ttl
        self._db: Any = None  # aiosqlite.Connection

        # In-process hit/miss counters for hit_rate reporting.
        self._hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _ensure_open(self) -> Any:
        """Return the open aiosqlite connection, opening it if necessary.

        Returns:
            An open ``aiosqlite.Connection`` instance.

        Raises:
            ImportError: If ``aiosqlite`` is not installed.
        """
        if self._db is not None:
            return self._db

        try:
            import aiosqlite  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "aiosqlite is required for SQLiteCache. "
                "Install it with: pip install aiosqlite"
            ) from exc

        db = await aiosqlite.connect(self._db_path)
        db.row_factory = aiosqlite.Row

        # Enable WAL mode for better read/write concurrency.
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute("PRAGMA synchronous=NORMAL;")
        await db.execute("PRAGMA foreign_keys=ON;")
        await db.execute(_CREATE_TABLE_SQL)
        await db.execute(_CREATE_INDEX_SQL)
        await db.commit()

        self._db = db
        return db

    async def close(self) -> None:
        """Close the underlying database connection.

        Safe to call even if the connection was never opened.
        """
        if self._db is not None:
            await self._db.close()
            self._db = None

    # ------------------------------------------------------------------
    # CacheBackend interface
    # ------------------------------------------------------------------

    async def get(self, cache_key: str) -> dict | None:
        """Return a cached response dict, or ``None`` on a miss or TTL expiry.

        Updates ``last_hit_at`` and increments ``hit_count`` on a successful
        hit. Deletes entries that have passed their TTL and returns ``None``.

        Args:
            cache_key: Opaque cache key string (SHA-256 hex digest).

        Returns:
            Deserialized response dict on a hit, or ``None``.

        Raises:
            relay.exceptions.CacheError: On unexpected storage errors.
        """
        from relay.exceptions import CacheError  # noqa: PLC0415

        db = await self._ensure_open()
        now = time.time()

        try:
            async with db.execute(
                "SELECT response_json, expires_at FROM cache_entries WHERE cache_key = ?",
                (cache_key,),
            ) as cursor:
                row = await cursor.fetchone()

            if row is None:
                self._misses += 1
                return None

            expires_at: float | None = row["expires_at"]
            if expires_at is not None and expires_at < now:
                # Entry has expired; delete and treat as a miss.
                await db.execute("DELETE FROM cache_entries WHERE cache_key = ?", (cache_key,))
                await db.commit()
                self._misses += 1
                return None

            # Update recency metadata.
            await db.execute(
                "UPDATE cache_entries SET last_hit_at = ?, hit_count = hit_count + 1 "
                "WHERE cache_key = ?",
                (now, cache_key),
            )
            await db.commit()

            self._hits += 1
            blob: bytes = row["response_json"]
            return _decompress_response(blob)

        except Exception as exc:
            raise CacheError(f"SQLiteCache.get failed: {exc}") from exc

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
        """Store a response in the cache, replacing any existing entry.

        If *max_size_gb* was set and would be exceeded after this write,
        LRU eviction runs automatically before committing.

        Args:
            cache_key: Opaque cache key string.
            provider: Provider identifier.
            model: Model identifier.
            response: Response dict to cache.
            input_tokens: Input token count for this request.
            output_tokens: Output token count for this request.
            ttl: Time-to-live in seconds. Falls back to *default_ttl* if
                ``None``. A resolved ``None`` means the entry never expires.

        Raises:
            relay.exceptions.CacheError: On write failure.
        """
        from relay.exceptions import CacheError  # noqa: PLC0415

        db = await self._ensure_open()
        now = time.time()
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expires_at = (now + effective_ttl) if effective_ttl is not None else None

        blob = _compress_response(response)
        size_bytes = len(blob)

        try:
            await db.execute(
                """
                INSERT INTO cache_entries
                    (cache_key, provider, model, response_json, input_tokens,
                     output_tokens, created_at, last_hit_at, hit_count,
                     expires_at, size_bytes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    provider      = excluded.provider,
                    model         = excluded.model,
                    response_json = excluded.response_json,
                    input_tokens  = excluded.input_tokens,
                    output_tokens = excluded.output_tokens,
                    created_at    = excluded.created_at,
                    last_hit_at   = excluded.last_hit_at,
                    hit_count     = 0,
                    expires_at    = excluded.expires_at,
                    size_bytes    = excluded.size_bytes
                """,
                (
                    cache_key, provider, model, blob,
                    input_tokens, output_tokens,
                    now, now, expires_at, size_bytes,
                ),
            )
            await db.commit()

            if self._max_size_bytes is not None:
                await self._evict_lru_if_needed(db)

        except Exception as exc:
            raise CacheError(f"SQLiteCache.put failed: {exc}") from exc

    async def invalidate(self, cache_key: str) -> None:
        """Remove a single entry from the cache.

        A no-op if the key does not exist.

        Args:
            cache_key: Opaque cache key string.

        Raises:
            relay.exceptions.CacheError: On deletion failure.
        """
        from relay.exceptions import CacheError  # noqa: PLC0415

        db = await self._ensure_open()
        try:
            await db.execute("DELETE FROM cache_entries WHERE cache_key = ?", (cache_key,))
            await db.commit()
        except Exception as exc:
            raise CacheError(f"SQLiteCache.invalidate failed: {exc}") from exc

    async def invalidate_job(self, job_id: str) -> None:
        """Remove all cache entries associated with *job_id*.

        The SQLite backend does not currently store job-level associations
        in the cache table, so this method is a deliberate no-op. Job-level
        invalidation should be performed through the relay job-persistence
        layer instead.

        Args:
            job_id: The relay job identifier.
        """
        logger.debug(
            "SQLiteCache.invalidate_job(%r) called but this backend does not "
            "store job associations; no entries removed.",
            job_id,
        )

    async def stats(self) -> dict:
        """Return a snapshot of cache statistics.

        Returns:
            Dict with keys ``"size_bytes"``, ``"entry_count"``,
            ``"hit_rate"``, ``"hits"``, ``"misses"``,
            ``"compression"`` (``"zstd"`` or ``"none"``).

        Raises:
            relay.exceptions.CacheError: On query failure.
        """
        from relay.exceptions import CacheError  # noqa: PLC0415

        db = await self._ensure_open()
        try:
            async with db.execute(
                "SELECT COUNT(*) as entry_count, COALESCE(SUM(size_bytes), 0) as size_bytes "
                "FROM cache_entries"
            ) as cursor:
                row = await cursor.fetchone()

            total_calls = self._hits + self._misses
            hit_rate = self._hits / total_calls if total_calls > 0 else 0.0

            return {
                "entry_count": row["entry_count"],
                "size_bytes": row["size_bytes"],
                "hit_rate": hit_rate,
                "hits": self._hits,
                "misses": self._misses,
                "compression": "zstd" if _ZSTD_AVAILABLE else "none",
            }
        except Exception as exc:
            raise CacheError(f"SQLiteCache.stats failed: {exc}") from exc

    async def vacuum(self) -> None:
        """Expire TTL-exceeded entries and run LRU eviction if needed.

        1. Deletes all entries whose ``expires_at`` is in the past.
        2. If *max_size_gb* is configured and still exceeded after TTL
           deletion, evicts the least-recently-used entries.

        Raises:
            relay.exceptions.CacheError: On maintenance failure.
        """
        from relay.exceptions import CacheError  # noqa: PLC0415

        db = await self._ensure_open()
        now = time.time()
        try:
            await db.execute(
                "DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now,),
            )
            await db.commit()

            if self._max_size_bytes is not None:
                await self._evict_lru_if_needed(db)

            # Reclaim unused pages from the database file.
            await db.execute("PRAGMA wal_checkpoint(PASSIVE);")
        except Exception as exc:
            raise CacheError(f"SQLiteCache.vacuum failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _evict_lru_if_needed(self, db: Any) -> None:
        """Evict least-recently-used entries until under *max_size_bytes*.

        Args:
            db: Open aiosqlite connection.
        """
        assert self._max_size_bytes is not None

        async with db.execute(
            "SELECT COALESCE(SUM(size_bytes), 0) FROM cache_entries"
        ) as cursor:
            row = await cursor.fetchone()
        current_size: int = row[0]

        if current_size <= self._max_size_bytes:
            return

        # Fetch LRU candidates ordered by last_hit_at ascending.
        async with db.execute(
            "SELECT cache_key, size_bytes FROM cache_entries ORDER BY last_hit_at ASC"
        ) as cursor:
            candidates = await cursor.fetchall()

        keys_to_delete: list[str] = []
        for candidate in candidates:
            if current_size <= self._max_size_bytes:
                break
            keys_to_delete.append(candidate[0])
            current_size -= candidate[1]

        if keys_to_delete:
            placeholders = ",".join("?" * len(keys_to_delete))
            await db.execute(
                f"DELETE FROM cache_entries WHERE cache_key IN ({placeholders})",
                keys_to_delete,
            )
            await db.commit()
            logger.debug("SQLiteCache: evicted %d LRU entries.", len(keys_to_delete))

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "SQLiteCache":
        await self._ensure_open()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()


# ---------------------------------------------------------------------------
# Compression helpers
# ---------------------------------------------------------------------------


def _compress_response(response: dict) -> bytes:
    """Serialize and optionally compress a response dict.

    Args:
        response: Response dict to compress.

    Returns:
        Compressed bytes if ``zstandard`` is available, otherwise raw
        UTF-8-encoded JSON bytes.
    """
    raw: bytes = json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if _ZSTD_AVAILABLE:
        return _zstd_compressor.compress(raw)
    return raw


def _decompress_response(blob: bytes) -> dict:
    """Decompress and deserialize a stored response blob.

    Args:
        blob: Raw bytes from the ``response_json`` column.

    Returns:
        Deserialized response dict.
    """
    if _ZSTD_AVAILABLE:
        try:
            raw = _zstd_decompressor.decompress(blob)
            return json.loads(raw)
        except Exception:
            # Fall through: blob might be uncompressed if stored before zstd
            # was available, or if compression was disabled at write time.
            pass
    return json.loads(blob)
