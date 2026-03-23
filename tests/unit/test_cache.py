"""Unit tests for the SQLiteCache backend."""

from __future__ import annotations

import time

import pytest
import pytest_asyncio

from relay.cache.sqlite import SQLiteCache


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def cache(tmp_path):
    db_path = tmp_path / "test_cache.db"
    c = SQLiteCache(str(db_path), max_size_gb=None, default_ttl=None)
    async with c:
        yield c


_SAMPLE_RESPONSE = {"content": "Paris", "stop_reason": "end_turn", "tokens": 5}


async def _put(cache: SQLiteCache, key: str = "key1", **kwargs) -> None:
    defaults = dict(
        cache_key=key,
        provider="anthropic",
        model="claude-opus-4-5",
        response=_SAMPLE_RESPONSE.copy(),
        input_tokens=10,
        output_tokens=5,
    )
    defaults.update(kwargs)
    await cache.put(**defaults)


# ── put / get cycle ───────────────────────────────────────────────────────────


class TestPutGet:
    @pytest.mark.asyncio
    async def test_put_then_get_returns_response(self, cache):
        await _put(cache, key="abc")
        result = await cache.get("abc")
        assert result is not None
        assert result["content"] == "Paris"

    @pytest.mark.asyncio
    async def test_get_miss_returns_none(self, cache):
        result = await cache.get("nonexistent_key_xyz")
        assert result is None

    @pytest.mark.asyncio
    async def test_put_overwrites_existing(self, cache):
        await _put(cache, key="k1", response={"content": "original"})
        await _put(cache, key="k1", response={"content": "updated"})
        result = await cache.get("k1")
        assert result["content"] == "updated"

    @pytest.mark.asyncio
    async def test_multiple_keys_independent(self, cache):
        await _put(cache, key="k1", response={"v": 1})
        await _put(cache, key="k2", response={"v": 2})
        assert (await cache.get("k1"))["v"] == 1
        assert (await cache.get("k2"))["v"] == 2

    @pytest.mark.asyncio
    async def test_response_roundtrip_preserves_types(self, cache):
        response = {"content": "test", "count": 42, "flag": True, "items": [1, 2, 3]}
        await _put(cache, key="rt", response=response)
        result = await cache.get("rt")
        assert result == response


# ── TTL expiry ────────────────────────────────────────────────────────────────


class TestTTLExpiry:
    @pytest.mark.asyncio
    async def test_entry_available_before_expiry(self, tmp_path):
        c = SQLiteCache(str(tmp_path / "ttl.db"), max_size_gb=None, default_ttl=3600)
        async with c:
            await _put(c, key="live")
            assert await c.get("live") is not None

    @pytest.mark.asyncio
    async def test_expired_entry_returns_none(self, tmp_path):
        c = SQLiteCache(str(tmp_path / "exp.db"), max_size_gb=None, default_ttl=None)
        async with c:
            # Put with a TTL of 1 second.
            await _put(c, key="short", ttl=1)
            assert await c.get("short") is not None

            # Manually expire by waiting slightly longer.
            # In tests we manipulate the expires_at by using a tiny ttl.
            await _put(c, key="short2", ttl=1)

            # Force-age the entry by directly updating expires_at in the db.
            async with c._db.execute(
                "UPDATE cache_entries SET expires_at = ? WHERE cache_key = ?",
                (time.time() - 1.0, "short2"),
            ):
                pass
            await c._db.commit()

            result = await c.get("short2")
            assert result is None

    @pytest.mark.asyncio
    async def test_no_ttl_entry_never_expires(self, tmp_path):
        c = SQLiteCache(str(tmp_path / "nexp.db"), max_size_gb=None, default_ttl=None)
        async with c:
            await _put(c, key="eternal", ttl=None)
            result = await c.get("eternal")
            assert result is not None


# ── Invalidation ──────────────────────────────────────────────────────────────


class TestInvalidation:
    @pytest.mark.asyncio
    async def test_invalidate_removes_entry(self, cache):
        await _put(cache, key="remove_me")
        assert await cache.get("remove_me") is not None

        await cache.invalidate("remove_me")
        assert await cache.get("remove_me") is None

    @pytest.mark.asyncio
    async def test_invalidate_missing_key_is_noop(self, cache):
        # Should not raise.
        await cache.invalidate("key_that_does_not_exist")

    @pytest.mark.asyncio
    async def test_vacuum_removes_expired_entries(self, tmp_path):
        c = SQLiteCache(str(tmp_path / "vac.db"), max_size_gb=None, default_ttl=None)
        async with c:
            await _put(c, key="exp", ttl=1)

            # Age the entry.
            async with c._db.execute(
                "UPDATE cache_entries SET expires_at = ? WHERE cache_key = ?",
                (time.time() - 2.0, "exp"),
            ):
                pass
            await c._db.commit()

            await c.vacuum()
            assert await c.get("exp") is None


# ── Stats ─────────────────────────────────────────────────────────────────────


class TestStats:
    @pytest.mark.asyncio
    async def test_stats_empty_cache(self, cache):
        s = await cache.stats()
        assert s["entry_count"] == 0
        assert s["size_bytes"] == 0
        assert s["hits"] == 0
        assert s["misses"] == 0
        assert s["hit_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_stats_after_put(self, cache):
        await _put(cache, key="s1")
        s = await cache.stats()
        assert s["entry_count"] == 1
        assert s["size_bytes"] > 0

    @pytest.mark.asyncio
    async def test_stats_hit_count(self, cache):
        await _put(cache, key="h1")
        await cache.get("h1")  # hit
        await cache.get("h1")  # hit
        await cache.get("missing")  # miss
        s = await cache.stats()
        assert s["hits"] == 2
        assert s["misses"] == 1

    @pytest.mark.asyncio
    async def test_stats_hit_rate_calculation(self, cache):
        await _put(cache, key="hr")
        await cache.get("hr")       # hit
        await cache.get("missing")  # miss
        s = await cache.stats()
        assert s["hit_rate"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_stats_compression_key(self, cache):
        s = await cache.stats()
        assert "compression" in s
        assert s["compression"] in ("zstd", "none")

    @pytest.mark.asyncio
    async def test_stats_multiple_entries(self, cache):
        for i in range(5):
            await _put(cache, key=f"entry_{i}")
        s = await cache.stats()
        assert s["entry_count"] == 5
