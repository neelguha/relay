"""Unit tests for the async retry decorator."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from relay.exceptions import AuthenticationError, RateLimitError, ServerError
from relay.utils.retry import _compute_delay, _should_retry, retry


# ── Helper exceptions ─────────────────────────────────────────────────────────


class _StatusError(Exception):
    """Generic exception with a status_code attribute."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


# ── retry decorator ───────────────────────────────────────────────────────────


class TestRetrySuccessful:
    @pytest.mark.asyncio
    async def test_success_not_retried(self):
        call_count = 0

        @retry(max_attempts=3, initial_backoff=0.01)
        async def fn():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await fn()
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_returns_value(self):
        @retry(max_attempts=2, initial_backoff=0.01)
        async def fn():
            return 42

        assert await fn() == 42


class TestRetryRetryableErrors:
    @pytest.mark.asyncio
    async def test_retries_rate_limit_error(self):
        call_count = 0

        @retry(max_attempts=3, initial_backoff=0.01, jitter="none")
        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RateLimitError("too many requests")
            return "done"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await fn()

        assert result == "done"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retries_server_error(self):
        call_count = 0

        @retry(max_attempts=3, initial_backoff=0.01, jitter="none")
        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ServerError("server blew up", status_code=500)
            return "recovered"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await fn()

        assert result == "recovered"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_exhausts_max_attempts_and_raises(self):
        @retry(max_attempts=3, initial_backoff=0.01, jitter="none")
        async def always_fails():
            raise RateLimitError("always rate limited")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RateLimitError):
                await always_fails()

    @pytest.mark.asyncio
    async def test_retry_on_status_code(self):
        call_count = 0

        @retry(max_attempts=4, initial_backoff=0.01, retry_on=(503,), jitter="none")
        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise _StatusError("service unavailable", status_code=503)
            return "ok"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await fn()

        assert result == "ok"
        assert call_count == 3


class TestRetryNonRetryableErrors:
    @pytest.mark.asyncio
    async def test_non_retryable_not_retried(self):
        call_count = 0

        @retry(max_attempts=5, initial_backoff=0.01)
        async def fn():
            nonlocal call_count
            call_count += 1
            raise AuthenticationError("invalid key", status_code=401)

        with pytest.raises(AuthenticationError):
            await fn()

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_value_error_not_retried(self):
        call_count = 0

        @retry(max_attempts=3, initial_backoff=0.01)
        async def fn():
            nonlocal call_count
            call_count += 1
            raise ValueError("bad input")

        with pytest.raises(ValueError):
            await fn()

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_status_code_not_in_retry_on_not_retried(self):
        call_count = 0

        @retry(max_attempts=3, initial_backoff=0.01, retry_on=(429,), jitter="none")
        async def fn():
            nonlocal call_count
            call_count += 1
            raise _StatusError("not found", status_code=404)

        with pytest.raises(_StatusError):
            await fn()

        assert call_count == 1


class TestRetryBackoffTiming:
    @pytest.mark.asyncio
    async def test_sleep_called_between_retries(self):
        @retry(max_attempts=3, initial_backoff=1.0, jitter="none")
        async def always_fails():
            raise RateLimitError("rate limited")

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(RateLimitError):
                await always_fails()

        # Should sleep twice (after attempt 0 and attempt 1, not after the final).
        assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_sleep_not_called_on_non_retryable(self):
        @retry(max_attempts=3, initial_backoff=1.0)
        async def fn():
            raise ValueError("nope")

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(ValueError):
                await fn()

        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_sleep_after_final_attempt(self):
        """Sleep should not be called after the last attempt."""

        @retry(max_attempts=2, initial_backoff=1.0, jitter="none")
        async def fn():
            raise RateLimitError("rate limited")

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(RateLimitError):
                await fn()

        # max_attempts=2 → 1 retry → 1 sleep call.
        assert mock_sleep.call_count == 1


class TestJitterModes:
    def test_invalid_jitter_raises(self):
        with pytest.raises(ValueError, match="jitter"):
            retry(jitter="random")

    def test_none_jitter_is_deterministic(self):
        delay1 = _compute_delay(
            attempt=0,
            initial_backoff=1.0,
            backoff_multiplier=2.0,
            max_backoff=60.0,
            jitter="none",
            exc=Exception(),
        )
        delay2 = _compute_delay(
            attempt=0,
            initial_backoff=1.0,
            backoff_multiplier=2.0,
            max_backoff=60.0,
            jitter="none",
            exc=Exception(),
        )
        assert delay1 == delay2 == 1.0

    def test_full_jitter_bounded(self):
        for _ in range(100):
            delay = _compute_delay(
                attempt=0,
                initial_backoff=4.0,
                backoff_multiplier=2.0,
                max_backoff=60.0,
                jitter="full",
                exc=Exception(),
            )
            assert 0.0 <= delay <= 4.0

    def test_equal_jitter_bounded(self):
        for _ in range(100):
            delay = _compute_delay(
                attempt=0,
                initial_backoff=4.0,
                backoff_multiplier=2.0,
                max_backoff=60.0,
                jitter="equal",
                exc=Exception(),
            )
            # Equal jitter: half + [0, half], so in [half, base].
            assert 2.0 <= delay <= 4.0

    def test_max_backoff_capped(self):
        delay = _compute_delay(
            attempt=10,  # 1.0 * 2^10 = 1024 > max_backoff
            initial_backoff=1.0,
            backoff_multiplier=2.0,
            max_backoff=30.0,
            jitter="none",
            exc=Exception(),
        )
        assert delay == 30.0

    def test_retry_after_honoured(self):
        exc = RateLimitError("rate limited", retry_after=20.0)
        delay = _compute_delay(
            attempt=0,
            initial_backoff=1.0,
            backoff_multiplier=2.0,
            max_backoff=60.0,
            jitter="none",
            exc=exc,
        )
        assert delay == 20.0


# ── _should_retry ─────────────────────────────────────────────────────────────


class TestShouldRetry:
    def test_rate_limit_error(self):
        assert _should_retry(RateLimitError("x"), retry_on=()) is True

    def test_server_error(self):
        assert _should_retry(ServerError("x"), retry_on=()) is True

    def test_status_code_in_retry_on(self):
        exc = _StatusError("x", status_code=503)
        assert _should_retry(exc, retry_on=(503,)) is True

    def test_status_code_not_in_retry_on(self):
        exc = _StatusError("x", status_code=404)
        assert _should_retry(exc, retry_on=(429, 500)) is False

    def test_plain_exception(self):
        assert _should_retry(ValueError("bad"), retry_on=(429,)) is False

    def test_max_attempts_less_than_1_raises(self):
        with pytest.raises(ValueError, match="max_attempts"):
            retry(max_attempts=0)
