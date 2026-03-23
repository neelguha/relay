"""Async retry decorator with exponential backoff for relay provider calls.

Provides a configurable ``retry`` decorator that wraps async functions and
automatically retries them on transient HTTP errors, using exponential
backoff with optional jitter to avoid thundering-herd problems.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
from collections.abc import Callable
from typing import ParamSpec, TypeVar

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

_JITTER_MODES = frozenset({"none", "full", "equal"})


def retry(
    max_attempts: int = 5,
    initial_backoff: float = 1.0,
    backoff_multiplier: float = 2.0,
    max_backoff: float = 60.0,
    jitter: str = "full",
    retry_on: tuple[int, ...] = (429, 500, 502, 503, 504),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Async retry decorator with exponential backoff.

    Wraps an async function so that it is automatically retried when it raises
    an exception whose ``status_code`` attribute (if present) matches one of
    the configured HTTP status codes, or when it raises
    ``relay.exceptions.RateLimitError`` or ``relay.exceptions.ServerError``.

    Backoff is computed as::

        delay = min(initial_backoff * backoff_multiplier ** attempt, max_backoff)

    and then optionally modified by the chosen jitter strategy:

    * ``"none"`` — no jitter; pure exponential backoff.
    * ``"full"`` — ``delay = random.uniform(0, delay)`` (recommended default).
    * ``"equal"`` — ``delay = delay / 2 + random.uniform(0, delay / 2)``.

    Args:
        max_attempts: Total number of attempts (including the first). Must be
            at least 1.
        initial_backoff: Backoff duration in seconds before the second attempt.
        backoff_multiplier: Multiplicative factor applied to the backoff on
            each subsequent attempt.
        max_backoff: Upper bound on the computed backoff duration in seconds.
        jitter: Jitter strategy. One of ``"none"``, ``"full"``, or
            ``"equal"``.
        retry_on: Tuple of HTTP status codes that should trigger a retry. Only
            checked when the raised exception exposes a ``status_code``
            attribute. ``RateLimitError`` and ``ServerError`` are always
            retried regardless of this tuple.

    Returns:
        A decorator that applies the retry logic to any async callable.

    Raises:
        ValueError: If ``jitter`` is not one of the accepted values or if
            ``max_attempts`` is less than 1.

    Example:
        >>> from relay.utils.retry import retry
        >>>
        >>> @retry(max_attempts=3, initial_backoff=0.5)
        ... async def call_api(url: str) -> dict:
        ...     ...
    """
    if jitter not in _JITTER_MODES:
        raise ValueError(f"jitter must be one of {sorted(_JITTER_MODES)!r}, got {jitter!r}")
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:  # type: ignore[return]
            last_exc: BaseException | None = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)  # type: ignore[return-value]
                except Exception as exc:
                    last_exc = exc
                    if not _should_retry(exc, retry_on):
                        raise
                    if attempt == max_attempts - 1:
                        # Final attempt exhausted.
                        raise
                    delay = _compute_delay(
                        attempt=attempt,
                        initial_backoff=initial_backoff,
                        backoff_multiplier=backoff_multiplier,
                        max_backoff=max_backoff,
                        jitter=jitter,
                        exc=exc,
                    )
                    logger.warning(
                        "relay retry: attempt %d/%d failed (%s). Retrying in %.2fs.",
                        attempt + 1,
                        max_attempts,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
            # Should never reach here, but satisfies type checker.
            assert last_exc is not None
            raise last_exc  # noqa: B904

        return wrapper  # type: ignore[return-value]

    return decorator  # type: ignore[return-value]


def _should_retry(exc: BaseException, retry_on: tuple[int, ...]) -> bool:
    """Determine whether *exc* is a retryable error.

    Args:
        exc: The exception that was raised.
        retry_on: HTTP status codes that are considered retryable.

    Returns:
        ``True`` if the call should be retried.
    """
    # Import here to avoid circular imports at module level.
    try:
        from relay.exceptions import RateLimitError, ServerError  # noqa: PLC0415

        if isinstance(exc, (RateLimitError, ServerError)):
            return True
    except ImportError:
        pass

    status_code: int | None = getattr(exc, "status_code", None)
    if status_code is not None and status_code in retry_on:
        return True

    return False


def _compute_delay(
    attempt: int,
    initial_backoff: float,
    backoff_multiplier: float,
    max_backoff: float,
    jitter: str,
    exc: BaseException,
) -> float:
    """Compute the backoff delay for a given attempt index.

    If the exception carries a ``retry_after`` attribute (as set by
    ``RateLimitError``), that value is used as the floor for the delay.

    Args:
        attempt: Zero-based attempt index (0 = first retry).
        initial_backoff: Base backoff in seconds.
        backoff_multiplier: Exponential growth factor.
        max_backoff: Maximum allowable backoff in seconds.
        jitter: Jitter strategy (``"none"``, ``"full"``, or ``"equal"``).
        exc: The exception that triggered this retry.

    Returns:
        The number of seconds to sleep before the next attempt.
    """
    base_delay = min(initial_backoff * (backoff_multiplier ** attempt), max_backoff)

    # Honor Retry-After if the provider supplied it.
    retry_after: float | None = getattr(exc, "retry_after", None)
    if retry_after is not None:
        base_delay = max(base_delay, float(retry_after))

    if jitter == "none":
        return base_delay
    if jitter == "full":
        return random.uniform(0.0, base_delay)
    # jitter == "equal"
    half = base_delay / 2.0
    return half + random.uniform(0.0, half)
