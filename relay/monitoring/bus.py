"""Internal event bus for relay monitoring.

Provides a thread-safe, async-compatible publish/subscribe mechanism that
fans events out to registered callbacks. All internal components emit events
through this bus; the metric collector, dashboards, and user callbacks all
subscribe to it.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from enum import Enum
from threading import Lock
from typing import Any, Callable

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Enumeration of all event types emitted by relay internals.

    Attributes:
        JOB_CREATED: A new BatchJob has been created locally.
        JOB_SUBMITTED: A BatchJob has been successfully submitted to a provider.
        JOB_PROGRESS: Incremental progress update for an in-flight job.
        JOB_COMPLETED: A BatchJob reached a terminal completed/partial state.
        JOB_FAILED: A BatchJob reached a terminal failed/cancelled state.
        REQUEST_COMPLETED: A single BatchRequest finished successfully.
        REQUEST_FAILED: A single BatchRequest returned an error.
        CACHE_HIT: A request was served from the cache.
        CACHE_MISS: A request was not found in the cache.
    """

    JOB_CREATED = "job_created"
    JOB_SUBMITTED = "job_submitted"
    JOB_PROGRESS = "job_progress"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    REQUEST_COMPLETED = "request_completed"
    REQUEST_FAILED = "request_failed"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"


# Type alias: a callback is either a plain callable or an async coroutine function.
Callback = Callable[[EventType, dict[str, Any]], Any]


class EventBus:
    """Thread-safe, async-compatible internal event bus.

    Subscribers register callbacks for specific event types or for all events.
    When an event is emitted, every matching callback is invoked. Async callbacks
    are scheduled on the running event loop (if any); sync callbacks are called
    directly. Errors inside individual callbacks are caught and logged so they
    never interrupt the emit chain.

    Example::

        bus = EventBus()

        def on_request_done(event_type, data):
            print(f"Request {data['request_id']} finished")

        bus.subscribe(EventType.REQUEST_COMPLETED, on_request_done)
        bus.emit(EventType.REQUEST_COMPLETED, {"request_id": "abc", "tokens": 42})
    """

    def __init__(self) -> None:
        # Maps event_type -> list of callbacks. None key = wildcard (all events).
        self._listeners: dict[EventType | None, list[Callback]] = defaultdict(list)
        self._lock: Lock = Lock()

    # ── Subscription ───────────────────────────────────────────────────────────

    def subscribe(
        self,
        event_type: EventType | str,
        callback: Callback,
    ) -> None:
        """Register a callback for a specific event type.

        The callback signature must be ``callback(event_type, data)`` where
        ``data`` is a plain :class:`dict`. Both synchronous and ``async def``
        callbacks are supported.

        Args:
            event_type: The :class:`EventType` (or its string value) to listen
                for.
            callback: Callable invoked whenever the event fires.

        Raises:
            ValueError: If *event_type* is not a recognised :class:`EventType`.
        """
        if isinstance(event_type, str):
            event_type = EventType(event_type)
        with self._lock:
            self._listeners[event_type].append(callback)

    def subscribe_all(self, callback: Callback) -> None:
        """Register a callback that receives every event regardless of type.

        Args:
            callback: Callable invoked for all emitted events.
        """
        with self._lock:
            self._listeners[None].append(callback)  # type: ignore[index]

    def unsubscribe(
        self,
        event_type: EventType | str,
        callback: Callback,
    ) -> None:
        """Remove a previously registered callback.

        No-op if the callback was never registered.

        Args:
            event_type: The event type the callback was registered against.
            callback: The exact callback object to remove.
        """
        if isinstance(event_type, str):
            event_type = EventType(event_type)
        with self._lock:
            listeners = self._listeners.get(event_type, [])
            try:
                listeners.remove(callback)
            except ValueError:
                pass

    # ── Emission ───────────────────────────────────────────────────────────────

    def emit(self, event_type: EventType | str, data: dict[str, Any]) -> None:
        """Emit an event synchronously, dispatching to all matching callbacks.

        Async callbacks are wrapped in :func:`asyncio.ensure_future` if a
        running event loop is detected, allowing them to run concurrently
        without blocking the caller. If no loop is running, async callbacks
        are skipped with a warning.

        Errors raised inside individual callbacks are caught, logged at WARNING
        level, and execution continues with the next callback.

        Args:
            event_type: The type of event to emit.
            data: Arbitrary payload dict forwarded verbatim to every callback.
        """
        if isinstance(event_type, str):
            event_type = EventType(event_type)

        with self._lock:
            # Collect matching + wildcard callbacks without holding the lock
            # while calling them (avoids deadlocks if a callback calls emit).
            callbacks: list[Callback] = list(self._listeners.get(event_type, []))
            callbacks += list(self._listeners.get(None, []))  # type: ignore[arg-type]

        for cb in callbacks:
            self._invoke(cb, event_type, data)

    async def emit_async(
        self, event_type: EventType | str, data: dict[str, Any]
    ) -> None:
        """Emit an event and await all async callbacks before returning.

        Unlike :meth:`emit`, this method awaits coroutine callbacks so callers
        can be sure all side effects (e.g. metrics writes) have completed before
        continuing.

        Args:
            event_type: The type of event to emit.
            data: Arbitrary payload dict forwarded to every callback.
        """
        if isinstance(event_type, str):
            event_type = EventType(event_type)

        with self._lock:
            callbacks: list[Callback] = list(self._listeners.get(event_type, []))
            callbacks += list(self._listeners.get(None, []))  # type: ignore[arg-type]

        for cb in callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(event_type, data)
                else:
                    cb(event_type, data)
            except Exception:
                logger.warning(
                    "EventBus callback %r raised an exception for event %s",
                    cb,
                    event_type,
                    exc_info=True,
                )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _invoke(
        self, callback: Callback, event_type: EventType, data: dict[str, Any]
    ) -> None:
        """Invoke a single callback, handling both sync and async cases.

        Args:
            callback: The callable to invoke.
            event_type: The event type being emitted.
            data: The event payload.
        """
        try:
            if asyncio.iscoroutinefunction(callback):
                try:
                    loop = asyncio.get_running_loop()
                    loop.call_soon_threadsafe(
                        lambda: asyncio.ensure_future(
                            callback(event_type, data), loop=loop
                        )
                    )
                except RuntimeError:
                    logger.warning(
                        "EventBus: async callback %r skipped — no running event loop",
                        callback,
                    )
            else:
                callback(event_type, data)
        except Exception:
            logger.warning(
                "EventBus callback %r raised an exception for event %s",
                callback,
                event_type,
                exc_info=True,
            )

    # ── Convenience ────────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Remove all registered callbacks. Primarily useful in tests."""
        with self._lock:
            self._listeners.clear()
