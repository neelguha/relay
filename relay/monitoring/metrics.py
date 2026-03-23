"""Metric collection for relay monitoring.

Maintains all counters, gauges, and rolling-window statistics defined in the
relay specification. Subscribes to the :class:`~relay.monitoring.bus.EventBus`
so metrics update automatically as events flow through the system.
"""

from __future__ import annotations

import math
import time
from collections import deque
from threading import Lock
from typing import Any

from relay.monitoring.bus import EventBus, EventType


# ── Rolling-window helpers ─────────────────────────────────────────────────────


class _RollingCounter:
    """Counts events within a sliding time window.

    Each call to :meth:`record` stamps the current time. :meth:`rate` returns
    the number of events recorded in the last *window_seconds*.

    Args:
        window_seconds: Length of the sliding window in seconds.
    """

    def __init__(self, window_seconds: float = 300.0) -> None:
        self._window = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = Lock()

    def record(self) -> None:
        """Record one event at the current time."""
        now = time.monotonic()
        with self._lock:
            self._timestamps.append(now)
            self._evict(now)

    def count(self) -> int:
        """Return the number of events within the rolling window."""
        now = time.monotonic()
        with self._lock:
            self._evict(now)
            return len(self._timestamps)

    def _evict(self, now: float) -> None:
        cutoff = now - self._window
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()


class _LatencyTracker:
    """Stores raw latency samples in a rolling window for percentile queries.

    Args:
        window_seconds: Maximum age of a sample to be included in calculations.
        max_samples: Hard cap on stored samples to bound memory usage.
    """

    def __init__(
        self, window_seconds: float = 300.0, max_samples: int = 10_000
    ) -> None:
        self._window = window_seconds
        self._max_samples = max_samples
        # Deque of (monotonic_time, latency_ms) pairs.
        self._samples: deque[tuple[float, float]] = deque()
        self._lock = Lock()

    def record(self, latency_ms: float) -> None:
        """Record a single latency observation.

        Args:
            latency_ms: Observed latency in milliseconds.
        """
        now = time.monotonic()
        with self._lock:
            self._samples.append((now, latency_ms))
            self._evict(now)
            # Trim to max_samples from the oldest end.
            while len(self._samples) > self._max_samples:
                self._samples.popleft()

    def percentile(self, p: float) -> float | None:
        """Return the p-th percentile latency in milliseconds.

        Args:
            p: Percentile in [0, 100].

        Returns:
            The percentile value, or ``None`` if no samples are available.
        """
        now = time.monotonic()
        with self._lock:
            self._evict(now)
            values = sorted(s[1] for s in self._samples)
        if not values:
            return None
        if len(values) == 1:
            return values[0]
        idx = (p / 100.0) * (len(values) - 1)
        lo, hi = int(math.floor(idx)), int(math.ceil(idx))
        if lo == hi:
            return values[lo]
        frac = idx - lo
        return values[lo] * (1 - frac) + values[hi] * frac

    def _evict(self, now: float) -> None:
        cutoff = now - self._window
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()


# ── MetricCollector ────────────────────────────────────────────────────────────


class MetricCollector:
    """Collects and aggregates all relay observability metrics.

    Subscribes to a :class:`~relay.monitoring.bus.EventBus` instance and
    updates internal counters and gauges as events arrive. Callers can obtain
    a point-in-time snapshot via :meth:`get_metrics`.

    Metrics tracked:

    - ``relay.requests.submitted`` — lifetime counter
    - ``relay.requests.completed`` — lifetime counter
    - ``relay.requests.failed`` — lifetime counter
    - ``relay.requests.cached`` — lifetime counter
    - ``relay.tokens.input`` — lifetime counter
    - ``relay.tokens.output`` — lifetime counter
    - ``relay.cost.estimated_usd`` — gauge (sum of active job estimates)
    - ``relay.cost.actual_usd`` — lifetime counter
    - ``relay.cache.hit_rate`` — rolling 5-minute ratio
    - ``relay.cache.size_bytes`` — gauge (set externally)
    - ``relay.jobs.active`` — gauge
    - ``relay.provider.latency_p50_ms`` — rolling 5-minute P50
    - ``relay.provider.latency_p99_ms`` — rolling 5-minute P99

    Example::

        bus = EventBus()
        collector = MetricCollector(bus)
        collector.record_request_submitted()
        snapshot = collector.get_metrics()

    Args:
        bus: The :class:`~relay.monitoring.bus.EventBus` to subscribe to.
            Pass ``None`` to operate without automatic event integration.
        rolling_window_seconds: Window length for rate/percentile metrics.
    """

    def __init__(
        self,
        bus: EventBus | None = None,
        rolling_window_seconds: float = 300.0,
    ) -> None:
        self._lock = Lock()
        self._window = rolling_window_seconds

        # ── Lifetime counters ──────────────────────────────────────────────────
        self._requests_submitted: int = 0
        self._requests_completed: int = 0
        self._requests_failed: int = 0
        self._requests_cached: int = 0
        self._tokens_input: int = 0
        self._tokens_output: int = 0
        self._cost_actual_usd: float = 0.0

        # ── Gauges ────────────────────────────────────────────────────────────
        self._cost_estimated_usd: float = 0.0
        self._cache_size_bytes: int = 0
        self._jobs_active: int = 0

        # ── Rolling windows ───────────────────────────────────────────────────
        self._cache_hits_window = _RollingCounter(rolling_window_seconds)
        self._cache_total_window = _RollingCounter(rolling_window_seconds)
        self._latency_tracker = _LatencyTracker(rolling_window_seconds)

        if bus is not None:
            self._subscribe(bus)

    # ── EventBus integration ───────────────────────────────────────────────────

    def _subscribe(self, bus: EventBus) -> None:
        """Register handlers on the event bus.

        Args:
            bus: The event bus to listen on.
        """
        bus.subscribe(EventType.JOB_CREATED, self._on_job_created)
        bus.subscribe(EventType.JOB_SUBMITTED, self._on_job_submitted)
        bus.subscribe(EventType.JOB_COMPLETED, self._on_job_completed)
        bus.subscribe(EventType.JOB_FAILED, self._on_job_failed)
        bus.subscribe(EventType.REQUEST_COMPLETED, self._on_request_completed)
        bus.subscribe(EventType.REQUEST_FAILED, self._on_request_failed)
        bus.subscribe(EventType.CACHE_HIT, self._on_cache_hit)
        bus.subscribe(EventType.CACHE_MISS, self._on_cache_miss)

    # -- Event handlers (all accept (event_type, data) per EventBus contract) --

    def _on_job_created(self, _event_type: EventType, data: dict[str, Any]) -> None:
        estimated = float(data.get("estimated_cost_usd", 0.0))
        with self._lock:
            self._cost_estimated_usd += estimated

    def _on_job_submitted(self, _event_type: EventType, data: dict[str, Any]) -> None:
        with self._lock:
            self._jobs_active += 1

    def _on_job_completed(self, _event_type: EventType, data: dict[str, Any]) -> None:
        actual = float(data.get("actual_cost_usd", 0.0))
        estimated = float(data.get("estimated_cost_usd", 0.0))
        with self._lock:
            self._jobs_active = max(0, self._jobs_active - 1)
            self._cost_actual_usd += actual
            self._cost_estimated_usd = max(0.0, self._cost_estimated_usd - estimated)

    def _on_job_failed(self, _event_type: EventType, data: dict[str, Any]) -> None:
        estimated = float(data.get("estimated_cost_usd", 0.0))
        with self._lock:
            self._jobs_active = max(0, self._jobs_active - 1)
            self._cost_estimated_usd = max(0.0, self._cost_estimated_usd - estimated)

    def _on_request_completed(
        self, _event_type: EventType, data: dict[str, Any]
    ) -> None:
        input_tokens = int(data.get("input_tokens", 0))
        output_tokens = int(data.get("output_tokens", 0))
        latency_ms = data.get("latency_ms")
        with self._lock:
            self._requests_completed += 1
            self._tokens_input += input_tokens
            self._tokens_output += output_tokens
        if latency_ms is not None:
            self._latency_tracker.record(float(latency_ms))

    def _on_request_failed(
        self, _event_type: EventType, data: dict[str, Any]
    ) -> None:
        latency_ms = data.get("latency_ms")
        with self._lock:
            self._requests_failed += 1
        if latency_ms is not None:
            self._latency_tracker.record(float(latency_ms))

    def _on_cache_hit(self, _event_type: EventType, _data: dict[str, Any]) -> None:
        self._cache_hits_window.record()
        self._cache_total_window.record()

    def _on_cache_miss(self, _event_type: EventType, _data: dict[str, Any]) -> None:
        self._cache_total_window.record()

    # ── Explicit record methods ────────────────────────────────────────────────

    def record_request_submitted(self, count: int = 1) -> None:
        """Increment the submitted-requests counter.

        Args:
            count: Number of requests submitted in this batch.
        """
        with self._lock:
            self._requests_submitted += count

    def record_request_completed(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float | None = None,
    ) -> None:
        """Record a successfully completed request.

        Args:
            input_tokens: Number of input tokens consumed.
            output_tokens: Number of output tokens generated.
            latency_ms: End-to-end request latency in milliseconds, if known.
        """
        with self._lock:
            self._requests_completed += 1
            self._tokens_input += input_tokens
            self._tokens_output += output_tokens
        if latency_ms is not None:
            self._latency_tracker.record(latency_ms)

    def record_request_failed(self, latency_ms: float | None = None) -> None:
        """Record a failed request.

        Args:
            latency_ms: End-to-end request latency in milliseconds, if known.
        """
        with self._lock:
            self._requests_failed += 1
        if latency_ms is not None:
            self._latency_tracker.record(latency_ms)

    def record_cache_hit(self) -> None:
        """Record that a request was served from the cache."""
        with self._lock:
            self._requests_cached += 1
        self._cache_hits_window.record()
        self._cache_total_window.record()

    def record_cache_miss(self) -> None:
        """Record that a request was not found in the cache."""
        self._cache_total_window.record()

    def record_latency(self, latency_ms: float) -> None:
        """Record a provider API latency observation.

        Args:
            latency_ms: Observed latency in milliseconds.
        """
        self._latency_tracker.record(latency_ms)

    def set_cost_estimated(self, usd: float) -> None:
        """Overwrite the current estimated-cost gauge.

        Args:
            usd: New estimated cost in US dollars.
        """
        with self._lock:
            self._cost_estimated_usd = usd

    def add_cost_actual(self, usd: float) -> None:
        """Increment the confirmed-spend counter.

        Args:
            usd: Actual cost to add, in US dollars.
        """
        with self._lock:
            self._cost_actual_usd += usd

    def set_cache_size_bytes(self, size: int) -> None:
        """Update the cache size gauge.

        Args:
            size: Current cache size in bytes.
        """
        with self._lock:
            self._cache_size_bytes = size

    def set_jobs_active(self, count: int) -> None:
        """Overwrite the active-jobs gauge.

        Args:
            count: Number of currently in-progress jobs.
        """
        with self._lock:
            self._jobs_active = count

    # ── Snapshot ───────────────────────────────────────────────────────────────

    def get_metrics(self) -> dict[str, Any]:
        """Return a point-in-time snapshot of all metrics.

        Returns:
            A dict mapping each metric name (using the ``relay.*`` dotted
            namespace) to its current value. Gauges reflect the moment the
            call is made; counters are cumulative from process start.

            Example output::

                {
                    "relay.requests.submitted": 1024,
                    "relay.requests.completed": 1000,
                    "relay.requests.failed": 10,
                    "relay.requests.cached": 14,
                    "relay.tokens.input": 512000,
                    "relay.tokens.output": 128000,
                    "relay.cost.estimated_usd": 0.75,
                    "relay.cost.actual_usd": 1.23,
                    "relay.cache.hit_rate": 0.52,
                    "relay.cache.size_bytes": 104857600,
                    "relay.jobs.active": 2,
                    "relay.provider.latency_p50_ms": 340.5,
                    "relay.provider.latency_p99_ms": 980.2,
                }
        """
        with self._lock:
            requests_submitted = self._requests_submitted
            requests_completed = self._requests_completed
            requests_failed = self._requests_failed
            requests_cached = self._requests_cached
            tokens_input = self._tokens_input
            tokens_output = self._tokens_output
            cost_estimated = self._cost_estimated_usd
            cost_actual = self._cost_actual_usd
            cache_size = self._cache_size_bytes
            jobs_active = self._jobs_active

        # Rolling-window calculations (lock-free; each tracker is internally safe).
        hits = self._cache_hits_window.count()
        total = self._cache_total_window.count()
        cache_hit_rate = hits / total if total > 0 else 0.0

        p50 = self._latency_tracker.percentile(50)
        p99 = self._latency_tracker.percentile(99)

        return {
            "relay.requests.submitted": requests_submitted,
            "relay.requests.completed": requests_completed,
            "relay.requests.failed": requests_failed,
            "relay.requests.cached": requests_cached,
            "relay.tokens.input": tokens_input,
            "relay.tokens.output": tokens_output,
            "relay.cost.estimated_usd": cost_estimated,
            "relay.cost.actual_usd": cost_actual,
            "relay.cache.hit_rate": cache_hit_rate,
            "relay.cache.size_bytes": cache_size,
            "relay.jobs.active": jobs_active,
            "relay.provider.latency_p50_ms": p50,
            "relay.provider.latency_p99_ms": p99,
        }

    def reset(self) -> None:
        """Reset all counters and gauges to zero. Primarily useful in tests."""
        with self._lock:
            self._requests_submitted = 0
            self._requests_completed = 0
            self._requests_failed = 0
            self._requests_cached = 0
            self._tokens_input = 0
            self._tokens_output = 0
            self._cost_actual_usd = 0.0
            self._cost_estimated_usd = 0.0
            self._cache_size_bytes = 0
            self._jobs_active = 0
        self._cache_hits_window = _RollingCounter(self._window)
        self._cache_total_window = _RollingCounter(self._window)
        self._latency_tracker = _LatencyTracker(self._window)
