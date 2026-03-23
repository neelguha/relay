"""Prometheus exposition-format metric exporter for relay monitoring.

Generates the standard Prometheus text format (version 0.0.4) from a relay
metrics snapshot. The rendered text can be served directly on an HTTP
``/metrics`` endpoint, either by the built-in web dashboard or by any WSGI/ASGI
framework.

No external dependencies are required; this module produces the text format
using only the Python standard library.

References:
    https://prometheus.io/docs/instrumenting/exposition_formats/
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from relay.monitoring.exporters.base import MetricExporter

# ── Metric descriptor table ────────────────────────────────────────────────────
# Each entry: (relay_key, prom_name, prom_type, help_text)
_METRIC_DESCRIPTORS: list[tuple[str, str, str, str]] = [
    (
        "relay.requests.submitted",
        "relay_requests_submitted_total",
        "counter",
        "Total number of requests submitted across all jobs.",
    ),
    (
        "relay.requests.completed",
        "relay_requests_completed_total",
        "counter",
        "Total number of requests completed successfully.",
    ),
    (
        "relay.requests.failed",
        "relay_requests_failed_total",
        "counter",
        "Total number of requests that returned an error.",
    ),
    (
        "relay.requests.cached",
        "relay_requests_cached_total",
        "counter",
        "Total number of requests served from the cache.",
    ),
    (
        "relay.tokens.input",
        "relay_tokens_input_total",
        "counter",
        "Total input tokens consumed across all completed requests.",
    ),
    (
        "relay.tokens.output",
        "relay_tokens_output_total",
        "counter",
        "Total output tokens generated across all completed requests.",
    ),
    (
        "relay.cost.estimated_usd",
        "relay_cost_estimated_usd",
        "gauge",
        "Estimated cost in USD for currently active jobs.",
    ),
    (
        "relay.cost.actual_usd",
        "relay_cost_actual_usd_total",
        "counter",
        "Confirmed spend in USD from provider invoices.",
    ),
    (
        "relay.cache.hit_rate",
        "relay_cache_hit_rate",
        "gauge",
        "Rolling 5-minute cache hit rate (0.0–1.0).",
    ),
    (
        "relay.cache.size_bytes",
        "relay_cache_size_bytes",
        "gauge",
        "Current total cache storage in bytes.",
    ),
    (
        "relay.jobs.active",
        "relay_jobs_active",
        "gauge",
        "Number of jobs currently IN_PROGRESS.",
    ),
    (
        "relay.provider.latency_p50_ms",
        "relay_provider_latency_p50_milliseconds",
        "gauge",
        "Rolling 5-minute median (P50) API response latency in milliseconds.",
    ),
    (
        "relay.provider.latency_p99_ms",
        "relay_provider_latency_p99_milliseconds",
        "gauge",
        "Rolling 5-minute 99th-percentile (P99) API response latency in milliseconds.",
    ),
]


class PrometheusExporter(MetricExporter):
    """Renders relay metrics as Prometheus exposition-format text.

    Each call to :meth:`export` updates an internal snapshot that is then
    served via :meth:`render`. The web dashboard can wire :meth:`render` to
    its ``GET /metrics`` route.

    No Prometheus client library is required; the text format is produced
    with plain string formatting.

    Args:
        labels: Optional dict of static label key-value pairs to attach to
            every metric (e.g. ``{"instance": "worker-1", "env": "prod"}``).

    Example::

        exporter = PrometheusExporter(labels={"service": "relay"})

        # Periodically push a snapshot:
        await exporter.export(collector.get_metrics())

        # Serve the /metrics endpoint:
        text = exporter.render()

        # In a FastAPI or Starlette app:
        # @app.get("/metrics")
        # async def metrics():
        #     return PlainTextResponse(exporter.render(), media_type=CONTENT_TYPE)

    Attributes:
        CONTENT_TYPE: The MIME type to use when serving the ``/metrics``
            endpoint, per the Prometheus exposition specification.
    """

    CONTENT_TYPE: str = (
        "text/plain; version=0.0.4; charset=utf-8"
    )

    def __init__(self, labels: dict[str, str] | None = None) -> None:
        self._labels: dict[str, str] = labels or {}
        self._snapshot: dict[str, Any] = {}
        self._last_updated: float | None = None
        self._lock = asyncio.Lock()

    # ── MetricExporter interface ───────────────────────────────────────────────

    async def export(self, metrics: dict[str, Any]) -> None:
        """Update the internal metrics snapshot.

        The snapshot is held in memory and served on demand via :meth:`render`.
        No I/O is performed.

        Args:
            metrics: Snapshot dict from
                :meth:`~relay.monitoring.metrics.MetricCollector.get_metrics`.
        """
        async with self._lock:
            self._snapshot = dict(metrics)
            self._last_updated = time.time()

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(self) -> str:
        """Render the current snapshot as Prometheus exposition text.

        Returns:
            A UTF-8 string in Prometheus text format 0.0.4, ready to be
            returned from a ``GET /metrics`` HTTP handler. Returns an empty
            string if :meth:`export` has not been called yet.
        """
        snapshot = self._snapshot
        if not snapshot:
            return ""

        label_str = self._format_labels(self._labels)
        lines: list[str] = []

        for relay_key, prom_name, prom_type, help_text in _METRIC_DESCRIPTORS:
            value = snapshot.get(relay_key)
            if value is None:
                continue

            lines.append(f"# HELP {prom_name} {help_text}")
            lines.append(f"# TYPE {prom_name} {prom_type}")

            formatted_value = self._format_value(value)
            if label_str:
                lines.append(f"{prom_name}{{{label_str}}} {formatted_value}")
            else:
                lines.append(f"{prom_name} {formatted_value}")

        # Append scrape metadata.
        if self._last_updated is not None:
            ts_ms = int(self._last_updated * 1000)
            lines.append("# HELP relay_scrape_timestamp_ms Unix timestamp of the last metric export in milliseconds.")
            lines.append("# TYPE relay_scrape_timestamp_ms gauge")
            if label_str:
                lines.append(f"relay_scrape_timestamp_ms{{{label_str}}} {ts_ms}")
            else:
                lines.append(f"relay_scrape_timestamp_ms {ts_ms}")

        return "\n".join(lines) + "\n"

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _format_value(value: Any) -> str:
        """Serialise a metric value to a Prometheus-compatible string.

        ``None`` values are emitted as ``NaN`` per the Prometheus spec for
        unknown gauge values. Integer counters are kept integral; floats use
        full precision.

        Args:
            value: The metric value to format.

        Returns:
            String representation suitable for the Prometheus text format.
        """
        if value is None:
            return "NaN"
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, int):
            return str(value)
        # float
        f = float(value)
        if f != f:  # NaN check (faster than math.isnan)
            return "NaN"
        return repr(f)

    @staticmethod
    def _format_labels(labels: dict[str, str]) -> str:
        """Render a label dict to the Prometheus label syntax ``k="v",...``.

        Args:
            labels: Dict of label names to values. Values are escaped per the
                Prometheus specification.

        Returns:
            Comma-separated ``key="value"`` pairs, or an empty string if
            *labels* is empty.
        """
        if not labels:
            return ""
        parts = []
        for k, v in labels.items():
            escaped = v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            parts.append(f'{k}="{escaped}"')
        return ", ".join(parts)

    @property
    def last_updated(self) -> float | None:
        """Unix timestamp of the most recent :meth:`export` call, or ``None``."""
        return self._last_updated
