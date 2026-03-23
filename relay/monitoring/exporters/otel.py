"""OpenTelemetry metric exporter for relay monitoring.

Emits relay metrics to any OTLP-compatible backend (Jaeger, Honeycomb,
Datadog, Grafana Cloud, etc.) when the OpenTelemetry Python SDK is installed.

This module is a *conditional stub*: it works correctly when
``opentelemetry-sdk`` and ``opentelemetry-exporter-otlp-proto-grpc`` (or the
HTTP variant) are installed, and degrades gracefully to a no-op when they are
not.  No import errors are raised in either case.

Configuration is driven by the standard OpenTelemetry environment variables:

- ``OTEL_EXPORTER_OTLP_ENDPOINT`` — OTLP receiver URL
  (e.g. ``http://localhost:4317`` for gRPC, ``http://localhost:4318`` for HTTP)
- ``OTEL_SERVICE_NAME`` — service name attached to all metrics
  (defaults to ``"relay"``)
- ``OTEL_EXPORTER_OTLP_PROTOCOL`` — ``"grpc"`` (default) or ``"http/protobuf"``
- ``OTEL_RESOURCE_ATTRIBUTES`` — additional resource attributes as
  ``key=value,key=value``

References:
    https://opentelemetry.io/docs/specs/otel/metrics/
    https://opentelemetry-python.readthedocs.io/
"""

from __future__ import annotations

import logging
import os
from typing import Any

from relay.monitoring.exporters.base import MetricExporter

logger = logging.getLogger(__name__)

# ── Optional SDK imports ───────────────────────────────────────────────────────

try:
    from opentelemetry import metrics as otel_metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OTEL_AVAILABLE = False

# ── Metric name -> OTel instrument type mapping ────────────────────────────────
# (relay_key, otel_name, instrument_kind, unit, description)
_INSTRUMENT_DEFS: list[tuple[str, str, str, str, str]] = [
    (
        "relay.requests.submitted",
        "relay.requests.submitted",
        "counter",
        "{requests}",
        "Total requests submitted across all jobs.",
    ),
    (
        "relay.requests.completed",
        "relay.requests.completed",
        "counter",
        "{requests}",
        "Total requests completed successfully.",
    ),
    (
        "relay.requests.failed",
        "relay.requests.failed",
        "counter",
        "{requests}",
        "Total requests that returned an error.",
    ),
    (
        "relay.requests.cached",
        "relay.requests.cached",
        "counter",
        "{requests}",
        "Total requests served from the cache.",
    ),
    (
        "relay.tokens.input",
        "relay.tokens.input",
        "counter",
        "{tokens}",
        "Total input tokens consumed.",
    ),
    (
        "relay.tokens.output",
        "relay.tokens.output",
        "counter",
        "{tokens}",
        "Total output tokens generated.",
    ),
    (
        "relay.cost.estimated_usd",
        "relay.cost.estimated",
        "gauge",
        "USD",
        "Estimated cost of active jobs in US dollars.",
    ),
    (
        "relay.cost.actual_usd",
        "relay.cost.actual",
        "counter",
        "USD",
        "Confirmed spend from provider invoices in US dollars.",
    ),
    (
        "relay.cache.hit_rate",
        "relay.cache.hit_rate",
        "gauge",
        "1",
        "Rolling 5-minute cache hit rate (0.0–1.0).",
    ),
    (
        "relay.cache.size_bytes",
        "relay.cache.size",
        "gauge",
        "By",
        "Current total cache storage in bytes.",
    ),
    (
        "relay.jobs.active",
        "relay.jobs.active",
        "gauge",
        "{jobs}",
        "Number of jobs currently IN_PROGRESS.",
    ),
    (
        "relay.provider.latency_p50_ms",
        "relay.provider.latency.p50",
        "gauge",
        "ms",
        "Rolling P50 API response latency in milliseconds.",
    ),
    (
        "relay.provider.latency_p99_ms",
        "relay.provider.latency.p99",
        "gauge",
        "ms",
        "Rolling P99 API response latency in milliseconds.",
    ),
]


class OtelExporter(MetricExporter):
    """Emits relay metrics via the OpenTelemetry SDK.

    When the OpenTelemetry SDK is **not** installed, all calls are silently
    skipped and a one-time warning is logged on the first :meth:`export` call.

    When the SDK is installed, a :class:`~opentelemetry.sdk.metrics.MeterProvider`
    is configured using environment variables on the first call to :meth:`export`.
    The provider is shared across all instances within a process.

    This exporter uses *observable gauge* and *observable counter* instruments
    driven by the snapshot values, because relay already computes all
    aggregations internally. A :class:`~opentelemetry.sdk.metrics.export.PeriodicExportingMetricReader`
    is created automatically when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set.

    Args:
        service_name: Value for the ``service.name`` resource attribute.
            Defaults to the ``OTEL_SERVICE_NAME`` environment variable, then
            falls back to ``"relay"``.
        extra_resource_attrs: Additional key-value pairs merged into the OTel
            resource (e.g. ``{"deployment.environment": "production"}``).
        export_interval_ms: How often the periodic reader pushes metrics to the
            backend, in milliseconds. Defaults to 60 000 (1 minute).

    Example::

        import os
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4317"

        exporter = OtelExporter(service_name="my-relay-worker")

        # Call periodically:
        await exporter.export(collector.get_metrics())

        # Flush on shutdown:
        await exporter.close()
    """

    def __init__(
        self,
        service_name: str | None = None,
        extra_resource_attrs: dict[str, str] | None = None,
        export_interval_ms: int = 60_000,
    ) -> None:
        self._service_name: str = (
            service_name
            or os.environ.get("OTEL_SERVICE_NAME", "relay")
        )
        self._extra_attrs: dict[str, str] = extra_resource_attrs or {}
        self._export_interval_ms = export_interval_ms

        # Populated lazily on first export.
        self._meter = None
        self._instruments: dict[str, Any] = {}
        self._snapshot: dict[str, Any] = {}
        self._initialised: bool = False
        self._warned_missing_sdk: bool = False
        self._provider: Any = None  # MeterProvider, held to allow shutdown

    # ── MetricExporter interface ───────────────────────────────────────────────

    async def export(self, metrics: dict[str, Any]) -> None:
        """Update the internal snapshot and (if SDK is available) push metrics.

        On the first call, the OTel MeterProvider and instruments are created.
        Subsequent calls update the snapshot; the periodic reader handles the
        actual export schedule.

        Args:
            metrics: Snapshot dict from
                :meth:`~relay.monitoring.metrics.MetricCollector.get_metrics`.
        """
        if not _OTEL_AVAILABLE:
            if not self._warned_missing_sdk:
                logger.warning(
                    "OtelExporter: opentelemetry-sdk is not installed. "
                    "Install it with: pip install relay[otel] "
                    "or: pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc"
                )
                self._warned_missing_sdk = True
            return

        # Store the latest snapshot so observable callbacks can read it.
        self._snapshot = dict(metrics)

        if not self._initialised:
            self._setup()

    async def close(self) -> None:
        """Flush pending metrics and shut down the MeterProvider.

        Should be called when the :class:`~relay.client.BatchClient` is
        shutting down to ensure all buffered data is sent.
        """
        if not _OTEL_AVAILABLE or self._provider is None:
            return
        try:
            self._provider.shutdown()
            logger.debug("OtelExporter: MeterProvider shut down cleanly.")
        except Exception:
            logger.exception("OtelExporter: error during shutdown")

    # ── Internal setup ─────────────────────────────────────────────────────────

    def _setup(self) -> None:
        """Initialise the MeterProvider, instruments, and OTLP reader.

        Called exactly once on the first :meth:`export` call. Any exception
        during setup is logged and swallowed so relay continues to operate
        without telemetry rather than crashing.
        """
        try:
            self._provider = self._build_provider()
            otel_metrics.set_meter_provider(self._provider)
            self._meter = otel_metrics.get_meter(
                name="relay",
                version="1.0.0",
            )
            self._register_instruments()
            self._initialised = True
            logger.info(
                "OtelExporter: initialised (service=%s, endpoint=%s)",
                self._service_name,
                os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "<none>"),
            )
        except Exception:
            logger.exception("OtelExporter: setup failed; metrics will not be exported")
            self._initialised = True  # Don't retry on every call.

    def _build_provider(self) -> Any:
        """Construct a MeterProvider with an OTLP reader if an endpoint is set.

        Returns:
            A configured :class:`~opentelemetry.sdk.metrics.MeterProvider`.
        """
        resource_attrs = {SERVICE_NAME: self._service_name, **self._extra_attrs}
        resource = Resource.create(resource_attrs)

        readers = []
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        if endpoint:
            otlp_reader = self._build_otlp_reader(endpoint)
            if otlp_reader is not None:
                readers.append(otlp_reader)

        return MeterProvider(resource=resource, metric_readers=readers)

    def _build_otlp_reader(self, endpoint: str) -> Any | None:
        """Attempt to build a PeriodicExportingMetricReader with an OTLP exporter.

        Tries the gRPC exporter first, then falls back to the HTTP/protobuf
        exporter. Returns ``None`` if neither is available.

        Args:
            endpoint: The OTLP receiver base URL.

        Returns:
            A :class:`~opentelemetry.sdk.metrics.export.PeriodicExportingMetricReader`,
            or ``None`` if no OTLP exporter package is installed.
        """
        protocol = os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc").lower()

        # Try gRPC exporter.
        if protocol == "grpc":
            try:
                from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (  # noqa: PLC0415
                    OTLPMetricExporter,
                )
                otlp_exporter = OTLPMetricExporter(endpoint=endpoint)
                return PeriodicExportingMetricReader(
                    otlp_exporter,
                    export_interval_millis=self._export_interval_ms,
                )
            except ImportError:
                logger.debug(
                    "OtelExporter: grpc exporter not available, trying http/protobuf"
                )

        # Fall back to HTTP/protobuf exporter.
        try:
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import (  # noqa: PLC0415
                OTLPMetricExporter as OTLPHttpMetricExporter,
            )
            otlp_exporter = OTLPHttpMetricExporter(endpoint=endpoint)
            return PeriodicExportingMetricReader(
                otlp_exporter,
                export_interval_millis=self._export_interval_ms,
            )
        except ImportError:
            logger.warning(
                "OtelExporter: neither opentelemetry-exporter-otlp-proto-grpc "
                "nor opentelemetry-exporter-otlp-proto-http is installed. "
                "Metrics will not be exported to %s.",
                endpoint,
            )
            return None

    def _register_instruments(self) -> None:
        """Register OTel observable instruments for every relay metric."""
        for relay_key, otel_name, kind, unit, description in _INSTRUMENT_DEFS:
            if kind == "counter":
                self._instruments[relay_key] = self._meter.create_observable_counter(
                    name=otel_name,
                    callbacks=[self._make_callback(relay_key)],
                    unit=unit,
                    description=description,
                )
            else:  # gauge
                self._instruments[relay_key] = self._meter.create_observable_gauge(
                    name=otel_name,
                    callbacks=[self._make_callback(relay_key)],
                    unit=unit,
                    description=description,
                )

    def _make_callback(self, relay_key: str):
        """Return an OTel observable callback closure for a given metric key.

        The callback reads from the latest snapshot stored in
        :attr:`_snapshot`, which is updated on every :meth:`export` call.

        Args:
            relay_key: The relay metric key (e.g. ``"relay.requests.submitted"``).

        Returns:
            A callable compatible with the OTel SDK observable callback
            protocol (accepts ``options`` and yields
            :class:`~opentelemetry.sdk.metrics.Observation` instances).
        """
        try:
            from opentelemetry.metrics import Observation  # noqa: PLC0415
        except ImportError:  # pragma: no cover
            return lambda _: []

        def callback(options):  # noqa: ANN001
            value = self._snapshot.get(relay_key)
            if value is not None:
                yield Observation(float(value))

        return callback

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """``True`` if the OpenTelemetry SDK is installed and usable."""
        return _OTEL_AVAILABLE

    @property
    def service_name(self) -> str:
        """The ``service.name`` resource attribute used for all metrics."""
        return self._service_name
