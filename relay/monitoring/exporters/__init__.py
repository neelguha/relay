"""Metric exporter plugins for relay monitoring.

All exporters implement :class:`MetricExporter`. Built-in implementations:

- :class:`~relay.monitoring.exporters.jsonl.JsonlExporter` — appends one JSON
  record per snapshot to a ``.jsonl`` file.
- :class:`~relay.monitoring.exporters.prometheus.PrometheusExporter` — renders
  Prometheus exposition-format text for the ``/metrics`` endpoint.
- :class:`~relay.monitoring.exporters.otel.OtelExporter` — emits metrics and
  traces via the OpenTelemetry SDK when installed.

Custom exporters can be created by subclassing :class:`MetricExporter` and
implementing :meth:`~MetricExporter.export`.

Example::

    from relay.monitoring.exporters import MetricExporter

    class MyExporter(MetricExporter):
        async def export(self, metrics: dict) -> None:
            send_to_my_backend(metrics)
"""

from relay.monitoring.exporters.base import MetricExporter

__all__ = ["MetricExporter"]
