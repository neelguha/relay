"""Monitoring and observability layer for relay.

This package provides the internal event bus, metric collection, and metric
exporters that power relay's observability features: the terminal dashboard,
the web dashboard, structured logging, Prometheus integration, and
OpenTelemetry export.

Public API::

    from relay.monitoring import EventBus, MetricCollector

    bus = EventBus()
    collector = MetricCollector(bus)
"""

from relay.monitoring.bus import EventBus, EventType
from relay.monitoring.metrics import MetricCollector

__all__ = [
    "EventBus",
    "EventType",
    "MetricCollector",
]
