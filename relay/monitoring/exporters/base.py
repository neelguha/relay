"""Abstract base class for relay metric exporters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MetricExporter(ABC):
    """Abstract base class for relay metric exporters.

    Subclasses receive a metrics snapshot dict (as returned by
    :meth:`~relay.monitoring.metrics.MetricCollector.get_metrics`) and are
    responsible for forwarding it to an external system or file.

    The :meth:`export` method is ``async`` so that I/O-bound exporters (network
    calls, file writes) can yield control back to the event loop without
    blocking the monitoring pipeline.

    Example::

        class MyExporter(MetricExporter):
            async def export(self, metrics: dict[str, Any]) -> None:
                await my_http_client.post("/ingest", json=metrics)

        exporter = MyExporter()
        await exporter.export(collector.get_metrics())
    """

    @abstractmethod
    async def export(self, metrics: dict[str, Any]) -> None:
        """Export a metrics snapshot to the target backend.

        Args:
            metrics: Point-in-time snapshot as returned by
                :meth:`~relay.monitoring.metrics.MetricCollector.get_metrics`.
                Keys use the ``relay.*`` dotted namespace.

        Raises:
            Exception: Implementations may raise any exception on failure.
                Callers are responsible for handling errors gracefully.
        """

    async def close(self) -> None:
        """Release any resources held by the exporter.

        Called when the :class:`~relay.client.BatchClient` is shutting down.
        The default implementation is a no-op; override to flush buffers or
        close connections.
        """
