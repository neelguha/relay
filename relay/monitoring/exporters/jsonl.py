"""JSONL metric exporter for relay monitoring.

Appends one JSON record per export call to a ``.jsonl`` file on disk. Each
record contains the full metrics snapshot plus a UTC timestamp, making the
file trivially queryable with pandas, DuckDB, or ``jq``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from relay.monitoring.exporters.base import MetricExporter

logger = logging.getLogger(__name__)


class JsonlExporter(MetricExporter):
    """Appends metric snapshots as newline-delimited JSON records to a file.

    Each call to :meth:`export` produces one JSON line of the form::

        {"timestamp": "2024-01-15T12:00:00Z", "relay.requests.submitted": 42, ...}

    Writes are performed via :func:`asyncio.to_thread` to avoid blocking the
    event loop during file I/O.

    The output file is created (including parent directories) if it does not
    already exist. It is opened in append mode so existing records are never
    overwritten.

    Args:
        path: Destination file path. Accepts a string or :class:`~pathlib.Path`.
            Defaults to ``./relay_metrics.jsonl`` in the current working
            directory.
        flush_every: Number of records to accumulate before flushing the file
            buffer to disk. Set to ``1`` (the default) for durability; higher
            values reduce I/O at the cost of potential data loss on crash.

    Example::

        exporter = JsonlExporter(path="~/.relay/logs/metrics.jsonl")
        await exporter.export(collector.get_metrics())
    """

    def __init__(
        self,
        path: str | Path = "relay_metrics.jsonl",
        flush_every: int = 1,
    ) -> None:
        self._path = Path(path).expanduser().resolve()
        self._flush_every = max(1, flush_every)
        self._pending_count: int = 0
        self._lock = asyncio.Lock()
        self._file_handle = None  # opened lazily on first export

    # ── MetricExporter interface ───────────────────────────────────────────────

    async def export(self, metrics: dict[str, Any]) -> None:
        """Append a metrics snapshot as a single JSON line.

        A ``"timestamp"`` key (ISO-8601, UTC) is injected automatically.

        Args:
            metrics: Snapshot dict from
                :meth:`~relay.monitoring.metrics.MetricCollector.get_metrics`.

        Raises:
            OSError: If the output file cannot be created or written.
        """
        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metrics,
        }
        line = json.dumps(record, default=_json_default) + "\n"
        await asyncio.to_thread(self._write_line, line)

    async def close(self) -> None:
        """Flush and close the underlying file handle."""
        await asyncio.to_thread(self._close_file)

    # ── Internal sync helpers (run inside asyncio.to_thread) ──────────────────

    def _write_line(self, line: str) -> None:
        """Write one line and conditionally flush.

        Args:
            line: The newline-terminated JSON string to write.
        """
        try:
            fh = self._get_file_handle()
            fh.write(line)
            self._pending_count += 1
            if self._pending_count >= self._flush_every:
                fh.flush()
                os.fsync(fh.fileno())
                self._pending_count = 0
        except OSError:
            logger.exception("JsonlExporter: failed to write to %s", self._path)
            raise

    def _get_file_handle(self):
        """Open the file handle on first call, creating parent dirs as needed."""
        if self._file_handle is None or self._file_handle.closed:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = open(self._path, "a", encoding="utf-8")  # noqa: SIM115
        return self._file_handle

    def _close_file(self) -> None:
        """Flush remaining buffered data and close the file."""
        if self._file_handle is not None and not self._file_handle.closed:
            try:
                self._file_handle.flush()
                self._file_handle.close()
            except OSError:
                logger.exception("JsonlExporter: error closing %s", self._path)
            finally:
                self._file_handle = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        """Resolved absolute path of the output file."""
        return self._path


def _json_default(obj: Any) -> Any:
    """Custom JSON serialiser for types not handled by the stdlib encoder.

    Args:
        obj: The object that could not be serialised.

    Returns:
        A JSON-serialisable representation of *obj*.

    Raises:
        TypeError: If no conversion is possible.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")
