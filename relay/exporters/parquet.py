"""Parquet exporter for BatchResult collections.

Writes a strongly-typed Apache Parquet file using ``pyarrow``.  The schema
is declared explicitly so column types are always stable regardless of the
data present in a particular batch.

``pyarrow`` is an optional dependency.  If it is not installed an
``ImportError`` is raised with an actionable install hint.
"""

from __future__ import annotations

import asyncio
import json
from functools import partial
from typing import Any

from relay.models import BatchResult


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _require_pyarrow() -> Any:
    """Import pyarrow and return the module, or raise with an install hint.

    Returns:
        The ``pyarrow`` module.

    Raises:
        ImportError: When ``pyarrow`` is not installed, with a pip install
            hint included in the error message.
    """
    try:
        import pyarrow  # type: ignore[import-untyped]

        return pyarrow
    except ModuleNotFoundError as exc:
        raise ImportError(
            "pyarrow is required for Parquet export but is not installed.\n"
            "Install it with:  pip install pyarrow\n"
            "Or install the relay extras:  pip install relay[parquet]"
        ) from exc


def _build_schema(pa: Any, include_metadata: bool) -> Any:
    """Construct the explicit pyarrow schema for a BatchResult table.

    Args:
        pa: The imported ``pyarrow`` module.
        include_metadata: When ``True``, adds cache provenance fields
            (``from_cache``, ``cached_at``) to the schema.

    Returns:
        A ``pyarrow.Schema`` object.
    """
    fields = [
        pa.field("request_id", pa.string(), nullable=False),
        pa.field("job_id", pa.string(), nullable=False),
        pa.field("content", pa.large_utf8(), nullable=False),
        pa.field("stop_reason", pa.string(), nullable=False),
        pa.field("input_tokens", pa.int64(), nullable=False),
        pa.field("output_tokens", pa.int64(), nullable=False),
        pa.field("model", pa.string(), nullable=False),
        # error columns
        pa.field("error_code", pa.string(), nullable=True),
        pa.field("error_message", pa.string(), nullable=True),
        pa.field("error_retryable", pa.bool_(), nullable=True),
    ]

    if include_metadata:
        fields.extend(
            [
                pa.field("from_cache", pa.bool_(), nullable=False),
                pa.field("cached_at", pa.timestamp("us", tz="UTC"), nullable=True),
            ]
        )

    return pa.schema(fields)


def _results_to_columns(
    results: list[BatchResult],
    include_metadata: bool,
) -> dict[str, list[Any]]:
    """Convert a list of BatchResults into columnar lists for pyarrow.

    Args:
        results: The BatchResult objects to convert.
        include_metadata: Whether to include cache provenance columns.

    Returns:
        A dictionary mapping column name to a list of values (one per row).
    """
    cols: dict[str, list[Any]] = {
        "request_id": [],
        "job_id": [],
        "content": [],
        "stop_reason": [],
        "input_tokens": [],
        "output_tokens": [],
        "model": [],
        "error_code": [],
        "error_message": [],
        "error_retryable": [],
    }

    if include_metadata:
        cols["from_cache"] = []
        cols["cached_at"] = []

    for r in results:
        cols["request_id"].append(r.request_id)
        cols["job_id"].append(r.job_id)
        cols["content"].append(r.content)
        cols["stop_reason"].append(r.stop_reason)
        cols["input_tokens"].append(r.input_tokens)
        cols["output_tokens"].append(r.output_tokens)
        cols["model"].append(r.model)

        if r.error is not None:
            cols["error_code"].append(r.error.code)
            cols["error_message"].append(r.error.message)
            cols["error_retryable"].append(r.error.retryable)
        else:
            cols["error_code"].append(None)
            cols["error_message"].append(None)
            cols["error_retryable"].append(None)

        if include_metadata:
            cols["from_cache"].append(r.from_cache)
            # pyarrow timestamp columns require timezone-aware datetimes when
            # tz="UTC" is specified.  We store the value in microseconds since
            # the Unix epoch to avoid the timezone dependency.
            if r.cached_at is not None:
                import datetime as _dt

                epoch = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)
                aware = r.cached_at.replace(tzinfo=_dt.timezone.utc) if r.cached_at.tzinfo is None else r.cached_at
                cols["cached_at"].append(int((aware - epoch).total_seconds() * 1_000_000))
            else:
                cols["cached_at"].append(None)

    return cols


def _write_parquet_sync(
    results: list[BatchResult],
    path: str,
    include_metadata: bool,
) -> None:
    """Synchronous Parquet write executed in a thread pool.

    Args:
        results: The BatchResult objects to serialize.
        path: Destination file path.
        include_metadata: Whether to include cache provenance columns.

    Raises:
        ImportError: If ``pyarrow`` is not installed.
    """
    pa = _require_pyarrow()
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    schema = _build_schema(pa, include_metadata)
    columns = _results_to_columns(results, include_metadata)

    arrays = []
    for field in schema:
        raw = columns[field.name]
        if field.type == pa.large_utf8():
            arr = pa.array(raw, type=pa.large_utf8())
        elif field.type == pa.timestamp("us", tz="UTC"):
            arr = pa.array(raw, type=pa.timestamp("us", tz="UTC"))
        else:
            arr = pa.array(raw, type=field.type)
        arrays.append(arr)

    table = pa.table(arrays, schema=schema)
    pq.write_table(table, path, compression="snappy")


async def export_parquet(
    results: list[BatchResult],
    path: str,
    include_metadata: bool = True,
) -> None:
    """Export a list of BatchResults to an Apache Parquet file asynchronously.

    The Parquet file uses Snappy compression and an explicit, strongly-typed
    schema so downstream tools (pandas, DuckDB, Spark, BigQuery) can
    introspect column types without scanning data.

    Schema (``include_metadata=True``)::

        request_id       string (non-null)
        job_id           string (non-null)
        content          large_utf8 (non-null)
        stop_reason      string (non-null)
        input_tokens     int64  (non-null)
        output_tokens    int64  (non-null)
        model            string (non-null)
        error_code       string (nullable)
        error_message    string (nullable)
        error_retryable  bool   (nullable)
        from_cache       bool   (non-null)
        cached_at        timestamp[us, UTC] (nullable)

    The ``error_*`` columns are ``null`` for successful requests.  When
    ``include_metadata=False``, ``from_cache`` and ``cached_at`` are omitted.

    The actual file write is dispatched to a thread-pool executor so the
    event loop is not blocked.

    Args:
        results: Ordered list of ``BatchResult`` objects to export.
        path: Destination file path.  The file is created or overwritten.
        include_metadata: When ``True`` (default), includes cache provenance
            columns (``from_cache``, ``cached_at``) in the output.

    Raises:
        ImportError: If ``pyarrow`` is not installed, with an install hint.
        OSError: If the destination path is not writable.

    Example:
        >>> await export_parquet(results, "./output/job_abc.parquet")
        >>> await export_parquet(
        ...     results, "./output/job_abc.parquet", include_metadata=False
        ... )
    """
    loop = asyncio.get_running_loop()
    fn = partial(_write_parquet_sync, results, path, include_metadata)
    await loop.run_in_executor(None, fn)
