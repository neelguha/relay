"""JSONL exporter for BatchResult collections.

Writes one JSON object per line, suitable for streaming consumption, log
ingestion pipelines, and tools like jq or DuckDB's ``read_json``.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import aiofiles

from relay.models import BatchResult


def _result_to_dict(
    result: BatchResult,
    include_metadata: bool,
    include_raw_response: bool,
) -> dict[str, Any]:
    """Serialize a single BatchResult to a plain dictionary.

    Args:
        result: The BatchResult to serialize.
        include_metadata: When True, includes ``from_cache`` and ``cached_at``
            fields in the output.
        include_raw_response: When True, includes the provider's raw API
            response dict under the ``raw_response`` key.

    Returns:
        A JSON-serializable dictionary representing the result.
    """
    record: dict[str, Any] = {
        "request_id": result.request_id,
        "job_id": result.job_id,
        "content": result.content,
        "stop_reason": result.stop_reason,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "model": result.model,
    }

    if include_metadata:
        record["from_cache"] = result.from_cache
        record["cached_at"] = (
            result.cached_at.isoformat() if result.cached_at is not None else None
        )

    if include_raw_response:
        record["raw_response"] = result.raw_response

    if result.error is not None:
        record["error"] = {
            "code": result.error.code,
            "message": result.error.message,
            "retryable": result.error.retryable,
        }
    else:
        record["error"] = None

    return record


async def export_jsonl(
    results: list[BatchResult],
    path: str,
    include_metadata: bool = True,
    include_raw_response: bool = False,
) -> None:
    """Export a list of BatchResults to a JSONL file asynchronously.

    Each ``BatchResult`` is serialized as a single JSON object on its own
    line.  The file is written atomically line-by-line using ``aiofiles`` so
    the event loop is never blocked by I/O.

    Args:
        results: Ordered list of ``BatchResult`` objects to export.
        path: Destination file path.  The file is created or overwritten.
        include_metadata: When ``True`` (default), writes ``from_cache`` and
            ``cached_at`` fields alongside every record.
        include_raw_response: When ``True``, writes the provider's raw API
            response dict under a ``raw_response`` key.  Defaults to
            ``False`` because raw responses can be very large.

    Raises:
        OSError: If the destination path is not writable.

    Example:
        >>> await export_jsonl(results, "./output/job_abc.jsonl")
        >>> await export_jsonl(
        ...     results,
        ...     "./output/job_abc.jsonl",
        ...     include_metadata=False,
        ...     include_raw_response=True,
        ... )
    """
    async with aiofiles.open(path, mode="w", encoding="utf-8") as fh:
        for result in results:
            record = _result_to_dict(result, include_metadata, include_raw_response)
            line = json.dumps(record, ensure_ascii=False, default=str)
            await fh.write(line + "\n")
