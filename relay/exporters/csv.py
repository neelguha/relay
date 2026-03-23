"""CSV exporter for BatchResult collections.

Writes a flat, header-prefixed CSV file where nested structures (``error``,
``raw_response``) are serialized as JSON strings so every row has a uniform
column layout.
"""

from __future__ import annotations

import csv
import json
import io
from typing import Any

import aiofiles

from relay.models import BatchResult


# Columns always present in the output, in declaration order.
_BASE_COLUMNS: list[str] = [
    "request_id",
    "job_id",
    "content",
    "stop_reason",
    "input_tokens",
    "output_tokens",
    "model",
    "error",
]

# Columns appended when include_metadata=True.
_METADATA_COLUMNS: list[str] = [
    "from_cache",
    "cached_at",
]


def _result_to_row(result: BatchResult, include_metadata: bool) -> dict[str, Any]:
    """Convert a BatchResult to a flat dict suitable for csv.DictWriter.

    Args:
        result: The BatchResult to convert.
        include_metadata: When ``True``, includes ``from_cache`` and
            ``cached_at`` columns.

    Returns:
        A flat dictionary whose keys match the chosen column set and whose
        values are all scalars or JSON strings.
    """
    error_value: str | None = None
    if result.error is not None:
        error_value = json.dumps(
            {
                "code": result.error.code,
                "message": result.error.message,
                "retryable": result.error.retryable,
            },
            ensure_ascii=False,
        )

    row: dict[str, Any] = {
        "request_id": result.request_id,
        "job_id": result.job_id,
        "content": result.content,
        "stop_reason": result.stop_reason,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "model": result.model,
        "error": error_value,
    }

    if include_metadata:
        row["from_cache"] = result.from_cache
        row["cached_at"] = (
            result.cached_at.isoformat() if result.cached_at is not None else None
        )

    return row


async def export_csv(
    results: list[BatchResult],
    path: str,
    include_metadata: bool = True,
) -> None:
    """Export a list of BatchResults to a CSV file asynchronously.

    The output file uses UTF-8 encoding with a BOM so it opens correctly in
    Excel without additional configuration.  All rows share the same column
    schema; nested objects are represented as JSON strings.

    Column layout (``include_metadata=True``)::

        request_id, job_id, content, stop_reason, input_tokens,
        output_tokens, model, error, from_cache, cached_at

    When ``include_metadata=False``, the ``from_cache`` and ``cached_at``
    columns are omitted.

    Args:
        results: Ordered list of ``BatchResult`` objects to export.
        path: Destination file path.  The file is created or overwritten.
        include_metadata: When ``True`` (default), includes cache provenance
            columns (``from_cache``, ``cached_at``) in the output.

    Raises:
        OSError: If the destination path is not writable.

    Example:
        >>> await export_csv(results, "./output/job_abc.csv")
        >>> await export_csv(results, "./output/job_abc.csv", include_metadata=False)
    """
    columns = _BASE_COLUMNS.copy()
    if include_metadata:
        columns.extend(_METADATA_COLUMNS)

    # Build the entire CSV in memory so we can flush a single write.  For very
    # large result sets callers should chunk the list themselves.
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=columns,
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    writer.writeheader()
    for result in results:
        writer.writerow(_result_to_row(result, include_metadata))

    csv_text = buf.getvalue()

    async with aiofiles.open(path, mode="w", encoding="utf-8-sig") as fh:
        await fh.write(csv_text)
