"""Result format exporters for the relay library.

Provides a unified :func:`export_job` entry-point that dispatches to the
appropriate format-specific exporter, as well as direct access to each
individual exporter function.

Supported formats
-----------------
``jsonl``
    One JSON object per line.  Zero extra dependencies.
``csv``
    Flat CSV with nested objects serialised as JSON strings.  Zero extra
    dependencies.
``parquet``
    Strongly-typed Apache Parquet via ``pyarrow``.  Install with
    ``pip install pyarrow`` or ``pip install relay[parquet]``.
``hf_dataset``
    HuggingFace Arrow dataset (``datasets`` library required).  Install with
    ``pip install datasets`` or ``pip install relay[hf]``.

Usage
-----
::

    from relay.exporters import export_job

    await export_job(
        results=batch_results,
        format="parquet",
        output_path="./output/job_abc.parquet",
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from relay.exporters.csv import export_csv
from relay.exporters.hf_dataset import export_hf_dataset
from relay.exporters.jsonl import export_jsonl
from relay.exporters.parquet import export_parquet

if TYPE_CHECKING:
    from relay.models import BatchResult

__all__ = [
    "export_job",
    "export_jsonl",
    "export_csv",
    "export_parquet",
    "export_hf_dataset",
]

# Literal type alias used in public API signatures.
ExportFormat = Literal["jsonl", "csv", "parquet", "hf_dataset"]


async def export_job(
    results: list[BatchResult],
    format: ExportFormat,  # noqa: A002  (shadows built-in intentionally)
    output_path: str,
    *,
    include_metadata: bool = True,
    include_raw_response: bool = False,
    push_to_hub: str | None = None,
) -> str:
    """Export a list of BatchResults to a file in the requested format.

    This is the primary entry-point for result export.  It dispatches to the
    appropriate format-specific exporter and returns the resolved output path
    so callers can log or display it.

    Args:
        results: Ordered list of :class:`~relay.models.BatchResult` objects
            to export.
        format: Output format identifier.  One of ``"jsonl"``, ``"csv"``,
            ``"parquet"``, or ``"hf_dataset"``.
        output_path: Destination file or directory path.  For ``"jsonl"`` and
            ``"csv"`` this should be a file path; for ``"parquet"`` a
            ``.parquet`` file path; for ``"hf_dataset"`` a directory path.
        include_metadata: When ``True`` (default), cache provenance fields
            (``from_cache``, ``cached_at``) are included in the output.
            Ignored for ``"hf_dataset"`` (always included).
        include_raw_response: When ``True``, the provider's raw API response
            dict is embedded in each record.  Only applies to ``"jsonl"``
            exports; ignored by other formats.
        push_to_hub: HuggingFace Hub repository identifier
            (e.g. ``"myorg/dataset-name"``).  Only applies when
            ``format="hf_dataset"``.  Ignored for all other formats.

    Returns:
        The resolved ``output_path`` string as provided by the caller.

    Raises:
        ValueError: If ``format`` is not one of the supported format strings.
        ImportError: If a format with optional dependencies (``"parquet"``,
            ``"hf_dataset"``) is requested but the required package is not
            installed.
        OSError: If the destination path is not writable.

    Example:
        >>> from relay.exporters import export_job
        >>>
        >>> # JSONL — no extra dependencies
        >>> await export_job(results, "jsonl", "./output/job_abc.jsonl")
        >>>
        >>> # Parquet — requires pyarrow
        >>> await export_job(results, "parquet", "./output/job_abc.parquet")
        >>>
        >>> # HuggingFace Dataset with Hub push
        >>> await export_job(
        ...     results,
        ...     "hf_dataset",
        ...     "./output/hf_job_abc/",
        ...     push_to_hub="myorg/llm-batch-results",
        ... )
    """
    if format == "jsonl":
        await export_jsonl(
            results,
            output_path,
            include_metadata=include_metadata,
            include_raw_response=include_raw_response,
        )
    elif format == "csv":
        await export_csv(
            results,
            output_path,
            include_metadata=include_metadata,
        )
    elif format == "parquet":
        await export_parquet(
            results,
            output_path,
            include_metadata=include_metadata,
        )
    elif format == "hf_dataset":
        await export_hf_dataset(
            results,
            output_path,
            push_to_hub=push_to_hub,
        )
    else:
        supported = ", ".join(["jsonl", "csv", "parquet", "hf_dataset"])
        raise ValueError(
            f"Unsupported export format {format!r}.  "
            f"Supported formats are: {supported}"
        )

    return output_path
