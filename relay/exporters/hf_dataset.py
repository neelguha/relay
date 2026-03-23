"""HuggingFace Dataset exporter for BatchResult collections.

Converts a list of BatchResults into a ``datasets.Dataset`` and saves it to
disk in Arrow format.  Optionally pushes the dataset to the HuggingFace Hub.

``datasets`` is an optional dependency.  If it is not installed an
``ImportError`` is raised with an actionable install hint.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
from functools import partial
from typing import Any

from relay.models import BatchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_datasets() -> Any:
    """Import the ``datasets`` package and return it, or raise with a hint.

    Returns:
        The ``datasets`` module.

    Raises:
        ImportError: When ``datasets`` is not installed, with a pip install
            hint included in the error message.
    """
    try:
        import datasets  # type: ignore[import-untyped]

        return datasets
    except ModuleNotFoundError as exc:
        raise ImportError(
            "The 'datasets' package is required for HuggingFace Dataset export "
            "but is not installed.\n"
            "Install it with:  pip install datasets\n"
            "Or install the relay extras:  pip install relay[hf]"
        ) from exc


def _results_to_records(results: list[BatchResult]) -> list[dict[str, Any]]:
    """Serialize BatchResults into a list of flat dicts for ``datasets``.

    Datetime fields are converted to ISO-8601 strings so the ``datasets``
    library can infer a ``Value("string")`` feature type without any
    additional schema hints.

    Args:
        results: The BatchResult objects to convert.

    Returns:
        A list of dictionaries, one per result, with all scalar values.
    """
    records: list[dict[str, Any]] = []
    for r in results:
        record: dict[str, Any] = {
            "request_id": r.request_id,
            "job_id": r.job_id,
            "content": r.content,
            "stop_reason": r.stop_reason,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "model": r.model,
            "from_cache": r.from_cache,
            "cached_at": r.cached_at.isoformat() if r.cached_at is not None else None,
            # Flatten error so each sub-field is its own column.
            "error_code": r.error.code if r.error is not None else None,
            "error_message": r.error.message if r.error is not None else None,
            "error_retryable": r.error.retryable if r.error is not None else None,
        }
        records.append(record)
    return records


def _write_hf_dataset_sync(
    results: list[BatchResult],
    path: str,
    push_to_hub: str | None,
) -> None:
    """Synchronous HF Dataset write executed in a thread pool.

    Args:
        results: The BatchResult objects to serialize.
        path: Local destination directory for the Arrow dataset.
        push_to_hub: If provided, the dataset is pushed to the HuggingFace
            Hub after saving.  Should be a Hub repo identifier in the form
            ``"owner/dataset-name"``.

    Raises:
        ImportError: If ``datasets`` is not installed.
        ValueError: If ``push_to_hub`` is given but the Hub upload fails.
    """
    datasets = _require_datasets()

    records = _results_to_records(results)
    dataset = datasets.Dataset.from_list(records)
    dataset.save_to_disk(path)

    if push_to_hub is not None:
        dataset.push_to_hub(push_to_hub)


async def export_hf_dataset(
    results: list[BatchResult],
    path: str,
    push_to_hub: str | None = None,
) -> None:
    """Export a list of BatchResults as a HuggingFace ``datasets.Dataset``.

    The dataset is saved to ``path`` in Arrow/Parquet format using
    ``Dataset.save_to_disk()``.  If ``push_to_hub`` is given the dataset is
    also pushed to the HuggingFace Hub after the local save completes.

    The following columns are always present in the dataset::

        request_id     (string)
        job_id         (string)
        content        (string)
        stop_reason    (string)
        input_tokens   (int64)
        output_tokens  (int64)
        model          (string)
        from_cache     (bool)
        cached_at      (string, ISO-8601 or null)
        error_code     (string or null)
        error_message  (string or null)
        error_retryable (bool or null)

    The disk write and optional Hub push are dispatched to a thread-pool
    executor so the event loop is not blocked.

    Args:
        results: Ordered list of ``BatchResult`` objects to export.
        path: Local destination directory path.  Created if absent.  The
            directory is populated with Arrow shard files and a
            ``dataset_info.json`` manifest.
        push_to_hub: Optional HuggingFace Hub repository identifier in
            ``"owner/dataset-name"`` format.  When provided, the dataset is
            pushed to the Hub after the local save.  Requires a valid
            ``HUGGING_FACE_HUB_TOKEN`` environment variable or a prior call
            to ``huggingface_hub.login()``.

    Raises:
        ImportError: If the ``datasets`` package is not installed, with an
            install hint.
        OSError: If ``path`` is not writable.

    Example:
        >>> # Save locally only
        >>> await export_hf_dataset(results, "./output/hf_job_abc/")

        >>> # Save locally and push to the Hub
        >>> await export_hf_dataset(
        ...     results,
        ...     "./output/hf_job_abc/",
        ...     push_to_hub="myorg/llm-batch-results",
        ... )
    """
    loop = asyncio.get_running_loop()
    fn = partial(_write_hf_dataset_sync, results, path, push_to_hub)
    await loop.run_in_executor(None, fn)
