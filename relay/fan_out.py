"""Multi-provider fan-out helper for relay.

Provides :func:`fan_out`, which submits an identical set of
:class:`~relay.models.BatchRequest` objects to multiple providers in parallel
and optionally waits for all results before returning.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Union

from relay.models import BatchConfig, BatchJob, BatchRequest, BatchResult

if TYPE_CHECKING:
    # Avoid a hard circular import; BatchClient is imported at call time.
    from relay.client import BatchClient

logger = logging.getLogger(__name__)


def _config_key(config: BatchConfig) -> str:
    """Return the canonical ``'<provider>/<model>'`` key for a config.

    Args:
        config: The batch configuration whose key should be produced.

    Returns:
        A string of the form ``'<provider>/<model>'``, e.g.
        ``'anthropic/claude-opus-4-5'``.
    """
    return f"{config.provider}/{config.model}"


async def _submit_one(
    client: "BatchClient",
    requests: list[BatchRequest],
    config: BatchConfig,
) -> tuple[str, BatchJob | Exception]:
    """Submit *requests* to a single provider described by *config*.

    Failures are caught and returned as :class:`Exception` instances so that
    one provider failing does not abort the entire fan-out.

    Args:
        client: An open :class:`~relay.client.BatchClient` context.
        requests: The list of requests to submit.
        config: Provider/model configuration for this submission.

    Returns:
        A 2-tuple of ``(key, result)`` where *key* is
        ``'<provider>/<model>'`` and *result* is either a
        :class:`~relay.models.BatchJob` on success or an
        :class:`Exception` on failure.
    """
    key = _config_key(config)
    try:
        job: BatchJob = await client.submit(requests, config)
        return key, job
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "fan_out: submission failed for %s — %s: %s",
            key,
            type(exc).__name__,
            exc,
        )
        return key, exc


async def _wait_one(
    client: "BatchClient",
    key: str,
    job: BatchJob,
) -> tuple[str, list[BatchResult] | Exception]:
    """Wait for *job* to finish and download its results.

    Failures are caught and returned as :class:`Exception` instances so that
    one provider failing does not abort the entire fan-out.

    Args:
        client: An open :class:`~relay.client.BatchClient` context.
        key: The ``'<provider>/<model>'`` string identifying this job.
        job: The :class:`~relay.models.BatchJob` to poll.

    Returns:
        A 2-tuple of ``(key, result)`` where *result* is either a
        ``list[BatchResult]`` on success or an :class:`Exception` on failure.
    """
    try:
        results: list[BatchResult] = await client.wait_and_download(job.id)
        return key, results
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "fan_out: wait/download failed for %s (job %s) — %s: %s",
            key,
            job.id,
            type(exc).__name__,
            exc,
        )
        return key, exc


async def fan_out(
    client: "BatchClient",
    requests: list[BatchRequest],
    configs: list[BatchConfig],
    *,
    wait: bool = True,
) -> dict[str, list[BatchResult] | BatchJob | Exception]:
    """Submit identical requests to multiple providers in parallel.

    Each configuration in *configs* receives the same *requests* list.
    Submissions are issued concurrently via :func:`asyncio.gather`.  If
    *wait* is ``True``, the function blocks until every job has reached a
    terminal state and its results have been downloaded.

    Partial failures are handled gracefully: if one provider's submission or
    download raises an exception, the corresponding value in the returned
    dictionary is the :class:`Exception` instance rather than results.  The
    other providers are unaffected.

    Args:
        client: An open :class:`~relay.client.BatchClient` context manager.
            Must already be entered (i.e. used inside ``async with
            BatchClient() as client``).
        requests: The list of :class:`~relay.models.BatchRequest` objects to
            send to every provider.  The same list object is passed to all
            providers; no defensive copy is made.
        configs: One :class:`~relay.models.BatchConfig` per target
            provider/model combination.  Keys in the returned dictionary are
            derived from these configs as ``'<provider>/<model>'``.
        wait: When ``True`` (default), wait for all jobs to complete and
            return ``dict[str, list[BatchResult]]`` — or
            ``dict[str, Exception]`` for any providers that failed.  When
            ``False``, return the :class:`~relay.models.BatchJob` handles
            immediately without polling, giving
            ``dict[str, BatchJob | Exception]``.

    Returns:
        A dictionary keyed by ``'<provider>/<model>'``.

        * If *wait* is ``True``: values are ``list[BatchResult]`` for
          successful providers or :class:`Exception` for failed ones.
        * If *wait* is ``False``: values are :class:`~relay.models.BatchJob`
          for successfully submitted jobs or :class:`Exception` for failed
          ones.

    Example::

        async with BatchClient() as client:
            results = await fan_out(
                client,
                requests=my_requests,
                configs=[
                    BatchConfig(provider='anthropic', model='claude-opus-4-5'),
                    BatchConfig(provider='openai',    model='gpt-4o'),
                    BatchConfig(provider='google',    model='gemini-2.0-flash'),
                    BatchConfig(provider='xai',       model='grok-3'),
                ],
                wait=True,
            )
            # results: dict[str, list[BatchResult]]
            # keys are '<provider>/<model>'
    """
    # ── Phase 1: parallel submission ──────────────────────────────────────────
    submission_tasks = [
        _submit_one(client, requests, config) for config in configs
    ]
    submission_outcomes: list[tuple[str, BatchJob | Exception]] = (
        await asyncio.gather(*submission_tasks)
    )

    # Separate successful jobs from submission errors.
    jobs: dict[str, BatchJob] = {}
    output: dict[str, list[BatchResult] | BatchJob | Exception] = {}

    for key, outcome in submission_outcomes:
        if isinstance(outcome, Exception):
            output[key] = outcome
        else:
            jobs[key] = outcome

    if not wait:
        # Return BatchJob handles for successful submissions; errors already in
        # output.
        output.update(jobs)
        return output

    # ── Phase 2: parallel wait + download ─────────────────────────────────────
    if not jobs:
        # Every provider failed during submission; nothing to wait on.
        return output

    wait_tasks = [
        _wait_one(client, key, job) for key, job in jobs.items()
    ]
    wait_outcomes: list[tuple[str, list[BatchResult] | Exception]] = (
        await asyncio.gather(*wait_tasks)
    )

    for key, outcome in wait_outcomes:
        output[key] = outcome

    return output
