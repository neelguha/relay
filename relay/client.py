"""BatchClient — central entry point for the relay library.

Owns the database connection, cache, provider adapters, and orchestrates the
full lifecycle of a batch job: validation, cost estimation, cache resolution,
submission, polling, result download, and export.

Example::

    from relay import BatchClient, BatchRequest, BatchConfig

    async with BatchClient(config='config.toml') as client:
        job = await client.submit(requests, config)
        results = await client.wait_and_download(job.id)
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator

from relay.cache.sqlite import SQLiteCache
from relay.config import load_config, RelayConfig
from relay.db.prices import get_batch_price_per_million, estimate_cost_usd
from relay.db.store import JobStore
from relay.exceptions import (
    BudgetExceeded,
    JobNotFound,
    ValidationError,
)
from relay.models import (
    BatchConfig,
    BatchError,
    BatchJob,
    BatchRequest,
    BatchResult,
    CostEstimate,
    JobProgress,
    JobStatus,
    ProviderStatus,
    RequestStatus,
)
from relay.providers import get_provider
from relay.utils.hashing import compute_cache_key
from relay.utils.tokenizer import estimate_tokens, count_message_tokens

logger = logging.getLogger(__name__)

_RESULT_COMMIT_BATCH_SIZE = 500


class BatchClient:
    """Async context manager that owns all relay resources.

    Manages the lifecycle of batch jobs across providers: validation, cost
    estimation, cache resolution, submission, polling, download, and export.

    Args:
        config: Path to a TOML config file, an already-loaded
            :class:`~relay.config.RelayConfig` instance, or ``None`` to
            discover config files automatically via the standard search order.

    Example::

        async with BatchClient(config='config.toml') as client:
            job = await client.submit(requests, batch_config)
            results = await client.wait_and_download(job.id)
    """

    def __init__(
        self,
        config: str | RelayConfig | None = None,
    ) -> None:
        if isinstance(config, RelayConfig):
            self._relay_config = config
        else:
            self._relay_config = load_config(config)

        self._store: JobStore | None = None
        self._cache: SQLiteCache | None = None
        self._providers: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "BatchClient":
        """Open the database and cache connections.

        Returns:
            The client instance, ready to use.
        """
        cfg = self._relay_config
        self._store = JobStore(cfg.db_path)
        await self._store.open()

        if cfg.cache.enabled:
            self._cache = SQLiteCache(
                db_path=cfg.db_path.replace(".db", "_cache.db"),
                max_size_gb=cfg.cache.max_size_gb,
                default_ttl=cfg.cache.ttl_seconds if cfg.cache.ttl_seconds else None,
            )
            await self._cache._ensure_open()

        return self

    async def __aexit__(self, *_: Any) -> None:
        """Close database and cache connections."""
        if self._store is not None:
            await self._store.close()
            self._store = None
        if self._cache is not None:
            await self._cache.close()
            self._cache = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cache(self) -> SQLiteCache | None:
        """Direct access to the underlying :class:`~relay.cache.sqlite.SQLiteCache`.

        Returns:
            The cache instance, or ``None`` if caching is disabled.
        """
        return self._cache

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _resolve_job_id(self, job_id_or_name: str) -> str:
        """Resolve a job ID or name to the canonical UUID.

        Args:
            job_id_or_name: A relay UUID or human-readable job name.

        Returns:
            The relay-internal UUID string.

        Raises:
            :class:`~relay.exceptions.JobNotFound`: If no match is found.
        """
        job = await self.get_job(job_id_or_name)
        return job.id

    def _require_store(self) -> JobStore:
        """Return the open job store, raising if the client is not entered.

        Returns:
            The open :class:`~relay.db.store.JobStore`.

        Raises:
            RuntimeError: If called outside the async context manager.
        """
        if self._store is None:
            raise RuntimeError(
                "BatchClient is not open. Use it as an async context manager."
            )
        return self._store

    def _get_provider(self, provider_name: str) -> Any:
        """Return a cached provider adapter, instantiating it if necessary.

        Args:
            provider_name: Provider identifier (e.g. ``"anthropic"``).

        Returns:
            An instantiated :class:`~relay.providers.base.BaseProvider`.
        """
        key = provider_name.lower()
        if key not in self._providers:
            cfg = self._relay_config
            provider_cfg_map: dict[str, Any] = {
                "anthropic": dataclasses.asdict(cfg.providers.anthropic),
                "openai": dataclasses.asdict(cfg.providers.openai),
                "google": dataclasses.asdict(cfg.providers.google),
                "xai": dataclasses.asdict(cfg.providers.xai),
            }
            provider_config = provider_cfg_map.get(key, {})
            self._providers[key] = get_provider(key, provider_config)
        return self._providers[key]

    @staticmethod
    def _validate_requests(requests: list[BatchRequest]) -> None:
        """Validate that all message content is text-only strings.

        Args:
            requests: The list of requests to validate.

        Raises:
            :class:`~relay.exceptions.ValidationError`: If any message
                content is not a plain string.
        """
        for req in requests:
            for msg in req.messages:
                content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
                if not isinstance(content, str):
                    raise ValidationError(
                        f"Request {req.id!r}: message content must be a plain text "
                        f"string, got {type(content).__name__!r}. "
                        "Relay supports text-only content."
                    )

    def _compute_request_cache_key(
        self, req: BatchRequest, config: BatchConfig
    ) -> str:
        """Compute the SHA-256 cache key for a single request.

        Args:
            req: The batch request.
            config: The batch-level configuration (supplies provider and model).

        Returns:
            64-character lowercase hex SHA-256 digest.
        """
        model = req.model or config.model
        messages: list[dict[str, str]] = (
            [dict(m) for m in req.messages]
            if req.messages and isinstance(req.messages[0], dict)
            else [dataclasses.asdict(m) for m in req.messages]
        )
        return compute_cache_key(
            provider=config.provider,
            model=model,
            system_prompt=req.system,
            user_messages=messages,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            top_p=req.top_p,
            stop_sequences=list(req.stop_sequences),
        )

    def _estimate_request_tokens(
        self, req: BatchRequest, config: BatchConfig
    ) -> tuple[int, int]:
        """Estimate input and output tokens for a single request.

        Uses the provider adapter when available, otherwise falls back to the
        character-based heuristic.

        Args:
            req: The batch request.
            config: The batch-level configuration.

        Returns:
            ``(input_tokens, output_tokens)`` tuple.
        """
        try:
            provider = self._get_provider(config.provider)
            return provider.estimate_tokens(req)
        except Exception:
            messages = req.messages or []
            input_tokens = count_message_tokens(
                [m if isinstance(m, dict) else dataclasses.asdict(m) for m in messages],
                model=req.model or config.model,
            )
            if req.system:
                input_tokens += estimate_tokens(req.system)
            output_tokens = req.max_tokens
            return input_tokens, output_tokens

    def _check_cost_limits(self, estimated_cost: float) -> None:
        """Enforce hard cost limits from configuration.

        Args:
            estimated_cost: Estimated batch cost in USD.

        Raises:
            :class:`~relay.exceptions.BudgetExceeded`: If the estimate
                exceeds the configured hard limit.
        """
        hard_limit = self._relay_config.cost.hard_limit_usd
        if hard_limit > 0 and estimated_cost > hard_limit:
            raise BudgetExceeded(
                f"Estimated cost ${estimated_cost:.4f} exceeds the configured "
                f"hard limit of ${hard_limit:.2f}. Adjust relay.cost.hard_limit_usd "
                "in your config to raise the limit."
            )
        warn_threshold = self._relay_config.cost.warn_threshold_usd
        if estimated_cost > warn_threshold:
            logger.warning(
                "Estimated batch cost $%.4f exceeds the warning threshold of $%.2f.",
                estimated_cost,
                warn_threshold,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit(
        self,
        requests: list[BatchRequest],
        config: BatchConfig,
    ) -> BatchJob:
        """Validate, estimate cost, resolve cache, and submit a batch.

        The job is returned immediately after submission. Use :meth:`wait`
        or :meth:`wait_and_download` to block until results are ready.

        Args:
            requests: The list of :class:`~relay.models.BatchRequest` objects
                to submit.
            config: Batch-level configuration including provider, model, and
                cache settings.

        Returns:
            A :class:`~relay.models.BatchJob` handle with the initial status.

        Raises:
            :class:`~relay.exceptions.ValidationError`: If any request
                contains non-text content.
            :class:`~relay.exceptions.BudgetExceeded`: If the estimated cost
                exceeds the configured hard limit.
        """
        store = self._require_store()

        if not requests:
            raise ValidationError("requests list must not be empty.")

        # --- PENDING ---
        job_id = str(uuid.uuid4())
        now = datetime.utcnow()
        job = BatchJob(
            id=job_id,
            provider_job_id="",
            provider=config.provider,
            model=config.model,
            project=config.project,
            status=JobStatus.PENDING,
            total_requests=len(requests),
            name=config.name,
            created_at=now,
            tags=list(config.tags),
        )
        config_dict = {
            k: v
            for k, v in dataclasses.asdict(config).items()
            if k not in {"on_progress", "on_complete"}
        }
        await store.create_job(job, config_dict)

        # --- VALIDATING ---
        job.status = JobStatus.VALIDATING
        await store.update_job_status(job_id, JobStatus.VALIDATING)

        self._validate_requests(requests)

        # Cost estimation
        total_input = 0
        total_output = 0
        for req in requests:
            inp, out = self._estimate_request_tokens(req, config)
            total_input += inp
            total_output += out

        try:
            estimated_cost = estimate_cost_usd(
                config.model, total_input, total_output, use_batch_pricing=True
            )
        except KeyError:
            logger.warning(
                "Model %r not found in pricing table; cost estimate will be 0.",
                config.model,
            )
            estimated_cost = 0.0

        job.input_tokens = total_input
        job.output_tokens = total_output
        job.estimated_cost_usd = estimated_cost
        await store.update_job(job)

        self._check_cost_limits(estimated_cost)

        # --- CACHE_RESOLVING ---
        job.status = JobStatus.CACHE_RESOLVING
        await store.update_job_status(job_id, JobStatus.CACHE_RESOLVING)

        requests_to_submit: list[BatchRequest] = []
        cache_results: list[BatchResult] = []
        cache_hit_count = 0

        use_cache = config.use_cache and self._cache is not None

        # Persist all requests first (pending), then resolve cache hits
        request_records = []
        cache_keys: dict[str, str] = {}  # request_id -> cache_key

        # Map original request IDs to job-scoped storage IDs to avoid
        # collisions when the same JSONL file is submitted multiple times.
        req_id_map: dict[str, str] = {}  # original_id -> storage_id

        for req in requests:
            storage_id = f"{job_id}:{req.id}"
            req_id_map[req.id] = storage_id
            ck = self._compute_request_cache_key(req, config)
            cache_keys[req.id] = ck
            request_records.append(
                {
                    "id": storage_id,
                    "job_id": job_id,
                    "payload": dataclasses.asdict(req),
                    "cache_key": ck,
                    "status": RequestStatus.PENDING.value,
                }
            )

        await store.create_requests_bulk(request_records)

        if use_cache:
            for req in requests:
                ck = cache_keys[req.id]
                cached = await self._cache.get(ck)  # type: ignore[union-attr]
                if cached is not None:
                    cache_hit_count += 1
                    cr = BatchResult(
                        request_id=req_id_map[req.id],
                        job_id=job_id,
                        content=cached.get("content", ""),
                        stop_reason=cached.get("stop_reason", ""),
                        input_tokens=cached.get("input_tokens", 0),
                        output_tokens=cached.get("output_tokens", 0),
                        model=cached.get("model", req.model or config.model),
                        from_cache=True,
                        cached_at=datetime.utcnow(),
                        raw_response=cached,
                    )
                    cache_results.append(cr)
                else:
                    requests_to_submit.append(req)
        else:
            requests_to_submit = list(requests)

        job.cached_hits = cache_hit_count
        await store.update_job(job)

        # Persist cache results immediately
        if cache_results:
            for batch_start in range(0, len(cache_results), _RESULT_COMMIT_BATCH_SIZE):
                chunk = cache_results[batch_start : batch_start + _RESULT_COMMIT_BATCH_SIZE]
                await store.create_results_bulk(chunk)

        # If everything was cached, complete immediately
        if not requests_to_submit:
            job.status = JobStatus.COMPLETED
            job.completed_requests = cache_hit_count
            job.completed_at = datetime.utcnow()
            job.actual_cost_usd = 0.0
            await store.update_job(job)
            logger.info(
                "Job %s: all %d requests served from cache. No API call made.",
                job_id,
                cache_hit_count,
            )
            return job

        # --- SUBMITTING ---
        job.status = JobStatus.SUBMITTING
        job.submitted_at = datetime.utcnow()
        await store.update_job(job)

        try:
            provider = self._get_provider(config.provider)
            serialised = [dataclasses.asdict(r) for r in requests_to_submit]
            provider_job_id = await provider.submit_batch(serialised)
        except Exception as exc:
            job.status = JobStatus.FAILED
            job.error = str(exc)
            await store.update_job(job)
            raise

        # --- IN_PROGRESS ---
        job.provider_job_id = provider_job_id
        job.status = JobStatus.IN_PROGRESS
        job.total_requests = len(requests)
        await store.update_job(job)

        logger.info(
            "Job %s submitted as provider job %r. %d requests pending (%d from cache).",
            job_id,
            provider_job_id,
            len(requests_to_submit),
            cache_hit_count,
        )
        return job

    async def get_job(
        self, job_id: str, *, project: str | None = None
    ) -> BatchJob:
        """Return the current snapshot of a job.

        Accepts either a relay-internal UUID or a human-readable job name.
        UUIDs are tried first; if no match is found the argument is treated
        as a name and looked up via
        :meth:`~relay.db.store.JobStore.get_job_by_name`.

        Args:
            job_id: The relay-internal job UUID **or** human-readable name.
            project: Optional project scope for name-based lookup.

        Returns:
            A :class:`~relay.models.BatchJob` with the latest field values.

        Raises:
            :class:`~relay.exceptions.JobNotFound`: If no job with the given
                ID or name exists in the store.
        """
        store = self._require_store()
        job = await store.get_job(job_id)
        if job is None:
            # Fall back to name-based lookup
            job = await store.get_job_by_name(job_id, project=project)
        if job is None:
            raise JobNotFound(f"Job {job_id!r} not found.")

        # Poll the provider for fresh status if the job is still in progress.
        if job.status == JobStatus.IN_PROGRESS and job.provider_job_id:
            try:
                provider = self._get_provider(job.provider)
                pstatus: ProviderStatus = await provider.get_status(
                    job.provider_job_id
                )
                job.completed_requests = pstatus.completed + job.cached_hits
                job.failed_requests = pstatus.failed

                provider_status_str = (pstatus.status or "").lower()
                if provider_status_str in {
                    "ended", "completed", "succeeded", "done",
                }:
                    job.status = JobStatus.DOWNLOADING
                elif provider_status_str in {"failed", "error", "expired"}:
                    job.status = JobStatus.FAILED
                    job.error = pstatus.error or provider_status_str
                elif provider_status_str in {"cancelled", "canceled"}:
                    job.status = JobStatus.CANCELLED

                await store.update_job(job)
            except Exception:
                logger.debug("Provider poll failed for job %s, returning stale status.", job.id)

        return job

    async def list_jobs(self, **filters: Any) -> list[BatchJob]:
        """Query jobs with optional filters.

        Keyword arguments are forwarded directly to
        :meth:`~relay.db.store.JobStore.list_jobs`.

        Args:
            **filters: Supported keys: ``provider``, ``model``, ``status``,
                ``project``, ``tags``, ``after``, ``before``, ``order_by``,
                ``descending``, ``limit``, ``offset``.

        Returns:
            A list of :class:`~relay.models.BatchJob` objects matching the
            filters.
        """
        store = self._require_store()
        return await store.list_jobs(**filters)

    async def wait(
        self,
        job_id: str,
        poll_interval: float = 5.0,
    ) -> AsyncGenerator[JobProgress, None]:
        """Async generator that yields progress until the job reaches a terminal state.

        The generator is safe to cancel. It polls the provider for status
        updates and yields a :class:`~relay.models.JobProgress` snapshot on
        each poll.

        Args:
            job_id: The relay-internal job UUID.
            poll_interval: Seconds between provider status polls. Defaults to
                5.0.

        Yields:
            :class:`~relay.models.JobProgress` snapshots until the job
            terminates.

        Raises:
            :class:`~relay.exceptions.JobNotFound`: If the job does not exist.
        """
        job_id = await self._resolve_job_id(job_id)
        store = self._require_store()

        start_time = time.monotonic()

        while True:
            job = await store.get_job(job_id)
            if job is None:
                raise JobNotFound(f"Job {job_id!r} disappeared.")

            elapsed = time.monotonic() - start_time

            # Compute ETA
            eta: float | None = None
            if job.completed_requests > 0 and job.total_requests > 0:
                rate = job.completed_requests / max(elapsed, 0.001)
                remaining = job.total_requests - job.completed_requests - job.failed_requests
                if rate > 0 and remaining > 0:
                    eta = remaining / rate

            progress = JobProgress(
                job_id=job_id,
                status=job.status,
                total=job.total_requests,
                completed=job.completed_requests,
                failed=job.failed_requests,
                cached=job.cached_hits,
                cost_so_far=job.actual_cost_usd or job.estimated_cost_usd,
                elapsed_seconds=elapsed,
                eta_seconds=eta,
            )

            yield progress

            if job.status.is_terminal:
                break

            # Poll provider for fresh status if job is in a non-terminal,
            # non-local state.
            if job.status == JobStatus.IN_PROGRESS and job.provider_job_id:
                try:
                    provider = self._get_provider(job.provider)
                    pstatus: ProviderStatus = await provider.get_status(
                        job.provider_job_id
                    )
                    job.completed_requests = pstatus.completed + job.cached_hits
                    job.failed_requests = pstatus.failed

                    # Map terminal provider states
                    provider_status_str = (pstatus.status or "").lower()
                    if provider_status_str in {"ended", "completed", "succeeded", "done"}:
                        job.status = JobStatus.DOWNLOADING
                    elif provider_status_str in {"failed", "error", "expired"}:
                        job.status = JobStatus.FAILED
                        job.error = pstatus.error or provider_status_str
                    elif provider_status_str in {"canceling", "cancelled", "canceled"}:
                        job.status = JobStatus.CANCELLED

                    await store.update_job(job)
                except Exception as exc:
                    logger.warning(
                        "Job %s: error polling provider status: %s", job_id, exc
                    )

            if not job.status.is_terminal:
                await asyncio.sleep(poll_interval)

    async def download(self, job_id: str) -> list[BatchResult]:
        """Download and parse results from the provider.

        Populates cache entries for each new result and commits to the
        database in batches of 500. If the job is not yet in a terminal or
        DOWNLOADING state, raises a :class:`~relay.exceptions.ValidationError`.

        Args:
            job_id: The relay-internal job UUID.

        Returns:
            A list of :class:`~relay.models.BatchResult` objects (including
            any cache hits stored during :meth:`submit`).

        Raises:
            :class:`~relay.exceptions.JobNotFound`: If the job does not exist.
            :class:`~relay.exceptions.ValidationError`: If the job is not in a
                downloadable state.
        """
        job_id = await self._resolve_job_id(job_id)
        store = self._require_store()
        job = await store.get_job(job_id)

        if job.status not in {
            JobStatus.DOWNLOADING,
            JobStatus.COMPLETED,
            JobStatus.PARTIAL,
            JobStatus.IN_PROGRESS,
        }:
            raise ValidationError(
                f"Job {job_id!r} is in status {job.status.value!r} and cannot be "
                "downloaded. Wait for the job to reach IN_PROGRESS or DOWNLOADING."
            )

        # Transition to DOWNLOADING
        if job.status != JobStatus.COMPLETED:
            job.status = JobStatus.DOWNLOADING
            await store.update_job_status(job_id, JobStatus.DOWNLOADING)

        # Load already-stored results (cache hits from submit)
        existing_results = await store.list_results(job_id, limit=10_000_000)
        existing_ids = {r["request_id"] for r in existing_results}

        batch_results: list[BatchResult] = []

        # Re-constitute cached results
        for row in existing_results:
            if row.get("from_cache"):
                br = BatchResult(
                    request_id=row["request_id"],
                    job_id=job_id,
                    content=row.get("content") or "",
                    stop_reason=row.get("stop_reason") or "",
                    input_tokens=row.get("input_tokens") or 0,
                    output_tokens=row.get("output_tokens") or 0,
                    model=job.model,
                    from_cache=True,
                    raw_response=row.get("response_json") or {},
                )
                batch_results.append(br)

        # Download new results from provider (skip if already terminal+downloaded)
        if job.status != JobStatus.COMPLETED and job.provider_job_id:
            try:
                provider = self._get_provider(job.provider)
                raw_results = await provider.download_results(job.provider_job_id)
            except Exception as exc:
                job.status = JobStatus.FAILED
                job.error = str(exc)
                await store.update_job(job)
                raise

            new_results: list[BatchResult] = []
            total_input = 0
            total_output = 0

            for raw in raw_results:
                orig_request_id = raw.get("request_id", "")
                # Map provider's request ID to our storage ID
                request_id = f"{job_id}:{orig_request_id}"
                if request_id in existing_ids:
                    continue

                error_data = raw.get("error")
                batch_error: BatchError | None = None
                if error_data:
                    batch_error = BatchError(
                        code=error_data.get("code", "unknown"),
                        message=error_data.get("message", ""),
                        retryable=error_data.get("retryable", False),
                    )

                inp_tok = int(raw.get("input_tokens") or 0)
                out_tok = int(raw.get("output_tokens") or 0)
                total_input += inp_tok
                total_output += out_tok

                br = BatchResult(
                    request_id=request_id,
                    job_id=job_id,
                    content=raw.get("content") or "",
                    stop_reason=raw.get("stop_reason") or "",
                    input_tokens=inp_tok,
                    output_tokens=out_tok,
                    model=raw.get("model") or job.model,
                    from_cache=False,
                    raw_response={k: v for k, v in raw.items() if k != "error"},
                    error=batch_error,
                )
                new_results.append(br)

            # Commit in batches of 500
            for batch_start in range(0, len(new_results), _RESULT_COMMIT_BATCH_SIZE):
                chunk = new_results[batch_start : batch_start + _RESULT_COMMIT_BATCH_SIZE]
                await store.create_results_bulk(chunk)

            # Populate cache for new results
            if self._cache is not None:
                # We need the original requests to compute cache keys
                request_rows = await store.list_requests(job_id, limit=10_000_000)
                key_map = {r["id"]: r.get("cache_key") for r in request_rows}
                cfg_row = None
                # Reconstruct a minimal config for cache TTL
                ttl = self._relay_config.cache.ttl_seconds or None

                for br in new_results:
                    if br.error is not None:
                        continue
                    ck = key_map.get(br.request_id)
                    if ck:
                        try:
                            await self._cache.put(
                                cache_key=ck,
                                provider=job.provider,
                                model=br.model,
                                response=br.raw_response,
                                input_tokens=br.input_tokens,
                                output_tokens=br.output_tokens,
                                ttl=ttl,
                            )
                        except Exception as exc:
                            logger.warning(
                                "Job %s: failed to cache result for request %r: %s",
                                job_id,
                                br.request_id,
                                exc,
                            )

            # Recalculate totals
            completed_count = sum(1 for r in new_results if r.error is None)
            failed_count = sum(1 for r in new_results if r.error is not None)

            # Determine terminal status
            if failed_count > 0 and completed_count > 0:
                final_status = JobStatus.PARTIAL
            elif failed_count > 0 and completed_count == 0:
                final_status = JobStatus.FAILED
            else:
                final_status = JobStatus.COMPLETED

            try:
                actual_cost = estimate_cost_usd(
                    job.model, total_input, total_output, use_batch_pricing=True
                )
            except KeyError:
                actual_cost = 0.0

            job.status = final_status
            job.completed_requests = completed_count + job.cached_hits
            job.failed_requests = failed_count
            job.input_tokens = total_input
            job.output_tokens = total_output
            job.actual_cost_usd = actual_cost
            job.completed_at = datetime.utcnow()
            await store.update_job(job)

            batch_results.extend(new_results)
        else:
            # Job was already COMPLETED (all from cache or re-download)
            for row in existing_results:
                if not row.get("from_cache"):
                    br = BatchResult(
                        request_id=row["request_id"],
                        job_id=job_id,
                        content=row.get("content") or "",
                        stop_reason=row.get("stop_reason") or "",
                        input_tokens=row.get("input_tokens") or 0,
                        output_tokens=row.get("output_tokens") or 0,
                        model=job.model,
                        from_cache=False,
                        raw_response=row.get("response_json") or {},
                    )
                    batch_results.append(br)

        logger.info(
            "Job %s: downloaded %d results (%d cached, %d new).",
            job_id,
            len(batch_results),
            job.cached_hits,
            len(batch_results) - job.cached_hits,
        )

        # Strip the job-scoped prefix from request IDs so the user sees
        # their original IDs (e.g. "req-001" not "job-uuid:req-001").
        prefix = f"{job_id}:"
        cleaned: list[BatchResult] = []
        for br in batch_results:
            rid = br.request_id
            if rid.startswith(prefix):
                rid = rid[len(prefix):]
            cleaned.append(dataclasses.replace(br, request_id=rid))
        return cleaned

    async def wait_and_download(self, job_id: str) -> list[BatchResult]:
        """Wait for a job to complete, then download and return results.

        Combines :meth:`wait` and :meth:`download` into a single convenience
        call. Progress events are logged at DEBUG level.

        Args:
            job_id: The relay-internal job UUID.

        Returns:
            A list of :class:`~relay.models.BatchResult` objects.

        Raises:
            :class:`~relay.exceptions.JobNotFound`: If the job does not exist.
        """
        async for progress in self.wait(job_id):
            logger.debug(
                "Job %s [%s]: %d/%d completed, %d failed, %d cached, "
                "elapsed=%.1fs",
                job_id,
                progress.status.value,
                progress.completed,
                progress.total,
                progress.failed,
                progress.cached,
                progress.elapsed_seconds,
            )
            if progress.status.is_terminal:
                break

        return await self.download(job_id)

    async def cancel(self, job_id: str) -> BatchJob:
        """Cancel an in-progress batch job.

        Cancellation is best-effort; requests already processed by the
        provider may still be billed.

        Args:
            job_id: The relay-internal job UUID.

        Returns:
            The updated :class:`~relay.models.BatchJob` with status
            ``CANCELLED``.

        Raises:
            :class:`~relay.exceptions.JobNotFound`: If the job does not exist.
            :class:`~relay.exceptions.ValidationError`: If the job is already
                in a terminal state.
        """
        job_id = await self._resolve_job_id(job_id)
        store = self._require_store()
        job = await store.get_job(job_id)

        if job.status.is_terminal:
            raise ValidationError(
                f"Job {job_id!r} is already in terminal state {job.status.value!r} "
                "and cannot be cancelled."
            )

        if job.provider_job_id:
            try:
                provider = self._get_provider(job.provider)
                await provider.cancel(job.provider_job_id)
            except Exception as exc:
                logger.warning(
                    "Job %s: provider cancel request failed: %s. "
                    "Marking as CANCELLED locally.",
                    job_id,
                    exc,
                )

        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.utcnow()
        await store.update_job(job)
        return job

    async def resubmit_failed(self, job_id: str) -> BatchJob:
        """Create a new job with only the failed requests from a completed job.

        Args:
            job_id: The relay-internal job UUID of the original job.

        Returns:
            A new :class:`~relay.models.BatchJob` containing only the failed
            requests.

        Raises:
            :class:`~relay.exceptions.JobNotFound`: If the original job does
                not exist.
            :class:`~relay.exceptions.ValidationError`: If the original job
                has not reached a terminal state, or has no failed requests.
        """
        job_id = await self._resolve_job_id(job_id)
        store = self._require_store()
        original = await store.get_job(job_id)

        if not original.status.is_terminal:
            raise ValidationError(
                f"Job {job_id!r} is not yet in a terminal state "
                f"(current: {original.status.value!r}). Wait for it to complete."
            )

        # Find failed request records
        failed_recs = await store.list_requests(
            job_id, status=RequestStatus.FAILED, limit=10_000_000
        )
        if not failed_recs:
            raise ValidationError(
                f"Job {job_id!r} has no failed requests to resubmit."
            )

        # Reconstruct BatchRequest objects from persisted payloads
        failed_requests: list[BatchRequest] = []
        for rec in failed_recs:
            payload = rec["payload_json"]
            failed_requests.append(
                BatchRequest(
                    id=payload.get("id", str(uuid.uuid4())),
                    messages=payload.get("messages", []),
                    system=payload.get("system"),
                    model=payload.get("model"),
                    max_tokens=payload.get("max_tokens", 1024),
                    temperature=payload.get("temperature", 1.0),
                    top_p=payload.get("top_p"),
                    stop_sequences=payload.get("stop_sequences", []),
                    metadata=payload.get("metadata", {}),
                    tags=payload.get("tags", []),
                )
            )

        # Reconstruct BatchConfig from stored config_json
        original_rec = await store.get_job(job_id)
        # Re-submit with the same config (recovered from tags/provider/model)
        new_config = BatchConfig(
            provider=original.provider,
            model=original.model,
            project=original.project,
            tags=list(original.tags),
            use_cache=False,  # skip cache so we actually re-run the failures
        )

        return await self.submit(failed_requests, new_config)

    async def export(
        self,
        job_id: str,
        format: str,
        path: str,
    ) -> None:
        """Write job results to disk in the specified format.

        Supported formats: ``"jsonl"``, ``"csv"``, ``"parquet"``,
        ``"hf_dataset"``.

        Args:
            job_id: The relay-internal job UUID.
            format: Output format string (case-insensitive).
            path: Filesystem path to write the output file or directory.

        Raises:
            :class:`~relay.exceptions.JobNotFound`: If the job does not exist.
            :class:`~relay.exceptions.ValidationError`: If *format* is not
                supported.
        """
        fmt = format.lower()
        supported = {"jsonl", "csv", "parquet", "hf_dataset"}
        if fmt not in supported:
            raise ValidationError(
                f"Unsupported export format {format!r}. "
                f"Supported formats: {sorted(supported)}"
            )

        job_id = await self._resolve_job_id(job_id)
        store = self._require_store()

        result_rows = await store.list_results(job_id, limit=10_000_000)
        records = [
            {
                "request_id": r["request_id"],
                "job_id": r["job_id"],
                "content": r.get("content"),
                "stop_reason": r.get("stop_reason"),
                "input_tokens": r.get("input_tokens"),
                "output_tokens": r.get("output_tokens"),
                "from_cache": bool(r.get("from_cache")),
            }
            for r in result_rows
        ]

        import asyncio
        loop = asyncio.get_event_loop()

        if fmt == "jsonl":
            await loop.run_in_executor(None, _write_jsonl, records, path)
        elif fmt == "csv":
            await loop.run_in_executor(None, _write_csv, records, path)
        elif fmt == "parquet":
            await loop.run_in_executor(None, _write_parquet, records, path)
        elif fmt == "hf_dataset":
            await loop.run_in_executor(None, _write_hf_dataset, records, path)

        logger.info("Job %s: exported %d results as %s to %r.", job_id, len(records), fmt, path)

    async def estimate_cost(
        self,
        requests: list[BatchRequest],
        config: BatchConfig,
    ) -> CostEstimate:
        """Dry-run cost estimation without submitting any requests.

        Checks the cache to determine how many requests already have results,
        then estimates the token count and USD cost for the remainder.

        Args:
            requests: The list of :class:`~relay.models.BatchRequest` objects
                to estimate.
            config: Batch-level configuration.

        Returns:
            A :class:`~relay.models.CostEstimate` with per-provider breakdown.
        """
        self._validate_requests(requests)

        cache_hits = 0
        use_cache = config.use_cache and self._cache is not None

        if use_cache:
            for req in requests:
                ck = self._compute_request_cache_key(req, config)
                cached = await self._cache.get(ck)  # type: ignore[union-attr]
                if cached is not None:
                    cache_hits += 1

        net_requests = len(requests) - cache_hits

        total_input = 0
        total_output = 0
        cache_input = 0
        cache_output = 0

        for i, req in enumerate(requests):
            inp, out = self._estimate_request_tokens(req, config)
            if use_cache and i < cache_hits:
                cache_input += inp
                cache_output += out
            else:
                total_input += inp
                total_output += out

        try:
            gross_usd = estimate_cost_usd(
                config.model,
                total_input + cache_input,
                total_output + cache_output,
                use_batch_pricing=True,
            )
            net_usd = estimate_cost_usd(
                config.model, total_input, total_output, use_batch_pricing=True
            )
            saved_usd = gross_usd - net_usd
        except KeyError:
            gross_usd = net_usd = saved_usd = 0.0

        return CostEstimate(
            total_requests=len(requests),
            cache_hits=cache_hits,
            net_requests=net_requests,
            input_tokens=total_input,
            estimated_output_tokens=total_output,
            gross_usd=gross_usd,
            saved_usd=saved_usd,
            net_usd=net_usd,
            per_provider={config.provider: net_usd},
        )


# ---------------------------------------------------------------------------
# Export helpers (synchronous, run in thread pool by export())
# ---------------------------------------------------------------------------


def _write_jsonl(records: list[dict], path: str) -> None:
    """Write records to a JSONL file.

    Args:
        records: List of result dicts.
        path: Output file path.
    """
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_csv(records: list[dict], path: str) -> None:
    """Write records to a CSV file.

    Args:
        records: List of result dicts.
        path: Output file path.
    """
    import csv

    if not records:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            fh.write("")
        return

    fieldnames = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _write_parquet(records: list[dict], path: str) -> None:
    """Write records to a Parquet file using pyarrow.

    Args:
        records: List of result dicts.
        path: Output file path.

    Raises:
        ImportError: If ``pyarrow`` is not installed.
    """
    try:
        import pyarrow as pa  # type: ignore[import-not-found]
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required for Parquet export. "
            "Install it with: pip install pyarrow"
        ) from exc

    table = pa.Table.from_pylist(records)
    pq.write_table(table, path)


def _write_hf_dataset(records: list[dict], path: str) -> None:
    """Write records to a HuggingFace Dataset directory.

    Args:
        records: List of result dicts.
        path: Output directory path.

    Raises:
        ImportError: If ``datasets`` is not installed.
    """
    try:
        from datasets import Dataset  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "datasets is required for HuggingFace Dataset export. "
            "Install it with: pip install datasets"
        ) from exc

    ds = Dataset.from_list(records)
    ds.save_to_disk(path)
