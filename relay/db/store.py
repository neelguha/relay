"""SQLite-backed job store for relay.

All public methods are async and use ``aiosqlite`` so they can be called from
any asyncio context without blocking the event loop.  The database is opened
with WAL journal mode and a 5-second busy-timeout to allow concurrent readers
while a write is in progress.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import aiosqlite

from relay.models import BatchJob, BatchResult, JobStatus, RequestStatus

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# Columns returned by SELECT * FROM jobs in definition order.
_JOB_COLS = (
    "id",
    "provider_job_id",
    "provider",
    "model",
    "project",
    "name",
    "description",
    "status",
    "total_requests",
    "completed_requests",
    "failed_requests",
    "cached_hits",
    "input_tokens",
    "output_tokens",
    "estimated_cost_usd",
    "actual_cost_usd",
    "created_at",
    "submitted_at",
    "completed_at",
    "config_json",
    "error",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _row_to_job(row: aiosqlite.Row, tags: list[str] | None = None) -> BatchJob:
    """Convert a raw SQLite row to a :class:`~relay.models.BatchJob`.

    Args:
        row: A row returned by an ``aiosqlite`` query on the ``jobs`` table.
        tags: Pre-fetched list of tag strings for this job.  Defaults to an
            empty list when ``None`` is supplied.

    Returns:
        A fully populated :class:`~relay.models.BatchJob` dataclass instance.
    """
    d = dict(zip(_JOB_COLS, row))
    return BatchJob(
        id=d["id"],
        provider_job_id=d["provider_job_id"] or "",
        provider=d["provider"],
        model=d["model"],
        project=d["project"],
        name=d.get("name"),
        status=JobStatus(d["status"]),
        total_requests=d["total_requests"],
        completed_requests=d["completed_requests"],
        failed_requests=d["failed_requests"],
        cached_hits=d["cached_hits"],
        input_tokens=d["input_tokens"],
        output_tokens=d["output_tokens"],
        estimated_cost_usd=d["estimated_cost_usd"] or 0.0,
        actual_cost_usd=d["actual_cost_usd"],
        created_at=d["created_at"],
        submitted_at=d["submitted_at"],
        completed_at=d["completed_at"],
        tags=tags or [],
        error=d["error"],
    )


def _now() -> float:
    """Return the current UTC time as a Unix timestamp."""
    return time.time()


# ---------------------------------------------------------------------------
# JobStore
# ---------------------------------------------------------------------------


class JobStore:
    """Async SQLite store for relay jobs, requests, results, and tags.

    The store owns a single ``aiosqlite`` connection.  Use it as an async
    context manager so that the connection is properly opened and closed::

        async with JobStore("~/.relay/jobs.db") as store:
            job = await store.get_job("abc-123")

    Alternatively, call :meth:`open` and :meth:`close` manually.

    Args:
        db_path: Path to the SQLite database file.  The parent directory is
            created automatically if it does not exist.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path).expanduser().resolve()
        self._conn: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def open(self) -> None:
        """Open the database connection and apply the schema.

        WAL mode and a 5-second busy-timeout are enabled after the connection
        is established.  The schema DDL is applied with ``IF NOT EXISTS``
        guards so it is safe to call on an already-initialised database.

        Raises:
            aiosqlite.Error: If the connection cannot be opened.
        """
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self._db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA busy_timeout=5000")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
        await self._conn.executescript(schema_sql)
        await self._conn.commit()

    async def close(self) -> None:
        """Close the database connection.

        Safe to call even if the connection was never opened.
        """
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self) -> "JobStore":
        await self.open()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    @property
    def _db(self) -> aiosqlite.Connection:
        """Return the active connection, raising if the store is not open."""
        if self._conn is None:
            raise RuntimeError(
                "JobStore is not open. Call open() or use it as an async context manager."
            )
        return self._conn

    # ------------------------------------------------------------------
    # Jobs — write operations
    # ------------------------------------------------------------------

    async def create_job(self, job: BatchJob, config: dict[str, Any]) -> None:
        """Persist a new job record.

        Args:
            job: The :class:`~relay.models.BatchJob` to store.
            config: Serialisable dictionary of the job's
                :class:`~relay.models.BatchConfig`.  Stored as JSON in the
                ``config_json`` column.

        Raises:
            aiosqlite.IntegrityError: If a job with the same ``id`` already
                exists.
        """
        await self._db.execute(
            """
            INSERT INTO jobs (
                id, provider_job_id, provider, model, project, name, description,
                status, total_requests, completed_requests, failed_requests,
                cached_hits, input_tokens, output_tokens, estimated_cost_usd,
                actual_cost_usd, created_at, submitted_at, completed_at,
                config_json, error
            ) VALUES (
                :id, :provider_job_id, :provider, :model, :project, :name, :description,
                :status, :total_requests, :completed_requests, :failed_requests,
                :cached_hits, :input_tokens, :output_tokens, :estimated_cost_usd,
                :actual_cost_usd, :created_at, :submitted_at, :completed_at,
                :config_json, :error
            )
            """,
            {
                "id": job.id,
                "provider_job_id": job.provider_job_id or None,
                "provider": job.provider,
                "model": job.model,
                "project": job.project,
                "name": job.name,
                "description": None,
                "status": job.status.value,
                "total_requests": job.total_requests,
                "completed_requests": job.completed_requests,
                "failed_requests": job.failed_requests,
                "cached_hits": job.cached_hits,
                "input_tokens": job.input_tokens,
                "output_tokens": job.output_tokens,
                "estimated_cost_usd": job.estimated_cost_usd,
                "actual_cost_usd": job.actual_cost_usd,
                "created_at": job.created_at if isinstance(job.created_at, float) else job.created_at.timestamp(),
                "submitted_at": (
                    job.submitted_at.timestamp()
                    if job.submitted_at and not isinstance(job.submitted_at, float)
                    else job.submitted_at
                ),
                "completed_at": (
                    job.completed_at.timestamp()
                    if job.completed_at and not isinstance(job.completed_at, float)
                    else job.completed_at
                ),
                "config_json": json.dumps(config),
                "error": job.error,
            },
        )
        if job.tags:
            await self._upsert_tags(job.id, job.tags)
        await self._db.commit()

    async def update_job(self, job: BatchJob) -> None:
        """Update mutable fields of an existing job.

        All counters, status, cost fields, timestamps, and error are updated.
        The ``provider``, ``model``, ``project``, and ``config_json`` columns
        are considered immutable after creation and are not touched.

        Args:
            job: The job with updated field values.  The ``id`` is used as the
                lookup key.

        Raises:
            aiosqlite.Error: On any database-level error.
        """
        await self._db.execute(
            """
            UPDATE jobs SET
                provider_job_id     = :provider_job_id,
                status              = :status,
                total_requests      = :total_requests,
                completed_requests  = :completed_requests,
                failed_requests     = :failed_requests,
                cached_hits         = :cached_hits,
                input_tokens        = :input_tokens,
                output_tokens       = :output_tokens,
                estimated_cost_usd  = :estimated_cost_usd,
                actual_cost_usd     = :actual_cost_usd,
                submitted_at        = :submitted_at,
                completed_at        = :completed_at,
                error               = :error
            WHERE id = :id
            """,
            {
                "id": job.id,
                "provider_job_id": job.provider_job_id or None,
                "status": job.status.value,
                "total_requests": job.total_requests,
                "completed_requests": job.completed_requests,
                "failed_requests": job.failed_requests,
                "cached_hits": job.cached_hits,
                "input_tokens": job.input_tokens,
                "output_tokens": job.output_tokens,
                "estimated_cost_usd": job.estimated_cost_usd,
                "actual_cost_usd": job.actual_cost_usd,
                "submitted_at": (
                    job.submitted_at.timestamp()
                    if job.submitted_at and not isinstance(job.submitted_at, float)
                    else job.submitted_at
                ),
                "completed_at": (
                    job.completed_at.timestamp()
                    if job.completed_at and not isinstance(job.completed_at, float)
                    else job.completed_at
                ),
                "error": job.error,
            },
        )
        await self._upsert_tags(job.id, job.tags)
        await self._db.commit()

    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        error: str | None = None,
    ) -> None:
        """Update only the status (and optional error) of a job.

        Args:
            job_id: The relay-internal job UUID.
            status: The new :class:`~relay.models.JobStatus`.
            error: Optional error message to store alongside the status.
        """
        now = _now()
        completed_at = now if status.is_terminal else None
        await self._db.execute(
            """
            UPDATE jobs
            SET status       = :status,
                error        = :error,
                completed_at = COALESCE(completed_at, :completed_at)
            WHERE id = :id
            """,
            {
                "id": job_id,
                "status": status.value,
                "error": error,
                "completed_at": completed_at,
            },
        )
        await self._db.commit()

    # ------------------------------------------------------------------
    # Jobs — read operations
    # ------------------------------------------------------------------

    async def get_job(self, job_id: str) -> BatchJob | None:
        """Fetch a single job by its relay-internal UUID.

        Args:
            job_id: The relay-internal job UUID.

        Returns:
            A :class:`~relay.models.BatchJob` or ``None`` if not found.
        """
        async with self._db.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        tags = await self._fetch_tags(job_id)
        return _row_to_job(row, tags)

    async def get_job_by_name(
        self, name: str, project: str | None = None
    ) -> BatchJob | None:
        """Fetch a job by its human-readable name.

        Args:
            name: The job name to look up.
            project: Optional project scope. When provided, searches only
                within that project. When ``None``, matches any project.

        Returns:
            A :class:`~relay.models.BatchJob` or ``None`` if not found.
            If multiple jobs share the same name (across projects), the
            most recently created one is returned.
        """
        if project is not None:
            sql = (
                "SELECT * FROM jobs WHERE name = ? AND project = ?"
                " ORDER BY created_at DESC LIMIT 1"
            )
            params: tuple[Any, ...] = (name, project)
        else:
            sql = (
                "SELECT * FROM jobs WHERE name = ?"
                " ORDER BY created_at DESC LIMIT 1"
            )
            params = (name,)

        async with self._db.execute(sql, params) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        job_id = row[0]
        tags = await self._fetch_tags(job_id)
        return _row_to_job(row, tags)

    async def list_jobs(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        status: str | Sequence[str] | None = None,
        project: str | None = None,
        tags: Sequence[str] | None = None,
        after: float | None = None,
        before: float | None = None,
        order_by: str = "created_at",
        descending: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[BatchJob]:
        """List jobs with optional filters.

        All filters are applied as AND conditions.  Tag filtering returns jobs
        that possess **all** of the specified tags.

        Args:
            provider: Filter by provider name (e.g. ``"anthropic"``).
            model: Filter by model identifier.
            status: Filter by one or more
                :class:`~relay.models.JobStatus` values (string or
                ``JobStatus`` instances).  A single string or a list of strings
                are both accepted.
            project: Filter by project name (exact match).
            tags: Only include jobs that have **all** of these tags.
            after: Unix timestamp lower bound on ``created_at`` (inclusive).
            before: Unix timestamp upper bound on ``created_at`` (exclusive).
            order_by: Column to sort by.  Must be one of ``"created_at"``,
                ``"submitted_at"``, ``"completed_at"``, ``"model"``,
                ``"provider"``, ``"status"``, ``"actual_cost_usd"``.
                Defaults to ``"created_at"``.
            descending: When ``True`` (default) sort newest first.
            limit: Maximum number of records to return.  Defaults to 100.
            offset: Number of records to skip for pagination.  Defaults to 0.

        Returns:
            A list of :class:`~relay.models.BatchJob` objects, each with its
            tags populated.

        Raises:
            ValueError: If *order_by* names an unknown column.
        """
        _allowed_order = {
            "created_at",
            "submitted_at",
            "completed_at",
            "model",
            "provider",
            "status",
            "actual_cost_usd",
        }
        if order_by not in _allowed_order:
            raise ValueError(
                f"Invalid order_by '{order_by}'. Must be one of: {sorted(_allowed_order)}"
            )

        clauses: list[str] = []
        params: list[Any] = []

        if provider is not None:
            clauses.append("j.provider = ?")
            params.append(provider)

        if model is not None:
            clauses.append("j.model = ?")
            params.append(model)

        if status is not None:
            if isinstance(status, str):
                status_list = [status]
            else:
                status_list = list(status)
            placeholders = ", ".join("?" * len(status_list))
            clauses.append(f"j.status IN ({placeholders})")
            params.extend(status_list)

        if project is not None:
            clauses.append("j.project = ?")
            params.append(project)

        if after is not None:
            clauses.append("j.created_at >= ?")
            params.append(after)

        if before is not None:
            clauses.append("j.created_at < ?")
            params.append(before)

        if tags:
            # Subquery: jobs that have ALL required tags
            tag_placeholders = ", ".join("?" * len(tags))
            clauses.append(
                f"""
                j.id IN (
                    SELECT job_id FROM job_tags
                    WHERE tag IN ({tag_placeholders})
                    GROUP BY job_id
                    HAVING COUNT(DISTINCT tag) = ?
                )
                """
            )
            params.extend(tags)
            params.append(len(tags))

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        direction = "DESC" if descending else "ASC"
        sql = f"""
            SELECT j.*
            FROM jobs j
            {where}
            ORDER BY j.{order_by} {direction}
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        rows: list[aiosqlite.Row] = []
        async with self._db.execute(sql, params) as cur:
            rows = await cur.fetchall()

        jobs: list[BatchJob] = []
        for row in rows:
            job_id = row[0]
            job_tags = await self._fetch_tags(job_id)
            jobs.append(_row_to_job(row, job_tags))
        return jobs

    # ------------------------------------------------------------------
    # Requests — write operations
    # ------------------------------------------------------------------

    async def create_request(
        self,
        request_id: str,
        job_id: str,
        payload: dict[str, Any],
        *,
        cache_key: str | None = None,
        status: RequestStatus = RequestStatus.PENDING,
    ) -> None:
        """Persist a single request record.

        Args:
            request_id: The caller-supplied or auto-generated UUID for this
                request.
            job_id: Parent job UUID.
            payload: The full :class:`~relay.models.BatchRequest` serialised
                as a dictionary.
            cache_key: SHA-256 cache key, or ``None`` if not yet computed.
            status: Initial :class:`~relay.models.RequestStatus`.
                Defaults to ``PENDING``.
        """
        await self._db.execute(
            """
            INSERT INTO requests (id, job_id, cache_key, status, payload_json, error)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                job_id,
                cache_key,
                status.value,
                json.dumps(payload),
                None,
            ),
        )
        await self._db.commit()

    async def create_requests_bulk(
        self,
        records: Sequence[dict[str, Any]],
    ) -> None:
        """Insert multiple request rows in a single transaction.

        Args:
            records: A sequence of dicts, each with the keys ``id``,
                ``job_id``, ``payload`` (dict), and optionally ``cache_key``
                and ``status``.
        """
        rows = [
            (
                r["id"],
                r["job_id"],
                r.get("cache_key"),
                r.get("status", RequestStatus.PENDING.value),
                json.dumps(r["payload"]),
                None,
            )
            for r in records
        ]
        await self._db.executemany(
            """
            INSERT INTO requests (id, job_id, cache_key, status, payload_json, error)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        await self._db.commit()

    async def update_request_status(
        self,
        request_id: str,
        status: RequestStatus,
        *,
        error: str | None = None,
    ) -> None:
        """Update the status (and optional error) of a single request.

        Args:
            request_id: The request UUID.
            status: New :class:`~relay.models.RequestStatus`.
            error: Optional error message.
        """
        await self._db.execute(
            "UPDATE requests SET status = ?, error = ? WHERE id = ?",
            (status.value, error, request_id),
        )
        await self._db.commit()

    # ------------------------------------------------------------------
    # Requests — read operations
    # ------------------------------------------------------------------

    async def get_request(self, request_id: str) -> dict[str, Any] | None:
        """Fetch a single request by its UUID.

        Args:
            request_id: The request UUID.

        Returns:
            A dictionary of column values, or ``None`` if not found.  The
            ``payload_json`` value is deserialised to a dict.
        """
        async with self._db.execute(
            "SELECT * FROM requests WHERE id = ?", (request_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        d = dict(row)
        d["payload_json"] = json.loads(d["payload_json"])
        return d

    async def list_requests(
        self,
        job_id: str,
        *,
        status: RequestStatus | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List requests belonging to a job.

        Args:
            job_id: Parent job UUID.
            status: Optional filter by request status.
            limit: Maximum number of records.  Defaults to 1000.
            offset: Records to skip for pagination.  Defaults to 0.

        Returns:
            A list of request dicts with ``payload_json`` deserialised.
        """
        if status is not None:
            sql = (
                "SELECT * FROM requests WHERE job_id = ? AND status = ?"
                " LIMIT ? OFFSET ?"
            )
            params: tuple[Any, ...] = (job_id, status.value, limit, offset)
        else:
            sql = "SELECT * FROM requests WHERE job_id = ? LIMIT ? OFFSET ?"
            params = (job_id, limit, offset)

        rows: list[dict[str, Any]] = []
        async with self._db.execute(sql, params) as cur:
            async for row in cur:
                d = dict(row)
                d["payload_json"] = json.loads(d["payload_json"])
                rows.append(d)
        return rows

    async def count_requests(
        self,
        job_id: str,
        *,
        status: RequestStatus | None = None,
    ) -> int:
        """Return the count of requests for a job, optionally filtered by status.

        Args:
            job_id: Parent job UUID.
            status: Optional status filter.

        Returns:
            Integer count.
        """
        if status is not None:
            sql = "SELECT COUNT(*) FROM requests WHERE job_id = ? AND status = ?"
            params_t: tuple[Any, ...] = (job_id, status.value)
        else:
            sql = "SELECT COUNT(*) FROM requests WHERE job_id = ?"
            params_t = (job_id,)

        async with self._db.execute(sql, params_t) as cur:
            row = await cur.fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Results — write operations
    # ------------------------------------------------------------------

    async def create_result(self, result: BatchResult) -> None:
        """Persist a single result record.

        Also updates the corresponding request status to
        ``CACHED`` if ``result.from_cache`` is ``True``, otherwise
        ``COMPLETED``.

        Args:
            result: The :class:`~relay.models.BatchResult` to store.
        """
        await self._db.execute(
            """
            INSERT OR REPLACE INTO results
                (request_id, job_id, content, stop_reason,
                 input_tokens, output_tokens, from_cache, response_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.request_id,
                result.job_id,
                result.content,
                result.stop_reason,
                result.input_tokens,
                result.output_tokens,
                1 if result.from_cache else 0,
                json.dumps(result.raw_response),
            ),
        )
        new_status = RequestStatus.CACHED if result.from_cache else RequestStatus.COMPLETED
        await self._db.execute(
            "UPDATE requests SET status = ? WHERE id = ?",
            (new_status.value, result.request_id),
        )
        await self._db.commit()

    async def create_results_bulk(self, results: Sequence[BatchResult]) -> None:
        """Insert multiple result rows in a single transaction.

        Requests are also updated to their terminal status in the same
        transaction.  Use this for efficient checkpoint commits.

        Args:
            results: A sequence of :class:`~relay.models.BatchResult` objects.
        """
        result_rows = [
            (
                r.request_id,
                r.job_id,
                r.content,
                r.stop_reason,
                r.input_tokens,
                r.output_tokens,
                1 if r.from_cache else 0,
                json.dumps(r.raw_response),
            )
            for r in results
        ]
        await self._db.executemany(
            """
            INSERT OR REPLACE INTO results
                (request_id, job_id, content, stop_reason,
                 input_tokens, output_tokens, from_cache, response_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            result_rows,
        )
        request_updates = [
            (
                RequestStatus.CACHED.value if r.from_cache else RequestStatus.COMPLETED.value,
                r.request_id,
            )
            for r in results
        ]
        await self._db.executemany(
            "UPDATE requests SET status = ? WHERE id = ?",
            request_updates,
        )
        await self._db.commit()

    # ------------------------------------------------------------------
    # Results — read operations
    # ------------------------------------------------------------------

    async def get_result(self, request_id: str) -> dict[str, Any] | None:
        """Fetch a single result by its request UUID.

        Args:
            request_id: The request UUID (also the primary key in ``results``).

        Returns:
            A dictionary of column values with ``response_json`` deserialised,
            or ``None`` if not found.
        """
        async with self._db.execute(
            "SELECT * FROM results WHERE request_id = ?", (request_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        d = dict(row)
        if d.get("response_json"):
            d["response_json"] = json.loads(d["response_json"])
        return d

    async def list_results(
        self,
        job_id: str,
        *,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List all results for a job.

        Args:
            job_id: Parent job UUID.
            limit: Maximum records to return.  Defaults to 1000.
            offset: Records to skip.  Defaults to 0.

        Returns:
            A list of result dicts with ``response_json`` deserialised.
        """
        rows: list[dict[str, Any]] = []
        async with self._db.execute(
            "SELECT * FROM results WHERE job_id = ? LIMIT ? OFFSET ?",
            (job_id, limit, offset),
        ) as cur:
            async for row in cur:
                d = dict(row)
                if d.get("response_json"):
                    d["response_json"] = json.loads(d["response_json"])
                rows.append(d)
        return rows

    async def count_results(self, job_id: str) -> int:
        """Return the number of stored results for a job.

        Args:
            job_id: Parent job UUID.

        Returns:
            Integer count.
        """
        async with self._db.execute(
            "SELECT COUNT(*) FROM results WHERE job_id = ?", (job_id,)
        ) as cur:
            row = await cur.fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Tags
    # ------------------------------------------------------------------

    async def _upsert_tags(self, job_id: str, tags: Sequence[str]) -> None:
        """Insert tags for a job, ignoring duplicates.

        Args:
            job_id: Parent job UUID.
            tags: Tag strings to insert.
        """
        if not tags:
            return
        await self._db.executemany(
            "INSERT OR IGNORE INTO job_tags (job_id, tag) VALUES (?, ?)",
            [(job_id, tag) for tag in tags],
        )

    async def _fetch_tags(self, job_id: str) -> list[str]:
        """Return sorted tag strings for a given job.

        Args:
            job_id: Parent job UUID.

        Returns:
            Sorted list of tag strings.
        """
        async with self._db.execute(
            "SELECT tag FROM job_tags WHERE job_id = ? ORDER BY tag",
            (job_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [r[0] for r in rows]

    async def add_tags(self, job_id: str, tags: Sequence[str]) -> None:
        """Add tags to an existing job.

        Tags that already exist are silently ignored (idempotent).

        Args:
            job_id: The relay-internal job UUID.
            tags: Tag strings to add.
        """
        await self._upsert_tags(job_id, tags)
        await self._db.commit()

    async def remove_tags(self, job_id: str, tags: Sequence[str]) -> None:
        """Remove specific tags from an existing job.

        Tags not present on the job are silently ignored.

        Args:
            job_id: The relay-internal job UUID.
            tags: Tag strings to remove.
        """
        await self._db.executemany(
            "DELETE FROM job_tags WHERE job_id = ? AND tag = ?",
            [(job_id, tag) for tag in tags],
        )
        await self._db.commit()

    async def get_tags(self, job_id: str) -> list[str]:
        """Return all tags associated with a job.

        Args:
            job_id: The relay-internal job UUID.

        Returns:
            Sorted list of tag strings.
        """
        return await self._fetch_tags(job_id)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    async def delete_job(self, job_id: str) -> None:
        """Delete a job and all its associated records.

        Cascading deletes remove rows from ``requests``, ``results``, and
        ``job_tags`` in the correct order to satisfy foreign-key constraints.

        Args:
            job_id: The relay-internal job UUID.
        """
        result_ids_sql = (
            "SELECT request_id FROM results WHERE job_id = ?"
        )
        async with self._db.execute(result_ids_sql, (job_id,)) as cur:
            result_ids = [r[0] for r in await cur.fetchall()]

        if result_ids:
            placeholders = ", ".join("?" * len(result_ids))
            await self._db.execute(
                f"DELETE FROM results WHERE request_id IN ({placeholders})",
                result_ids,
            )

        await self._db.execute("DELETE FROM requests WHERE job_id = ?", (job_id,))
        await self._db.execute("DELETE FROM job_tags WHERE job_id = ?", (job_id,))
        await self._db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        await self._db.commit()

    async def job_exists(self, job_id: str) -> bool:
        """Check whether a job record exists.

        Args:
            job_id: The relay-internal job UUID.

        Returns:
            ``True`` if the job is present in the database.
        """
        async with self._db.execute(
            "SELECT 1 FROM jobs WHERE id = ? LIMIT 1", (job_id,)
        ) as cur:
            return await cur.fetchone() is not None

    async def vacuum(self) -> None:
        """Run SQLite VACUUM to reclaim free pages.

        This is a blocking operation that should be called infrequently (e.g.
        after bulk deletions or during maintenance windows).
        """
        await self._db.execute("VACUUM")
