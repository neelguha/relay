"""Jobs routes for the relay web dashboard.

Provides:
- ``GET /jobs`` — paginated, searchable/filterable job list.
- ``GET /jobs/{job_id}`` — detailed view of a single job with config, timing,
  and token breakdown.
- ``POST /jobs/{job_id}/cancel`` — mark a job as cancelled (HTMX action).
- ``POST /jobs/{job_id}/resubmit`` — stub for resubmit action (HTMX action).

All list/detail data is read directly from the SQLite database via
``aiosqlite`` to avoid coupling the dashboard to the async BatchClient
context.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

import logging

import aiosqlite
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs")

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

_JOB_COLS = (
    "id", "provider_job_id", "provider", "model", "project", "name", "description",
    "status", "total_requests", "completed_requests", "failed_requests",
    "cached_hits", "input_tokens", "output_tokens", "estimated_cost_usd",
    "actual_cost_usd", "created_at", "submitted_at", "completed_at",
    "config_json", "error",
)

_ALLOWED_STATUSES = (
    "PENDING", "CACHE_RESOLVING", "VALIDATING", "SUBMITTING",
    "IN_PROGRESS", "DOWNLOADING", "COMPLETED", "PARTIAL", "FAILED", "CANCELLED",
)

_PAGE_SIZE = 25


def _row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
    """Convert a raw SQLite row to a plain dictionary.

    Args:
        row: A row from an ``aiosqlite`` query on the ``jobs`` table.

    Returns:
        A dictionary whose keys match :data:`_JOB_COLS` and whose timestamp
        values have been converted from Unix floats to ISO-8601 strings.
    """
    d = dict(zip(_JOB_COLS, row))
    # Ensure numeric columns are the correct type (SELECT * returns untyped rows).
    for int_col in (
        "total_requests", "completed_requests", "failed_requests",
        "cached_hits", "input_tokens", "output_tokens",
    ):
        try:
            d[int_col] = int(d.get(int_col) or 0)
        except (TypeError, ValueError):
            d[int_col] = 0
    for float_col in ("estimated_cost_usd", "actual_cost_usd"):
        val = d.get(float_col)
        if val is not None:
            try:
                d[float_col] = float(val)
            except (TypeError, ValueError):
                d[float_col] = 0.0
    for ts_col in ("created_at", "submitted_at", "completed_at"):
        val = d.get(ts_col)
        if val is not None:
            try:
                d[ts_col] = datetime.utcfromtimestamp(float(val)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            except (TypeError, ValueError, OSError):
                pass
    # Parse stored config_json so templates can iterate it.
    raw_cfg = d.get("config_json")
    if raw_cfg:
        try:
            d["config"] = json.loads(raw_cfg)
        except (TypeError, ValueError):
            d["config"] = {}
    else:
        d["config"] = {}
    return d


async def _refresh_in_progress_jobs(db_path: str) -> None:
    """Poll providers for status updates on all IN_PROGRESS jobs.

    This keeps the dashboard live without requiring the user to manually
    check each job.  Errors are logged and swallowed so the dashboard
    always renders.
    """
    try:
        from relay.client import BatchClient

        async with BatchClient() as client:
            store = client._require_store()
            jobs = await store.list_jobs(status="IN_PROGRESS", limit=50)
            for job in jobs:
                try:
                    # get_job polls the provider and persists updated status
                    await client.get_job(job.id)
                except Exception:
                    logger.debug("Failed to refresh job %s", job.id)
    except Exception:
        logger.debug("Could not refresh in-progress jobs", exc_info=True)


async def _list_jobs(
    db_path: str,
    *,
    search: str = "",
    status_filter: str = "",
    provider_filter: str = "",
    page: int = 1,
) -> tuple[list[dict[str, Any]], int]:
    """Fetch a paginated, filtered slice of jobs from the database.

    Args:
        db_path: Absolute path to the relay SQLite database file.
        search: Free-text search applied to ``id``, ``model``, ``project``,
            and ``provider`` columns (case-insensitive LIKE).
        status_filter: Exact status string to filter by.  Empty string means
            no filter.
        provider_filter: Exact provider name to filter by.  Empty string means
            no filter.
        page: 1-based page number.

    Returns:
        A 2-tuple of ``(jobs, total_count)`` where ``jobs`` is a list of row
        dictionaries and ``total_count`` is the unfiltered total matching the
        current filters (used to compute pagination).
    """
    clauses: list[str] = []
    params: list[Any] = []

    if search:
        like = f"%{search}%"
        clauses.append(
            "(id LIKE ? OR model LIKE ? OR COALESCE(project,'') LIKE ? OR provider LIKE ?)"
        )
        params.extend([like, like, like, like])

    if status_filter and status_filter in _ALLOWED_STATUSES:
        clauses.append("status = ?")
        params.append(status_filter)

    if provider_filter:
        clauses.append("provider = ?")
        params.append(provider_filter)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    count_sql = f"SELECT COUNT(*) FROM jobs {where}"
    data_sql = (
        f"SELECT * FROM jobs {where} "
        f"ORDER BY created_at DESC LIMIT ? OFFSET ?"
    )
    offset = (_PAGE_SIZE * (page - 1))

    async with aiosqlite.connect(db_path) as conn:
        async with conn.execute(count_sql, params) as cur:
            row = await cur.fetchone()
            total = row[0] if row else 0

        async with conn.execute(data_sql, params + [_PAGE_SIZE, offset]) as cur:
            rows = await cur.fetchall()

    jobs = [_row_to_dict(row) for row in rows]
    return jobs, total


async def _get_job(db_path: str, job_id: str) -> dict[str, Any] | None:
    """Fetch a single job by its relay-internal UUID.

    Args:
        db_path: Absolute path to the relay SQLite database file.
        job_id: The relay-internal job UUID.

    Returns:
        A dictionary of job fields, or ``None`` if not found.
    """
    async with aiosqlite.connect(db_path) as conn:
        async with conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)) as cur:
            row = await cur.fetchone()
        if row is None:
            return None

        # Fetch tags.
        async with conn.execute(
            "SELECT tag FROM job_tags WHERE job_id = ?", (job_id,)
        ) as cur:
            tag_rows = await cur.fetchall()
        tags = [r[0] for r in tag_rows]

    d = _row_to_dict(row)
    d["tags"] = tags
    # Compute elapsed seconds for display.
    created_raw = row[_JOB_COLS.index("created_at")]
    completed_raw = row[_JOB_COLS.index("completed_at")]
    if created_raw is not None:
        end_ts = float(completed_raw) if completed_raw else time.time()
        d["elapsed_seconds"] = round(end_ts - float(created_raw))
    else:
        d["elapsed_seconds"] = None

    # Progress percentage.
    total_req = d.get("total_requests") or 0
    completed_req = d.get("completed_requests") or 0
    d["progress_pct"] = round((completed_req / total_req * 100) if total_req > 0 else 0)

    return d


async def _get_distinct_providers(db_path: str) -> list[str]:
    """Return all distinct provider names stored in the database.

    Args:
        db_path: Absolute path to the relay SQLite database file.

    Returns:
        Sorted list of provider name strings.
    """
    async with aiosqlite.connect(db_path) as conn:
        async with conn.execute(
            "SELECT DISTINCT provider FROM jobs ORDER BY provider"
        ) as cur:
            rows = await cur.fetchall()
    return [r[0] for r in rows if r[0]]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("", response_class=HTMLResponse)
async def jobs_list(
    request: Request,
    page: int = 1,
    search: str = "",
    status: str = "",
    provider: str = "",
) -> HTMLResponse:
    """Render the paginated jobs list page.

    Accepts optional query parameters for filtering:
    - ``page``: Page number (1-based).
    - ``search``: Free-text search across id, model, project, provider.
    - ``status``: Exact status filter.
    - ``provider``: Exact provider filter.

    Args:
        request: The incoming FastAPI/Starlette request object.
        page: 1-based page index; defaults to 1.
        search: Free-text filter string; defaults to empty (no filter).
        status: Status string filter; defaults to empty (no filter).
        provider: Provider name filter; defaults to empty (no filter).

    Returns:
        An HTML response with the rendered ``jobs.html`` template.
    """
    db_path: str = request.app.state.db_path
    await _refresh_in_progress_jobs(db_path)
    jobs, total = await _list_jobs(
        db_path,
        search=search,
        status_filter=status,
        provider_filter=provider,
        page=page,
    )
    providers = await _get_distinct_providers(db_path)
    total_pages = max(1, (total + _PAGE_SIZE - 1) // _PAGE_SIZE)

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "jobs.html",
        {
            "request": request,
            "jobs": jobs,
            "total": total,
            "page": page,
            "total_pages": total_pages,
            "search": search,
            "status_filter": status,
            "provider_filter": provider,
            "all_statuses": _ALLOWED_STATUSES,
            "all_providers": providers,
            "page_size": _PAGE_SIZE,
        },
    )


@router.get("/{job_id}", response_class=HTMLResponse)
async def job_detail(request: Request, job_id: str) -> HTMLResponse:
    """Render the detail page for a single job.

    Fetches the full job record including config JSON, timing, token
    breakdown, and tags, then renders ``job_detail.html``.

    Args:
        request: The incoming FastAPI/Starlette request object.
        job_id: The relay-internal UUID of the job.

    Returns:
        An HTML response with the rendered ``job_detail.html`` template, or
        a plain 404 HTML response if the job is not found.
    """
    db_path: str = request.app.state.db_path
    job = await _get_job(db_path, job_id)
    if job is None:
        return HTMLResponse(content="<h1>Job not found</h1>", status_code=404)

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "job_detail.html",
        {"request": request, "job": job},
    )


@router.post("/{job_id}/cancel", response_class=HTMLResponse)
async def cancel_job(request: Request, job_id: str) -> HTMLResponse:
    """Mark a job as CANCELLED in the database.

    This is a best-effort action that sets ``status = 'CANCELLED'`` directly
    in the database.  It does not attempt to contact the provider API to
    cancel any in-flight remote batch.

    Args:
        request: The incoming FastAPI/Starlette request object.
        job_id: The relay-internal UUID of the job to cancel.

    Returns:
        An HTMX-compatible HTML fragment showing the updated status badge.
    """
    db_path: str = request.app.state.db_path
    now = time.time()
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            """
            UPDATE jobs
            SET status = 'CANCELLED', completed_at = COALESCE(completed_at, ?)
            WHERE id = ? AND status NOT IN ('COMPLETED','PARTIAL','FAILED','CANCELLED')
            """,
            (now, job_id),
        )
        await conn.commit()

    return HTMLResponse(
        content='<span class="badge badge-cancelled">CANCELLED</span>',
        status_code=200,
    )


@router.post("/{job_id}/resubmit", response_class=HTMLResponse)
async def resubmit_job(request: Request, job_id: str) -> HTMLResponse:
    """Stub endpoint for resubmitting a failed job.

    Resubmission requires access to the original request payloads, which are
    not stored in the jobs database.  This endpoint returns a user-friendly
    message indicating that resubmission must be performed via the Python API
    or CLI.

    Args:
        request: The incoming FastAPI/Starlette request object.
        job_id: The relay-internal UUID of the job to resubmit.

    Returns:
        An HTMX-compatible HTML fragment with an informational message.
    """
    return HTMLResponse(
        content=(
            '<span class="text-yellow-400">'
            "Resubmit via the relay CLI or Python API."
            "</span>"
        ),
        status_code=200,
    )
