"""Overview route for the relay web dashboard.

Serves the ``/`` endpoint, which displays a summary of the relay job database:
active jobs, cost today, cache hit rate, and errors in the last hour.  An HTMX
partial endpoint (``/overview/cards``) is also exposed so the page can
auto-refresh its summary cards every five seconds without a full page reload.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import logging

import aiosqlite
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _fetch_summary(db_path: str) -> dict:
    """Query the SQLite database for overview summary statistics.

    Args:
        db_path: Absolute path to the relay SQLite database file.

    Returns:
        A dictionary with the keys:
        - ``active_jobs`` (int): Count of jobs in a non-terminal state.
        - ``cost_today_usd`` (float): Sum of ``estimated_cost_usd`` for jobs
          created in the current UTC calendar day.
        - ``cache_hit_rate`` (float): Fraction of all completed requests that
          were served from cache (0.0 – 1.0).
        - ``errors_last_hour`` (int): Number of jobs whose status is ``FAILED``
          and whose ``completed_at`` timestamp falls within the last 3600
          seconds.
        - ``total_jobs`` (int): Total number of job rows.
        - ``completed_jobs`` (int): Jobs in a terminal state.
    """
    active_statuses = ("PENDING", "CACHE_RESOLVING", "VALIDATING", "SUBMITTING",
                       "IN_PROGRESS", "DOWNLOADING")
    active_placeholders = ",".join("?" * len(active_statuses))

    # Start of today in UTC as a Unix timestamp.
    now = time.time()
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    ).timestamp()
    one_hour_ago = now - 3600.0

    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row

        # Active jobs count.
        async with conn.execute(
            f"SELECT COUNT(*) FROM jobs WHERE status IN ({active_placeholders})",
            active_statuses,
        ) as cur:
            row = await cur.fetchone()
            active_jobs: int = row[0] if row else 0

        # Cost today.
        async with conn.execute(
            "SELECT COALESCE(SUM(estimated_cost_usd), 0.0) FROM jobs WHERE created_at >= ?",
            (today_start,),
        ) as cur:
            row = await cur.fetchone()
            cost_today_usd: float = row[0] if row else 0.0

        # Cache hit rate: sum(cached_hits) / sum(total_requests) across all jobs.
        async with conn.execute(
            "SELECT COALESCE(SUM(cached_hits), 0), COALESCE(SUM(total_requests), 0) FROM jobs"
        ) as cur:
            row = await cur.fetchone()
            total_cached: int = row[0] if row else 0
            total_requests: int = row[1] if row else 0
        cache_hit_rate = (total_cached / total_requests) if total_requests > 0 else 0.0

        # Errors in last hour.
        async with conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE status = 'FAILED' AND completed_at >= ?",
            (one_hour_ago,),
        ) as cur:
            row = await cur.fetchone()
            errors_last_hour: int = row[0] if row else 0

        # Total and completed jobs.
        async with conn.execute("SELECT COUNT(*) FROM jobs") as cur:
            row = await cur.fetchone()
            total_jobs: int = row[0] if row else 0

        terminal = ("COMPLETED", "PARTIAL", "FAILED", "CANCELLED")
        terminal_placeholders = ",".join("?" * len(terminal))
        async with conn.execute(
            f"SELECT COUNT(*) FROM jobs WHERE status IN ({terminal_placeholders})",
            terminal,
        ) as cur:
            row = await cur.fetchone()
            completed_jobs: int = row[0] if row else 0

    return {
        "active_jobs": active_jobs,
        "cost_today_usd": cost_today_usd,
        "cache_hit_rate": cache_hit_rate,
        "errors_last_hour": errors_last_hour,
        "total_jobs": total_jobs,
        "completed_jobs": completed_jobs,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
async def overview_page(request: Request) -> HTMLResponse:
    """Render the full overview page.

    Fetches summary statistics from the database and renders ``overview.html``
    with the full base layout.  The page includes an HTMX polling trigger that
    refreshes the ``#summary-cards`` partial every 5 seconds.

    Args:
        request: The incoming FastAPI/Starlette request object.

    Returns:
        An HTML response containing the rendered ``overview.html`` template.
    """
    db_path: str = request.app.state.db_path
    summary = await _fetch_summary(db_path)
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "overview.html",
        {"request": request, **summary},
    )


@router.get("/overview/cards", response_class=HTMLResponse)
async def overview_cards_partial(request: Request) -> HTMLResponse:
    """Return only the summary-cards HTML fragment for HTMX polling.

    This endpoint is called by the HTMX ``hx-get`` trigger embedded in
    ``overview.html`` every 5 seconds.  It renders only the ``_cards``
    sub-template and returns it so HTMX can swap the ``#summary-cards`` element
    in place without reloading the full page.

    Args:
        request: The incoming FastAPI/Starlette request object.

    Returns:
        An HTML response containing only the summary-cards fragment.
    """
    db_path: str = request.app.state.db_path
    # Refresh in-progress jobs from providers before rendering.
    try:
        from relay.dashboard.web.routes.jobs import _refresh_in_progress_jobs
        await _refresh_in_progress_jobs(db_path)
    except Exception:
        logger.debug("Could not refresh in-progress jobs", exc_info=True)
    summary = await _fetch_summary(db_path)
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "_cards.html",
        {"request": request, **summary},
    )
