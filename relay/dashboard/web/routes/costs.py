"""Cost analytics route for the relay web dashboard.

Provides the ``GET /costs`` endpoint, which renders a cost analytics page
showing cumulative spending broken down by provider and a daily spending
time-series for the last 30 days.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import aiosqlite
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter(prefix="/costs")

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


async def _fetch_cost_by_provider(db_path: str) -> list[dict[str, Any]]:
    """Aggregate total cost and token usage grouped by provider.

    Args:
        db_path: Absolute path to the relay SQLite database file.

    Returns:
        A list of dictionaries, each containing ``provider``,
        ``total_usd``, ``input_tokens``, ``output_tokens``, and
        ``job_count`` keys, sorted descending by ``total_usd``.
    """
    sql = """
        SELECT
            provider,
            COALESCE(SUM(estimated_cost_usd), 0.0) AS total_usd,
            COALESCE(SUM(input_tokens), 0)          AS input_tokens,
            COALESCE(SUM(output_tokens), 0)         AS output_tokens,
            COUNT(*)                                AS job_count
        FROM jobs
        GROUP BY provider
        ORDER BY total_usd DESC
    """
    async with aiosqlite.connect(db_path) as conn:
        async with conn.execute(sql) as cur:
            rows = await cur.fetchall()
    return [
        {
            "provider": row[0],
            "total_usd": round(float(row[1]), 4),
            "input_tokens": int(row[2]),
            "output_tokens": int(row[3]),
            "job_count": int(row[4]),
        }
        for row in rows
        if row[0]
    ]


async def _fetch_daily_spending(
    db_path: str, days: int = 30
) -> list[dict[str, Any]]:
    """Return per-day spending totals for the last *days* calendar days (UTC).

    Each day is represented even if there are zero jobs, making it easy to
    render a continuous bar chart.

    Args:
        db_path: Absolute path to the relay SQLite database file.
        days: Number of past calendar days to include.  Defaults to 30.

    Returns:
        A list of dictionaries with ``date`` (``"YYYY-MM-DD"``) and
        ``total_usd`` (float) keys, ordered chronologically (oldest first).
    """
    # Collect per-day totals from the database using strftime.
    sql = """
        SELECT
            strftime('%Y-%m-%d', datetime(created_at, 'unixepoch')) AS day,
            COALESCE(SUM(estimated_cost_usd), 0.0)                  AS total_usd
        FROM jobs
        WHERE created_at >= ?
        GROUP BY day
        ORDER BY day ASC
    """
    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=days)
    ).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

    async with aiosqlite.connect(db_path) as conn:
        async with conn.execute(sql, (cutoff,)) as cur:
            rows = await cur.fetchall()

    db_by_day = {row[0]: round(float(row[1]), 4) for row in rows}

    # Fill in missing days with zero so the chart is continuous.
    today = datetime.now(timezone.utc).date()
    result = []
    for i in range(days):
        day = (today - timedelta(days=days - 1 - i)).isoformat()
        result.append({"date": day, "total_usd": db_by_day.get(day, 0.0)})

    return result


async def _fetch_cost_by_model(db_path: str) -> list[dict[str, Any]]:
    """Aggregate total cost grouped by model identifier.

    Args:
        db_path: Absolute path to the relay SQLite database file.

    Returns:
        A list of dictionaries with ``model``, ``total_usd``, and
        ``job_count`` keys, sorted descending by ``total_usd``.
    """
    sql = """
        SELECT
            model,
            COALESCE(SUM(estimated_cost_usd), 0.0) AS total_usd,
            COUNT(*)                                AS job_count
        FROM jobs
        GROUP BY model
        ORDER BY total_usd DESC
        LIMIT 20
    """
    async with aiosqlite.connect(db_path) as conn:
        async with conn.execute(sql) as cur:
            rows = await cur.fetchall()
    return [
        {
            "model": row[0],
            "total_usd": round(float(row[1]), 4),
            "job_count": int(row[2]),
        }
        for row in rows
        if row[0]
    ]


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.get("", response_class=HTMLResponse)
async def costs_page(request: Request) -> HTMLResponse:
    """Render the cost analytics page.

    Fetches provider-level and model-level cost breakdowns, plus a 30-day
    daily spending series, then renders ``costs.html``.

    Args:
        request: The incoming FastAPI/Starlette request object.

    Returns:
        An HTML response with the rendered ``costs.html`` template.
    """
    db_path: str = request.app.state.db_path

    by_provider, daily, by_model = (
        await _fetch_cost_by_provider(db_path),
        await _fetch_daily_spending(db_path),
        await _fetch_cost_by_model(db_path),
    )

    # Grand total across all providers.
    grand_total = round(sum(r["total_usd"] for r in by_provider), 4)

    # Max daily value for proportional bar widths.
    max_daily = max((d["total_usd"] for d in daily), default=0.0) or 1.0

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "costs.html",
        {
            "request": request,
            "by_provider": by_provider,
            "daily": daily,
            "by_model": by_model,
            "grand_total": grand_total,
            "max_daily": max_daily,
        },
    )
