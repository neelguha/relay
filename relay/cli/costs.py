"""costs subcommand group for the relay CLI.

Provides commands to report spending across providers, models, and projects.

Commands::

    relay costs today -- show costs accumulated today (or since a given date)

Example::

    $ relay costs today
    $ relay costs today --since 2024-01-01
    $ relay costs today --group-by model --format csv
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
from datetime import datetime, timezone
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from relay.client import BatchClient
from relay.models import BatchJob, JobStatus

app = typer.Typer(help="Report LLM spending.")
console = Console()
err_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _start_of_today_utc() -> datetime:
    """Return midnight UTC for today.

    Returns:
        A timezone-aware :class:`datetime` at ``00:00:00 UTC`` for the current
        calendar date.
    """
    now = datetime.now(tz=timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _group_jobs(
    jobs: list[BatchJob],
    group_by: str,
) -> dict[str, dict]:
    """Aggregate cost and token totals from a list of jobs by a grouping key.

    Args:
        jobs: The list of :class:`~relay.models.BatchJob` objects to aggregate.
        group_by: Aggregation dimension — one of ``"provider"``, ``"model"``,
            or ``"project"``.

    Returns:
        A dict mapping each group label to a nested dict with keys
        ``"total_usd"``, ``"input_tokens"``, ``"output_tokens"``, and
        ``"job_count"``.
    """
    groups: dict[str, dict] = {}
    for job in jobs:
        if group_by == "provider":
            key = job.provider
        elif group_by == "model":
            key = job.model
        elif group_by == "project":
            key = job.project or "(none)"
        else:
            key = job.provider

        if key not in groups:
            groups[key] = {
                "total_usd": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "job_count": 0,
            }
        cost = job.actual_cost_usd if job.actual_cost_usd is not None else job.estimated_cost_usd
        groups[key]["total_usd"] += cost
        groups[key]["input_tokens"] += job.input_tokens
        groups[key]["output_tokens"] += job.output_tokens
        groups[key]["job_count"] += 1

    return groups


def _render_table(groups: dict[str, dict], group_by: str, total_usd: float) -> Table:
    """Build a Rich cost-summary table.

    Args:
        groups: Aggregated group data from :func:`_group_jobs`.
        group_by: The dimension used for grouping (used as a column header).
        total_usd: Grand total cost across all groups.

    Returns:
        A :class:`rich.table.Table` ready to print.
    """
    table = Table(show_header=True, header_style="bold cyan", expand=True)
    table.add_column(group_by.title(), style="bold")
    table.add_column("Jobs", justify="right")
    table.add_column("In tokens", justify="right")
    table.add_column("Out tokens", justify="right")
    table.add_column("Cost (USD)", justify="right", style="green")

    for label, data in sorted(groups.items(), key=lambda x: -x[1]["total_usd"]):
        table.add_row(
            label,
            str(data["job_count"]),
            f"{data['input_tokens']:,}",
            f"{data['output_tokens']:,}",
            f"${data['total_usd']:.4f}",
        )

    table.add_section()
    table.add_row(
        "[bold]TOTAL[/bold]",
        "",
        "",
        "",
        f"[bold]${total_usd:.4f}[/bold]",
    )
    return table


def _render_csv(groups: dict[str, dict], group_by: str) -> str:
    """Render aggregated cost data as a CSV string.

    Args:
        groups: Aggregated group data from :func:`_group_jobs`.
        group_by: The dimension used for grouping (used as the first column
            header).

    Returns:
        A UTF-8 CSV string with a header row followed by one row per group.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([group_by, "jobs", "input_tokens", "output_tokens", "cost_usd"])
    for label, data in sorted(groups.items(), key=lambda x: -x[1]["total_usd"]):
        writer.writerow([
            label,
            data["job_count"],
            data["input_tokens"],
            data["output_tokens"],
            f"{data['total_usd']:.6f}",
        ])
    return buf.getvalue()


# ---------------------------------------------------------------------------
# today command
# ---------------------------------------------------------------------------


@app.command("today")
def today(
    since: Optional[str] = typer.Option(
        None,
        "--since",
        help=(
            "Show costs since this ISO-8601 timestamp instead of midnight UTC. "
            "Example: 2024-01-01 or 2024-01-01T12:00:00."
        ),
    ),
    group_by: str = typer.Option(
        "provider",
        "--group-by",
        "-g",
        help="Aggregation dimension: provider, model, or project.",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, csv, or json.",
    ),
) -> None:
    """Show today's LLM spending, optionally grouped and filtered.

    Queries all completed jobs created since midnight UTC (or since the
    ``--since`` timestamp) and aggregates their costs. Terminal jobs with
    FAILED or CANCELLED status contribute their estimated cost where no
    actual cost was recorded.

    Args:
        since: Return costs for jobs created after this ISO-8601 timestamp.
            Defaults to midnight UTC of the current calendar day.
        group_by: Aggregation dimension — ``provider``, ``model``, or
            ``project``.
        format: Output format — ``table`` (default), ``csv``, or ``json``.
    """
    if group_by not in {"provider", "model", "project"}:
        err_console.print(f"[red]Unknown --group-by {group_by!r}. Choose from: provider, model, project[/red]")
        raise typer.Exit(1)

    if format not in {"table", "csv", "json"}:
        err_console.print(f"[red]Unknown --format {format!r}. Choose from: table, csv, json[/red]")
        raise typer.Exit(1)

    # Resolve the start timestamp
    if since:
        try:
            since_dt = datetime.fromisoformat(since)
            if since_dt.tzinfo is None:
                since_dt = since_dt.replace(tzinfo=timezone.utc)
        except ValueError:
            err_console.print(f"[red]Invalid --since timestamp: {since!r}[/red]")
            raise typer.Exit(1)
    else:
        since_dt = _start_of_today_utc()

    since_ts = since_dt.timestamp()

    async def _run() -> None:
        async with BatchClient() as client:
            jobs = await client.list_jobs(
                after=since_ts,
                limit=10_000,
                descending=False,
            )

        if not jobs:
            console.print(
                f"[dim]No jobs found since {since_dt.strftime('%Y-%m-%d %H:%M UTC')}.[/dim]"
            )
            return

        groups = _group_jobs(jobs, group_by)
        total_usd = sum(d["total_usd"] for d in groups.values())

        label_since = since_dt.strftime("%Y-%m-%d %H:%M UTC")

        if format == "table":
            console.print(
                f"[bold cyan]Costs since {label_since}[/bold cyan]  "
                f"([dim]{len(jobs)} job(s)[/dim])"
            )
            console.print(_render_table(groups, group_by, total_usd))

        elif format == "csv":
            console.print(_render_csv(groups, group_by), end="")

        elif format == "json":
            output = {
                "since": since_dt.isoformat(),
                "group_by": group_by,
                "total_usd": round(total_usd, 6),
                "job_count": len(jobs),
                "groups": [
                    {
                        group_by: label,
                        "job_count": data["job_count"],
                        "input_tokens": data["input_tokens"],
                        "output_tokens": data["output_tokens"],
                        "total_usd": round(data["total_usd"], 6),
                    }
                    for label, data in sorted(groups.items(), key=lambda x: -x[1]["total_usd"])
                ],
            }
            console.print_json(json.dumps(output))

    asyncio.run(_run())
