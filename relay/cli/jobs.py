"""jobs subcommand group for the relay CLI.

Provides commands to list, inspect, cancel, and resubmit batch jobs.

Commands::

    relay jobs list      -- list jobs with optional filters
    relay jobs status    -- show status of a single job
    relay jobs cancel    -- cancel an in-progress job
    relay jobs resubmit-failed -- resubmit all failed requests from a job

Example::

    $ relay jobs list --provider anthropic --status IN_PROGRESS
    $ relay jobs status <job-id>
    $ relay jobs cancel <job-id>
    $ relay jobs resubmit-failed <job-id>
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from relay.client import BatchClient
from relay.models import BatchJob, JobStatus

app = typer.Typer(help="Manage batch jobs.")
console = Console()
err_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_dt(dt: datetime | str | None) -> str:
    """Format a datetime as a compact UTC string, returning '—' for None.

    Args:
        dt: A :class:`datetime` instance, an ISO-format string, or ``None``.

    Returns:
        A human-readable UTC timestamp string or ``'—'`` when ``None``.
    """
    if dt is None:
        return "—"
    if isinstance(dt, (int, float)):
        dt = datetime.fromtimestamp(dt)
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except ValueError:
            return dt
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _status_color(status: JobStatus) -> str:
    """Return a Rich markup color tag for a given job status.

    Args:
        status: A :class:`~relay.models.JobStatus` enum value.

    Returns:
        A Rich markup string wrapping the status value with an appropriate
        color.
    """
    colors = {
        JobStatus.PENDING: "dim",
        JobStatus.CACHE_RESOLVING: "yellow",
        JobStatus.VALIDATING: "yellow",
        JobStatus.SUBMITTING: "yellow",
        JobStatus.IN_PROGRESS: "cyan",
        JobStatus.DOWNLOADING: "blue",
        JobStatus.COMPLETED: "green",
        JobStatus.PARTIAL: "dark_orange",
        JobStatus.FAILED: "red",
        JobStatus.CANCELLED: "dim red",
    }
    color = colors.get(status, "white")
    return f"[{color}]{status.value}[/{color}]"


def _jobs_table(jobs: list[BatchJob]) -> Table:
    """Build a Rich table displaying a list of jobs.

    Args:
        jobs: Jobs to display.

    Returns:
        A :class:`rich.table.Table` ready to print.
    """
    table = Table(show_header=True, header_style="bold cyan", expand=True)
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Name")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Project")
    table.add_column("Status")
    table.add_column("Progress", justify="right")
    table.add_column("Cost (USD)", justify="right")
    table.add_column("Created", no_wrap=True)

    for job in jobs:
        table.add_row(
            job.id[:8] + "…",
            job.name or "—",
            job.provider,
            job.model,
            job.project or "—",
            _status_color(job.status),
            f"{job.completed_requests}/{job.total_requests}",
            f"${job.actual_cost_usd or job.estimated_cost_usd:.4f}",
            _format_dt(job.created_at),
        )
    return table


def _job_detail_table(job: BatchJob) -> Table:
    """Build a Rich key-value table for a single job.

    Args:
        job: The job to display.

    Returns:
        A :class:`rich.table.Table` with field-value rows.
    """
    table = Table(show_header=False, expand=False, box=None, padding=(0, 2))
    table.add_column("Field", style="bold")
    table.add_column("Value")

    rows = [
        ("Job ID", job.id),
        ("Name", job.name or "—"),
        ("Provider Job ID", job.provider_job_id or "—"),
        ("Provider", job.provider),
        ("Model", job.model),
        ("Project", job.project or "—"),
        ("Status", _status_color(job.status)),
        ("Total requests", str(job.total_requests)),
        ("Completed", str(job.completed_requests)),
        ("Failed", str(job.failed_requests)),
        ("Cache hits", str(job.cached_hits)),
        ("Input tokens", f"{job.input_tokens:,}"),
        ("Output tokens", f"{job.output_tokens:,}"),
        ("Est. cost (USD)", f"${job.estimated_cost_usd:.4f}"),
        ("Actual cost (USD)", f"${job.actual_cost_usd:.4f}" if job.actual_cost_usd is not None else "—"),
        ("Created at", _format_dt(job.created_at)),
        ("Submitted at", _format_dt(job.submitted_at)),
        ("Completed at", _format_dt(job.completed_at)),
        ("Tags", ", ".join(job.tags) if job.tags else "—"),
        ("Error", job.error or "—"),
    ]

    for field, value in rows:
        table.add_row(field, value)

    return table


# ---------------------------------------------------------------------------
# list command
# ---------------------------------------------------------------------------


@app.command("list")
def list_jobs(
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Filter by provider."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Filter by model."),
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status (e.g. IN_PROGRESS)."),
    project: Optional[str] = typer.Option(None, "--project", help="Filter by project label."),
    since: Optional[str] = typer.Option(
        None,
        "--since",
        help="Return jobs created after this ISO-8601 timestamp (e.g. 2024-01-01T00:00:00).",
    ),
    limit: int = typer.Option(50, "--limit", "-n", help="Maximum number of jobs to return."),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table or json."),
) -> None:
    """List batch jobs with optional filters.

    Prints a table (or JSON array) of jobs matching the supplied filter
    criteria. Results are ordered by creation time, newest first.

    Args:
        provider: Restrict output to jobs from this provider.
        model: Restrict output to jobs using this model identifier.
        status: Restrict output to jobs in this status (e.g. ``IN_PROGRESS``).
        project: Restrict output to jobs tagged with this project label.
        since: Only return jobs created after this ISO-8601 timestamp.
        limit: Maximum number of jobs to return (default 50).
        format: Output format — ``table`` (default) or ``json``.
    """
    if format not in {"table", "json"}:
        err_console.print(f"[red]Unknown format {format!r}. Choose from: table, json[/red]")
        raise typer.Exit(1)

    filters: dict = {}
    if provider:
        filters["provider"] = provider
    if model:
        filters["model"] = model
    if status:
        try:
            filters["status"] = JobStatus(status.upper())
        except ValueError:
            valid = ", ".join(s.value for s in JobStatus)
            err_console.print(f"[red]Unknown status {status!r}. Valid values: {valid}[/red]")
            raise typer.Exit(1)
    if project:
        filters["project"] = project
    if since:
        try:
            filters["after"] = datetime.fromisoformat(since)
        except ValueError:
            err_console.print(f"[red]Invalid --since timestamp: {since!r}[/red]")
            raise typer.Exit(1)

    filters["limit"] = limit
    filters["descending"] = True

    async def _run() -> None:
        async with BatchClient() as client:
            jobs = await client.list_jobs(**filters)

        if format == "json":
            import dataclasses
            output = [
                {k: (v.value if isinstance(v, JobStatus) else str(v) if isinstance(v, datetime) else v)
                 for k, v in dataclasses.asdict(job).items()}
                for job in jobs
            ]
            console.print_json(json.dumps(output))
        else:
            if not jobs:
                console.print("[dim]No jobs found.[/dim]")
                return
            console.print(_jobs_table(jobs))
            console.print(f"\n[dim]{len(jobs)} job(s) shown.[/dim]")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------


@app.command("status")
def job_status(
    job_id: str = typer.Argument(..., help="Relay job ID (UUID)."),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table or json."),
) -> None:
    """Show detailed status of a single job.

    Fetches and displays all fields of the :class:`~relay.models.BatchJob`
    record for the given job ID.

    Args:
        job_id: The relay-internal job UUID.
        format: Output format — ``table`` (default) or ``json``.
    """
    if format not in {"table", "json"}:
        err_console.print(f"[red]Unknown format {format!r}. Choose from: table, json[/red]")
        raise typer.Exit(1)

    async def _run() -> None:
        async with BatchClient() as client:
            job = await client.get_job(job_id)

        if format == "json":
            import dataclasses
            data = {
                k: (v.value if isinstance(v, JobStatus) else str(v) if isinstance(v, datetime) else v)
                for k, v in dataclasses.asdict(job).items()
            }
            console.print_json(json.dumps(data))
        else:
            console.print(_job_detail_table(job))

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# cancel command
# ---------------------------------------------------------------------------


@app.command("cancel")
def cancel(
    job_id: str = typer.Argument(..., help="Relay job ID (UUID) to cancel."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Cancel an in-progress batch job.

    Sends a cancellation request to the provider. Requests already processed
    by the provider may still be billed. Prompts for confirmation unless
    ``--yes`` is supplied.

    Args:
        job_id: The relay-internal job UUID of the job to cancel.
        yes: Skip the interactive confirmation prompt when set.
    """
    if not yes:
        confirmed = typer.confirm(f"Cancel job {job_id}?")
        if not confirmed:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(0)

    async def _run() -> None:
        async with BatchClient() as client:
            job = await client.cancel(job_id)
        console.print(f"[green]Job {job.id} cancelled.[/green] Status: {job.status.value}")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# resubmit-failed command
# ---------------------------------------------------------------------------


@app.command("resubmit-failed")
def resubmit_failed(
    job_id: str = typer.Argument(..., help="Relay job ID (UUID) whose failed requests to resubmit."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Resubmit failed requests from a completed job.

    Creates a new batch job containing only the failed requests from the
    specified completed job. The new job inherits the provider, model, and
    project from the original.

    Args:
        job_id: The relay-internal job UUID of the original job.
        yes: Skip the interactive confirmation prompt when set.
    """
    if not yes:
        confirmed = typer.confirm(f"Resubmit failed requests from job {job_id}?")
        if not confirmed:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(0)

    async def _run() -> None:
        async with BatchClient() as client:
            new_job = await client.resubmit_failed(job_id)
        console.print(
            f"[green]New job created:[/green] [bold]{new_job.id}[/bold] "
            f"({new_job.total_requests} request(s))"
        )

    asyncio.run(_run())
