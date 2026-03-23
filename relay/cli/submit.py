"""submit and run commands for the relay CLI.

Provides two top-level commands:

* ``relay submit`` — read a JSONL file, submit a batch job, and return the
  job ID immediately.
* ``relay run`` — submit, wait for completion, and download results in a
  single command with optional Rich live progress display.

Example::

    $ relay submit requests.jsonl --provider anthropic --model claude-opus-4-5
    $ relay run requests.jsonl --provider openai --model gpt-4o --watch
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from relay.client import BatchClient
from relay.models import BatchConfig, BatchRequest, JobProgress, JobStatus

app = typer.Typer(help="Submit batch jobs and wait for results.")
console = Console()
err_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_requests(input_file: Path) -> list[BatchRequest]:
    """Parse a JSONL file into a list of BatchRequest objects.

    Args:
        input_file: Path to a ``.jsonl`` file where each line is a JSON object
            matching the :class:`~relay.models.BatchRequest` schema.

    Returns:
        An ordered list of :class:`~relay.models.BatchRequest` instances.

    Raises:
        typer.Exit: If the file cannot be read or contains invalid JSON.
    """
    requests: list[BatchRequest] = []
    try:
        with input_file.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    err_console.print(
                        f"[red]JSON parse error on line {lineno}:[/red] {exc}"
                    )
                    raise typer.Exit(1) from exc
                requests.append(BatchRequest(**data))
    except OSError as exc:
        err_console.print(f"[red]Cannot read {input_file}:[/red] {exc}")
        raise typer.Exit(1) from exc

    if not requests:
        err_console.print(f"[red]No requests found in {input_file}[/red]")
        raise typer.Exit(1)

    return requests


def _build_config(
    provider: str,
    model: str,
    project: Optional[str],
    tags: List[str],
    no_cache: bool,
    output_dir: Optional[str],
    name: Optional[str] = None,
) -> BatchConfig:
    """Construct a :class:`~relay.models.BatchConfig` from CLI arguments.

    Args:
        provider: Provider identifier (e.g. ``"anthropic"``).
        model: Model identifier (e.g. ``"claude-opus-4-5"``).
        project: Optional project label for cost grouping.
        tags: Arbitrary string tags to attach to the job.
        no_cache: When ``True``, disable cache resolution for this job.
        output_dir: Optional directory to write downloaded results to.
        name: Optional human-readable name for easy lookup later.

    Returns:
        A fully populated :class:`~relay.models.BatchConfig`.
    """
    return BatchConfig(
        provider=provider,
        model=model,
        name=name,
        project=project,
        tags=list(tags),
        use_cache=not no_cache,
        output_dir=output_dir,
    )


def _make_progress_table(progress: JobProgress) -> Table:
    """Build a Rich table snapshot from a JobProgress object.

    Args:
        progress: A :class:`~relay.models.JobProgress` snapshot from the
            ``wait`` async generator.

    Returns:
        A :class:`rich.table.Table` ready to render inside a :class:`rich.live.Live`
        context.
    """
    elapsed = progress.elapsed_seconds
    eta = progress.eta_seconds
    total = max(progress.total, 1)
    done = progress.completed
    cached = progress.cached
    failed = progress.failed

    requests_per_min = (done / max(elapsed, 1)) * 60

    cache_hit_pct = (cached / total) * 100 if total > 0 else 0.0
    eta_str = f"{int(eta)}s" if eta is not None else "—"

    table = Table(show_header=True, header_style="bold cyan", expand=True)
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    table.add_row("Status", str(progress.status.value))
    table.add_row("Progress", f"{done}/{total}")
    table.add_row("Failed", str(failed))
    table.add_row("Cached", f"{cached} ({cache_hit_pct:.1f}%)")
    table.add_row("Elapsed", f"{int(elapsed)}s")
    table.add_row("ETA", eta_str)
    table.add_row("Req/min", f"{requests_per_min:.1f}")
    table.add_row("Cost so far", f"${progress.cost_so_far:.4f}")

    return table


# ---------------------------------------------------------------------------
# submit command
# ---------------------------------------------------------------------------


@app.command("submit")
def submit(
    input_file: Path = typer.Argument(..., help="Path to JSONL file of batch requests."),
    provider: str = typer.Option(..., "--provider", "-p", help="Provider name (anthropic, openai, google, xai)."),
    model: str = typer.Option(..., "--model", "-m", help="Model identifier."),
    project: Optional[str] = typer.Option(None, "--project", help="Project label for cost grouping."),
    tags: Optional[List[str]] = typer.Option(None, "--tag", help="Tags to attach to the job (repeatable)."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache resolution for this job."),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="Directory to write downloaded results."),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Human-readable name for easy lookup later."),
    confirm_above: Optional[float] = typer.Option(
        None,
        "--confirm-above",
        help="Prompt for confirmation if estimated cost exceeds this USD threshold.",
    ),
) -> None:
    """Submit a JSONL batch job and print the job ID.

    Reads *input_file*, validates each request, resolves cache hits, submits
    the remainder to *provider*, and prints the new job ID to stdout. Use
    ``relay run`` to submit and wait for completion in a single step.

    Args:
        input_file: Path to a ``.jsonl`` file of batch requests.
        provider: Provider name (``anthropic``, ``openai``, ``google``, ``xai``).
        model: Model identifier (e.g. ``claude-opus-4-5``).
        project: Optional project label used in cost reports.
        tags: Arbitrary string tags attached to the job (repeatable).
        no_cache: Disable cache resolution when set.
        output_dir: Optional directory to persist downloaded results.
        name: Human-readable name for the job. Use this to check on results
            later without saving the UUID.
        confirm_above: Prompt for confirmation when estimated cost exceeds this
            USD value.
    """
    requests = _load_requests(input_file)
    config = _build_config(
        provider=provider,
        model=model,
        project=project,
        tags=tags or [],
        no_cache=no_cache,
        output_dir=output_dir,
        name=name,
    )

    async def _run() -> None:
        async with BatchClient() as client:
            # Estimate cost upfront for confirmation gate
            if confirm_above is not None:
                estimate = await client.estimate(requests, config)
                if estimate.net_usd > confirm_above:
                    confirm = typer.confirm(
                        f"Estimated cost ${estimate.net_usd:.4f} exceeds "
                        f"--confirm-above ${confirm_above:.2f}. Proceed?"
                    )
                    if not confirm:
                        console.print("[yellow]Aborted.[/yellow]")
                        raise typer.Exit(0)

            job = await client.submit(requests, config)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------


@app.command("run")
def run(
    input_file: Path = typer.Argument(..., help="Path to JSONL file of batch requests."),
    provider: str = typer.Option(..., "--provider", "-p", help="Provider name (anthropic, openai, google, xai)."),
    model: str = typer.Option(..., "--model", "-m", help="Model identifier."),
    project: Optional[str] = typer.Option(None, "--project", help="Project label for cost grouping."),
    tags: Optional[List[str]] = typer.Option(None, "--tag", help="Tags to attach to the job (repeatable)."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache resolution for this job."),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="Directory to write downloaded results."),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Human-readable name for easy lookup later."),
    confirm_above: Optional[float] = typer.Option(
        None,
        "--confirm-above",
        help="Prompt for confirmation if estimated cost exceeds this USD threshold.",
    ),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path for results."),
    format: str = typer.Option("jsonl", "--format", "-f", help="Output format: jsonl, csv, or parquet."),
    watch: bool = typer.Option(False, "--watch", help="Show live Rich progress table while waiting."),
) -> None:
    """Submit, wait, and download results in one command.

    Combines ``relay submit``, waiting for the job to reach a terminal state,
    and downloading the results. When ``--watch`` is given, a live Rich table
    is displayed showing elapsed time, ETA, requests/min, cost, and cache hit
    percentage.

    Args:
        input_file: Path to a ``.jsonl`` file of batch requests.
        provider: Provider name (``anthropic``, ``openai``, ``google``, ``xai``).
        model: Model identifier.
        project: Optional project label used in cost reports.
        tags: Arbitrary string tags attached to the job (repeatable).
        no_cache: Disable cache resolution when set.
        output_dir: Optional directory to persist downloaded results.
        name: Human-readable name for the job.
        confirm_above: Prompt for confirmation when estimated cost exceeds this
            USD value.
        output: Destination file for exported results.
        format: Export format — one of ``jsonl``, ``csv``, or ``parquet``.
        watch: Display a live Rich progress table while waiting for the job.
    """
    requests = _load_requests(input_file)
    config = _build_config(
        provider=provider,
        model=model,
        project=project,
        tags=tags or [],
        no_cache=no_cache,
        output_dir=output_dir,
        name=name,
    )

    if format not in {"jsonl", "csv", "parquet"}:
        err_console.print(f"[red]Unknown format {format!r}. Choose from: jsonl, csv, parquet[/red]")
        raise typer.Exit(1)

    async def _run() -> None:
        from relay.exporters import export_job

        async with BatchClient() as client:
            # Estimate cost upfront for confirmation gate
            if confirm_above is not None:
                estimate = await client.estimate(requests, config)
                if estimate.net_usd > confirm_above:
                    confirm = typer.confirm(
                        f"Estimated cost ${estimate.net_usd:.4f} exceeds "
                        f"--confirm-above ${confirm_above:.2f}. Proceed?"
                    )
                    if not confirm:
                        console.print("[yellow]Aborted.[/yellow]")
                        raise typer.Exit(0)

            console.print(f"[bold]Submitting {len(requests)} requests[/bold] via [cyan]{provider}[/cyan] / [cyan]{model}[/cyan]...")
            job = await client.submit(requests, config)
            job_id = job.id
            console.print(f"[green]Job submitted:[/green] [bold]{job_id}[/bold]")

            if watch:
                with Live(console=console, refresh_per_second=2) as live:
                    async for progress in client.wait(job_id):
                        live.update(_make_progress_table(progress))
                        if progress.status.is_terminal:
                            break
            else:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("{task.completed}/{task.total}"),
                    TimeElapsedColumn(),
                    console=console,
                ) as bar:
                    task = bar.add_task("Waiting...", total=len(requests))
                    async for progress in client.wait(job_id):
                        bar.update(task, completed=progress.completed, description=f"[cyan]{progress.status.value}[/cyan]")
                        if progress.status.is_terminal:
                            break

            final_job = await client.get_job(job_id)
            console.print(f"\n[bold]Status:[/bold] {final_job.status.value}")
            console.print(f"[bold]Completed:[/bold] {final_job.completed_requests}/{final_job.total_requests}")
            console.print(f"[bold]Cost:[/bold] ${final_job.actual_cost_usd or 0.0:.4f}")

            if final_job.status not in {JobStatus.FAILED, JobStatus.CANCELLED}:
                console.print("\n[dim]Downloading results...[/dim]")
                results = await client.download(job_id)
                console.print(f"[green]Downloaded {len(results)} results.[/green]")

                if output:
                    out_path = str(output)
                    await export_job(results, format, out_path)
                    console.print(f"[green]Exported to {out_path}[/green]")

    asyncio.run(_run())
