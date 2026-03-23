"""export command for the relay CLI.

Exports downloaded results for a completed batch job to a local file or a
HuggingFace Hub dataset.

Supported formats::

    jsonl       One JSON object per line.  No extra dependencies.
    csv         Flat CSV with nested objects serialised as JSON strings.
    parquet     Apache Parquet via ``pyarrow`` (``pip install pyarrow``).
    hf_dataset  HuggingFace Arrow dataset (``pip install datasets``).

Example::

    $ relay export <job-id> --format jsonl --output results.jsonl
    $ relay export <job-id> --format parquet --output results.parquet
    $ relay export <job-id> --format hf_dataset --output ./hf_out/ --push myorg/my-dataset
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from relay.client import BatchClient
from relay.exporters import export_job

app = typer.Typer(help="Export job results to file or HuggingFace Hub.")
console = Console()
err_console = Console(stderr=True)

_VALID_FORMATS = {"jsonl", "csv", "parquet", "hf_dataset"}


@app.command("export")
def export(
    job_id: str = typer.Argument(..., help="Relay job ID (UUID) whose results to export."),
    format: str = typer.Option(
        "jsonl",
        "--format",
        "-f",
        help="Output format: jsonl, csv, parquet, or hf_dataset.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help=(
            "Output path.  For jsonl/csv/parquet this should be a file path; "
            "for hf_dataset a directory path.  Defaults to "
            "'<job-id>.<format>'."
        ),
    ),
    push: Optional[str] = typer.Option(
        None,
        "--push",
        help=(
            "HuggingFace Hub repository to push the dataset to "
            "(e.g. 'myorg/my-dataset').  Only used when --format hf_dataset."
        ),
    ),
    include_raw: bool = typer.Option(
        False,
        "--include-raw",
        help="Embed the provider's raw API response in each record (jsonl only).",
    ),
) -> None:
    """Export job results to a local file or HuggingFace Hub dataset.

    Downloads results for *job_id* from the local relay database (they must
    have already been downloaded via ``relay run`` or ``relay jobs status``)
    and writes them to *output* in the requested *format*.

    For ``hf_dataset`` format, the ``datasets`` library must be installed::

        pip install datasets

    For ``parquet`` format, ``pyarrow`` must be installed::

        pip install pyarrow

    When ``--push`` is provided with ``--format hf_dataset``, the dataset is
    also pushed to the HuggingFace Hub. You must be authenticated with the Hub
    (``huggingface-cli login``) before using this option.

    Args:
        job_id: The relay-internal job UUID of the job to export.
        format: Output format — ``jsonl``, ``csv``, ``parquet``, or
            ``hf_dataset``.
        output: Destination file or directory path.  Defaults to
            ``<job-id>.<format>``.
        push: HuggingFace Hub repository identifier.  Only used when
            ``format="hf_dataset"``.
        include_raw: When set, include the provider's raw API response dict
            in each record (``jsonl`` format only).
    """
    fmt = format.lower()
    if fmt not in _VALID_FORMATS:
        valid = ", ".join(sorted(_VALID_FORMATS))
        err_console.print(f"[red]Unknown format {format!r}. Choose from: {valid}[/red]")
        raise typer.Exit(1)

    if push and fmt != "hf_dataset":
        err_console.print("[red]--push is only valid with --format hf_dataset[/red]")
        raise typer.Exit(1)

    # Resolve default output path
    if output is None:
        ext = fmt if fmt != "hf_dataset" else "hf"
        output = Path(f"{job_id}.{ext}")

    output_str = str(output)

    async def _run() -> None:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(f"Downloading results for [bold]{job_id[:8]}…[/bold]")

            async with BatchClient() as client:
                results = await client.download(job_id)
                progress.update(task, description=f"Exporting {len(results)} result(s) as [bold]{fmt}[/bold]…")

                await export_job(
                    results=results,
                    format=fmt,  # type: ignore[arg-type]
                    output_path=output_str,
                    include_metadata=True,
                    include_raw_response=include_raw,
                    push_to_hub=push,
                )

        console.print(
            f"[green]Exported {len(results)} result(s)[/green] "
            f"to [bold]{output_str}[/bold] "
            f"([dim]{fmt}[/dim])"
        )
        if push:
            console.print(f"[green]Pushed to HuggingFace Hub:[/green] [bold]{push}[/bold]")

    asyncio.run(_run())
