"""estimate command for the relay CLI.

Reads a JSONL file and estimates token usage and cost without submitting
any requests to a provider. Cache hits are checked so the estimate reflects
the actual net cost that would be incurred.

Example::

    $ relay estimate requests.jsonl --provider anthropic --model claude-opus-4-5
    $ relay estimate requests.jsonl --provider openai --model gpt-4o --no-cache
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from relay.client import BatchClient
from relay.models import BatchConfig, BatchRequest

app = typer.Typer(help="Estimate tokens and cost without submitting.")
console = Console()
err_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# estimate command
# ---------------------------------------------------------------------------


@app.command("estimate")
def estimate(
    input_file: Path = typer.Argument(..., help="Path to JSONL file of batch requests."),
    provider: str = typer.Option(..., "--provider", "-p", help="Provider name (anthropic, openai, google, xai)."),
    model: str = typer.Option(..., "--model", "-m", help="Model identifier."),
    project: Optional[str] = typer.Option(None, "--project", help="Project label (for grouping only, not submitted)."),
    tags: Optional[List[str]] = typer.Option(None, "--tag", help="Tags (for grouping only, not submitted)."),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Ignore the cache when estimating (treat all requests as net-new).",
    ),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table or json."),
) -> None:
    """Estimate token usage and cost for a JSONL batch without submitting.

    Reads *input_file*, checks the relay cache for existing results, and
    computes the number of cache hits, net (new) requests, input/output token
    counts, and gross/net USD cost using the current pricing table.

    No requests are sent to the provider. The estimate is based on the
    configured batch pricing rates and a character-level token heuristic for
    requests where the provider tokeniser is unavailable.

    Args:
        input_file: Path to a ``.jsonl`` file of batch requests.
        provider: Provider name (``anthropic``, ``openai``, ``google``,
            ``xai``).
        model: Model identifier (e.g. ``claude-opus-4-5``).
        project: Project label attached to the estimate (not submitted).
        tags: Arbitrary tags attached to the estimate (not submitted).
        no_cache: When set, treat all requests as net-new regardless of what
            is in the cache.
        format: Output format — ``table`` (default) or ``json``.
    """
    if format not in {"table", "json"}:
        err_console.print(f"[red]Unknown format {format!r}. Choose from: table, json[/red]")
        raise typer.Exit(1)

    # Load requests from JSONL
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

    config = BatchConfig(
        provider=provider,
        model=model,
        project=project,
        tags=list(tags or []),
        use_cache=not no_cache,
    )

    async def _run() -> None:
        async with BatchClient() as client:
            estimate_result = await client.estimate_cost(requests, config)

        if format == "json":
            output = {
                "total_requests": estimate_result.total_requests,
                "cache_hits": estimate_result.cache_hits,
                "net_requests": estimate_result.net_requests,
                "input_tokens": estimate_result.input_tokens,
                "estimated_output_tokens": estimate_result.estimated_output_tokens,
                "gross_usd": round(estimate_result.gross_usd, 6),
                "saved_usd": round(estimate_result.saved_usd, 6),
                "net_usd": round(estimate_result.net_usd, 6),
                "per_provider": {
                    k: round(v, 6) for k, v in estimate_result.per_provider.items()
                },
            }
            console.print_json(json.dumps(output))
            return

        # Table output
        table = Table(show_header=False, expand=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        cache_hit_pct = (
            estimate_result.cache_hits / estimate_result.total_requests * 100
            if estimate_result.total_requests > 0
            else 0.0
        )

        table.add_row("Provider", provider)
        table.add_row("Model", model)
        table.add_row("Total requests", f"{estimate_result.total_requests:,}")
        table.add_row(
            "Cache hits",
            f"{estimate_result.cache_hits:,} ({cache_hit_pct:.1f}%)",
        )
        table.add_row("Net (new) requests", f"{estimate_result.net_requests:,}")
        table.add_row("Est. input tokens", f"{estimate_result.input_tokens:,}")
        table.add_row("Est. output tokens", f"{estimate_result.estimated_output_tokens:,}")
        table.add_row("Gross cost (USD)", f"${estimate_result.gross_usd:.4f}")
        table.add_row("Cache savings (USD)", f"-${estimate_result.saved_usd:.4f}")
        table.add_row("[bold green]Net cost (USD)[/bold green]", f"[bold green]${estimate_result.net_usd:.4f}[/bold green]")

        console.print("[bold cyan]Cost Estimate[/bold cyan]")
        console.print(table)

        if estimate_result.saved_usd > 0:
            console.print(
                f"\n[dim]Cache would save ${estimate_result.saved_usd:.4f} "
                f"({cache_hit_pct:.1f}% of requests already cached).[/dim]"
            )

    asyncio.run(_run())
