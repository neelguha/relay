"""cache subcommand group for the relay CLI.

Provides commands to inspect, manage, and evict entries from the relay
SQLite response cache.

Commands::

    relay cache stats       -- show size, hit rate, and entry count
    relay cache list        -- list cache entries with optional provider filter
    relay cache invalidate  -- delete one entry by cache_key or all entries for a job
    relay cache vacuum      -- force TTL expiry + LRU eviction
    relay cache clear       -- delete all cache entries

Example::

    $ relay cache stats
    $ relay cache list --provider anthropic
    $ relay cache invalidate <cache-key>
    $ relay cache vacuum
    $ relay cache clear
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from relay.client import BatchClient

app = typer.Typer(help="Inspect and manage the response cache.")
console = Console()
err_console = Console(stderr=True)

_GB = 1024 ** 3
_MB = 1024 ** 2


def _human_bytes(size: int) -> str:
    """Format a byte count as a human-readable string.

    Args:
        size: Size in bytes.

    Returns:
        A string such as ``"12.3 MB"`` or ``"1.01 GB"``.
    """
    if size >= _GB:
        return f"{size / _GB:.2f} GB"
    if size >= _MB:
        return f"{size / _MB:.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"


# ---------------------------------------------------------------------------
# stats command
# ---------------------------------------------------------------------------


@app.command("stats")
def stats() -> None:
    """Show cache statistics: size, entry count, and hit rate.

    Queries the SQLite cache for aggregate storage and performance metrics and
    displays them as a formatted key-value table.
    """
    async def _run() -> None:
        async with BatchClient() as client:
            cache = client.cache
            if cache is None:
                console.print("[yellow]Cache is disabled in the current configuration.[/yellow]")
                return

            data = await cache.stats()

        table = Table(show_header=False, expand=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        hit_rate_pct = data.get("hit_rate", 0.0) * 100
        table.add_row("Entry count", f"{data.get('entry_count', 0):,}")
        table.add_row("Total size", _human_bytes(data.get("size_bytes", 0)))
        table.add_row("Hit rate (session)", f"{hit_rate_pct:.1f}%")
        table.add_row("Hits", f"{data.get('hits', 0):,}")
        table.add_row("Misses", f"{data.get('misses', 0):,}")
        table.add_row("Compression", data.get("compression", "none"))

        console.print("[bold cyan]Cache Statistics[/bold cyan]")
        console.print(table)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# list command
# ---------------------------------------------------------------------------


@app.command("list")
def list_entries(
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Filter entries by provider."),
    limit: int = typer.Option(50, "--limit", "-n", help="Maximum number of entries to show."),
) -> None:
    """List cache entries with optional provider filter.

    Queries the raw ``cache_entries`` SQLite table and displays each entry's
    key prefix, provider, model, token counts, size, and last access time.

    Args:
        provider: When supplied, only entries for this provider are shown.
        limit: Maximum number of rows to return (default 50).
    """
    async def _run() -> None:
        async with BatchClient() as client:
            cache = client.cache
            if cache is None:
                console.print("[yellow]Cache is disabled in the current configuration.[/yellow]")
                return

            # Access the raw aiosqlite connection for the list query
            db = await cache._ensure_open()

            where_clause = ""
            params: tuple = ()
            if provider:
                where_clause = "WHERE provider = ?"
                params = (provider,)

            async with db.execute(
                f"""
                SELECT cache_key, provider, model,
                       input_tokens, output_tokens,
                       size_bytes, last_hit_at, hit_count, expires_at
                FROM cache_entries
                {where_clause}
                ORDER BY last_hit_at DESC
                LIMIT ?
                """,
                (*params, limit),
            ) as cursor:
                rows = await cursor.fetchall()

        if not rows:
            console.print("[dim]No cache entries found.[/dim]")
            return

        table = Table(show_header=True, header_style="bold cyan", expand=True)
        table.add_column("Key (prefix)", style="dim", no_wrap=True)
        table.add_column("Provider")
        table.add_column("Model")
        table.add_column("In tok", justify="right")
        table.add_column("Out tok", justify="right")
        table.add_column("Size", justify="right")
        table.add_column("Hits", justify="right")
        table.add_column("Last hit", no_wrap=True)
        table.add_column("Expires", no_wrap=True)

        now = time.time()
        for row in rows:
            key = row["cache_key"]
            last_hit = time.strftime("%Y-%m-%d %H:%M", time.localtime(row["last_hit_at"]))
            expires_at = row["expires_at"]
            if expires_at is None:
                expires_str = "never"
            elif expires_at < now:
                expires_str = "[red]expired[/red]"
            else:
                secs_left = int(expires_at - now)
                if secs_left > 86400:
                    expires_str = f"{secs_left // 86400}d"
                elif secs_left > 3600:
                    expires_str = f"{secs_left // 3600}h"
                else:
                    expires_str = f"{secs_left // 60}m"

            table.add_row(
                key[:16] + "…",
                row["provider"],
                row["model"],
                f"{row['input_tokens']:,}",
                f"{row['output_tokens']:,}",
                _human_bytes(row["size_bytes"]),
                str(row["hit_count"]),
                last_hit,
                expires_str,
            )

        console.print(table)
        console.print(f"\n[dim]{len(rows)} entry/entries shown.[/dim]")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# invalidate command
# ---------------------------------------------------------------------------


@app.command("invalidate")
def invalidate(
    cache_key: Optional[str] = typer.Argument(
        None, help="Cache key (SHA-256 hex) to delete."
    ),
    job: Optional[str] = typer.Option(
        None, "--job", help="Delete all cache entries associated with this job ID."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Delete one cache entry by key, or all entries for a job.

    Provide either a positional *cache_key* argument or the ``--job`` flag.
    At least one of the two must be given.

    Args:
        cache_key: SHA-256 hex digest identifying a single cache entry to
            remove.
        job: Relay job ID — removes all cache entries associated with that
            job via :meth:`~relay.cache.sqlite.SQLiteCache.invalidate_job`.
        yes: Skip the interactive confirmation prompt when set.
    """
    if cache_key is None and job is None:
        err_console.print("[red]Provide a cache_key argument or --job <job-id>.[/red]")
        raise typer.Exit(1)

    target = f"key {cache_key[:16]}…" if cache_key else f"job {job}"
    if not yes:
        confirmed = typer.confirm(f"Delete cache entries for {target}?")
        if not confirmed:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(0)

    async def _run() -> None:
        async with BatchClient() as client:
            cache = client.cache
            if cache is None:
                console.print("[yellow]Cache is disabled in the current configuration.[/yellow]")
                return

            if cache_key:
                await cache.invalidate(cache_key)
                console.print(f"[green]Deleted cache entry:[/green] {cache_key[:16]}…")
            elif job:
                await cache.invalidate_job(job)
                console.print(f"[green]Invalidated cache entries for job:[/green] {job}")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# vacuum command
# ---------------------------------------------------------------------------


@app.command("vacuum")
def vacuum() -> None:
    """Force TTL expiry and LRU eviction.

    Runs a maintenance pass that:

    1. Deletes all entries whose ``expires_at`` has passed.
    2. Evicts the least-recently-used entries if the total on-disk size still
       exceeds the configured ``max_size_gb`` limit.
    """
    async def _run() -> None:
        async with BatchClient() as client:
            cache = client.cache
            if cache is None:
                console.print("[yellow]Cache is disabled in the current configuration.[/yellow]")
                return

            before = await cache.stats()
            await cache.vacuum()
            after = await cache.stats()

        removed = before.get("entry_count", 0) - after.get("entry_count", 0)
        freed = before.get("size_bytes", 0) - after.get("size_bytes", 0)
        console.print(
            f"[green]Vacuum complete.[/green] "
            f"Removed {removed} entr{'ies' if removed != 1 else 'y'}, "
            f"freed {_human_bytes(max(freed, 0))}."
        )

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# clear command
# ---------------------------------------------------------------------------


@app.command("clear")
def clear(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Delete all cache entries.

    Permanently removes every row from the ``cache_entries`` table. This
    cannot be undone. Use ``relay cache vacuum`` to perform a softer cleanup
    based on TTL and size limits.

    Args:
        yes: Skip the interactive confirmation prompt when set.
    """
    if not yes:
        confirmed = typer.confirm(
            "[bold red]Delete ALL cache entries?[/bold red] This cannot be undone."
        )
        if not confirmed:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(0)

    async def _run() -> None:
        async with BatchClient() as client:
            cache = client.cache
            if cache is None:
                console.print("[yellow]Cache is disabled in the current configuration.[/yellow]")
                return

            db = await cache._ensure_open()
            async with db.execute("DELETE FROM cache_entries") as cursor:
                count = cursor.rowcount
            await db.commit()

        console.print(
            f"[green]Cleared {count if count >= 0 else 'all'} cache "
            f"entr{'ies' if count != 1 else 'y'}.[/green]"
        )

    asyncio.run(_run())
