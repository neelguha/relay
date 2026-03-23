"""relay CLI — Multi-provider LLM batch prediction tool.

Provides the root Typer application and registers all sub-command groups and
standalone commands.  The ``relay`` entry-point is configured in
``pyproject.toml`` (or ``setup.py``) as::

    [project.scripts]
    relay = "relay.cli:main"

Sub-command layout::

    relay submit    <input.jsonl>   -- submit a batch job
    relay run       <input.jsonl>   -- submit + wait + download
    relay estimate  <input.jsonl>   -- dry-run cost estimate
    relay export    <job-id>        -- export results to file or Hub
    relay jobs      list / status / cancel / resubmit-failed
    relay cache     stats / list / invalidate / vacuum / clear
    relay costs     today

Example::

    $ relay submit requests.jsonl --provider anthropic --model claude-opus-4-5
    $ relay jobs list --status IN_PROGRESS
    $ relay cache stats
"""

from __future__ import annotations

import typer

# Sub-command group apps
from relay.cli.cache import app as _cache_app
from relay.cli.costs import app as _costs_app
from relay.cli.jobs import app as _jobs_app

# Standalone command functions
from relay.cli.estimate import estimate
from relay.cli.export import export
from relay.cli.submit import run, submit

app = typer.Typer(
    name="relay",
    help="Multi-provider LLM batch prediction tool.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)

# ── Sub-command groups ────────────────────────────────────────────────────────

app.add_typer(_jobs_app, name="jobs", help="Manage batch jobs.")
app.add_typer(_cache_app, name="cache", help="Inspect and manage the response cache.")
app.add_typer(_costs_app, name="costs", help="Report LLM spending.")

# ── Standalone commands ───────────────────────────────────────────────────────

app.command("submit")(submit)
app.command("run")(run)
app.command("estimate")(estimate)
app.command("export")(export)


# ── Dashboard command ────────────────────────────────────────────────────────


@app.command("dashboard")
def dashboard(
    tui: bool = typer.Option(False, "--tui", help="Launch terminal UI instead of web dashboard."),
    host: str = typer.Option("127.0.0.1", "--host", help="Web dashboard bind address."),
    port: int = typer.Option(7860, "--port", help="Web dashboard port."),
    open_browser: bool = typer.Option(False, "--open", help="Open browser automatically."),
) -> None:
    """Launch the monitoring dashboard (web or terminal)."""
    import asyncio
    from relay.config import load_config

    cfg = load_config()

    if tui:
        try:
            from relay.dashboard.tui.app import RelayDashboard
        except ImportError:
            typer.echo("TUI dashboard requires 'textual'. Install with: pip install relay[dashboard]")
            raise typer.Exit(1)
        dashboard_app = RelayDashboard(db_path=cfg.db_path)
        dashboard_app.run()
    else:
        try:
            import uvicorn
            from relay.dashboard.web.app import create_app
        except ImportError:
            typer.echo("Web dashboard requires 'fastapi' and 'uvicorn'. Install with: pip install relay[dashboard]")
            raise typer.Exit(1)

        web_app = create_app(db_path=cfg.db_path)

        if open_browser:
            import webbrowser
            webbrowser.open(f"http://{host}:{port}")

        uvicorn.run(web_app, host=host, port=port)


# ── Entry-point ───────────────────────────────────────────────────────────────


def main() -> None:
    """Entry-point for the ``relay`` CLI executable.

    Invoked when the package is run as ``relay`` from the command line, as
    configured in ``pyproject.toml``::

        [project.scripts]
        relay = "relay.cli:main"
    """
    app()


if __name__ == "__main__":
    main()
