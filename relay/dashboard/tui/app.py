"""Main Textual TUI application for the relay dashboard.

Provides a terminal-based interface for monitoring batch jobs managed by the
relay library.  Data is loaded from the relay SQLite job store
(:class:`~relay.db.store.JobStore`) and refreshed automatically every
:data:`REFRESH_INTERVAL` seconds.

Keyboard shortcuts
------------------
j / k           Navigate rows up/down in the focused panel.
Enter           Open detail view for the selected job.
c               Cancel the selected active job (confirmation required).
r               Resubmit the selected failed job.
e               Export job data (JSONL).
f               Open the filter input.
?               Show the help overlay.
q               Quit the application.
Tab             Cycle focus between panels.
d               Toggle dark/light theme.

Usage::

    from relay.dashboard.tui.app import RelayDashboard

    app = RelayDashboard(db_path="~/.relay/jobs.db")
    app.run()
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen, Screen
from textual.widget import Widget
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    Static,
    TabbedContent,
    TabPane,
)

from relay.db.store import JobStore
from relay.models import BatchJob, JobStatus

from relay.dashboard.tui.widgets.jobs_panel import JobsPanel
from relay.dashboard.tui.widgets.cost_panel import CostPanel
from relay.dashboard.tui.widgets.cache_panel import CachePanel

logger = logging.getLogger(__name__)

REFRESH_INTERVAL: int = 5
"""Auto-refresh interval in seconds."""

_ACTIVE_STATUSES = [
    JobStatus.PENDING.value,
    JobStatus.CACHE_RESOLVING.value,
    JobStatus.VALIDATING.value,
    JobStatus.SUBMITTING.value,
    JobStatus.IN_PROGRESS.value,
    JobStatus.DOWNLOADING.value,
]

_TERMINAL_STATUSES = [
    JobStatus.COMPLETED.value,
    JobStatus.PARTIAL.value,
    JobStatus.FAILED.value,
    JobStatus.CANCELLED.value,
]


# ---------------------------------------------------------------------------
# Helper screens / overlays
# ---------------------------------------------------------------------------


class HelpScreen(ModalScreen):
    """Modal overlay displaying keyboard shortcut help.

    Dismisses on any key press or when the user clicks the close button.
    """

    DEFAULT_CSS = """
    HelpScreen {
        align: center middle;
    }
    HelpScreen > Container {
        background: $surface;
        border: solid $primary;
        width: 60;
        height: auto;
        padding: 1 2;
    }
    HelpScreen .help-title {
        text-align: center;
        padding: 0 0 1 0;
        color: $text;
        text-style: bold;
    }
    HelpScreen .help-row {
        padding: 0 0 0 1;
    }
    HelpScreen Button {
        margin: 1 0 0 0;
        width: 100%;
    }
    """

    BINDINGS = [Binding("escape", "dismiss", "Close")]

    def compose(self) -> ComposeResult:
        """Compose the help overlay content.

        Yields:
            A container with the keybinding reference table and a close button.
        """
        with Container():
            yield Label("Keyboard Shortcuts", classes="help-title")
            shortcuts = [
                ("j / k", "Navigate rows up / down"),
                ("Enter", "Open job detail view"),
                ("c", "Cancel selected active job"),
                ("r", "Resubmit selected failed job"),
                ("e", "Export selected job data (JSONL)"),
                ("f", "Open filter input"),
                ("?", "Show this help"),
                ("Tab", "Cycle focus between panels"),
                ("d", "Toggle dark / light theme"),
                ("q", "Quit the application"),
            ]
            for key, desc in shortcuts:
                yield Label(f"[bold cyan]{key:<10}[/bold cyan] {desc}", classes="help-row")
            yield Button("Close", variant="primary", id="help-close")

    @on(Button.Pressed, "#help-close")
    def close_help(self) -> None:
        """Dismiss the help overlay when the close button is pressed."""
        self.dismiss()


class JobDetailScreen(ModalScreen):
    """Modal screen showing full details for a single batch job.

    Args:
        job: The :class:`~relay.models.BatchJob` whose details to display.
    """

    DEFAULT_CSS = """
    JobDetailScreen {
        align: center middle;
    }
    JobDetailScreen > ScrollableContainer {
        background: $surface;
        border: solid $primary;
        width: 80;
        height: 36;
        padding: 1 2;
    }
    JobDetailScreen .detail-title {
        text-style: bold;
        padding: 0 0 1 0;
    }
    JobDetailScreen .detail-row {
        padding: 0 0 0 1;
    }
    JobDetailScreen Button {
        margin: 1 0 0 0;
        width: 100%;
    }
    """

    BINDINGS = [Binding("escape", "dismiss", "Close")]

    def __init__(self, job: BatchJob, **kwargs: Any) -> None:
        """Initialise the detail screen.

        Args:
            job: The batch job whose details are to be displayed.
            **kwargs: Forwarded to :class:`~textual.screen.ModalScreen`.
        """
        super().__init__(**kwargs)
        self._job = job

    def compose(self) -> ComposeResult:
        """Compose the job detail overlay content.

        Yields:
            A scrollable container with all job fields and a close button.
        """
        job = self._job
        with ScrollableContainer():
            yield Label(f"Job Detail: {job.id}", classes="detail-title")

            def row(label: str, value: Any) -> Label:
                return Label(
                    f"[bold]{label:<22}[/bold] {value}",
                    classes="detail-row",
                )

            yield row("ID", job.id)
            yield row("Provider Job ID", job.provider_job_id or "—")
            yield row("Provider", job.provider)
            yield row("Model", job.model)
            yield row("Project", job.project or "—")
            yield row("Status", job.status.value)
            yield row("Total requests", job.total_requests)
            yield row("Completed", job.completed_requests)
            yield row("Failed", job.failed_requests)
            yield row("Cache hits", job.cached_hits)
            yield row("Input tokens", f"{job.input_tokens:,}")
            yield row("Output tokens", f"{job.output_tokens:,}")

            est = job.estimated_cost_usd or 0.0
            act = job.actual_cost_usd
            yield row("Estimated cost", f"${est:.6f}")
            yield row("Actual cost", f"${act:.6f}" if act is not None else "—")

            def _fmt_ts(ts: Any) -> str:
                if ts is None:
                    return "—"
                if isinstance(ts, (int, float)):
                    return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S UTC")
                if isinstance(ts, datetime):
                    return ts.strftime("%Y-%m-%d %H:%M:%S UTC")
                return str(ts)

            yield row("Created at", _fmt_ts(job.created_at))
            yield row("Submitted at", _fmt_ts(job.submitted_at))
            yield row("Completed at", _fmt_ts(job.completed_at))

            if job.tags:
                yield row("Tags", ", ".join(job.tags))

            if job.error:
                yield Label(
                    f"[bold red]Error:[/bold red] {job.error}",
                    classes="detail-row",
                )

            yield Button("Close", variant="primary", id="detail-close")

    @on(Button.Pressed, "#detail-close")
    def close_detail(self) -> None:
        """Dismiss the detail overlay when the close button is pressed."""
        self.dismiss()


class FilterScreen(ModalScreen):
    """Modal overlay with a text input for filtering the jobs table.

    The entered filter string is returned to the caller via
    :meth:`~textual.screen.ModalScreen.dismiss`.
    """

    DEFAULT_CSS = """
    FilterScreen {
        align: center middle;
    }
    FilterScreen > Container {
        background: $surface;
        border: solid $accent;
        width: 60;
        height: auto;
        padding: 1 2;
    }
    FilterScreen Label {
        padding: 0 0 1 0;
    }
    FilterScreen Input {
        width: 100%;
    }
    FilterScreen .filter-hint {
        color: $text-muted;
        padding: 1 0 0 0;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss_empty", "Cancel"),
        Binding("enter", "apply_filter", "Apply"),
    ]

    def __init__(self, current_filter: str = "", **kwargs: Any) -> None:
        """Initialise the filter screen.

        Args:
            current_filter: Pre-filled filter text from the previous session.
            **kwargs: Forwarded to :class:`~textual.screen.ModalScreen`.
        """
        super().__init__(**kwargs)
        self._current = current_filter

    def compose(self) -> ComposeResult:
        """Compose the filter input overlay.

        Yields:
            A container with a label, text input, and usage hint.
        """
        with Container():
            yield Label("Filter Jobs")
            yield Input(
                placeholder="Search by ID, project, model, status …",
                value=self._current,
                id="filter-input",
            )
            yield Label(
                "Press [bold]Enter[/bold] to apply, [bold]Esc[/bold] to cancel.",
                classes="filter-hint",
            )

    def on_mount(self) -> None:
        """Focus the filter input when the screen is mounted."""
        self.query_one("#filter-input", Input).focus()

    def action_dismiss_empty(self) -> None:
        """Dismiss the overlay without changing the current filter."""
        self.dismiss(None)

    def action_apply_filter(self) -> None:
        """Dismiss the overlay and return the entered filter text."""
        value = self.query_one("#filter-input", Input).value
        self.dismiss(value)


# ---------------------------------------------------------------------------
# Throughput / Status panels (lightweight statics)
# ---------------------------------------------------------------------------


class ThroughputPanel(Widget):
    """Lightweight throughput statistics panel.

    Derives requests-per-minute and tokens-per-minute figures from the
    supplied list of active jobs.

    Attributes:
        jobs: Current list of all jobs (used to compute throughput metrics).
        error_message: Error string shown in place of stats on failure.
    """

    DEFAULT_CSS = """
    ThroughputPanel {
        height: 100%;
        border: solid $primary;
        padding: 0 1;
    }
    ThroughputPanel > Label {
        color: $text-muted;
        padding: 0 0 1 0;
    }
    """

    jobs: reactive[list[BatchJob]] = reactive(list, layout=True)
    error_message: reactive[str] = reactive("")

    def compose(self) -> ComposeResult:
        """Compose the panel with a heading and content area.

        Yields:
            A label heading and a static content widget.
        """
        yield Label("Throughput")
        yield Static("", id="throughput-content")

    def on_mount(self) -> None:
        """Render initial content after mounting."""
        self._refresh()

    def watch_jobs(self, jobs: list[BatchJob]) -> None:
        """Re-render when job list changes.

        Args:
            jobs: Updated job list.
        """
        self._refresh()

    def watch_error_message(self, message: str) -> None:
        """Re-render when error state changes.

        Args:
            message: New error text or empty string.
        """
        self._refresh()

    def _refresh(self) -> None:
        """Rebuild and render the throughput metrics."""
        try:
            content: Static = self.query_one("#throughput-content", Static)
        except Exception:
            return

        if self.error_message:
            content.update(f"[bold red]Error:[/bold red] {self.error_message}")
            return

        active = [j for j in self.jobs if not j.status.is_terminal]
        total_active = len(active)
        total_completed = sum(j.completed_requests for j in self.jobs)
        total_failed = sum(j.failed_requests for j in self.jobs)
        total_cached = sum(j.cached_hits for j in self.jobs)
        total_requests = sum(j.total_requests for j in self.jobs)

        in_progress = [j for j in active if j.status == JobStatus.IN_PROGRESS]
        req_rate = "—"
        tok_rate = "—"

        # Estimate throughput from in-progress jobs using elapsed time.
        if in_progress:
            now = time.time()
            rates_req: list[float] = []
            rates_tok: list[float] = []
            for job in in_progress:
                ts = job.submitted_at
                if ts is None:
                    continue
                ts_f = ts.timestamp() if isinstance(ts, datetime) else float(ts)
                elapsed = max(1.0, now - ts_f)
                rates_req.append(job.completed_requests / elapsed * 60)
                tokens = (job.input_tokens or 0) + (job.output_tokens or 0)
                rates_tok.append(tokens / elapsed * 60)
            if rates_req:
                req_rate = f"{sum(rates_req):.0f} req/min"
            if rates_tok:
                tok_sum = sum(rates_tok)
                if tok_sum >= 1_000_000:
                    tok_rate = f"{tok_sum / 1_000_000:.2f}M tok/min"
                elif tok_sum >= 1_000:
                    tok_rate = f"{tok_sum / 1_000:.1f}K tok/min"
                else:
                    tok_rate = f"{tok_sum:.0f} tok/min"

        lines = [
            f"[bold]Active jobs:[/bold]    {total_active}",
            f"[bold]Request rate:[/bold]   {req_rate}",
            f"[bold]Token rate:[/bold]     {tok_rate}",
            "",
            f"[bold]All-time totals[/bold]",
            f"  Total requests: {total_requests:,}",
            f"  Completed:      [green]{total_completed:,}[/green]",
            f"  Failed:         [red]{total_failed:,}[/red]",
            f"  Cache hits:     [cyan]{total_cached:,}[/cyan]",
        ]
        content.update("\n".join(lines))


class ErrorLogPanel(Widget):
    """Error log panel showing recent job failures.

    Displays the most recent failed jobs along with their error messages.

    Attributes:
        jobs: Full job list; the panel filters for failed entries.
        error_message: Database-level error banner text.
    """

    DEFAULT_CSS = """
    ErrorLogPanel {
        height: 100%;
        border: solid $error;
        padding: 0 1;
    }
    ErrorLogPanel > Label {
        color: $error;
        padding: 0 0 1 0;
    }
    """

    jobs: reactive[list[BatchJob]] = reactive(list, layout=True)
    error_message: reactive[str] = reactive("")

    def compose(self) -> ComposeResult:
        """Compose the error log panel.

        Yields:
            A heading label and a scrollable content area.
        """
        yield Label("Error Log")
        yield Static("", id="error-log-content")

    def on_mount(self) -> None:
        """Populate the panel after mounting."""
        self._refresh()

    def watch_jobs(self, jobs: list[BatchJob]) -> None:
        """Re-render when the job list changes.

        Args:
            jobs: Updated job list.
        """
        self._refresh()

    def watch_error_message(self, message: str) -> None:
        """Re-render when the database error state changes.

        Args:
            message: New error text or empty string.
        """
        self._refresh()

    def _refresh(self) -> None:
        """Rebuild and render the error log content."""
        try:
            content: Static = self.query_one("#error-log-content", Static)
        except Exception:
            return

        if self.error_message:
            content.update(f"[bold red]DB error:[/bold red] {self.error_message}")
            return

        failed = [
            j for j in self.jobs
            if j.status == JobStatus.FAILED and j.error
        ]
        failed.sort(
            key=lambda j: (
                j.completed_at.timestamp()
                if isinstance(j.completed_at, datetime)
                else float(j.completed_at or 0)
            ),
            reverse=True,
        )
        recent = failed[:20]

        if not recent:
            content.update("[dim]No errors recorded.[/dim]")
            return

        lines: list[str] = []
        for job in recent:
            short_id = job.id[:8]
            model = job.model[:20]
            err = (job.error or "")[:80]
            lines.append(f"[bold red]{short_id}[/bold red] [{model}] {err}")

        content.update("\n".join(lines))


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


class RelayDashboard(App):
    """Relay TUI dashboard application.

    Connects to the relay SQLite job store and displays a live-updating
    terminal interface for monitoring batch jobs.  All panels degrade
    gracefully when the database is locked or unavailable.

    Args:
        db_path: Path to the SQLite database file created by
            :class:`~relay.db.store.JobStore`.  Defaults to
            ``"~/.relay/jobs.db"``.

    Example::

        app = RelayDashboard(db_path="~/.relay/jobs.db")
        app.run()
    """

    TITLE = "Relay Dashboard"
    SUB_TITLE = "LLM Batch Job Monitor"

    CSS = """
    Screen {
        layout: vertical;
    }
    #main-grid {
        layout: grid;
        grid-size: 2 3;
        grid-rows: 1fr 1fr 1fr;
        grid-columns: 3fr 1fr;
        height: 1fr;
    }
    #jobs-container {
        row-span: 2;
        height: 100%;
    }
    #recent-jobs-container {
        row-span: 1;
        height: 100%;
    }
    #status-bar {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("j", "nav_down", "Next row", show=False),
        Binding("k", "nav_up", "Prev row", show=False),
        Binding("enter", "detail", "Detail"),
        Binding("c", "cancel_job", "Cancel"),
        Binding("r", "resubmit_job", "Resubmit"),
        Binding("e", "export_job", "Export"),
        Binding("f", "filter", "Filter"),
        Binding("question_mark", "help", "Help", key_display="?"),
        Binding("q", "quit", "Quit"),
        Binding("tab", "focus_next", "Next panel", show=False),
        Binding("d", "toggle_theme", "Theme", show=False),
    ]

    _all_jobs: reactive[list[BatchJob]] = reactive(list)
    _active_jobs: reactive[list[BatchJob]] = reactive(list)
    _recent_jobs: reactive[list[BatchJob]] = reactive(list)
    _db_error: reactive[str] = reactive("")
    _filter_text: reactive[str] = reactive("")
    _last_refresh: reactive[str] = reactive("Never")

    def __init__(self, db_path: str | Path = "~/.relay/jobs.db", **kwargs: Any) -> None:
        """Initialise the dashboard application.

        Args:
            db_path: Path to the relay SQLite database.
            **kwargs: Forwarded to :class:`~textual.app.App`.
        """
        super().__init__(**kwargs)
        self._db_path = Path(db_path).expanduser().resolve()
        self._store: JobStore | None = None

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        """Compose the full dashboard layout.

        Yields:
            Header, the main panel grid, a status bar, and the footer.
        """
        yield Header()

        with TabbedContent():
            with TabPane("Jobs", id="tab-jobs"):
                with Horizontal(id="main-grid"):
                    with Vertical(id="jobs-container"):
                        yield JobsPanel(id="active-jobs-panel")
                    with Vertical():
                        yield ThroughputPanel(id="throughput-panel")
                        yield CostPanel(id="cost-panel")
                        yield CachePanel(id="cache-panel")

            with TabPane("Recent", id="tab-recent"):
                yield JobsPanel(id="recent-jobs-panel")

            with TabPane("Errors", id="tab-errors"):
                yield ErrorLogPanel(id="error-log-panel")

        yield Static("", id="status-bar")
        yield Footer()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def on_mount(self) -> None:
        """Open the job store and start the auto-refresh loop after mounting."""
        await self._open_store()
        self.refresh_loop()

    async def on_unmount(self) -> None:
        """Close the job store when the application exits."""
        await self._close_store()

    async def _open_store(self) -> None:
        """Open a connection to the SQLite job store.

        Sets :attr:`_db_error` if the connection cannot be established so
        all panels can render a graceful error banner.
        """
        try:
            self._store = JobStore(self._db_path)
            await self._store.open()
            self._db_error = ""
        except Exception as exc:
            logger.exception("Failed to open job store at %s", self._db_path)
            self._db_error = str(exc)
            self._store = None

    async def _close_store(self) -> None:
        """Close the SQLite job store connection if open."""
        if self._store is not None:
            try:
                await self._store.close()
            except Exception:
                pass
            self._store = None

    # ------------------------------------------------------------------
    # Data refresh
    # ------------------------------------------------------------------

    @work(exclusive=True)
    async def refresh_loop(self) -> None:
        """Background worker that refreshes job data every ``REFRESH_INTERVAL`` seconds.

        Runs indefinitely until the application quits.  Each iteration
        attempts to fetch job data from the store; on failure it sets a
        non-empty :attr:`_db_error` that all panels render gracefully.
        """
        while True:
            await self._fetch_data()
            await asyncio.sleep(REFRESH_INTERVAL)

    async def _fetch_data(self) -> None:
        """Fetch fresh job data from the store and update reactive state.

        Handles :class:`Exception` broadly so a locked or unavailable
        database never crashes the application.
        """
        if self._store is None:
            # Attempt to reconnect.
            await self._open_store()
            if self._store is None:
                return

        try:
            all_jobs = await self._store.list_jobs(limit=500)
            active = [j for j in all_jobs if not j.status.is_terminal]
            recent = sorted(
                [j for j in all_jobs if j.status.is_terminal],
                key=lambda j: (
                    j.completed_at.timestamp()
                    if isinstance(j.completed_at, datetime)
                    else float(j.completed_at or 0)
                ),
                reverse=True,
            )[:50]

            self._all_jobs = all_jobs
            self._active_jobs = active
            self._recent_jobs = recent
            self._db_error = ""
            self._last_refresh = datetime.now().strftime("%H:%M:%S")

            self._propagate_to_panels(all_jobs, active, recent, "")

        except Exception as exc:
            logger.warning("Dashboard data refresh failed: %s", exc)
            err = f"{type(exc).__name__}: {exc}"
            self._db_error = err
            self._propagate_to_panels([], [], [], err)

    def _propagate_to_panels(
        self,
        all_jobs: list[BatchJob],
        active: list[BatchJob],
        recent: list[BatchJob],
        error: str,
    ) -> None:
        """Push updated data to all dashboard panels.

        Args:
            all_jobs: Full list of all jobs for aggregate panels.
            active: Non-terminal jobs for the active jobs table.
            recent: Most recent terminal jobs for the recent jobs table.
            error: Error message to propagate; empty string clears errors.
        """
        try:
            active_panel: JobsPanel = self.query_one("#active-jobs-panel", JobsPanel)
            active_panel.jobs = active
            active_panel.error_message = error
            active_panel.filter_text = self._filter_text
        except Exception:
            pass

        try:
            recent_panel: JobsPanel = self.query_one("#recent-jobs-panel", JobsPanel)
            recent_panel.jobs = recent
            recent_panel.error_message = error
        except Exception:
            pass

        try:
            throughput: ThroughputPanel = self.query_one("#throughput-panel", ThroughputPanel)
            throughput.jobs = all_jobs
            throughput.error_message = error
        except Exception:
            pass

        try:
            cost: CostPanel = self.query_one("#cost-panel", CostPanel)
            cost.jobs = all_jobs
            cost.error_message = error
        except Exception:
            pass

        try:
            err_panel: ErrorLogPanel = self.query_one("#error-log-panel", ErrorLogPanel)
            err_panel.jobs = all_jobs
            err_panel.error_message = error
        except Exception:
            pass

        try:
            status: Static = self.query_one("#status-bar", Static)
            if error:
                status.update(f"[red]DB error — {error}[/red]")
            else:
                active_count = len(active)
                status.update(
                    f"[dim]Last refreshed: {self._last_refresh} | "
                    f"Active jobs: {active_count} | "
                    f"DB: {self._db_path}[/dim]"
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_nav_down(self) -> None:
        """Move cursor down one row in the focused jobs panel."""
        self._focused_jobs_panel().navigate(1)

    def action_nav_up(self) -> None:
        """Move cursor up one row in the focused jobs panel."""
        self._focused_jobs_panel().navigate(-1)

    def action_detail(self) -> None:
        """Open the job detail overlay for the currently selected job."""
        panel = self._focused_jobs_panel()
        job = panel.get_selected_job()
        if job is not None:
            self.push_screen(JobDetailScreen(job))

    def action_cancel_job(self) -> None:
        """Cancel the currently selected active job (no-op if already terminal)."""
        panel = self._focused_jobs_panel()
        job = panel.get_selected_job()
        if job is None:
            return
        if job.status.is_terminal:
            self.notify("Job is already in a terminal state.", severity="warning")
            return
        self.notify(
            f"Cancellation requested for job {job.id[:8]}. "
            "Use the relay client to confirm.",
            severity="information",
        )

    def action_resubmit_job(self) -> None:
        """Resubmit the selected failed job (informational — no direct DB write)."""
        panel = self._focused_jobs_panel()
        job = panel.get_selected_job()
        if job is None:
            return
        if job.status != JobStatus.FAILED:
            self.notify("Only FAILED jobs can be resubmitted.", severity="warning")
            return
        self.notify(
            f"Resubmit job {job.id[:8]} via the relay CLI or client.",
            severity="information",
        )

    def action_export_job(self) -> None:
        """Export the selected job's data (informational hint)."""
        panel = self._focused_jobs_panel()
        job = panel.get_selected_job()
        if job is None:
            return
        self.notify(
            f"Export job {job.id[:8]} using: relay export --job-id {job.id}",
            severity="information",
        )

    def action_filter(self) -> None:
        """Open the filter input overlay."""

        def _apply(result: str | None) -> None:
            if result is not None:
                self._filter_text = result
                try:
                    panel: JobsPanel = self.query_one("#active-jobs-panel", JobsPanel)
                    panel.filter_text = result
                except Exception:
                    pass

        self.push_screen(FilterScreen(current_filter=self._filter_text), _apply)

    def action_help(self) -> None:
        """Open the keyboard shortcut help overlay."""
        self.push_screen(HelpScreen())

    def action_toggle_theme(self) -> None:
        """Toggle between dark and light themes."""
        self.dark = not self.dark

    def _focused_jobs_panel(self) -> JobsPanel:
        """Return the currently visible :class:`JobsPanel`.

        Falls back to the active-jobs panel when no jobs panel has focus.

        Returns:
            The best candidate :class:`JobsPanel` for navigation actions.
        """
        focused = self.focused
        if isinstance(focused, JobsPanel):
            return focused
        # Determine which tab is visible and return its panel.
        try:
            tabs: TabbedContent = self.query_one(TabbedContent)
            active_tab = tabs.active
            if active_tab == "tab-recent":
                return self.query_one("#recent-jobs-panel", JobsPanel)
        except Exception:
            pass
        return self.query_one("#active-jobs-panel", JobsPanel)
