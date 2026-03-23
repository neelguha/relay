"""Active jobs table widget for the relay TUI dashboard.

Displays a sortable, filterable table of batch jobs fetched from the
relay SQLite job store.  The panel handles database errors gracefully by
rendering an inline error message rather than crashing the application.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import DataTable, Label, Static

from relay.models import BatchJob, JobStatus

if TYPE_CHECKING:
    pass

# Status colour mapping used for Rich markup inside table cells.
_STATUS_STYLES: dict[str, str] = {
    JobStatus.PENDING.value: "dim",
    JobStatus.CACHE_RESOLVING.value: "cyan",
    JobStatus.VALIDATING.value: "cyan",
    JobStatus.SUBMITTING.value: "yellow",
    JobStatus.IN_PROGRESS.value: "green",
    JobStatus.DOWNLOADING.value: "blue",
    JobStatus.COMPLETED.value: "bold green",
    JobStatus.PARTIAL.value: "bold yellow",
    JobStatus.FAILED.value: "bold red",
    JobStatus.CANCELLED.value: "dim red",
}

_COLUMNS = (
    ("ID", 10),
    ("Project", 16),
    ("Provider", 10),
    ("Model", 20),
    ("Status", 16),
    ("Progress", 14),
    ("Cost (USD)", 12),
    ("Created", 20),
)


def _truncate(value: str, width: int) -> str:
    """Truncate *value* to *width* characters, appending an ellipsis.

    Args:
        value: The string to truncate.
        width: Maximum allowed character length (including ellipsis).

    Returns:
        The original string if it fits, otherwise a truncated version ending
        in ``"…"``.
    """
    if len(value) <= width:
        return value
    return value[: width - 1] + "…"


def _format_age(ts: float | datetime | None) -> str:
    """Format a Unix timestamp or datetime as a human-readable age string.

    Args:
        ts: Unix timestamp (float) or :class:`datetime` object, or ``None``.

    Returns:
        A concise elapsed-time string such as ``"3m 12s"`` or ``"2h 5m"``,
        or ``"—"`` when *ts* is ``None``.
    """
    if ts is None:
        return "—"
    if isinstance(ts, datetime):
        ts = ts.timestamp()
    elapsed = int(time.time() - ts)
    if elapsed < 60:
        return f"{elapsed}s"
    if elapsed < 3600:
        m, s = divmod(elapsed, 60)
        return f"{m}m {s}s"
    h, rem = divmod(elapsed, 3600)
    m = rem // 60
    return f"{h}h {m}m"


class JobsPanel(Widget):
    """Active jobs table widget.

    Renders a :class:`~textual.widgets.DataTable` populated with
    :class:`~relay.models.BatchJob` records supplied via :attr:`jobs`.
    The selected row index is exposed via :attr:`selected_index` so the
    parent application can open a detail view.

    Attributes:
        jobs: Current list of batch jobs to display.  Setting this reactive
            attribute triggers an automatic table refresh.
        error_message: When non-empty, an error banner is shown instead of
            the table.
        filter_text: Case-insensitive substring filter applied to job IDs,
            projects, models, and statuses.
        selected_index: Zero-based index of the currently highlighted row.
    """

    DEFAULT_CSS = """
    JobsPanel {
        height: 100%;
        border: solid $primary;
        padding: 0 1;
    }
    JobsPanel > Label {
        color: $text-muted;
        padding: 0 0 1 0;
    }
    JobsPanel .error-banner {
        color: $error;
        padding: 1;
        border: solid $error;
        margin: 1;
    }
    """

    jobs: reactive[list[BatchJob]] = reactive(list, layout=True)
    error_message: reactive[str] = reactive("")
    filter_text: reactive[str] = reactive("")
    selected_index: reactive[int] = reactive(0)

    def compose(self) -> ComposeResult:
        """Compose the widget's child elements.

        Yields:
            A heading label and the data table.
        """
        yield Label("Active Jobs")
        yield DataTable(id="jobs-table", zebra_stripes=True, cursor_type="row")

    def on_mount(self) -> None:
        """Initialise table columns after the widget is mounted."""
        table: DataTable = self.query_one("#jobs-table", DataTable)
        for header, width in _COLUMNS:
            table.add_column(header, width=width)

    def watch_jobs(self, jobs: list[BatchJob]) -> None:
        """Rebuild the table whenever the :attr:`jobs` list changes.

        Args:
            jobs: The updated list of :class:`~relay.models.BatchJob` objects.
        """
        self._rebuild_table(jobs)

    def watch_error_message(self, message: str) -> None:
        """Show or hide the error banner when :attr:`error_message` changes.

        Args:
            message: The new error message string, or empty to clear.
        """
        self._apply_error(message)

    def watch_filter_text(self, text: str) -> None:
        """Re-filter the table when the filter text changes.

        Args:
            text: New filter substring.
        """
        self._rebuild_table(self.jobs)

    def _apply_error(self, message: str) -> None:
        """Render an error banner, hiding the table if there is an error.

        Args:
            message: Error message to display, or empty string to clear.
        """
        try:
            banner = self.query_one(".error-banner")
            banner.remove()
        except Exception:
            pass

        table = self.query_one("#jobs-table", DataTable)
        if message:
            table.display = False
            self.mount(Static(f"[bold red]Database error:[/] {message}", classes="error-banner"))
        else:
            table.display = True

    def _filtered_jobs(self, jobs: list[BatchJob]) -> list[BatchJob]:
        """Apply :attr:`filter_text` to *jobs* and return matching entries.

        Matches against job ID, project, provider, model, and status fields
        using a case-insensitive substring search.

        Args:
            jobs: Full list of :class:`~relay.models.BatchJob` objects.

        Returns:
            Filtered list containing only jobs that match the current filter.
        """
        ft = self.filter_text.lower().strip()
        if not ft:
            return jobs
        result = []
        for job in jobs:
            haystack = " ".join([
                job.id,
                job.project or "",
                job.provider,
                job.model,
                job.status.value,
            ]).lower()
            if ft in haystack:
                result.append(job)
        return result

    def _rebuild_table(self, jobs: list[BatchJob]) -> None:
        """Clear and repopulate the data table from *jobs*.

        Args:
            jobs: Current list of batch jobs (before filtering).
        """
        try:
            table: DataTable = self.query_one("#jobs-table", DataTable)
        except Exception:
            return

        table.clear()
        visible = self._filtered_jobs(jobs)

        for job in visible:
            total = job.total_requests or 1
            done = job.completed_requests
            pct = int(done / total * 100)
            progress_bar = f"{done}/{total} ({pct}%)"

            status_style = _STATUS_STYLES.get(job.status.value, "")
            status_cell = (
                f"[{status_style}]{job.status.value}[/{status_style}]"
                if status_style
                else job.status.value
            )

            cost_val = job.actual_cost_usd if job.actual_cost_usd is not None else job.estimated_cost_usd
            cost_str = f"${cost_val:.4f}"

            table.add_row(
                _truncate(job.id[:8], 10),
                _truncate(job.project or "—", 16),
                _truncate(job.provider, 10),
                _truncate(job.model, 20),
                status_cell,
                progress_bar,
                cost_str,
                _format_age(job.created_at),
            )

        if visible and 0 <= self.selected_index < len(visible):
            table.move_cursor(row=self.selected_index)

    def navigate(self, direction: int) -> None:
        """Move the cursor up or down by *direction* rows.

        Args:
            direction: ``-1`` to move up, ``1`` to move down.
        """
        try:
            table: DataTable = self.query_one("#jobs-table", DataTable)
        except Exception:
            return

        new_idx = max(0, min(self.selected_index + direction, table.row_count - 1))
        self.selected_index = new_idx
        table.move_cursor(row=new_idx)

    def get_selected_job(self) -> BatchJob | None:
        """Return the :class:`~relay.models.BatchJob` at the selected row.

        Returns:
            The selected job, or ``None`` if the jobs list is empty or the
            index is out of range.
        """
        visible = self._filtered_jobs(self.jobs)
        if not visible or self.selected_index >= len(visible):
            return None
        return visible[self.selected_index]
