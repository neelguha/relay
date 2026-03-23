"""Cost summary widget for the relay TUI dashboard.

Aggregates cost and token usage statistics from a list of
:class:`~relay.models.BatchJob` objects and renders them in a compact
panel.  Database errors are displayed inline so the application never
crashes due to a locked or unavailable database.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Label, Static

from relay.models import BatchJob

if TYPE_CHECKING:
    pass


def _fmt_tokens(n: int) -> str:
    """Format a token count as a compact human-readable string.

    Args:
        n: Raw token count (non-negative integer).

    Returns:
        A string such as ``"1.23M"``, ``"456K"``, or ``"123"`` depending on
        the magnitude of *n*.
    """
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


class CostPanel(Widget):
    """Cost summary widget.

    Computes aggregate cost and token usage from the supplied
    :attr:`jobs` list and renders a structured summary broken down by
    provider and model.  A database-error state is shown when
    :attr:`error_message` is set.

    Attributes:
        jobs: List of :class:`~relay.models.BatchJob` objects from which cost
            and token aggregates are derived.
        error_message: Non-empty string triggers an error banner in place of
            the summary content.
    """

    DEFAULT_CSS = """
    CostPanel {
        height: 100%;
        border: solid $primary;
        padding: 0 1;
    }
    CostPanel > Label {
        color: $text-muted;
        padding: 0 0 1 0;
    }
    CostPanel .cost-section {
        padding: 0 0 1 0;
    }
    CostPanel .error-banner {
        color: $error;
        padding: 1;
        border: solid $error;
        margin: 1;
    }
    CostPanel .summary-line {
        padding: 0 0 0 1;
    }
    """

    jobs: reactive[list[BatchJob]] = reactive(list, layout=True)
    error_message: reactive[str] = reactive("")

    def compose(self) -> ComposeResult:
        """Compose the widget's child elements.

        Yields:
            A heading label followed by the summary content container.
        """
        yield Label("Cost Summary")
        yield Static("", id="cost-content")

    def on_mount(self) -> None:
        """Populate the cost summary after the widget is mounted."""
        self._refresh_content()

    def watch_jobs(self, jobs: list[BatchJob]) -> None:
        """Rebuild the summary whenever the :attr:`jobs` list changes.

        Args:
            jobs: Updated list of batch jobs.
        """
        self._refresh_content()

    def watch_error_message(self, message: str) -> None:
        """Show or clear the error banner when :attr:`error_message` changes.

        Args:
            message: New error message, or empty string to clear.
        """
        self._refresh_content()

    def _aggregate(
        self,
        jobs: list[BatchJob],
    ) -> tuple[float, float, int, int, int, dict[str, float], dict[str, float]]:
        """Compute aggregate metrics from *jobs*.

        Args:
            jobs: List of :class:`~relay.models.BatchJob` objects.

        Returns:
            A 7-tuple of:
              - ``total_estimated``: Sum of estimated costs in USD.
              - ``total_actual``: Sum of actual costs in USD (where known).
              - ``total_input_tokens``: Aggregate input token count.
              - ``total_output_tokens``: Aggregate output token count.
              - ``total_cache_hits``: Aggregate cache-hit count.
              - ``by_provider``: Dict mapping provider name to actual/estimated
                cost in USD.
              - ``by_model``: Dict mapping model identifier to actual/estimated
                cost in USD.
        """
        total_estimated = 0.0
        total_actual = 0.0
        total_input = 0
        total_output = 0
        total_cache = 0
        by_provider: dict[str, float] = defaultdict(float)
        by_model: dict[str, float] = defaultdict(float)

        for job in jobs:
            est = job.estimated_cost_usd or 0.0
            act = job.actual_cost_usd if job.actual_cost_usd is not None else est
            total_estimated += est
            total_actual += act
            total_input += job.input_tokens or 0
            total_output += job.output_tokens or 0
            total_cache += job.cached_hits or 0
            by_provider[job.provider] += act
            by_model[job.model] += act

        return (
            total_estimated,
            total_actual,
            total_input,
            total_output,
            total_cache,
            dict(by_provider),
            dict(by_model),
        )

    def _build_markup(self, jobs: list[BatchJob]) -> str:
        """Build Rich markup text for the cost summary.

        Args:
            jobs: Current list of batch jobs.

        Returns:
            A multi-line Rich markup string ready to be passed to a
            :class:`~textual.widgets.Static` widget.
        """
        if not jobs:
            return "[dim]No jobs recorded.[/dim]"

        (
            total_est,
            total_act,
            total_input,
            total_output,
            total_cache,
            by_provider,
            by_model,
        ) = self._aggregate(jobs)

        lines: list[str] = []

        lines.append("[bold]Overall[/bold]")
        lines.append(f"  Estimated:    [yellow]${total_est:.4f}[/yellow]")
        lines.append(f"  Actual:       [green]${total_act:.4f}[/green]")
        lines.append(f"  Input tokens: {_fmt_tokens(total_input)}")
        lines.append(f"  Output tokens:{_fmt_tokens(total_output)}")
        lines.append(f"  Cache hits:   {total_cache}")
        lines.append("")

        if by_provider:
            lines.append("[bold]By Provider[/bold]")
            for provider, cost in sorted(by_provider.items(), key=lambda x: -x[1]):
                lines.append(f"  {provider:<16} [green]${cost:.4f}[/green]")
            lines.append("")

        if by_model:
            lines.append("[bold]By Model[/bold]")
            for model, cost in sorted(by_model.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"  {model[:20]:<20} [green]${cost:.4f}[/green]")
            if len(by_model) > 5:
                lines.append(f"  [dim]... and {len(by_model) - 5} more[/dim]")

        return "\n".join(lines)

    def _refresh_content(self) -> None:
        """Rebuild and render the cost content or error banner."""
        try:
            content: Static = self.query_one("#cost-content", Static)
        except Exception:
            return

        if self.error_message:
            content.update(
                f"[bold red]Database error:[/bold red]\n{self.error_message}"
            )
            return

        markup = self._build_markup(self.jobs)
        content.update(markup)
