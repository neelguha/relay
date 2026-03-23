"""Cache statistics widget for the relay TUI dashboard.

Displays live cache metrics (hit rate, entry count, storage size, etc.)
sourced from a :class:`~relay.cache.base.CacheBackend` implementation.
When the cache backend is unavailable or raises an error the panel
renders a graceful error message instead of propagating the exception.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Label, Static


def _fmt_bytes(n: int) -> str:
    """Format a byte count as a compact human-readable string.

    Args:
        n: Number of bytes (non-negative integer).

    Returns:
        A string such as ``"1.23 GB"``, ``"456.0 MB"``, ``"12.3 KB"``, or
        ``"999 B"`` depending on magnitude.
    """
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.2f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def _fmt_pct(value: float) -> str:
    """Format a fraction in ``[0.0, 1.0]`` as a percentage string.

    Args:
        value: Float in the range ``[0.0, 1.0]``.

    Returns:
        A string such as ``"83.5%"``.
    """
    return f"{value * 100:.1f}%"


class CachePanel(Widget):
    """Cache statistics widget.

    Renders a summary of the current cache state using statistics
    provided via :attr:`stats`.  Database or backend errors are
    rendered inline via :attr:`error_message` without crashing the
    application.

    Attributes:
        stats: Dict of cache statistics as returned by
            :meth:`~relay.cache.base.CacheBackend.stats`.  Common keys are
            ``"hit_rate"``, ``"entry_count"``, ``"size_bytes"``, and
            ``"eviction_count"``.  The widget degrades gracefully for missing
            keys by displaying ``"—"``.
        error_message: When non-empty, an error banner replaces the stats
            display.
    """

    DEFAULT_CSS = """
    CachePanel {
        height: 100%;
        border: solid $primary;
        padding: 0 1;
    }
    CachePanel > Label {
        color: $text-muted;
        padding: 0 0 1 0;
    }
    CachePanel .error-banner {
        color: $error;
        padding: 1;
        border: solid $error;
        margin: 1;
    }
    """

    stats: reactive[dict[str, Any]] = reactive(dict, layout=True)
    error_message: reactive[str] = reactive("")

    def compose(self) -> ComposeResult:
        """Compose the widget's child elements.

        Yields:
            A heading label followed by the stats content container.
        """
        yield Label("Cache Stats")
        yield Static("", id="cache-content")

    def on_mount(self) -> None:
        """Populate the stats display after the widget is mounted."""
        self._refresh_content()

    def watch_stats(self, stats: dict[str, Any]) -> None:
        """Rebuild the display whenever :attr:`stats` changes.

        Args:
            stats: New statistics dictionary from the cache backend.
        """
        self._refresh_content()

    def watch_error_message(self, message: str) -> None:
        """Show or clear the error banner when :attr:`error_message` changes.

        Args:
            message: New error text, or empty string to clear.
        """
        self._refresh_content()

    def _build_markup(self, stats: dict[str, Any]) -> str:
        """Build Rich markup for the cache statistics display.

        Handles missing keys gracefully by substituting ``"—"``.

        Args:
            stats: Statistics dictionary as returned by the cache backend.

        Returns:
            Multi-line Rich markup string suitable for a
            :class:`~textual.widgets.Static` widget.
        """
        if not stats:
            return "[dim]Cache statistics unavailable.[/dim]"

        def _get(key: str, default: Any = None) -> Any:
            return stats.get(key, default)

        hit_rate = _get("hit_rate")
        entry_count = _get("entry_count")
        size_bytes = _get("size_bytes")
        eviction_count = _get("eviction_count")
        compression_ratio = _get("compression_ratio")
        total_gets = _get("total_gets")
        total_puts = _get("total_puts")

        lines: list[str] = []

        lines.append("[bold]Cache Backend[/bold]")

        if hit_rate is not None:
            pct = _fmt_pct(hit_rate)
            colour = "green" if hit_rate >= 0.5 else "yellow" if hit_rate >= 0.2 else "red"
            lines.append(f"  Hit rate:     [{colour}]{pct}[/{colour}]")
        else:
            lines.append("  Hit rate:     —")

        if entry_count is not None:
            lines.append(f"  Entries:      {entry_count:,}")
        else:
            lines.append("  Entries:      —")

        if size_bytes is not None:
            lines.append(f"  Size:         {_fmt_bytes(size_bytes)}")
        else:
            lines.append("  Size:         —")

        if eviction_count is not None:
            lines.append(f"  Evictions:    {eviction_count:,}")

        if compression_ratio is not None:
            lines.append(f"  Compression:  {compression_ratio:.2f}x")

        if total_gets is not None or total_puts is not None:
            lines.append("")
            lines.append("[bold]Operations[/bold]")
            if total_gets is not None:
                lines.append(f"  Total GETs:   {total_gets:,}")
            if total_puts is not None:
                lines.append(f"  Total PUTs:   {total_puts:,}")

        # Surface any additional keys the backend may provide.
        known_keys = {
            "hit_rate", "entry_count", "size_bytes", "eviction_count",
            "compression_ratio", "total_gets", "total_puts",
        }
        extras = {k: v for k, v in stats.items() if k not in known_keys}
        if extras:
            lines.append("")
            lines.append("[bold]Additional[/bold]")
            for key, val in sorted(extras.items()):
                label = key.replace("_", " ").title()
                lines.append(f"  {label:<20} {val}")

        return "\n".join(lines)

    def _refresh_content(self) -> None:
        """Rebuild and render the cache stats or error banner."""
        try:
            content: Static = self.query_one("#cache-content", Static)
        except Exception:
            return

        if self.error_message:
            content.update(
                f"[bold red]Cache backend error:[/bold red]\n{self.error_message}"
            )
            return

        markup = self._build_markup(self.stats)
        content.update(markup)
