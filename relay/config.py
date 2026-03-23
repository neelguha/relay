"""Configuration loading and management for relay.

Loads configuration from multiple sources in priority order, merges them, and
exposes a single typed ``RelayConfig`` dataclass consumed by the rest of the
library.

Sources (highest priority first):
    1. Path supplied via ``--config`` on the CLI (or passed directly to
       :func:`load_config`).
    2. Path stored in the ``RELAY_CONFIG`` environment variable.
    3. ``~/.relay/config.toml`` (user-level defaults).
    4. ``pyproject.toml`` under the ``[tool.relay]`` table (project-level
       defaults).

Environment variable interpolation:
    Any ``${VAR_NAME}`` token in a raw TOML file is replaced with the
    corresponding environment variable value *before* the file is parsed.
    Missing variables are left as-is so that downstream code can detect and
    report them.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Python 3.11+ ships tomllib in the standard library; fall back to the
# third-party tomli package for earlier versions.
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover – Python < 3.11
    try:
        import tomli as tomllib  # type: ignore[no-reintimplicit-reexport]
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "relay requires either Python 3.11+ (tomllib) or the 'tomli' "
            "package (`pip install tomli`) for TOML parsing."
        ) from exc


# ── Helpers ────────────────────────────────────────────────────────────────────

_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _interpolate_env_vars(raw: str) -> str:
    """Replace ``${VAR_NAME}`` tokens with their environment variable values.

    Args:
        raw: Raw TOML source text that may contain ``${VAR_NAME}`` tokens.

    Returns:
        The source text with every recognised token replaced.  Tokens whose
        variable is not set are left unchanged so that callers can detect
        missing credentials explicitly.
    """

    def _replace(match: re.Match) -> str:  # type: ignore[type-arg]
        return os.environ.get(match.group(1), match.group(0))

    return _ENV_VAR_PATTERN.sub(_replace, raw)


def _read_toml(path: Path) -> dict[str, Any]:
    """Read, interpolate, and parse a TOML file.

    Args:
        path: Absolute or relative path to the ``.toml`` file.

    Returns:
        Parsed TOML document as a nested dictionary.

    Raises:
        FileNotFoundError: If *path* does not exist.
        tomllib.TOMLDecodeError: If the file contains invalid TOML.
    """
    raw = path.read_text(encoding="utf-8")
    interpolated = _interpolate_env_vars(raw)
    return tomllib.loads(interpolated)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict.

    Scalar values in *override* shadow those in *base*.  Nested dicts are
    merged recursively rather than replaced wholesale.

    Args:
        base: Lower-priority dictionary.
        override: Higher-priority dictionary whose values win on conflict.

    Returns:
        A new merged dictionary (neither input is mutated).
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ── Config dataclasses ─────────────────────────────────────────────────────────


@dataclass
class CostConfig:
    """Spend-guard thresholds.

    Attributes:
        warn_threshold_usd: Log a warning when estimated cost exceeds this.
        require_confirmation_usd: Prompt the user for confirmation above this.
        hard_limit_usd: Refuse to submit jobs that would exceed this limit.
    """

    warn_threshold_usd: float = 1.00
    require_confirmation_usd: float = 10.00
    hard_limit_usd: float = 100.00


@dataclass
class CacheConfig:
    """Result-caching settings.

    Attributes:
        enabled: Whether the cache layer is active.
        backend: Storage backend identifier (``"sqlite"`` or ``"redis"``).
        redis_url: Connection URL used when *backend* is ``"redis"``.
        ttl_seconds: Time-to-live for cached entries (30 days by default).
        max_size_gb: Maximum on-disk cache size in gigabytes.
        compress: Compress cached payloads with zlib when ``True``.
    """

    enabled: bool = True
    backend: str = "sqlite"
    redis_url: str = "redis://localhost:6379/0"
    ttl_seconds: int = 2_592_000  # 30 days
    max_size_gb: float = 5.0
    compress: bool = True


@dataclass
class RetryConfig:
    """Exponential-backoff retry policy.

    Attributes:
        max_attempts: Maximum number of total attempts (first try + retries).
        initial_backoff_seconds: Delay before the first retry.
        backoff_multiplier: Factor applied to the delay after each attempt.
        max_backoff_seconds: Upper bound on the computed delay.
        jitter: Jitter strategy – ``"full"`` randomises the whole interval,
            ``"equal"`` adds half an interval of random jitter, ``"none"``
            applies no jitter.
        retry_on: HTTP status codes that trigger a retry.
    """

    max_attempts: int = 5
    initial_backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 60.0
    jitter: str = "full"
    retry_on: list[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])


@dataclass
class AnthropicProviderConfig:
    """Configuration for the Anthropic provider.

    Attributes:
        api_key: Anthropic API key (typically ``${ANTHROPIC_API_KEY}``).
        default_model: Model identifier used when none is specified per-request.
        max_concurrent_batches: Maximum number of in-flight batch jobs.
    """

    api_key: str = ""
    default_model: str = "claude-opus-4-5"
    max_concurrent_batches: int = 5


@dataclass
class OpenAIProviderConfig:
    """Configuration for the OpenAI provider.

    Attributes:
        api_key: OpenAI API key (typically ``${OPENAI_API_KEY}``).
        default_model: Model identifier used when none is specified per-request.
        max_concurrent_batches: Maximum number of in-flight batch jobs.
    """

    api_key: str = ""
    default_model: str = "gpt-4o"
    max_concurrent_batches: int = 10


@dataclass
class GoogleProviderConfig:
    """Configuration for the Google (Gemini) provider.

    Attributes:
        api_key: Google AI API key (typically ``${GOOGLE_API_KEY}``).
        project_id: GCP project identifier (typically ``${GCP_PROJECT_ID}``).
        default_model: Model identifier used when none is specified per-request.
    """

    api_key: str = ""
    project_id: str = ""
    default_model: str = "gemini-2.0-flash"


@dataclass
class XAIProviderConfig:
    """Configuration for the xAI (Grok) provider.

    Attributes:
        api_key: xAI API key (typically ``${XAI_API_KEY}``).
        default_model: Model identifier used when none is specified per-request.
    """

    api_key: str = ""
    default_model: str = "grok-3"


@dataclass
class ProvidersConfig:
    """Aggregated provider configurations.

    Attributes:
        anthropic: Settings for the Anthropic provider.
        openai: Settings for the OpenAI provider.
        google: Settings for the Google provider.
        xai: Settings for the xAI provider.
    """

    anthropic: AnthropicProviderConfig = field(default_factory=AnthropicProviderConfig)
    openai: OpenAIProviderConfig = field(default_factory=OpenAIProviderConfig)
    google: GoogleProviderConfig = field(default_factory=GoogleProviderConfig)
    xai: XAIProviderConfig = field(default_factory=XAIProviderConfig)


@dataclass
class DashboardConfig:
    """Web dashboard settings.

    Attributes:
        enabled: Launch the dashboard server on startup when ``True``.
        host: Interface address to bind to.
        port: TCP port to listen on.
        theme: UI colour theme (``"dark"`` or ``"light"``).
        auto_refresh_seconds: Polling interval for live status updates.
    """

    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 7860
    theme: str = "dark"
    auto_refresh_seconds: int = 5


@dataclass
class RelayConfig:
    """Top-level relay configuration.

    All settings have sensible defaults so the library can be used without
    any configuration file.

    Attributes:
        db_path: Path to the SQLite database used for job persistence.
        log_level: Standard logging level string (``"DEBUG"``, ``"INFO"``, …).
        log_dir: Directory where rotating log files are written.
        output_dir: Default directory for downloaded batch results.
        cost: Spend-guard thresholds.
        cache: Result-caching settings.
        retry: Retry policy.
        providers: Per-provider API credentials and defaults.
        dashboard: Web dashboard settings.
    """

    db_path: str = "~/.relay/jobs.db"
    log_level: str = "INFO"
    log_dir: str = "~/.relay/logs"
    output_dir: str = "./relay_results"
    cost: CostConfig = field(default_factory=CostConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    providers: ProvidersConfig = field(default_factory=ProvidersConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)


# ── Dataclass construction from raw dicts ──────────────────────────────────────


def _build_cost(data: dict[str, Any]) -> CostConfig:
    return CostConfig(
        warn_threshold_usd=data.get("warn_threshold_usd", CostConfig.warn_threshold_usd),
        require_confirmation_usd=data.get(
            "require_confirmation_usd", CostConfig.require_confirmation_usd
        ),
        hard_limit_usd=data.get("hard_limit_usd", CostConfig.hard_limit_usd),
    )


def _build_cache(data: dict[str, Any]) -> CacheConfig:
    defaults = CacheConfig()
    return CacheConfig(
        enabled=data.get("enabled", defaults.enabled),
        backend=data.get("backend", defaults.backend),
        redis_url=data.get("redis_url", defaults.redis_url),
        ttl_seconds=data.get("ttl_seconds", defaults.ttl_seconds),
        max_size_gb=data.get("max_size_gb", defaults.max_size_gb),
        compress=data.get("compress", defaults.compress),
    )


def _build_retry(data: dict[str, Any]) -> RetryConfig:
    defaults = RetryConfig()
    return RetryConfig(
        max_attempts=data.get("max_attempts", defaults.max_attempts),
        initial_backoff_seconds=data.get(
            "initial_backoff_seconds", defaults.initial_backoff_seconds
        ),
        backoff_multiplier=data.get("backoff_multiplier", defaults.backoff_multiplier),
        max_backoff_seconds=data.get("max_backoff_seconds", defaults.max_backoff_seconds),
        jitter=data.get("jitter", defaults.jitter),
        retry_on=data.get("retry_on", list(defaults.retry_on)),
    )


def _build_providers(data: dict[str, Any]) -> ProvidersConfig:
    anthropic_data = data.get("anthropic", {})
    openai_data = data.get("openai", {})
    google_data = data.get("google", {})
    xai_data = data.get("xai", {})

    anthropic_defaults = AnthropicProviderConfig()
    openai_defaults = OpenAIProviderConfig()
    google_defaults = GoogleProviderConfig()
    xai_defaults = XAIProviderConfig()

    return ProvidersConfig(
        anthropic=AnthropicProviderConfig(
            api_key=anthropic_data.get("api_key", anthropic_defaults.api_key),
            default_model=anthropic_data.get(
                "default_model", anthropic_defaults.default_model
            ),
            max_concurrent_batches=anthropic_data.get(
                "max_concurrent_batches", anthropic_defaults.max_concurrent_batches
            ),
        ),
        openai=OpenAIProviderConfig(
            api_key=openai_data.get("api_key", openai_defaults.api_key),
            default_model=openai_data.get("default_model", openai_defaults.default_model),
            max_concurrent_batches=openai_data.get(
                "max_concurrent_batches", openai_defaults.max_concurrent_batches
            ),
        ),
        google=GoogleProviderConfig(
            api_key=google_data.get("api_key", google_defaults.api_key),
            project_id=google_data.get("project_id", google_defaults.project_id),
            default_model=google_data.get("default_model", google_defaults.default_model),
        ),
        xai=XAIProviderConfig(
            api_key=xai_data.get("api_key", xai_defaults.api_key),
            default_model=xai_data.get("default_model", xai_defaults.default_model),
        ),
    )


def _build_dashboard(data: dict[str, Any]) -> DashboardConfig:
    defaults = DashboardConfig()
    return DashboardConfig(
        enabled=data.get("enabled", defaults.enabled),
        host=data.get("host", defaults.host),
        port=data.get("port", defaults.port),
        theme=data.get("theme", defaults.theme),
        auto_refresh_seconds=data.get(
            "auto_refresh_seconds", defaults.auto_refresh_seconds
        ),
    )


def _build_relay_config(relay_data: dict[str, Any]) -> RelayConfig:
    """Construct a :class:`RelayConfig` from the ``[relay]`` TOML subtree.

    Args:
        relay_data: The value of the ``relay`` key in a merged TOML document.

    Returns:
        A fully populated :class:`RelayConfig` instance.
    """
    defaults = RelayConfig()
    return RelayConfig(
        db_path=relay_data.get("db_path", defaults.db_path),
        log_level=relay_data.get("log_level", defaults.log_level),
        log_dir=relay_data.get("log_dir", defaults.log_dir),
        output_dir=relay_data.get("output_dir", defaults.output_dir),
        cost=_build_cost(relay_data.get("cost", {})),
        cache=_build_cache(relay_data.get("cache", {})),
        retry=_build_retry(relay_data.get("retry", {})),
        providers=_build_providers(relay_data.get("providers", {})),
        dashboard=_build_dashboard(relay_data.get("dashboard", {})),
    )


# ── Source discovery ───────────────────────────────────────────────────────────

_USER_CONFIG = Path.home() / ".relay" / "config.toml"
_PYPROJECT = Path("pyproject.toml")


def _collect_sources(cli_path: str | os.PathLike | None = None) -> list[Path]:
    """Return config file paths in ascending priority order (lowest first).

    The caller should merge them left-to-right so that higher-priority sources
    override lower-priority ones.

    Sources (lowest → highest priority):
        1. ``pyproject.toml`` (``[tool.relay]`` table)
        2. ``~/.relay/config.toml``
        3. ``$RELAY_CONFIG``
        4. *cli_path*

    Args:
        cli_path: Path supplied by the caller (e.g. from ``--config``).

    Returns:
        List of :class:`~pathlib.Path` objects for files that actually exist,
        in ascending priority order.
    """
    candidates: list[Path | None] = [
        _PYPROJECT,
        _USER_CONFIG,
        Path(os.environ["RELAY_CONFIG"]) if "RELAY_CONFIG" in os.environ else None,
        Path(cli_path) if cli_path is not None else None,
    ]
    return [p for p in candidates if p is not None and p.exists()]


def _extract_relay_table(document: dict[str, Any], path: Path) -> dict[str, Any]:
    """Pull the ``[relay]`` (or ``[tool.relay]``) table from a parsed document.

    For ``pyproject.toml`` files the relay config lives under
    ``tool → relay``; for all other files it lives directly under ``relay``.

    Args:
        document: Fully parsed TOML document.
        path: Source path, used to detect ``pyproject.toml`` files.

    Returns:
        The relay configuration subtree, or an empty dict if absent.
    """
    if path.name == "pyproject.toml":
        return document.get("tool", {}).get("relay", {})
    return document.get("relay", {})


# ── Public API ─────────────────────────────────────────────────────────────────


def load_config(cli_path: str | os.PathLike | None = None) -> RelayConfig:
    """Load, merge, and return the relay configuration.

    Reads all available configuration sources, merges them in priority order
    (lowest wins; higher-priority sources override lower-priority ones), and
    returns a single :class:`RelayConfig` dataclass.

    Args:
        cli_path: Optional path to a TOML config file supplied on the command
            line (``--config``).  When provided this source takes highest
            priority.

    Returns:
        A :class:`RelayConfig` populated from all discovered sources.

    Raises:
        tomllib.TOMLDecodeError: If any discovered config file contains invalid
            TOML syntax.
        FileNotFoundError: If *cli_path* is provided but does not exist.

    Example::

        from relay.config import load_config

        cfg = load_config()
        print(cfg.db_path)            # "~/.relay/jobs.db"
        print(cfg.providers.anthropic.default_model)  # "claude-opus-4-5"
    """
    if cli_path is not None:
        cli_path = Path(cli_path)
        if not cli_path.exists():
            raise FileNotFoundError(
                f"Config file specified via --config does not exist: {cli_path}"
            )

    sources = _collect_sources(cli_path)

    merged_relay: dict[str, Any] = {}
    for source_path in sources:
        try:
            document = _read_toml(source_path)
        except Exception as exc:
            raise type(exc)(
                f"Failed to parse config file '{source_path}': {exc}"
            ) from exc
        relay_table = _extract_relay_table(document, source_path)
        merged_relay = _deep_merge(merged_relay, relay_table)

    return _build_relay_config(merged_relay)


# Convenience singleton – lazily initialised on first import access.
# Use ``get_config()`` rather than accessing this directly.
_config: RelayConfig | None = None


def get_config(cli_path: str | os.PathLike | None = None) -> RelayConfig:
    """Return the global :class:`RelayConfig`, loading it on first call.

    Subsequent calls return the cached instance, so config files are only read
    once per process.  Pass *cli_path* on the first call; it is ignored on
    later calls.

    Args:
        cli_path: Forwarded to :func:`load_config` on the first call only.

    Returns:
        The process-wide :class:`RelayConfig` singleton.
    """
    global _config
    if _config is None:
        _config = load_config(cli_path)
    return _config


def reset_config() -> None:
    """Clear the cached global config, forcing a reload on the next call.

    Primarily useful in tests that need to exercise different configuration
    scenarios within the same process.
    """
    global _config
    _config = None
