"""Provider registry for relay.

Provides a central registry mapping provider name strings to their adapter
classes, and the :func:`get_provider` factory function used by the
``BatchClient`` to instantiate adapters at runtime.

Supported built-in providers: ``anthropic``, ``openai``, ``google``, ``xai``.

Custom providers can be added at runtime via Python entry points under the
group ``relay.providers`` in a third-party package's ``pyproject.toml``,
or by calling :func:`register_provider` directly.

Example::

    from relay.providers import get_provider

    provider = get_provider("anthropic", {"api_key": "sk-ant-..."})
    job_id = await provider.submit_batch(requests)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from relay.exceptions import ValidationError

if TYPE_CHECKING:
    from relay.providers.base import BaseProvider

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Maps provider name -> adapter class (lazy import to avoid hard dependencies
# on provider SDKs at import time).
PROVIDERS: dict[str, str] = {
    "anthropic": "relay.providers.anthropic.AnthropicProvider",
    "openai": "relay.providers.openai.OpenAIProvider",
    "google": "relay.providers.google.GoogleProvider",
    "xai": "relay.providers.xai.XAIProvider",
}

# Cache of already-imported classes to avoid repeated dynamic imports.
_resolved: dict[str, type[BaseProvider]] = {}


def register_provider(name: str, cls: type[BaseProvider]) -> None:
    """Register a custom provider adapter class under the given name.

    This can be used by third-party packages to add providers without
    modifying the relay source code.  Names are case-insensitive and
    are normalised to lowercase before storage.

    Args:
        name: The provider name string (e.g. ``"myprovider"``).
        cls: A concrete subclass of :class:`~relay.providers.base.BaseProvider`.

    Raises:
        TypeError: If ``cls`` is not a subclass of ``BaseProvider``.
    """
    from relay.providers.base import BaseProvider as _Base  # local import

    if not (isinstance(cls, type) and issubclass(cls, _Base)):
        raise TypeError(
            f"cls must be a subclass of BaseProvider, got {cls!r}"
        )
    _resolved[name.lower()] = cls


def _import_provider(dotted_path: str) -> type[BaseProvider]:
    """Import and return a provider class from a dotted module path.

    Args:
        dotted_path: A fully-qualified class path such as
            ``"relay.providers.anthropic.AnthropicProvider"``.

    Returns:
        The imported provider class.

    Raises:
        ImportError: If the module or class cannot be found.
    """
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_provider(name: str, config: dict) -> BaseProvider:
    """Instantiate and return a provider adapter by name.

    Args:
        name: The provider name, case-insensitive.  Must be one of the
            registered providers (``"anthropic"``, ``"openai"``,
            ``"google"``, ``"xai"``), or a name registered via
            :func:`register_provider` or Python entry points.
        config: Provider configuration dict.  Passed directly to the
            adapter's ``__init__``.  At minimum should contain
            ``api_key``.

    Returns:
        An instantiated :class:`~relay.providers.base.BaseProvider` subclass.

    Raises:
        relay.exceptions.ValidationError: If ``name`` is not a registered
            provider.
        ImportError: If the provider's optional SDK dependency is not
            installed.

    Example::

        provider = get_provider("openai", {"api_key": "sk-..."})
        job_id = await provider.submit_batch(serialised_requests)
    """
    key = name.lower()

    if key in _resolved:
        return _resolved[key](config)

    if key not in PROVIDERS:
        available = ", ".join(sorted(PROVIDERS))
        raise ValidationError(
            f"Unknown provider {name!r}. Available providers: {available}. "
            "To add a custom provider, call relay.providers.register_provider()."
        )

    cls = _import_provider(PROVIDERS[key])
    _resolved[key] = cls
    return cls(config)


# ---------------------------------------------------------------------------
# Entry-point discovery (optional, runs at import time)
# ---------------------------------------------------------------------------

def _load_entry_points() -> None:
    """Load third-party providers registered via Python entry points.

    Entry points must be declared under the group ``relay.providers`` in
    a package's ``pyproject.toml``::

        [project.entry-points."relay.providers"]
        myprovider = "mypackage.providers:MyProvider"

    Errors during entry point loading are silently ignored so that a
    broken third-party plugin does not prevent the rest of relay from
    functioning.
    """
    try:
        from importlib.metadata import entry_points
        eps = entry_points(group="relay.providers")
        for ep in eps:
            try:
                cls = ep.load()
                _resolved[ep.name.lower()] = cls
                PROVIDERS[ep.name.lower()] = f"{cls.__module__}.{cls.__qualname__}"
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass


_load_entry_points()

__all__ = [
    "PROVIDERS",
    "get_provider",
    "register_provider",
]
