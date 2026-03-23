"""Token pricing table and lookup utilities for relay.

Prices are loaded from prices.json at import time. The JSON file ships a
static snapshot; run ``relay db update-prices`` to refresh from live provider
documentation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PRICES_PATH = Path(__file__).parent / "prices.json"


def _load_prices() -> dict[str, Any]:
    """Load the pricing table from the bundled JSON file.

    Returns:
        The full parsed content of prices.json, with a ``"models"`` key
        mapping model name to pricing metadata.

    Raises:
        FileNotFoundError: If prices.json cannot be found next to this module.
        json.JSONDecodeError: If the file is malformed.
    """
    with _PRICES_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


# Module-level cache: loaded once, shared for the lifetime of the process.
_PRICE_TABLE: dict[str, dict[str, Any]] = _load_prices()["models"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_price_per_million(model: str) -> tuple[float, float]:
    """Return the standard (non-batch) input and output price per million tokens.

    Args:
        model: The model identifier exactly as used in relay
            (e.g. ``"claude-opus-4-5"``, ``"gpt-4o"``).

    Returns:
        A ``(input_price_usd, output_price_usd)`` tuple where both values are
        expressed in US dollars per **one million** tokens.

    Raises:
        KeyError: If *model* is not present in the pricing table.

    Example:
        >>> inp, out = get_price_per_million("gpt-4o")
        >>> print(inp, out)
        2.5 10.0
    """
    entry = _PRICE_TABLE[model]
    return entry["input_per_million"], entry["output_per_million"]


def get_batch_price_per_million(model: str) -> tuple[float, float]:
    """Return the batch-discounted input and output price per million tokens.

    The batch discount is applied as a percentage reduction to the standard
    price.  For providers that do not offer a batch discount the standard price
    is returned unchanged.

    Args:
        model: The model identifier (e.g. ``"claude-sonnet-4-6"``).

    Returns:
        A ``(input_price_usd, output_price_usd)`` tuple of batch-adjusted
        prices in US dollars per one million tokens.

    Raises:
        KeyError: If *model* is not present in the pricing table.
    """
    entry = _PRICE_TABLE[model]
    discount = entry.get("batch_discount_pct", 0.0) / 100.0
    multiplier = 1.0 - discount
    return (
        entry["input_per_million"] * multiplier,
        entry["output_per_million"] * multiplier,
    )


def get_batch_discount_pct(model: str) -> float:
    """Return the batch discount percentage for a model (0–100).

    Args:
        model: The model identifier.

    Returns:
        The batch discount as a percentage (e.g. ``50.0`` for 50 % off).

    Raises:
        KeyError: If *model* is not present in the pricing table.
    """
    return _PRICE_TABLE[model].get("batch_discount_pct", 0.0)


def list_models() -> list[str]:
    """Return the list of all model identifiers in the pricing table.

    Returns:
        A sorted list of model name strings.
    """
    return sorted(_PRICE_TABLE.keys())


def get_model_info(model: str) -> dict[str, Any]:
    """Return the full pricing metadata entry for a model.

    Args:
        model: The model identifier.

    Returns:
        A dictionary with keys ``provider``, ``input_per_million``,
        ``output_per_million``, and ``batch_discount_pct``.

    Raises:
        KeyError: If *model* is not present in the pricing table.
    """
    return dict(_PRICE_TABLE[model])


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    use_batch_pricing: bool = True,
) -> float:
    """Estimate the total cost in USD for a given token count.

    Args:
        model: The model identifier.
        input_tokens: Number of input (prompt) tokens.
        output_tokens: Number of output (completion) tokens.
        use_batch_pricing: When ``True`` (default) apply the batch discount if
            one is available for the model.

    Returns:
        Estimated cost in US dollars as a non-negative float.

    Raises:
        KeyError: If *model* is not present in the pricing table.
    """
    if use_batch_pricing:
        inp_price, out_price = get_batch_price_per_million(model)
    else:
        inp_price, out_price = get_price_per_million(model)

    return (input_tokens * inp_price + output_tokens * out_price) / 1_000_000
