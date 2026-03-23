"""Deterministic cache key computation for relay requests.

All parameters that affect the LLM response are included in the hash.
Metadata and tags are excluded so that decorative fields do not invalidate
cached results.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _normalize_value(value: Any) -> Any:
    """Recursively normalize a value for canonical JSON serialization.

    Floats are rounded to 6 decimal places to prevent floating-point
    representation differences from producing different cache keys.
    Dict keys are sorted at every level of nesting.

    Args:
        value: The value to normalize.

    Returns:
        A normalized, JSON-serializable representation of the value.
    """
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {k: _normalize_value(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    return value


def compute_cache_key(
    provider: str,
    model: str,
    system_prompt: str | None,
    user_messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float | None,
    stop_sequences: list[str] | None,
) -> str:
    """Compute a deterministic SHA-256 cache key for a relay request.

    The key is derived from all parameters that influence the LLM output.
    Metadata and tags are intentionally excluded so that adding or changing
    decorative fields does not invalidate a cached response.

    Float values are normalized to 6 decimal places and all dict keys are
    sorted recursively before serialization to guarantee identical output
    regardless of the order in which arguments were constructed.

    Args:
        provider: Provider identifier string (e.g. ``"anthropic"``,
            ``"openai"``).
        model: Model identifier string (e.g. ``"claude-opus-4-5"``).
        system_prompt: Optional system-level instruction text. ``None``
            and an empty string are treated as distinct values.
        user_messages: List of message dicts, each containing at minimum
            ``"role"`` and ``"content"`` keys.
        temperature: Sampling temperature. Normalized to 6 decimal places.
        max_tokens: Maximum number of tokens to generate.
        top_p: Nucleus sampling probability cutoff, or ``None`` if unused.
        stop_sequences: List of stop strings, or ``None`` if unused.

    Returns:
        A 64-character lowercase hexadecimal SHA-256 digest string.

    Example:
        >>> key = compute_cache_key(
        ...     provider="anthropic",
        ...     model="claude-opus-4-5",
        ...     system_prompt="You are a helpful assistant.",
        ...     user_messages=[{"role": "user", "content": "Hello"}],
        ...     temperature=1.0,
        ...     max_tokens=1024,
        ...     top_p=None,
        ...     stop_sequences=[],
        ... )
        >>> len(key)
        64
    """
    payload: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "system_prompt": system_prompt,
        "user_messages": user_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stop_sequences": stop_sequences if stop_sequences is not None else [],
    }

    normalized = _normalize_value(payload)
    canonical_json = json.dumps(normalized, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return digest
