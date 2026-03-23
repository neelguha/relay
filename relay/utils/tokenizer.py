"""Token counting utilities for relay.

Provides three levels of token counting accuracy:

1. ``estimate_tokens`` — fast, dependency-free word-based heuristic.
2. ``count_tokens_tiktoken`` — accurate BPE count via the optional ``tiktoken``
   library. Falls back to the heuristic if tiktoken is not installed.
3. ``count_message_tokens`` — aggregate token count across a list of message
   dicts, accounting for per-message overhead where possible.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Average characters per token across a wide range of English text.
_CHARS_PER_TOKEN: float = 4.0

# Per-message overhead used by OpenAI-compatible chat models (in tokens).
_MESSAGE_OVERHEAD: int = 4  # role + framing tokens
_REPLY_OVERHEAD: int = 3   # priming tokens for the assistant turn


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in *text* using a character-based heuristic.

    Uses the rule-of-thumb that English text averages approximately 4 characters
    per token. This is fast and requires no external dependencies, but is only
    a rough estimate — accuracy degrades for code, non-English text, or highly
    structured content.

    Args:
        text: Input string to estimate token count for.

    Returns:
        Estimated token count as a non-negative integer. Returns ``0`` for an
        empty string.

    Example:
        >>> estimate_tokens("Hello, world!")
        3
    """
    if not text:
        return 0
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def count_tokens_tiktoken(text: str, model: str) -> int:
    """Count tokens in *text* using the ``tiktoken`` BPE tokenizer.

    If ``tiktoken`` is not installed, falls back to :func:`estimate_tokens`
    and logs a debug-level warning.

    The function first attempts to load the encoding registered for the exact
    *model* name. If the model is not recognised by tiktoken (e.g. Anthropic or
    Google models), it falls back to the ``cl100k_base`` encoding, which gives
    a reasonable approximation for most modern LLMs.

    Args:
        text: Input string to tokenize.
        model: Model identifier used to select the appropriate BPE vocabulary
            (e.g. ``"gpt-4o"``, ``"claude-opus-4-5"``).

    Returns:
        Exact BPE token count when tiktoken is available, otherwise a
        character-based estimate.

    Example:
        >>> count_tokens_tiktoken("Hello, world!", model="gpt-4o")
        4
    """
    try:
        import tiktoken  # type: ignore[import-not-found]
    except ImportError:
        logger.debug(
            "tiktoken is not installed; falling back to character-based token estimate. "
            "Install it with: pip install tiktoken"
        )
        return estimate_tokens(text)

    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.debug(
            "Model %r not recognised by tiktoken; using cl100k_base encoding.",
            model,
        )
        enc = tiktoken.get_encoding("cl100k_base")

    return len(enc.encode(text))


def count_message_tokens(
    messages: list[dict[str, Any]],
    model: str | None = None,
) -> int:
    """Count the total number of tokens consumed by a list of chat messages.

    Aggregates token counts for each message's ``"content"`` field and, when a
    *model* is provided, applies per-message framing overhead consistent with
    OpenAI's chat format specification.

    For models where tiktoken provides an exact encoding (typically OpenAI
    models), the per-token counts are precise. For other models the function
    falls back to :func:`estimate_tokens`.

    Args:
        messages: List of message dicts. Each dict should have at minimum a
            ``"content"`` key whose value is a string. An optional ``"role"``
            key is used to account for role-token overhead.
        model: Optional model identifier. When provided, it is passed to
            :func:`count_tokens_tiktoken` for higher accuracy and is used to
            decide whether to add per-message overhead tokens.

    Returns:
        Total estimated or exact token count across all messages, including
        framing overhead when *model* is specified.

    Example:
        >>> msgs = [
        ...     {"role": "user", "content": "What is the capital of France?"},
        ...     {"role": "assistant", "content": "Paris."},
        ... ]
        >>> count_message_tokens(msgs, model="gpt-4o")
        20
    """
    total = 0

    for message in messages:
        content: str = message.get("content") or ""
        role: str = message.get("role") or ""

        if model is not None:
            content_tokens = count_tokens_tiktoken(content, model)
            role_tokens = count_tokens_tiktoken(role, model) if role else 1
        else:
            content_tokens = estimate_tokens(content)
            role_tokens = 1

        total += content_tokens + role_tokens

        if model is not None:
            # Add per-message structural overhead (start/end delimiters).
            total += _MESSAGE_OVERHEAD

    if model is not None:
        # Overhead for the assistant reply priming that is prepended to every
        # API request by the model server.
        total += _REPLY_OVERHEAD

    return total
