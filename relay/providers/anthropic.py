"""Anthropic provider adapter for relay.

Uses the Anthropic Message Batches API (``/v1/messages/batches``) to submit,
poll, and download results for large-scale batch prediction jobs.

Key characteristics:
- Chunks requests into batches of up to 10,000 (API maximum).
- When a job spans multiple chunks, returns a comma-joined string of all
  chunk batch IDs as the ``provider_job_id``.
- Streams result JSONL files to minimise peak memory usage.
- Applies the 50% batch API discount in cost estimates.
- Maps SDK exceptions to the relay exception hierarchy.

Dependencies:
    pip install relay[anthropic]   # installs anthropic>=0.43
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from relay.exceptions import (
    AuthenticationError,
    BatchExpiredError,
    ProviderError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from relay.models import BatchRequest, ProviderStatus
from relay.providers.base import BaseProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing table (batch API prices = 50% of standard)
# Source: Anthropic pricing page, early 2025.
# ---------------------------------------------------------------------------

_ANTHROPIC_PRICES: dict[str, tuple[float, float]] = {
    # model                  (input $/M, output $/M)  — batch-discounted
    "claude-opus-4-5":       (7.50,  37.50),
    "claude-sonnet-4-6":     (1.50,   7.50),
    "claude-haiku-4-5":      (0.40,   2.00),
    # Aliases
    "claude-3-5-sonnet-20241022": (1.50, 7.50),
    "claude-3-5-haiku-20241022":  (0.40, 2.00),
    "claude-3-opus-20240229":     (7.50, 37.50),
}

_MAX_REQUESTS_PER_BATCH = 10_000

# Approximate chars-per-token ratio used when the SDK counter is unavailable.
_CHARS_PER_TOKEN = 4


def _require_anthropic() -> Any:
    """Import and return the ``anthropic`` module, raising a helpful error if absent.

    Returns:
        The ``anthropic`` module object.

    Raises:
        ImportError: With install instructions if the SDK is not installed.
    """
    try:
        import anthropic  # type: ignore[import]
        return anthropic
    except ImportError as exc:
        raise ImportError(
            "The 'anthropic' package is required for the Anthropic provider. "
            "Install it with: pip install relay[anthropic]"
        ) from exc


def _map_anthropic_error(exc: Exception, provider: str = "anthropic") -> ProviderError:
    """Map an Anthropic SDK exception to a relay exception.

    Args:
        exc: The original exception raised by the Anthropic SDK.
        provider: Provider name string for error context.

    Returns:
        A :class:`~relay.exceptions.ProviderError` subclass instance.
    """
    anthropic = _require_anthropic()
    msg = str(exc)

    if isinstance(exc, anthropic.AuthenticationError):
        return AuthenticationError(msg, provider=provider, status_code=401)
    if isinstance(exc, anthropic.RateLimitError):
        retry_after: float | None = None
        if hasattr(exc, "response") and exc.response is not None:
            retry_after_hdr = exc.response.headers.get("retry-after")
            if retry_after_hdr:
                try:
                    retry_after = float(retry_after_hdr)
                except ValueError:
                    pass
        return RateLimitError(msg, retry_after=retry_after, provider=provider, status_code=429)
    if isinstance(exc, anthropic.BadRequestError):
        return ValidationError(msg)
    if isinstance(exc, anthropic.InternalServerError):
        status = getattr(getattr(exc, "response", None), "status_code", 500)
        return ServerError(msg, provider=provider, status_code=status)
    if isinstance(exc, anthropic.APIStatusError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return ProviderError(msg, provider=provider, status_code=status)
    return ProviderError(msg, provider=provider)


def _build_anthropic_request(request: dict, model: str) -> dict:
    """Convert a serialised ``BatchRequest`` dict to an Anthropic batch request object.

    The relay message format (OpenAI-style ``[{"role": ..., "content": ...}]``) is
    translated to the Anthropic ``messages`` format.  System prompts are handled
    via the top-level ``system`` parameter.

    Args:
        request: A ``BatchRequest`` serialised as a dict (via ``dataclasses.asdict``).
        model: The model identifier to use for this request.

    Returns:
        A dict matching the Anthropic ``Request`` schema accepted by the
        Message Batches API.
    """
    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in request.get("messages", [])
        if m.get("role") != "system"
    ]

    params: dict[str, Any] = {
        "model": request.get("model") or model,
        "max_tokens": request.get("max_tokens", 1024),
        "messages": messages,
    }

    system = request.get("system")
    if system:
        params["system"] = system

    temperature = request.get("temperature")
    if temperature is not None:
        params["temperature"] = temperature

    top_p = request.get("top_p")
    if top_p is not None:
        params["top_p"] = top_p

    stop_sequences = request.get("stop_sequences") or []
    if stop_sequences:
        params["stop_sequences"] = stop_sequences

    return {
        "custom_id": request["id"],
        "params": params,
    }


class AnthropicProvider(BaseProvider):
    """Provider adapter for the Anthropic Message Batches API.

    Handles chunking, submission, polling, and result download for Anthropic
    batch jobs.  When more than 10,000 requests are submitted, they are split
    into multiple batches automatically; the returned ``provider_job_id`` is a
    comma-separated string of all chunk batch IDs.

    Args:
        config: Configuration dict.  Recognised keys:

            - ``api_key`` (str): Anthropic API key.  Falls back to the
              ``ANTHROPIC_API_KEY`` environment variable.
            - ``model`` (str): Default model identifier.
            - ``max_concurrent_batches`` (int): Maximum concurrent batch
              submissions (default: 5).
            - ``base_url`` (str | None): Override the Anthropic API base URL
              (useful for proxies/testing).
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        anthropic = _require_anthropic()

        api_key = config.get("api_key")
        base_url = config.get("base_url")
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        self._client = anthropic.AsyncAnthropic(**kwargs)
        self._default_model: str = config.get("model", "claude-opus-4-5")
        self._max_concurrent: int = int(config.get("max_concurrent_batches", 5))

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    async def submit_batch(self, requests: list[dict]) -> str:
        """Submit requests to the Anthropic Message Batches API.

        Automatically chunks the list into groups of at most 10,000 entries.
        When multiple chunks are created, each is submitted concurrently (up to
        ``max_concurrent_batches`` at a time) and the returned ID is a
        comma-separated string of all chunk batch IDs.

        Args:
            requests: Serialised ``BatchRequest`` dicts.

        Returns:
            A provider batch ID string.  When chunked, this is a
            comma-separated list of IDs (one per chunk).

        Raises:
            relay.exceptions.AuthenticationError: On API authentication failure.
            relay.exceptions.RateLimitError: On rate limit responses.
            relay.exceptions.ServerError: On 5xx responses.
            relay.exceptions.ValidationError: On malformed request payloads.
            relay.exceptions.ProviderError: On any other API error.
        """
        chunks = [
            requests[i : i + _MAX_REQUESTS_PER_BATCH]
            for i in range(0, len(requests), _MAX_REQUESTS_PER_BATCH)
        ]

        semaphore = asyncio.Semaphore(self._max_concurrent)

        async def _submit_chunk(chunk: list[dict]) -> str:
            async with semaphore:
                anthropic_requests = [
                    _build_anthropic_request(r, self._default_model) for r in chunk
                ]
                try:
                    batch = await self._client.messages.batches.create(
                        requests=anthropic_requests
                    )
                    return batch.id
                except Exception as exc:
                    raise _map_anthropic_error(exc) from exc

        batch_ids = await asyncio.gather(*[_submit_chunk(c) for c in chunks])
        return ",".join(batch_ids)

    # ------------------------------------------------------------------
    # Status polling
    # ------------------------------------------------------------------

    async def get_status(self, provider_job_id: str) -> ProviderStatus:
        """Poll the Anthropic API for the current batch status.

        Handles comma-separated multi-chunk job IDs by polling each chunk
        individually and aggregating the counters.

        Args:
            provider_job_id: The ID returned by :meth:`submit_batch`.

        Returns:
            An aggregated :class:`~relay.models.ProviderStatus`.

        Raises:
            relay.exceptions.ProviderError: On any API error.
        """
        batch_ids = [bid.strip() for bid in provider_job_id.split(",")]
        total_completed = 0
        total_failed = 0
        total_requests = 0
        aggregate_status = "in_progress"
        error_msg: str | None = None

        for bid in batch_ids:
            try:
                batch = await self._client.messages.batches.retrieve(bid)
            except Exception as exc:
                raise _map_anthropic_error(exc) from exc

            counts = batch.request_counts
            chunk_completed = getattr(counts, "succeeded", 0)
            chunk_errored = getattr(counts, "errored", 0)
            chunk_expired = getattr(counts, "expired", 0)
            chunk_processing = getattr(counts, "processing", 0)
            chunk_canceled = getattr(counts, "canceled", 0)

            chunk_total = (
                chunk_completed + chunk_errored + chunk_expired
                + chunk_processing + chunk_canceled
            )
            total_completed += chunk_completed
            total_failed += chunk_errored + chunk_expired
            total_requests += chunk_total

            native = batch.processing_status  # "in_progress" | "ended"
            if native == "ended":
                if batch.cancel_initiated_at is not None:
                    aggregate_status = "canceling"
                elif chunk_expired > 0 and chunk_completed == 0:
                    aggregate_status = "expired"
                    error_msg = "Batch expired before completion."
                else:
                    aggregate_status = "ended"
            # If any chunk is still in_progress, overall stays in_progress.

        return ProviderStatus(
            provider_job_id=provider_job_id,
            status=aggregate_status,
            completed=total_completed,
            failed=total_failed,
            total=total_requests,
            error=error_msg,
        )

    # ------------------------------------------------------------------
    # Result retrieval
    # ------------------------------------------------------------------

    async def download_results(self, provider_job_id: str) -> list[dict]:
        """Download and parse results for a completed Anthropic batch job.

        Iterates the result stream returned by the SDK.  Each result is
        normalised into the relay result dict format.

        Args:
            provider_job_id: The ID returned by :meth:`submit_batch`.

        Returns:
            A list of result dicts with keys: ``request_id``, ``content``,
            ``stop_reason``, ``input_tokens``, ``output_tokens``, ``model``,
            and optionally ``error``.

        Raises:
            relay.exceptions.ProviderError: On any API error.
        """
        batch_ids = [bid.strip() for bid in provider_job_id.split(",")]
        all_results: list[dict] = []

        for bid in batch_ids:
            try:
                async for result in await self._client.messages.batches.results(bid):
                    all_results.append(_parse_anthropic_result(result))
            except Exception as exc:
                raise _map_anthropic_error(exc) from exc

        return all_results

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    async def cancel(self, provider_job_id: str) -> bool:
        """Request cancellation of an Anthropic batch job.

        When the job spans multiple chunks, attempts to cancel all chunks
        and returns ``True`` if at least one cancellation was accepted.

        Args:
            provider_job_id: The ID returned by :meth:`submit_batch`.

        Returns:
            ``True`` if cancellation was accepted; ``False`` if the job
            was already in a terminal state.

        Raises:
            relay.exceptions.ProviderError: On any API error.
        """
        batch_ids = [bid.strip() for bid in provider_job_id.split(",")]
        any_cancelled = False

        for bid in batch_ids:
            try:
                batch = await self._client.messages.batches.cancel(bid)
                if batch.cancel_initiated_at is not None:
                    any_cancelled = True
            except Exception as exc:
                mapped = _map_anthropic_error(exc)
                # If already ended, treat as non-cancellable rather than error.
                if isinstance(mapped, ProviderError) and getattr(mapped, "status_code", None) == 422:
                    continue
                raise mapped from exc

        return any_cancelled

    # ------------------------------------------------------------------
    # Token / cost estimation
    # ------------------------------------------------------------------

    def estimate_tokens(self, request: BatchRequest) -> tuple[int, int]:
        """Estimate input and output token counts for a single ``BatchRequest``.

        Uses a character-based heuristic (4 chars ≈ 1 token) since the
        Anthropic SDK's ``count_tokens`` endpoint is async and would be
        too slow to call synchronously for large batches.

        Args:
            request: The :class:`~relay.models.BatchRequest` to estimate.

        Returns:
            ``(estimated_input_tokens, estimated_output_tokens)`` tuple.
            ``estimated_output_tokens`` is always ``request.max_tokens``.
        """
        text = ""
        if request.system:
            text += request.system
        for msg in request.messages:
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            text += content

        input_tokens = max(1, len(text) // _CHARS_PER_TOKEN)
        return input_tokens, request.max_tokens

    def get_price_per_million(self, model: str) -> tuple[float, float]:
        """Return the batch-discounted price per million tokens for the given model.

        Args:
            model: Anthropic model identifier.

        Returns:
            ``(input_price_usd, output_price_usd)`` per million tokens,
            reflecting the 50% Message Batches API discount.

        Raises:
            relay.exceptions.ValidationError: If the model is not in the
                pricing table.
        """
        price = _ANTHROPIC_PRICES.get(model)
        if price is None:
            # Fall back to a prefix match for versioned aliases.
            for key, val in _ANTHROPIC_PRICES.items():
                if model.startswith(key) or key.startswith(model):
                    return val
            raise ValidationError(
                f"Unknown Anthropic model {model!r}. "
                f"Known models: {', '.join(_ANTHROPIC_PRICES)}"
            )
        return price


# ---------------------------------------------------------------------------
# Result parsing helper
# ---------------------------------------------------------------------------

def _parse_anthropic_result(result: Any) -> dict:
    """Convert an Anthropic batch result object to a relay result dict.

    Args:
        result: A single result object from the Anthropic batch results stream.

    Returns:
        A normalised result dict compatible with relay's internal format.
    """
    custom_id: str = result.custom_id
    outcome = result.result

    if outcome.type == "succeeded":
        message = outcome.message
        content_blocks = message.content
        text_content = " ".join(
            block.text for block in content_blocks
            if hasattr(block, "text")
        )
        usage = message.usage
        return {
            "request_id": custom_id,
            "content": text_content,
            "stop_reason": message.stop_reason or "end_turn",
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
            "model": message.model,
            "error": None,
            "raw_response": {},
        }

    if outcome.type == "errored":
        err = outcome.error
        return {
            "request_id": custom_id,
            "content": "",
            "stop_reason": "error",
            "input_tokens": 0,
            "output_tokens": 0,
            "model": "",
            "error": {
                "code": getattr(err, "type", "api_error"),
                "message": str(err),
                "retryable": False,
            },
            "raw_response": {},
        }

    # "expired" or unknown
    return {
        "request_id": custom_id,
        "content": "",
        "stop_reason": "expired",
        "input_tokens": 0,
        "output_tokens": 0,
        "model": "",
        "error": {
            "code": outcome.type,
            "message": f"Request ended with status: {outcome.type}",
            "retryable": True,
        },
        "raw_response": {},
    }
