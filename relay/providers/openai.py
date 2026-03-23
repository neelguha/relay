"""OpenAI provider adapter for relay.

Uses the OpenAI Batch API (``/v1/batches``) together with the Files API
(``/v1/files``) to submit, poll, and download results for large-scale
batch prediction jobs.

Key characteristics:
- Uploads requests as a JSONL file via the Files API.
- Splits into chunks of at most 50,000 requests (API maximum).
- Uses ``tiktoken`` for accurate token counting when available.
- Falls back to a character-based heuristic when ``tiktoken`` is not installed.
- Applies the 50% batch API discount in cost estimates.
- Maps OpenAI SDK exceptions to the relay exception hierarchy.

Dependencies:
    pip install relay[openai]   # installs openai>=1.50, tiktoken>=0.7
"""

from __future__ import annotations

import asyncio
import io
import json
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
# Pricing table (batch API prices = 50% of standard, early 2025)
# ---------------------------------------------------------------------------

_OPENAI_PRICES: dict[str, tuple[float, float]] = {
    # model               (input $/M, output $/M)  — batch-discounted
    "gpt-4o":             (1.25,  5.00),
    "gpt-4o-mini":        (0.075, 0.30),
    "gpt-4o-2024-11-20":  (1.25,  5.00),
    "gpt-4o-mini-2024-07-18": (0.075, 0.30),
    "o3":                 (5.00,  20.00),
    "o3-mini":            (0.55,  2.20),
    "o1":                 (7.50,  30.00),
    "o1-mini":            (1.50,  6.00),
    "gpt-4-turbo":        (5.00,  15.00),
    "gpt-3.5-turbo":      (0.25,  0.75),
}

_MAX_REQUESTS_PER_BATCH = 50_000
_CHARS_PER_TOKEN = 4


def _require_openai() -> Any:
    """Import and return the ``openai`` module, raising a helpful error if absent.

    Returns:
        The ``openai`` module object.

    Raises:
        ImportError: With install instructions if the SDK is not installed.
    """
    try:
        import openai  # type: ignore[import]
        return openai
    except ImportError as exc:
        raise ImportError(
            "The 'openai' package is required for the OpenAI provider. "
            "Install it with: pip install relay[openai]"
        ) from exc


def _get_tiktoken_encoder(model: str) -> Any | None:
    """Return a tiktoken encoder for the given model, or ``None`` if unavailable.

    Args:
        model: An OpenAI model identifier.

    Returns:
        A tiktoken ``Encoding`` object, or ``None`` if tiktoken is not installed
        or does not recognise the model.
    """
    try:
        import tiktoken  # type: ignore[import]
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")
    except ImportError:
        return None


def _map_openai_error(exc: Exception, provider: str = "openai") -> ProviderError:
    """Map an OpenAI SDK exception to a relay exception.

    Args:
        exc: The original exception raised by the OpenAI SDK.
        provider: Provider name string for error context.

    Returns:
        A :class:`~relay.exceptions.ProviderError` subclass instance.
    """
    openai = _require_openai()
    msg = str(exc)

    if isinstance(exc, openai.AuthenticationError):
        return AuthenticationError(msg, provider=provider, status_code=401)
    if isinstance(exc, openai.RateLimitError):
        retry_after: float | None = None
        if hasattr(exc, "response") and exc.response is not None:
            hdr = exc.response.headers.get("retry-after")
            if hdr:
                try:
                    retry_after = float(hdr)
                except ValueError:
                    pass
        return RateLimitError(msg, retry_after=retry_after, provider=provider, status_code=429)
    if isinstance(exc, openai.BadRequestError):
        return ValidationError(msg)
    if isinstance(exc, openai.InternalServerError):
        status = getattr(getattr(exc, "response", None), "status_code", 500)
        return ServerError(msg, provider=provider, status_code=status)
    if isinstance(exc, openai.APIStatusError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return ProviderError(msg, provider=provider, status_code=status)
    return ProviderError(msg, provider=provider)


def _build_openai_jsonl(requests: list[dict], model: str) -> bytes:
    """Serialise a list of relay request dicts into an OpenAI Batch API JSONL payload.

    Each line conforms to the OpenAI batch input format::

        {"custom_id": "...", "method": "POST", "url": "/v1/chat/completions",
         "body": {...}}

    Args:
        requests: Serialised ``BatchRequest`` dicts.
        model: Default model to use when the request does not override it.

    Returns:
        UTF-8 encoded JSONL bytes suitable for upload via the Files API.
    """
    lines: list[str] = []
    for req in requests:
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in req.get("messages", [])
            if m.get("role") != "system"
        ]
        # Prepend system message in OpenAI format.
        system = req.get("system")
        if system:
            messages.insert(0, {"role": "system", "content": system})

        body: dict[str, Any] = {
            "model": req.get("model") or model,
            "messages": messages,
            "max_tokens": req.get("max_tokens", 1024),
        }

        temperature = req.get("temperature")
        if temperature is not None:
            body["temperature"] = temperature

        top_p = req.get("top_p")
        if top_p is not None:
            body["top_p"] = top_p

        stop = req.get("stop_sequences") or []
        if stop:
            body["stop"] = stop

        entry = {
            "custom_id": req["id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        lines.append(json.dumps(entry, ensure_ascii=False))

    return "\n".join(lines).encode("utf-8")


class OpenAIProvider(BaseProvider):
    """Provider adapter for the OpenAI Batch API.

    Uploads request payloads as JSONL files via the Files API, then creates
    a batch job.  When more than 50,000 requests are submitted, they are
    automatically split into multiple batches; all chunk job IDs are returned
    as a comma-separated ``provider_job_id``.

    Args:
        config: Configuration dict.  Recognised keys:

            - ``api_key`` (str): OpenAI API key.  Falls back to the
              ``OPENAI_API_KEY`` environment variable.
            - ``model`` (str): Default model identifier.
            - ``organization`` (str | None): OpenAI organisation ID.
            - ``base_url`` (str | None): Override the OpenAI API base URL.
            - ``completion_window`` (str): Batch completion window
              (default: ``"24h"``).
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        openai = _require_openai()

        api_key = config.get("api_key")
        organization = config.get("organization")
        base_url = config.get("base_url")

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if organization:
            kwargs["organization"] = organization
        if base_url:
            kwargs["base_url"] = base_url

        self._client = openai.AsyncOpenAI(**kwargs)
        self._default_model: str = config.get("model", "gpt-4o")
        self._completion_window: str = config.get("completion_window", "24h")
        # Cache tiktoken encoder per model to avoid re-loading.
        self._encoder: Any | None = None

    def _get_encoder(self, model: str | None = None) -> Any | None:
        """Return (and cache) a tiktoken encoder for the current model.

        Args:
            model: Model identifier to look up.  Defaults to the configured
                default model.

        Returns:
            A tiktoken ``Encoding`` object or ``None``.
        """
        if self._encoder is None:
            self._encoder = _get_tiktoken_encoder(model or self._default_model)
        return self._encoder

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    async def submit_batch(self, requests: list[dict]) -> str:
        """Upload requests as JSONL and create an OpenAI batch job.

        Chunks requests into groups of at most 50,000, uploads each chunk
        to the Files API, then creates one batch job per chunk.

        Args:
            requests: Serialised ``BatchRequest`` dicts.

        Returns:
            A comma-separated string of OpenAI batch job IDs (one per chunk).

        Raises:
            relay.exceptions.AuthenticationError: On authentication failure.
            relay.exceptions.RateLimitError: On rate limit responses.
            relay.exceptions.ServerError: On 5xx responses.
            relay.exceptions.ValidationError: On malformed payloads.
            relay.exceptions.ProviderError: On any other API error.
        """
        chunks = [
            requests[i : i + _MAX_REQUESTS_PER_BATCH]
            for i in range(0, len(requests), _MAX_REQUESTS_PER_BATCH)
        ]
        job_ids: list[str] = []

        for chunk in chunks:
            job_id = await self._submit_chunk(chunk)
            job_ids.append(job_id)

        return ",".join(job_ids)

    async def _submit_chunk(self, chunk: list[dict]) -> str:
        """Upload a single chunk and create one OpenAI batch job.

        Args:
            chunk: A subset of serialised request dicts (at most 50,000).

        Returns:
            The OpenAI batch job ID.

        Raises:
            relay.exceptions.ProviderError: On any API error.
        """
        jsonl_bytes = _build_openai_jsonl(chunk, self._default_model)

        try:
            file_obj = await self._client.files.create(
                file=("batch_input.jsonl", io.BytesIO(jsonl_bytes), "application/jsonl"),
                purpose="batch",
            )
            batch = await self._client.batches.create(
                input_file_id=file_obj.id,
                endpoint="/v1/chat/completions",
                completion_window=self._completion_window,
            )
            return batch.id
        except Exception as exc:
            raise _map_openai_error(exc) from exc

    # ------------------------------------------------------------------
    # Status polling
    # ------------------------------------------------------------------

    async def get_status(self, provider_job_id: str) -> ProviderStatus:
        """Poll the OpenAI API for the current batch status.

        Handles comma-separated multi-chunk IDs by polling each chunk and
        aggregating progress counters.

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
        terminal_count = 0
        error_msg: str | None = None

        for bid in batch_ids:
            try:
                batch = await self._client.batches.retrieve(bid)
            except Exception as exc:
                raise _map_openai_error(exc) from exc

            counts = batch.request_counts
            total_completed += getattr(counts, "completed", 0) or 0
            total_failed += getattr(counts, "failed", 0) or 0
            total_requests += getattr(counts, "total", 0) or 0

            # OpenAI terminal statuses: "completed", "failed", "expired", "cancelled"
            if batch.status in {"completed", "failed", "expired", "cancelled"}:
                terminal_count += 1
            if batch.status == "expired":
                error_msg = "Batch expired before completion."
            if batch.errors and batch.errors.data:
                msgs = [e.message for e in batch.errors.data if e.message]
                if msgs:
                    error_msg = "; ".join(msgs)

        if terminal_count == len(batch_ids):
            aggregate_status = "ended"
        else:
            aggregate_status = "in_progress"

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
        """Download and parse results for a completed OpenAI batch job.

        Retrieves the output file from the Files API and parses each JSONL line.

        Args:
            provider_job_id: The ID returned by :meth:`submit_batch`.

        Returns:
            A list of normalised result dicts.

        Raises:
            relay.exceptions.ProviderError: On any API error.
        """
        batch_ids = [bid.strip() for bid in provider_job_id.split(",")]
        all_results: list[dict] = []

        for bid in batch_ids:
            try:
                batch = await self._client.batches.retrieve(bid)
                output_file_id = batch.output_file_id
                error_file_id = batch.error_file_id

                if output_file_id:
                    content = await self._client.files.content(output_file_id)
                    raw_bytes = content.read()
                    for line in raw_bytes.decode("utf-8").splitlines():
                        line = line.strip()
                        if line:
                            parsed = json.loads(line)
                            all_results.append(_parse_openai_result(parsed))

                if error_file_id:
                    content = await self._client.files.content(error_file_id)
                    raw_bytes = content.read()
                    for line in raw_bytes.decode("utf-8").splitlines():
                        line = line.strip()
                        if line:
                            parsed = json.loads(line)
                            all_results.append(_parse_openai_result(parsed, is_error=True))

            except Exception as exc:
                raise _map_openai_error(exc) from exc

        return all_results

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    async def cancel(self, provider_job_id: str) -> bool:
        """Request cancellation of an OpenAI batch job.

        Args:
            provider_job_id: The ID returned by :meth:`submit_batch`.

        Returns:
            ``True`` if at least one chunk was successfully cancelled.

        Raises:
            relay.exceptions.ProviderError: On any API error.
        """
        batch_ids = [bid.strip() for bid in provider_job_id.split(",")]
        any_cancelled = False

        for bid in batch_ids:
            try:
                batch = await self._client.batches.cancel(bid)
                if batch.status == "cancelling":
                    any_cancelled = True
            except Exception as exc:
                mapped = _map_openai_error(exc)
                status = getattr(mapped, "status_code", None)
                if status in {400, 422}:
                    # Already completed/cancelled — not an error.
                    continue
                raise mapped from exc

        return any_cancelled

    # ------------------------------------------------------------------
    # Token / cost estimation
    # ------------------------------------------------------------------

    def estimate_tokens(self, request: BatchRequest) -> tuple[int, int]:
        """Estimate input and output token counts for a single ``BatchRequest``.

        Uses ``tiktoken`` when available for accurate counts; falls back to a
        character-based heuristic (4 chars ≈ 1 token) otherwise.

        Args:
            request: The :class:`~relay.models.BatchRequest` to estimate.

        Returns:
            ``(estimated_input_tokens, estimated_output_tokens)`` tuple.
        """
        model = request.model or self._default_model
        encoder = self._get_encoder(model)

        text_parts: list[str] = []
        if request.system:
            text_parts.append(request.system)
        for msg in request.messages:
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            text_parts.append(content)

        combined = " ".join(text_parts)

        if encoder is not None:
            try:
                input_tokens = len(encoder.encode(combined))
            except Exception:
                input_tokens = max(1, len(combined) // _CHARS_PER_TOKEN)
        else:
            input_tokens = max(1, len(combined) // _CHARS_PER_TOKEN)

        return input_tokens, request.max_tokens

    def get_price_per_million(self, model: str) -> tuple[float, float]:
        """Return the batch-discounted price per million tokens for the given model.

        Args:
            model: OpenAI model identifier.

        Returns:
            ``(input_price_usd, output_price_usd)`` per million tokens.

        Raises:
            relay.exceptions.ValidationError: If the model is not in the
                pricing table.
        """
        price = _OPENAI_PRICES.get(model)
        if price is None:
            # Fine-tuned model prefix match.
            for key, val in _OPENAI_PRICES.items():
                if model.startswith(key) or model.startswith(f"ft:{key}"):
                    return val
            raise ValidationError(
                f"Unknown OpenAI model {model!r}. "
                f"Known models: {', '.join(_OPENAI_PRICES)}"
            )
        return price


# ---------------------------------------------------------------------------
# Result parsing helper
# ---------------------------------------------------------------------------

def _parse_openai_result(data: dict, is_error: bool = False) -> dict:
    """Convert an OpenAI batch output line to a relay result dict.

    Args:
        data: A single parsed JSONL object from the batch output file.
        is_error: If ``True``, treat this line as an error record.

    Returns:
        A normalised result dict.
    """
    custom_id: str = data.get("custom_id", "")
    response = data.get("response") or {}
    body = response.get("body") or {}
    error = data.get("error")

    if error or is_error:
        err_msg = (
            (error.get("message") if error else None)
            or str(data)
        )
        return {
            "request_id": custom_id,
            "content": "",
            "stop_reason": "error",
            "input_tokens": 0,
            "output_tokens": 0,
            "model": "",
            "error": {
                "code": (error.get("code") if error else "batch_error") or "batch_error",
                "message": err_msg,
                "retryable": False,
            },
            "raw_response": data,
        }

    choices = body.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    content = message.get("content") or ""
    stop_reason = choice.get("finish_reason") or "stop"

    usage = body.get("usage") or {}
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    model = body.get("model", "")

    return {
        "request_id": custom_id,
        "content": content,
        "stop_reason": stop_reason,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model": model,
        "error": None,
        "raw_response": data,
    }
