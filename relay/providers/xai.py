"""XAI (Grok) provider adapter for relay.

Uses plain ``httpx`` with the XAI REST API, which exposes an OpenAI-compatible
``/v1/chat/completions`` endpoint.  Because XAI does not yet offer a native
batch API, this adapter submits all requests concurrently using an
``asyncio.Semaphore``-bounded pool of up to 50 simultaneous HTTP calls.

Key characteristics:
- OpenAI-compatible request format (same JSONL body as OpenAI's chat endpoint).
- Concurrent pool with configurable size (default: 50).
- Respects ``Retry-After`` headers on HTTP 429 responses.
- Exponential back-off with jitter on transient failures (5xx).
- Maps HTTP errors to the relay exception hierarchy.
- No external SDK dependency beyond ``httpx`` (already a core relay dep).

Dependencies:
    pip install relay[xai]   # installs httpx>=0.27 (already in core deps)
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
import uuid
from typing import Any

import httpx

from relay.exceptions import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from relay.models import BatchRequest, ProviderStatus
from relay.providers.base import BaseProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing table (XAI Grok, early 2025; batch discount TBD)
# ---------------------------------------------------------------------------

_XAI_PRICES: dict[str, tuple[float, float]] = {
    # model           (input $/M, output $/M)
    "grok-3":         (3.00, 15.00),
    "grok-3-mini":    (0.30,  0.50),
    "grok-2":         (2.00, 10.00),
    "grok-2-mini":    (0.20,  0.40),
    "grok-beta":      (5.00, 15.00),
}

_DEFAULT_BASE_URL = "https://api.x.ai/v1"
_DEFAULT_MAX_CONCURRENT = 50
_CHARS_PER_TOKEN = 4

# Retry configuration.
_MAX_RETRY_ATTEMPTS = 5
_INITIAL_BACKOFF = 1.0
_BACKOFF_MULTIPLIER = 2.0
_MAX_BACKOFF = 60.0


def _map_http_error(
    response: httpx.Response | None,
    exc: Exception | None,
    provider: str = "xai",
) -> ProviderError:
    """Map an httpx response or exception to a relay exception.

    Args:
        response: The HTTP response object, if available.
        exc: The original exception, if no response was received.
        provider: Provider name string for error context.

    Returns:
        A :class:`~relay.exceptions.ProviderError` subclass instance.
    """
    if response is not None:
        status = response.status_code
        try:
            body = response.json()
            msg = body.get("error", {}).get("message", response.text) if isinstance(body, dict) else response.text
        except Exception:
            msg = response.text

        if status == 401:
            return AuthenticationError(msg, provider=provider, status_code=status)
        if status == 403:
            return AuthenticationError(msg, provider=provider, status_code=status)
        if status == 422:
            return ValidationError(msg)
        if status == 429:
            retry_after: float | None = None
            hdr = response.headers.get("retry-after")
            if hdr:
                try:
                    retry_after = float(hdr)
                except ValueError:
                    pass
            return RateLimitError(msg, retry_after=retry_after, provider=provider, status_code=status)
        if status >= 500:
            return ServerError(msg, provider=provider, status_code=status)
        return ProviderError(msg, provider=provider, status_code=status)

    # Network-level error — no response.
    return ProviderError(str(exc) if exc else "Unknown network error", provider=provider)


def _build_chat_body(request: dict, model: str) -> dict:
    """Build an OpenAI-compatible ``/v1/chat/completions`` request body.

    Args:
        request: A serialised ``BatchRequest`` dict.
        model: Default model to use if the request does not override it.

    Returns:
        A dict suitable for JSON-encoding and posting to ``/v1/chat/completions``.
    """
    messages: list[dict] = []

    system = request.get("system")
    if system:
        messages.append({"role": "system", "content": system})

    for msg in request.get("messages", []):
        if msg.get("role") == "system":
            # Deduplicate if system was already in the messages list.
            continue
        messages.append({"role": msg["role"], "content": msg.get("content", "")})

    body: dict[str, Any] = {
        "model": request.get("model") or model,
        "messages": messages,
        "max_tokens": request.get("max_tokens", 1024),
    }

    temperature = request.get("temperature")
    if temperature is not None:
        body["temperature"] = temperature

    top_p = request.get("top_p")
    if top_p is not None:
        body["top_p"] = top_p

    stop = request.get("stop_sequences") or []
    if stop:
        body["stop"] = stop

    return body


class XAIProvider(BaseProvider):
    """Provider adapter for the XAI Grok API.

    Submits requests concurrently via an ``asyncio.Semaphore``-bounded pool
    (default: 50 simultaneous connections).  A synthetic UUID is used as the
    ``provider_job_id``; results are tracked in a class-level in-memory store.

    Rate limit responses (HTTP 429) are handled with automatic back-off,
    honouring the ``Retry-After`` response header when present.

    Args:
        config: Configuration dict.  Recognised keys:

            - ``api_key`` (str): XAI API key.  Falls back to the
              ``XAI_API_KEY`` environment variable.
            - ``model`` (str): Default model identifier.
            - ``base_url`` (str): XAI API base URL
              (default: ``"https://api.x.ai/v1"``).
            - ``max_concurrent`` (int): Concurrent request pool size
              (default: 50).
            - ``timeout`` (float): Per-request timeout in seconds
              (default: 120.0).
    """

    # In-memory job store: job_id -> {"status": ProviderStatus, "results": list[dict]}
    _job_store: dict[str, dict[str, Any]] = {}

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        api_key = config.get("api_key") or ""
        self._base_url = config.get("base_url", _DEFAULT_BASE_URL).rstrip("/")
        self._default_model: str = config.get("model", "grok-3")
        self._max_concurrent: int = int(config.get("max_concurrent", _DEFAULT_MAX_CONCURRENT))
        self._timeout: float = float(config.get("timeout", 120.0))

        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    async def submit_batch(self, requests: list[dict]) -> str:
        """Submit all requests to the XAI API using a concurrent pool.

        Args:
            requests: Serialised ``BatchRequest`` dicts.

        Returns:
            A synthetic UUID string used as the ``provider_job_id``.

        Raises:
            relay.exceptions.AuthenticationError: On 401/403 responses.
            relay.exceptions.RateLimitError: On rate limit responses.
            relay.exceptions.ServerError: On persistent 5xx responses.
            relay.exceptions.ValidationError: On 422 responses.
            relay.exceptions.ProviderError: On any other API error.
        """
        job_id = str(uuid.uuid4())
        XAIProvider._job_store[job_id] = {
            "status": ProviderStatus(
                provider_job_id=job_id,
                status="in_progress",
                completed=0,
                failed=0,
                total=len(requests),
            ),
            "results": [],
        }

        asyncio.ensure_future(self._run_pool(job_id, requests))
        return job_id

    async def _run_pool(self, job_id: str, requests: list[dict]) -> None:
        """Drive the concurrent request pool and populate the job store.

        This coroutine is scheduled as a background task by :meth:`submit_batch`.

        Args:
            job_id: The synthetic job identifier.
            requests: Serialised ``BatchRequest`` dicts.
        """
        semaphore = asyncio.Semaphore(self._max_concurrent)
        results: list[dict] = []
        completed = 0
        failed = 0

        async with httpx.AsyncClient(
            headers=self._headers,
            timeout=self._timeout,
        ) as client:

            async def _call_one(req: dict) -> dict:
                async with semaphore:
                    return await self._call_with_retry(client, req)

            tasks = [_call_one(req) for req in requests]
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                if result.get("error"):
                    failed += 1
                else:
                    completed += 1
                XAIProvider._job_store[job_id]["status"] = ProviderStatus(
                    provider_job_id=job_id,
                    status="in_progress",
                    completed=completed,
                    failed=failed,
                    total=len(requests),
                )

        XAIProvider._job_store[job_id]["results"] = results
        XAIProvider._job_store[job_id]["status"] = ProviderStatus(
            provider_job_id=job_id,
            status="ended",
            completed=completed,
            failed=failed,
            total=len(requests),
        )

    async def _call_with_retry(self, client: httpx.AsyncClient, request: dict) -> dict:
        """Submit a single request to the XAI chat completions endpoint with retry.

        Implements exponential back-off with full jitter.  Respects
        ``Retry-After`` headers for HTTP 429 responses.

        Args:
            client: A shared ``httpx.AsyncClient`` instance.
            request: A serialised ``BatchRequest`` dict.

        Returns:
            A normalised relay result dict.
        """
        request_id: str = request.get("id", str(uuid.uuid4()))
        body = _build_chat_body(request, self._default_model)
        url = f"{self._base_url}/chat/completions"

        attempt = 0
        backoff = _INITIAL_BACKOFF

        while attempt < _MAX_RETRY_ATTEMPTS:
            attempt += 1
            response: httpx.Response | None = None
            try:
                response = await client.post(url, json=body)

                if response.status_code == 429:
                    retry_after_hdr = response.headers.get("retry-after")
                    if retry_after_hdr:
                        try:
                            wait = float(retry_after_hdr)
                        except ValueError:
                            wait = backoff
                    else:
                        wait = backoff
                    wait = wait + random.uniform(0, wait * 0.1)
                    logger.warning(
                        "XAI rate limit on request %s; retrying in %.1fs (attempt %d/%d)",
                        request_id, wait, attempt, _MAX_RETRY_ATTEMPTS,
                    )
                    await asyncio.sleep(wait)
                    backoff = min(backoff * _BACKOFF_MULTIPLIER, _MAX_BACKOFF)
                    continue

                if response.status_code >= 500 and attempt < _MAX_RETRY_ATTEMPTS:
                    wait = backoff + random.uniform(0, backoff * 0.5)
                    logger.warning(
                        "XAI server error %d on request %s; retrying in %.1fs (attempt %d/%d)",
                        response.status_code, request_id, wait, attempt, _MAX_RETRY_ATTEMPTS,
                    )
                    await asyncio.sleep(wait)
                    backoff = min(backoff * _BACKOFF_MULTIPLIER, _MAX_BACKOFF)
                    continue

                if response.status_code not in (200, 201):
                    mapped = _map_http_error(response, None)
                    return {
                        "request_id": request_id,
                        "content": "",
                        "stop_reason": "error",
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "model": body.get("model", ""),
                        "error": {
                            "code": type(mapped).__name__,
                            "message": str(mapped),
                            "retryable": isinstance(mapped, (RateLimitError, ServerError)),
                        },
                        "raw_response": {},
                    }

                return _parse_xai_response(request_id, response.json())

            except httpx.TimeoutException as exc:
                if attempt < _MAX_RETRY_ATTEMPTS:
                    wait = backoff + random.uniform(0, backoff * 0.5)
                    logger.warning(
                        "XAI timeout on request %s; retrying in %.1fs (attempt %d/%d)",
                        request_id, wait, attempt, _MAX_RETRY_ATTEMPTS,
                    )
                    await asyncio.sleep(wait)
                    backoff = min(backoff * _BACKOFF_MULTIPLIER, _MAX_BACKOFF)
                    continue
                return {
                    "request_id": request_id,
                    "content": "",
                    "stop_reason": "error",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "model": body.get("model", ""),
                    "error": {
                        "code": "timeout",
                        "message": str(exc),
                        "retryable": True,
                    },
                    "raw_response": {},
                }
            except httpx.NetworkError as exc:
                if attempt < _MAX_RETRY_ATTEMPTS:
                    wait = backoff + random.uniform(0, backoff * 0.5)
                    await asyncio.sleep(wait)
                    backoff = min(backoff * _BACKOFF_MULTIPLIER, _MAX_BACKOFF)
                    continue
                return {
                    "request_id": request_id,
                    "content": "",
                    "stop_reason": "error",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "model": body.get("model", ""),
                    "error": {
                        "code": "network_error",
                        "message": str(exc),
                        "retryable": True,
                    },
                    "raw_response": {},
                }

        # Exhausted all retries.
        return {
            "request_id": request_id,
            "content": "",
            "stop_reason": "error",
            "input_tokens": 0,
            "output_tokens": 0,
            "model": body.get("model", ""),
            "error": {
                "code": "max_retries_exceeded",
                "message": f"Request {request_id} failed after {_MAX_RETRY_ATTEMPTS} attempts.",
                "retryable": False,
            },
            "raw_response": {},
        }

    # ------------------------------------------------------------------
    # Status polling
    # ------------------------------------------------------------------

    async def get_status(self, provider_job_id: str) -> ProviderStatus:
        """Return the current status of an XAI batch job.

        Reads from the in-memory job store populated by the background pool task.

        Args:
            provider_job_id: The synthetic UUID returned by :meth:`submit_batch`.

        Returns:
            A :class:`~relay.models.ProviderStatus` snapshot.

        Raises:
            relay.exceptions.ProviderError: If the job ID is not found.
        """
        entry = XAIProvider._job_store.get(provider_job_id)
        if entry is None:
            raise ProviderError(
                f"XAI job {provider_job_id!r} not found in session store. "
                "Results are stored in memory only; the job may have been submitted "
                "in a different process or session.",
                provider="xai",
            )
        return entry["status"]

    # ------------------------------------------------------------------
    # Result retrieval
    # ------------------------------------------------------------------

    async def download_results(self, provider_job_id: str) -> list[dict]:
        """Return results for a completed XAI batch job.

        Polls the in-memory store until the background pool task finishes.

        Args:
            provider_job_id: The synthetic UUID returned by :meth:`submit_batch`.

        Returns:
            A list of normalised result dicts.

        Raises:
            relay.exceptions.ProviderError: If the job ID is not found.
        """
        for _ in range(86400):  # Up to 24 hours of 1-second polling.
            entry = XAIProvider._job_store.get(provider_job_id)
            if entry is None:
                raise ProviderError(
                    f"XAI job {provider_job_id!r} not found.",
                    provider="xai",
                )
            if entry["status"].status == "ended":
                return entry["results"]
            await asyncio.sleep(1)

        raise ProviderError(
            f"XAI job {provider_job_id!r} did not complete within the timeout.",
            provider="xai",
        )

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    async def cancel(self, provider_job_id: str) -> bool:
        """Attempt to cancel an XAI batch job.

        Because cancellation of in-flight HTTP requests is not possible once
        dispatched, this method marks the job as cancelled in the in-memory
        store only.  Already-dispatched requests may still complete and be billed.

        Args:
            provider_job_id: The synthetic UUID returned by :meth:`submit_batch`.

        Returns:
            ``True`` if the job was found and is not yet complete;
            ``False`` otherwise.
        """
        entry = XAIProvider._job_store.get(provider_job_id)
        if entry is None:
            return False
        if entry["status"].status == "ended":
            return False
        status = entry["status"]
        entry["status"] = ProviderStatus(
            provider_job_id=provider_job_id,
            status="cancelled",
            completed=status.completed,
            failed=status.failed,
            total=status.total,
        )
        return True

    # ------------------------------------------------------------------
    # Token / cost estimation
    # ------------------------------------------------------------------

    def estimate_tokens(self, request: BatchRequest) -> tuple[int, int]:
        """Estimate input and output token counts for a single ``BatchRequest``.

        Uses a character-based heuristic (4 chars ≈ 1 token) since the XAI API
        does not expose a local tokenizer.

        Args:
            request: The :class:`~relay.models.BatchRequest` to estimate.

        Returns:
            ``(estimated_input_tokens, estimated_output_tokens)`` tuple.
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
        """Return the price per million tokens for the given XAI model.

        Args:
            model: XAI model identifier.

        Returns:
            ``(input_price_usd, output_price_usd)`` per million tokens.

        Raises:
            relay.exceptions.ValidationError: If the model is not in the
                pricing table.
        """
        price = _XAI_PRICES.get(model)
        if price is None:
            for key, val in _XAI_PRICES.items():
                if model.startswith(key) or key.startswith(model.split("-")[0]):
                    return val
            raise ValidationError(
                f"Unknown XAI model {model!r}. "
                f"Known models: {', '.join(_XAI_PRICES)}"
            )
        return price


# ---------------------------------------------------------------------------
# Response parsing helper
# ---------------------------------------------------------------------------

def _parse_xai_response(request_id: str, body: dict) -> dict:
    """Convert an XAI ``/v1/chat/completions`` response body to a relay result dict.

    Args:
        request_id: The relay request identifier.
        body: The parsed JSON response body from the XAI API.

    Returns:
        A normalised relay result dict.
    """
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
        "request_id": request_id,
        "content": content,
        "stop_reason": stop_reason,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model": model,
        "error": None,
        "raw_response": body,
    }
