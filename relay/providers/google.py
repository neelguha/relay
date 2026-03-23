"""Google Gemini provider adapter for relay.

Uses the ``google-generativeai`` SDK (Gemini Developer API) to submit batch
prediction jobs.  Supports API key authentication.

Key characteristics:
- Maps relay messages (OpenAI-style role/content) to the Gemini
  ``contents`` / ``parts`` format.
- System instructions are passed via ``system_instruction`` at the top level
  of the ``GenerateContentRequest``.
- Because the Gemini Developer API does not yet expose a native asynchronous
  batch endpoint equivalent to Anthropic's or OpenAI's, this adapter submits
  all requests concurrently (up to a configurable pool size) using
  ``asyncio.Semaphore``, tracking them under a single synthetic job ID.
- Result state is held in memory; for large jobs users should rely on relay's
  job persistence layer.
- Maps SDK exceptions to the relay exception hierarchy.

Dependencies:
    pip install relay[google]   # installs google-generativeai>=0.8
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

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
# Pricing table (Gemini Developer API, no native batch discount, early 2025)
# ---------------------------------------------------------------------------

_GOOGLE_PRICES: dict[str, tuple[float, float]] = {
    # model                (input $/M, output $/M)
    "gemini-2.0-flash":    (0.075, 0.30),
    "gemini-2.0-flash-lite": (0.075, 0.30),
    "gemini-2.5-pro":      (1.25, 10.00),
    "gemini-2.5-flash":    (0.075, 0.30),
    "gemini-1.5-pro":      (1.25, 5.00),
    "gemini-1.5-flash":    (0.075, 0.30),
    "gemini-1.0-pro":      (0.50, 1.50),
}

_CHARS_PER_TOKEN = 4
_DEFAULT_MAX_CONCURRENT = 20


def _require_genai() -> Any:
    """Import and return the ``google.generativeai`` module.

    Returns:
        The ``google.generativeai`` module object.

    Raises:
        ImportError: With install instructions if the SDK is not installed.
    """
    try:
        import google.generativeai as genai  # type: ignore[import]
        return genai
    except ImportError as exc:
        raise ImportError(
            "The 'google-generativeai' package is required for the Google provider. "
            "Install it with: pip install relay[google]"
        ) from exc


def _map_google_error(exc: Exception, provider: str = "google") -> ProviderError:
    """Map a google-generativeai SDK exception to a relay exception.

    Args:
        exc: The original exception from the Google SDK.
        provider: Provider name string for error context.

    Returns:
        A :class:`~relay.exceptions.ProviderError` subclass instance.
    """
    msg = str(exc)
    exc_type = type(exc).__name__

    # google.api_core exceptions
    try:
        from google.api_core import exceptions as gexc  # type: ignore[import]
        if isinstance(exc, gexc.Unauthenticated):
            return AuthenticationError(msg, provider=provider, status_code=401)
        if isinstance(exc, gexc.PermissionDenied):
            return AuthenticationError(msg, provider=provider, status_code=403)
        if isinstance(exc, gexc.ResourceExhausted):
            return RateLimitError(msg, provider=provider, status_code=429)
        if isinstance(exc, gexc.InvalidArgument):
            return ValidationError(msg)
        if isinstance(exc, gexc.InternalServerError):
            return ServerError(msg, provider=provider, status_code=500)
        if isinstance(exc, gexc.ServiceUnavailable):
            return ServerError(msg, provider=provider, status_code=503)
    except ImportError:
        pass

    # Fallback: inspect exception class name for clues.
    lower_msg = msg.lower()
    if "401" in msg or "unauthenticated" in lower_msg or "api_key" in lower_msg:
        return AuthenticationError(msg, provider=provider, status_code=401)
    if "429" in msg or "quota" in lower_msg or "rate" in lower_msg:
        return RateLimitError(msg, provider=provider, status_code=429)
    if any(code in msg for code in ("500", "502", "503", "504")):
        return ServerError(msg, provider=provider, status_code=500)

    return ProviderError(msg, provider=provider)


def _build_gemini_contents(request: dict) -> list[dict]:
    """Convert relay messages to the Gemini ``contents`` format.

    System messages (role ``"system"``) are excluded; they are handled
    separately via the ``system_instruction`` parameter.  The Gemini API
    uses ``"model"`` for assistant turns.

    Args:
        request: A serialised ``BatchRequest`` dict.

    Returns:
        A list of ``{"role": ..., "parts": [{"text": ...}]}`` dicts.
    """
    role_map = {"user": "user", "assistant": "model"}
    contents = []
    for msg in request.get("messages", []):
        role = msg.get("role", "user")
        if role == "system":
            continue
        gemini_role = role_map.get(role, "user")
        contents.append({
            "role": gemini_role,
            "parts": [{"text": msg.get("content", "")}],
        })
    return contents


def _build_generate_config(request: dict) -> dict:
    """Build a ``GenerationConfig`` dict from a relay request.

    Args:
        request: A serialised ``BatchRequest`` dict.

    Returns:
        A dict that can be passed as ``generation_config`` to the Gemini SDK.
    """
    cfg: dict[str, Any] = {
        "max_output_tokens": request.get("max_tokens", 1024),
    }
    temperature = request.get("temperature")
    if temperature is not None:
        cfg["temperature"] = temperature
    top_p = request.get("top_p")
    if top_p is not None:
        cfg["top_p"] = top_p
    stop_sequences = request.get("stop_sequences") or []
    if stop_sequences:
        cfg["stop_sequences"] = stop_sequences
    return cfg


class GoogleProvider(BaseProvider):
    """Provider adapter for the Google Gemini Developer API.

    Submits requests concurrently using ``asyncio.Semaphore`` (since the
    Gemini Developer API does not currently offer a native batch endpoint).
    A synthetic ``provider_job_id`` (a UUID) is returned immediately on
    submission; results are gathered and stored in memory keyed on this ID.

    Args:
        config: Configuration dict.  Recognised keys:

            - ``api_key`` (str): Google API key.  Falls back to the
              ``GOOGLE_API_KEY`` environment variable (via the SDK).
            - ``model`` (str): Default model identifier.
            - ``max_concurrent`` (int): Maximum concurrent API calls
              (default: 20).
    """

    # In-memory result store: job_id -> list[dict]
    _result_store: dict[str, list[dict]] = {}
    _status_store: dict[str, ProviderStatus] = {}

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        genai = _require_genai()

        api_key = config.get("api_key")
        if api_key:
            genai.configure(api_key=api_key)

        self._genai = genai
        self._default_model: str = config.get("model", "gemini-2.0-flash")
        self._max_concurrent: int = int(
            config.get("max_concurrent", _DEFAULT_MAX_CONCURRENT)
        )

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    async def submit_batch(self, requests: list[dict]) -> str:
        """Submit a list of requests to the Gemini API concurrently.

        Requests are dispatched via an ``asyncio.Semaphore``-bounded pool.
        A synthetic job ID is returned immediately; results are stored in
        a class-level in-memory dict keyed on that ID.

        Args:
            requests: Serialised ``BatchRequest`` dicts.

        Returns:
            A synthetic UUID string used as the ``provider_job_id``.

        Raises:
            relay.exceptions.AuthenticationError: On authentication failure.
            relay.exceptions.RateLimitError: On quota exhaustion.
            relay.exceptions.ServerError: On 5xx responses.
            relay.exceptions.ValidationError: On malformed payloads.
            relay.exceptions.ProviderError: On any other API error.
        """
        job_id = str(uuid.uuid4())
        # Initialise status immediately so get_status() never KeyErrors.
        GoogleProvider._status_store[job_id] = ProviderStatus(
            provider_job_id=job_id,
            status="in_progress",
            completed=0,
            failed=0,
            total=len(requests),
        )
        GoogleProvider._result_store[job_id] = []

        asyncio.ensure_future(self._run_batch(job_id, requests))
        return job_id

    async def _run_batch(self, job_id: str, requests: list[dict]) -> None:
        """Execute all requests in the batch and populate the result store.

        This coroutine is scheduled as a background task by :meth:`submit_batch`.
        It updates :attr:`_status_store` and :attr:`_result_store` in place.

        Args:
            job_id: The synthetic job identifier.
            requests: Serialised ``BatchRequest`` dicts.
        """
        semaphore = asyncio.Semaphore(self._max_concurrent)
        results: list[dict] = []
        completed = 0
        failed = 0

        async def _call_one(req: dict) -> dict:
            async with semaphore:
                return await self._generate_one(req)

        tasks = [_call_one(req) for req in requests]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            if result.get("error"):
                failed += 1
            else:
                completed += 1
            GoogleProvider._status_store[job_id] = ProviderStatus(
                provider_job_id=job_id,
                status="in_progress",
                completed=completed,
                failed=failed,
                total=len(requests),
            )

        GoogleProvider._result_store[job_id] = results
        GoogleProvider._status_store[job_id] = ProviderStatus(
            provider_job_id=job_id,
            status="ended",
            completed=completed,
            failed=failed,
            total=len(requests),
        )

    async def _generate_one(self, request: dict) -> dict:
        """Call the Gemini ``generate_content`` API for a single request.

        Args:
            request: A serialised ``BatchRequest`` dict.

        Returns:
            A normalised relay result dict.
        """
        model_name = request.get("model") or self._default_model
        contents = _build_gemini_contents(request)
        generation_config = _build_generate_config(request)
        system_instruction = request.get("system")

        request_id: str = request.get("id", str(uuid.uuid4()))

        try:
            model = self._genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_instruction,
                generation_config=generation_config,
            )
            response = await model.generate_content_async(contents)

            text = ""
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    text = "".join(
                        part.text for part in candidate.content.parts
                        if hasattr(part, "text")
                    )
                finish_reason = str(getattr(candidate, "finish_reason", "STOP"))
            else:
                finish_reason = "STOP"

            usage = getattr(response, "usage_metadata", None)
            input_tokens = getattr(usage, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage, "candidates_token_count", 0) or 0

            return {
                "request_id": request_id,
                "content": text,
                "stop_reason": finish_reason,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "model": model_name,
                "error": None,
                "raw_response": {},
            }
        except Exception as exc:
            mapped = _map_google_error(exc)
            return {
                "request_id": request_id,
                "content": "",
                "stop_reason": "error",
                "input_tokens": 0,
                "output_tokens": 0,
                "model": model_name,
                "error": {
                    "code": type(mapped).__name__,
                    "message": str(mapped),
                    "retryable": isinstance(mapped, (RateLimitError, ServerError)),
                },
                "raw_response": {},
            }

    # ------------------------------------------------------------------
    # Status polling
    # ------------------------------------------------------------------

    async def get_status(self, provider_job_id: str) -> ProviderStatus:
        """Return the current status of a submitted Google batch job.

        Reads from the in-memory status store populated by the background
        task started in :meth:`submit_batch`.

        Args:
            provider_job_id: The synthetic UUID returned by :meth:`submit_batch`.

        Returns:
            A :class:`~relay.models.ProviderStatus` snapshot.

        Raises:
            relay.exceptions.ProviderError: If the job ID is not found.
        """
        status = GoogleProvider._status_store.get(provider_job_id)
        if status is None:
            raise ProviderError(
                f"Google job {provider_job_id!r} not found in session store. "
                "Results are stored in memory only; the job may have been submitted "
                "in a different process or session.",
                provider="google",
            )
        return status

    # ------------------------------------------------------------------
    # Result retrieval
    # ------------------------------------------------------------------

    async def download_results(self, provider_job_id: str) -> list[dict]:
        """Return results for a completed Google batch job.

        Retrieves results from the in-memory result store.  Blocks until
        the background task has set the status to ``"ended"``.

        Args:
            provider_job_id: The synthetic UUID returned by :meth:`submit_batch`.

        Returns:
            A list of normalised result dicts.

        Raises:
            relay.exceptions.ProviderError: If the job ID is not found.
        """
        # Poll until complete.
        for _ in range(3600):
            status = GoogleProvider._status_store.get(provider_job_id)
            if status is None:
                raise ProviderError(
                    f"Google job {provider_job_id!r} not found.",
                    provider="google",
                )
            if status.status == "ended":
                break
            await asyncio.sleep(1)

        results = GoogleProvider._result_store.get(provider_job_id)
        if results is None:
            raise ProviderError(
                f"Google job {provider_job_id!r} has no results.",
                provider="google",
            )
        return results

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    async def cancel(self, provider_job_id: str) -> bool:
        """Attempt to cancel a Google batch job.

        Because the adapter uses a concurrent request pool rather than a
        native batch API, cancellation is best-effort: requests already
        dispatched to the Gemini API cannot be recalled.  This method marks
        the job as cancelled in the status store.

        Args:
            provider_job_id: The synthetic UUID returned by :meth:`submit_batch`.

        Returns:
            ``True`` if the job was found and marked cancelled;
            ``False`` otherwise.
        """
        status = GoogleProvider._status_store.get(provider_job_id)
        if status is None:
            return False
        if status.status == "ended":
            return False
        GoogleProvider._status_store[provider_job_id] = ProviderStatus(
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

        Uses a character-based heuristic (4 chars ≈ 1 token).

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
        """Return the price per million tokens for the given Gemini model.

        Args:
            model: Google model identifier.

        Returns:
            ``(input_price_usd, output_price_usd)`` per million tokens.

        Raises:
            relay.exceptions.ValidationError: If the model is not in the
                pricing table.
        """
        price = _GOOGLE_PRICES.get(model)
        if price is None:
            for key, val in _GOOGLE_PRICES.items():
                if model.startswith(key) or key.startswith(model.split("-")[0]):
                    return val
            raise ValidationError(
                f"Unknown Google model {model!r}. "
                f"Known models: {', '.join(_GOOGLE_PRICES)}"
            )
        return price
