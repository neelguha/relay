"""Abstract base class for all relay provider adapters.

Defines the ``BaseProvider`` interface that every provider adapter must
implement.  Each adapter is responsible for translating relay's unified
request format into the provider's native batch API calls, handling
authentication, chunking, polling, and mapping errors to the relay
exception hierarchy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from relay.models import BatchRequest, ProviderStatus


class BaseProvider(ABC):
    """Abstract base class for relay provider adapters.

    Subclasses must implement all abstract methods.  Adapters must never
    raise bare exceptions; all errors must be mapped to types defined in
    ``relay.exceptions``.

    Args:
        config: Provider-specific configuration dictionary.  At minimum
            this should contain an ``api_key`` entry.  Additional keys
            are adapter-defined.
    """

    def __init__(self, config: dict) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    @abstractmethod
    async def submit_batch(self, requests: list[dict]) -> str:
        """Submit a list of requests to the provider's batch API.

        The ``requests`` list contains dicts with at minimum the fields
        present on :class:`~relay.models.BatchRequest` (serialised via
        ``dataclasses.asdict``).  Adapters should chunk the list if the
        provider imposes a per-batch request limit.

        Args:
            requests: Serialised ``BatchRequest`` dicts to submit.

        Returns:
            The provider-assigned batch job identifier (``provider_job_id``).

        Raises:
            relay.exceptions.AuthenticationError: On 401/403 responses.
            relay.exceptions.RateLimitError: On 429 responses.
            relay.exceptions.ServerError: On 5xx responses.
            relay.exceptions.ValidationError: If the provider rejects the
                request payload.
            relay.exceptions.ProviderError: For any other provider error.
        """

    # ------------------------------------------------------------------
    # Status polling
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_status(self, provider_job_id: str) -> ProviderStatus:
        """Return the current status of a submitted batch job.

        Args:
            provider_job_id: The identifier returned by :meth:`submit_batch`.

        Returns:
            A :class:`~relay.models.ProviderStatus` snapshot including
            progress counters.

        Raises:
            relay.exceptions.ProviderError: On any API error.
            relay.exceptions.JobNotFound: If the provider no longer
                recognises the job ID.
        """

    # ------------------------------------------------------------------
    # Result retrieval
    # ------------------------------------------------------------------

    @abstractmethod
    async def download_results(self, provider_job_id: str) -> list[dict]:
        """Download and parse results for a completed batch job.

        Args:
            provider_job_id: The identifier returned by :meth:`submit_batch`.

        Returns:
            A list of result dicts.  Each dict contains at minimum:
            ``request_id``, ``content``, ``stop_reason``,
            ``input_tokens``, ``output_tokens``, ``model``, and
            optionally ``error`` (a dict with ``code`` and ``message``).

        Raises:
            relay.exceptions.ProviderError: On any API error.
        """

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    @abstractmethod
    async def cancel(self, provider_job_id: str) -> bool:
        """Request cancellation of an in-progress batch.

        Cancellation is best-effort; some requests may have already been
        processed and billed.

        Args:
            provider_job_id: The identifier returned by :meth:`submit_batch`.

        Returns:
            ``True`` if the cancellation request was accepted,
            ``False`` if the job had already reached a terminal state.

        Raises:
            relay.exceptions.ProviderError: On any API error.
        """

    # ------------------------------------------------------------------
    # Token / cost estimation (synchronous — used before submission)
    # ------------------------------------------------------------------

    @abstractmethod
    def estimate_tokens(self, request: BatchRequest) -> tuple[int, int]:
        """Estimate input and output token counts for a single request.

        This method must be synchronous so it can be called in bulk
        without an event loop.  Implementations should use the provider's
        local tokenizer when available, falling back to a character-based
        heuristic otherwise.

        Args:
            request: The :class:`~relay.models.BatchRequest` to estimate.

        Returns:
            A ``(input_tokens, output_tokens)`` tuple.  ``output_tokens``
            is always an estimate; for most adapters this returns
            ``request.max_tokens`` as the upper bound.
        """

    @abstractmethod
    def get_price_per_million(self, model: str) -> tuple[float, float]:
        """Return the per-million-token price for a given model.

        Prices reflect the batch API discount where applicable.

        Args:
            model: The model identifier string (e.g. ``"claude-opus-4-5"``).

        Returns:
            A ``(input_price_usd, output_price_usd)`` tuple where each
            value is the USD cost per *million* tokens.

        Raises:
            relay.exceptions.ValidationError: If the model is unknown.
        """
