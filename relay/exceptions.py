"""Exception hierarchy for the relay library."""


class RelayError(Exception):
    """Base exception for all relay errors."""


class ProviderError(RelayError):
    """Error returned by a provider API."""

    def __init__(self, message: str, provider: str | None = None, status_code: int | None = None):
        self.provider = provider
        self.status_code = status_code
        super().__init__(message)


class RateLimitError(ProviderError):
    """HTTP 429 — retry-able."""

    def __init__(self, message: str, retry_after: float | None = None, **kwargs):
        self.retry_after = retry_after
        super().__init__(message, **kwargs)


class AuthenticationError(ProviderError):
    """HTTP 401/403 — not retry-able."""


class ServerError(ProviderError):
    """HTTP 5xx — retry-able."""


class BatchExpiredError(ProviderError):
    """Provider-side expiry before completion."""


class ValidationError(RelayError):
    """Malformed request."""


class BudgetExceeded(RelayError):
    """Hard cost limit reached."""


class BudgetConfirmationRequired(RelayError):
    """Interactive confirmation not possible in non-interactive mode."""


class CacheError(RelayError):
    """Storage backend failure."""


class JobNotFound(RelayError):
    """Unknown job ID queried."""
