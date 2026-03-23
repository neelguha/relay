"""Unit tests for the relay exception hierarchy."""

from __future__ import annotations

import pytest

from relay.exceptions import (
    AuthenticationError,
    BatchExpiredError,
    BudgetConfirmationRequired,
    BudgetExceeded,
    CacheError,
    JobNotFound,
    ProviderError,
    RateLimitError,
    RelayError,
    ServerError,
    ValidationError,
)


# ── Inheritance ───────────────────────────────────────────────────────────────


class TestInheritanceHierarchy:
    @pytest.mark.parametrize("exc_class", [
        ProviderError,
        RateLimitError,
        AuthenticationError,
        ServerError,
        BatchExpiredError,
        ValidationError,
        BudgetExceeded,
        BudgetConfirmationRequired,
        CacheError,
        JobNotFound,
    ])
    def test_inherits_from_relay_error(self, exc_class):
        exc = exc_class("test message")
        assert isinstance(exc, RelayError)
        assert isinstance(exc, Exception)

    @pytest.mark.parametrize("exc_class", [
        RateLimitError,
        AuthenticationError,
        ServerError,
        BatchExpiredError,
    ])
    def test_provider_subclasses_inherit_from_provider_error(self, exc_class):
        exc = exc_class("test")
        assert isinstance(exc, ProviderError)

    def test_validation_error_not_provider_error(self):
        exc = ValidationError("bad request")
        assert not isinstance(exc, ProviderError)

    def test_budget_exceeded_not_provider_error(self):
        assert not isinstance(BudgetExceeded("over budget"), ProviderError)

    def test_cache_error_not_provider_error(self):
        assert not isinstance(CacheError("cache down"), ProviderError)

    def test_job_not_found_not_provider_error(self):
        assert not isinstance(JobNotFound("no such job"), ProviderError)


# ── RelayError ────────────────────────────────────────────────────────────────


class TestRelayError:
    def test_message(self):
        exc = RelayError("something went wrong")
        assert str(exc) == "something went wrong"

    def test_can_be_raised_and_caught(self):
        with pytest.raises(RelayError, match="boom"):
            raise RelayError("boom")


# ── ProviderError ─────────────────────────────────────────────────────────────


class TestProviderError:
    def test_message_only(self):
        exc = ProviderError("upstream error")
        assert str(exc) == "upstream error"
        assert exc.provider is None
        assert exc.status_code is None

    def test_with_provider(self):
        exc = ProviderError("error", provider="anthropic")
        assert exc.provider == "anthropic"
        assert exc.status_code is None

    def test_with_status_code(self):
        exc = ProviderError("error", status_code=503)
        assert exc.status_code == 503
        assert exc.provider is None

    def test_with_all_attrs(self):
        exc = ProviderError("error", provider="openai", status_code=429)
        assert exc.provider == "openai"
        assert exc.status_code == 429

    def test_catchable_as_relay_error(self):
        with pytest.raises(RelayError):
            raise ProviderError("provider gone", provider="google")


# ── RateLimitError ────────────────────────────────────────────────────────────


class TestRateLimitError:
    def test_defaults(self):
        exc = RateLimitError("too many requests")
        assert exc.retry_after is None
        assert exc.provider is None
        assert exc.status_code is None

    def test_retry_after(self):
        exc = RateLimitError("slow down", retry_after=30.5)
        assert exc.retry_after == 30.5

    def test_retry_after_zero(self):
        exc = RateLimitError("slow down", retry_after=0.0)
        assert exc.retry_after == 0.0

    def test_retry_after_with_provider(self):
        exc = RateLimitError("rate limited", retry_after=60.0, provider="anthropic")
        assert exc.retry_after == 60.0
        assert exc.provider == "anthropic"

    def test_catchable_as_provider_error(self):
        with pytest.raises(ProviderError):
            raise RateLimitError("429")

    def test_catchable_as_relay_error(self):
        with pytest.raises(RelayError):
            raise RateLimitError("429")


# ── AuthenticationError ───────────────────────────────────────────────────────


class TestAuthenticationError:
    def test_basic(self):
        exc = AuthenticationError("invalid key", provider="openai", status_code=401)
        assert exc.provider == "openai"
        assert exc.status_code == 401
        assert "invalid key" in str(exc)


# ── ServerError ───────────────────────────────────────────────────────────────


class TestServerError:
    def test_basic(self):
        exc = ServerError("internal server error", status_code=500)
        assert exc.status_code == 500


# ── Other relay errors ────────────────────────────────────────────────────────


class TestOtherErrors:
    def test_validation_error(self):
        exc = ValidationError("field 'model' is required")
        assert "model" in str(exc)

    def test_budget_exceeded(self):
        exc = BudgetExceeded("limit $100 reached")
        assert isinstance(exc, RelayError)

    def test_budget_confirmation_required(self):
        exc = BudgetConfirmationRequired("non-interactive mode")
        assert isinstance(exc, RelayError)

    def test_cache_error(self):
        exc = CacheError("disk full")
        assert isinstance(exc, RelayError)

    def test_job_not_found(self):
        exc = JobNotFound("job-abc-123")
        assert "job-abc-123" in str(exc)

    def test_batch_expired_error(self):
        exc = BatchExpiredError("batch expired after 24h", provider="anthropic")
        assert isinstance(exc, ProviderError)
        assert exc.provider == "anthropic"
