"""Unit tests for relay data models."""

from __future__ import annotations

import dataclasses
from datetime import datetime

import pytest

from relay.models import (
    BatchJob,
    BatchRequest,
    JobStatus,
    Message,
    RequestStatus,
)


# ── BatchRequest ──────────────────────────────────────────────────────────────


class TestBatchRequest:
    def test_defaults(self):
        req = BatchRequest()
        assert isinstance(req.id, str)
        assert len(req.id) == 36  # UUID4 with dashes
        assert req.messages == []
        assert req.system is None
        assert req.model is None
        assert req.max_tokens == 1024
        assert req.temperature == 1.0
        assert req.top_p is None
        assert req.stop_sequences == []
        assert req.metadata == {}
        assert req.tags == []

    def test_frozen(self):
        req = BatchRequest(max_tokens=512)
        with pytest.raises((dataclasses.FrozenInstanceError, TypeError, AttributeError)):
            req.max_tokens = 2048  # type: ignore[misc]

    def test_id_is_unique(self):
        ids = {BatchRequest().id for _ in range(50)}
        assert len(ids) == 50

    def test_explicit_fields(self):
        msgs = [{"role": "user", "content": "Hello"}]
        req = BatchRequest(
            messages=msgs,
            system="Be helpful.",
            model="claude-opus-4-5",
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            stop_sequences=["STOP"],
            metadata={"key": "value"},
            tags=["test"],
        )
        assert req.messages == msgs
        assert req.system == "Be helpful."
        assert req.model == "claude-opus-4-5"
        assert req.max_tokens == 256
        assert req.temperature == 0.7
        assert req.top_p == 0.9
        assert req.stop_sequences == ["STOP"]
        assert req.metadata == {"key": "value"}
        assert req.tags == ["test"]

    def test_custom_id(self):
        req = BatchRequest(id="my-custom-id")
        assert req.id == "my-custom-id"


# ── JobStatus ─────────────────────────────────────────────────────────────────


class TestJobStatusIsTerminal:
    @pytest.mark.parametrize("status", [
        JobStatus.COMPLETED,
        JobStatus.PARTIAL,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    ])
    def test_terminal_states(self, status: JobStatus):
        assert status.is_terminal is True

    @pytest.mark.parametrize("status", [
        JobStatus.PENDING,
        JobStatus.CACHE_RESOLVING,
        JobStatus.VALIDATING,
        JobStatus.SUBMITTING,
        JobStatus.IN_PROGRESS,
        JobStatus.DOWNLOADING,
    ])
    def test_non_terminal_states(self, status: JobStatus):
        assert status.is_terminal is False

    def test_all_statuses_covered(self):
        # Ensure every member of JobStatus is tested above and classified.
        for status in JobStatus:
            # Just verify the property exists and returns a bool.
            assert isinstance(status.is_terminal, bool)

    def test_str_value(self):
        assert JobStatus.COMPLETED == "COMPLETED"
        assert JobStatus.FAILED == "FAILED"


# ── BatchJob ──────────────────────────────────────────────────────────────────


class TestBatchJob:
    def _make_job(self, **kwargs) -> BatchJob:
        defaults = dict(
            id="job-abc",
            provider_job_id="prov-123",
            provider="anthropic",
            model="claude-opus-4-5",
            project=None,
            status=JobStatus.PENDING,
            total_requests=10,
        )
        defaults.update(kwargs)
        return BatchJob(**defaults)

    def test_creation_with_required_fields(self):
        job = self._make_job()
        assert job.id == "job-abc"
        assert job.provider == "anthropic"
        assert job.model == "claude-opus-4-5"
        assert job.status == JobStatus.PENDING
        assert job.total_requests == 10

    def test_counter_defaults(self):
        job = self._make_job()
        assert job.completed_requests == 0
        assert job.failed_requests == 0
        assert job.cached_hits == 0
        assert job.input_tokens == 0
        assert job.output_tokens == 0
        assert job.estimated_cost_usd == 0.0
        assert job.actual_cost_usd is None

    def test_timestamp_defaults(self):
        before = datetime.utcnow()
        job = self._make_job()
        after = datetime.utcnow()
        assert before <= job.created_at <= after
        assert job.submitted_at is None
        assert job.completed_at is None

    def test_tags_and_error_defaults(self):
        job = self._make_job()
        assert job.tags == []
        assert job.error is None

    def test_mutable(self):
        job = self._make_job()
        job.status = JobStatus.COMPLETED
        assert job.status == JobStatus.COMPLETED

    def test_with_tags(self):
        job = self._make_job(tags=["prod", "experiment"])
        assert "prod" in job.tags
        assert "experiment" in job.tags


# ── Message ───────────────────────────────────────────────────────────────────


class TestMessage:
    def test_frozen(self):
        msg = Message(role="user", content="Hello")
        with pytest.raises((dataclasses.FrozenInstanceError, TypeError, AttributeError)):
            msg.role = "assistant"  # type: ignore[misc]

    def test_fields(self):
        msg = Message(role="system", content="Be terse.")
        assert msg.role == "system"
        assert msg.content == "Be terse."


# ── RequestStatus ─────────────────────────────────────────────────────────────


class TestRequestStatus:
    def test_values(self):
        assert RequestStatus.PENDING == "pending"
        assert RequestStatus.COMPLETED == "completed"
        assert RequestStatus.FAILED == "failed"
        assert RequestStatus.CACHED == "cached"
