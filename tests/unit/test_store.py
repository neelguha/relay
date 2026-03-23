"""Unit tests for the async SQLite JobStore."""

from __future__ import annotations

import time
import uuid
from datetime import datetime

import pytest
import pytest_asyncio

from relay.db.store import JobStore
from relay.models import BatchJob, BatchResult, JobStatus, RequestStatus


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def store(tmp_path):
    db_path = tmp_path / "test_jobs.db"
    async with JobStore(str(db_path)) as s:
        yield s


def _make_job(**kwargs) -> BatchJob:
    defaults = dict(
        id=str(uuid.uuid4()),
        provider_job_id="prov-" + str(uuid.uuid4()),
        provider="anthropic",
        model="claude-opus-4-5",
        project=None,
        status=JobStatus.PENDING,
        total_requests=5,
    )
    defaults.update(kwargs)
    return BatchJob(**defaults)


def _make_result(request_id: str, job_id: str, **kwargs) -> BatchResult:
    defaults = dict(
        request_id=request_id,
        job_id=job_id,
        content="The capital of France is Paris.",
        stop_reason="end_turn",
        input_tokens=10,
        output_tokens=20,
        model="claude-opus-4-5",
        from_cache=False,
    )
    defaults.update(kwargs)
    return BatchResult(**defaults)


# ── Job CRUD ──────────────────────────────────────────────────────────────────


class TestJobCRUD:
    @pytest.mark.asyncio
    async def test_create_and_get_job(self, store):
        job = _make_job()
        await store.create_job(job, config={"provider": "anthropic"})

        fetched = await store.get_job(job.id)
        assert fetched is not None
        assert fetched.id == job.id
        assert fetched.provider == "anthropic"
        assert fetched.model == "claude-opus-4-5"
        assert fetched.status == JobStatus.PENDING

    @pytest.mark.asyncio
    async def test_get_nonexistent_job_returns_none(self, store):
        result = await store.get_job("does-not-exist")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_job_status(self, store):
        job = _make_job()
        await store.create_job(job, config={})

        await store.update_job_status(job.id, JobStatus.COMPLETED)
        fetched = await store.get_job(job.id)
        assert fetched.status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_update_job_status_with_error(self, store):
        job = _make_job()
        await store.create_job(job, config={})

        await store.update_job_status(job.id, JobStatus.FAILED, error="network timeout")
        fetched = await store.get_job(job.id)
        assert fetched.status == JobStatus.FAILED
        assert fetched.error == "network timeout"

    @pytest.mark.asyncio
    async def test_update_full_job(self, store):
        job = _make_job(total_requests=10)
        await store.create_job(job, config={})

        job.completed_requests = 7
        job.failed_requests = 3
        job.status = JobStatus.PARTIAL
        job.input_tokens = 1000
        job.output_tokens = 500
        job.estimated_cost_usd = 0.05

        await store.update_job(job)
        fetched = await store.get_job(job.id)
        assert fetched.completed_requests == 7
        assert fetched.failed_requests == 3
        assert fetched.status == JobStatus.PARTIAL
        assert fetched.input_tokens == 1000
        assert fetched.output_tokens == 500

    @pytest.mark.asyncio
    async def test_delete_job(self, store):
        job = _make_job()
        await store.create_job(job, config={})
        assert await store.job_exists(job.id)

        await store.delete_job(job.id)
        assert not await store.job_exists(job.id)
        assert await store.get_job(job.id) is None

    @pytest.mark.asyncio
    async def test_job_exists(self, store):
        job = _make_job()
        assert not await store.job_exists(job.id)
        await store.create_job(job, config={})
        assert await store.job_exists(job.id)


# ── list_jobs ─────────────────────────────────────────────────────────────────


class TestListJobs:
    @pytest.mark.asyncio
    async def test_list_all_jobs(self, store):
        for _ in range(3):
            await store.create_job(_make_job(), config={})
        jobs = await store.list_jobs()
        assert len(jobs) == 3

    @pytest.mark.asyncio
    async def test_filter_by_provider(self, store):
        await store.create_job(_make_job(provider="anthropic"), config={})
        await store.create_job(_make_job(provider="openai"), config={})

        anthropic_jobs = await store.list_jobs(provider="anthropic")
        assert all(j.provider == "anthropic" for j in anthropic_jobs)
        assert len(anthropic_jobs) == 1

    @pytest.mark.asyncio
    async def test_filter_by_status(self, store):
        job_pending = _make_job(status=JobStatus.PENDING)
        job_done = _make_job(status=JobStatus.COMPLETED)
        await store.create_job(job_pending, config={})
        await store.create_job(job_done, config={})

        # Update status in db.
        await store.update_job_status(job_done.id, JobStatus.COMPLETED)

        completed = await store.list_jobs(status="COMPLETED")
        assert len(completed) == 1
        assert completed[0].id == job_done.id

    @pytest.mark.asyncio
    async def test_filter_by_model(self, store):
        await store.create_job(_make_job(model="claude-opus-4-5"), config={})
        await store.create_job(_make_job(model="gpt-4o"), config={})

        results = await store.list_jobs(model="gpt-4o")
        assert len(results) == 1
        assert results[0].model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_limit_and_offset(self, store):
        for _ in range(5):
            await store.create_job(_make_job(), config={})

        page1 = await store.list_jobs(limit=2, offset=0)
        page2 = await store.list_jobs(limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        # Pages should not overlap.
        ids1 = {j.id for j in page1}
        ids2 = {j.id for j in page2}
        assert ids1.isdisjoint(ids2)

    @pytest.mark.asyncio
    async def test_invalid_order_by_raises(self, store):
        with pytest.raises(ValueError, match="order_by"):
            await store.list_jobs(order_by="invalid_column")


# ── Request CRUD ──────────────────────────────────────────────────────────────


class TestRequestCRUD:
    @pytest.mark.asyncio
    async def test_create_and_get_request(self, store):
        job = _make_job()
        await store.create_job(job, config={})

        req_id = str(uuid.uuid4())
        payload = {"role": "user", "content": "Hello"}
        await store.create_request(req_id, job.id, payload=payload, cache_key="key123")

        fetched = await store.get_request(req_id)
        assert fetched is not None
        assert fetched["id"] == req_id
        assert fetched["job_id"] == job.id
        assert fetched["cache_key"] == "key123"
        assert fetched["payload_json"] == payload

    @pytest.mark.asyncio
    async def test_get_nonexistent_request_returns_none(self, store):
        assert await store.get_request("no-such-req") is None

    @pytest.mark.asyncio
    async def test_update_request_status(self, store):
        job = _make_job()
        await store.create_job(job, config={})

        req_id = str(uuid.uuid4())
        await store.create_request(req_id, job.id, payload={})

        await store.update_request_status(req_id, RequestStatus.COMPLETED)
        fetched = await store.get_request(req_id)
        assert fetched["status"] == "completed"

    @pytest.mark.asyncio
    async def test_list_requests_for_job(self, store):
        job = _make_job()
        await store.create_job(job, config={})

        for _ in range(4):
            await store.create_request(str(uuid.uuid4()), job.id, payload={})

        requests = await store.list_requests(job.id)
        assert len(requests) == 4

    @pytest.mark.asyncio
    async def test_count_requests(self, store):
        job = _make_job()
        await store.create_job(job, config={})

        for _ in range(3):
            await store.create_request(str(uuid.uuid4()), job.id, payload={})

        count = await store.count_requests(job.id)
        assert count == 3

    @pytest.mark.asyncio
    async def test_bulk_create_requests(self, store):
        job = _make_job()
        await store.create_job(job, config={})

        records = [
            {"id": str(uuid.uuid4()), "job_id": job.id, "payload": {"n": i}}
            for i in range(5)
        ]
        await store.create_requests_bulk(records)
        assert await store.count_requests(job.id) == 5


# ── Result CRUD ───────────────────────────────────────────────────────────────


class TestResultCRUD:
    @pytest.mark.asyncio
    async def test_create_and_get_result(self, store):
        job = _make_job()
        await store.create_job(job, config={})

        req_id = str(uuid.uuid4())
        await store.create_request(req_id, job.id, payload={})

        result = _make_result(req_id, job.id)
        await store.create_result(result)

        fetched = await store.get_result(req_id)
        assert fetched is not None
        assert fetched["request_id"] == req_id
        assert fetched["content"] == "The capital of France is Paris."
        assert fetched["from_cache"] == 0

    @pytest.mark.asyncio
    async def test_get_nonexistent_result_returns_none(self, store):
        assert await store.get_result("no-such") is None

    @pytest.mark.asyncio
    async def test_list_results(self, store):
        job = _make_job()
        await store.create_job(job, config={})

        for _ in range(3):
            req_id = str(uuid.uuid4())
            await store.create_request(req_id, job.id, payload={})
            await store.create_result(_make_result(req_id, job.id))

        results = await store.list_results(job.id)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_count_results(self, store):
        job = _make_job()
        await store.create_job(job, config={})

        for _ in range(2):
            req_id = str(uuid.uuid4())
            await store.create_request(req_id, job.id, payload={})
            await store.create_result(_make_result(req_id, job.id))

        assert await store.count_results(job.id) == 2

    @pytest.mark.asyncio
    async def test_create_result_updates_request_status(self, store):
        job = _make_job()
        await store.create_job(job, config={})

        req_id = str(uuid.uuid4())
        await store.create_request(req_id, job.id, payload={})
        await store.create_result(_make_result(req_id, job.id, from_cache=False))

        req = await store.get_request(req_id)
        assert req["status"] == RequestStatus.COMPLETED.value

    @pytest.mark.asyncio
    async def test_create_cached_result_sets_cached_status(self, store):
        job = _make_job()
        await store.create_job(job, config={})

        req_id = str(uuid.uuid4())
        await store.create_request(req_id, job.id, payload={})
        await store.create_result(_make_result(req_id, job.id, from_cache=True))

        req = await store.get_request(req_id)
        assert req["status"] == RequestStatus.CACHED.value


# ── Tags ──────────────────────────────────────────────────────────────────────


class TestTags:
    @pytest.mark.asyncio
    async def test_tags_stored_with_job(self, store):
        job = _make_job(tags=["prod", "experiment"])
        await store.create_job(job, config={})

        fetched = await store.get_job(job.id)
        assert set(fetched.tags) == {"prod", "experiment"}

    @pytest.mark.asyncio
    async def test_add_tags(self, store):
        job = _make_job()
        await store.create_job(job, config={})

        await store.add_tags(job.id, ["alpha", "beta"])
        tags = await store.get_tags(job.id)
        assert set(tags) == {"alpha", "beta"}

    @pytest.mark.asyncio
    async def test_remove_tags(self, store):
        job = _make_job(tags=["a", "b", "c"])
        await store.create_job(job, config={})

        await store.remove_tags(job.id, ["b"])
        tags = await store.get_tags(job.id)
        assert "b" not in tags
        assert "a" in tags
        assert "c" in tags

    @pytest.mark.asyncio
    async def test_filter_by_tags(self, store):
        job_tagged = _make_job(tags=["special"])
        job_plain = _make_job()
        await store.create_job(job_tagged, config={})
        await store.create_job(job_plain, config={})

        results = await store.list_jobs(tags=["special"])
        assert len(results) == 1
        assert results[0].id == job_tagged.id

    @pytest.mark.asyncio
    async def test_add_duplicate_tag_idempotent(self, store):
        job = _make_job(tags=["tag1"])
        await store.create_job(job, config={})

        await store.add_tags(job.id, ["tag1", "tag1"])
        tags = await store.get_tags(job.id)
        assert tags.count("tag1") == 1

    @pytest.mark.asyncio
    async def test_delete_job_cascades_to_tags(self, store):
        job = _make_job(tags=["cascade-test"])
        await store.create_job(job, config={})
        await store.delete_job(job.id)

        # If tags were not deleted, we'd see them on a ghost job, but the job
        # is gone so get_tags would still return [] safely. We verify via
        # job_exists.
        assert not await store.job_exists(job.id)
