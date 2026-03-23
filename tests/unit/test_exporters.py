"""Unit tests for JSONL and CSV exporters."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import pytest

from relay.exporters.csv import _result_to_row, export_csv
from relay.exporters.jsonl import _result_to_dict, export_jsonl
from relay.models import BatchError, BatchResult


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_result(**kwargs) -> BatchResult:
    defaults = dict(
        request_id="req-001",
        job_id="job-001",
        content="The answer is 42.",
        stop_reason="end_turn",
        input_tokens=15,
        output_tokens=8,
        model="claude-opus-4-5",
        from_cache=False,
        cached_at=None,
        raw_response={"id": "resp-1"},
        error=None,
    )
    defaults.update(kwargs)
    return BatchResult(**defaults)


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── _result_to_dict ───────────────────────────────────────────────────────────


class TestResultToDict:
    def test_base_fields_present(self):
        d = _result_to_dict(_make_result(), include_metadata=False, include_raw_response=False)
        assert d["request_id"] == "req-001"
        assert d["job_id"] == "job-001"
        assert d["content"] == "The answer is 42."
        assert d["stop_reason"] == "end_turn"
        assert d["input_tokens"] == 15
        assert d["output_tokens"] == 8
        assert d["model"] == "claude-opus-4-5"

    def test_metadata_excluded_when_false(self):
        d = _result_to_dict(_make_result(), include_metadata=False, include_raw_response=False)
        assert "from_cache" not in d
        assert "cached_at" not in d

    def test_metadata_included_when_true(self):
        d = _result_to_dict(_make_result(), include_metadata=True, include_raw_response=False)
        assert "from_cache" in d
        assert "cached_at" in d
        assert d["from_cache"] is False
        assert d["cached_at"] is None

    def test_metadata_cached_at_iso_format(self):
        ts = datetime(2025, 1, 15, 12, 0, 0)
        result = _make_result(from_cache=True, cached_at=ts)
        d = _result_to_dict(result, include_metadata=True, include_raw_response=False)
        assert d["cached_at"] == ts.isoformat()

    def test_raw_response_excluded_when_false(self):
        d = _result_to_dict(_make_result(), include_metadata=False, include_raw_response=False)
        assert "raw_response" not in d

    def test_raw_response_included_when_true(self):
        d = _result_to_dict(_make_result(), include_metadata=False, include_raw_response=True)
        assert d["raw_response"] == {"id": "resp-1"}

    def test_error_none_when_no_error(self):
        d = _result_to_dict(_make_result(), include_metadata=False, include_raw_response=False)
        assert d["error"] is None

    def test_error_serialized_when_present(self):
        err = BatchError(code="rate_limit", message="Too many requests", retryable=True)
        result = _make_result(error=err)
        d = _result_to_dict(result, include_metadata=False, include_raw_response=False)
        assert d["error"]["code"] == "rate_limit"
        assert d["error"]["message"] == "Too many requests"
        assert d["error"]["retryable"] is True


# ── export_jsonl ──────────────────────────────────────────────────────────────


class TestExportJsonl:
    @pytest.mark.asyncio
    async def test_write_and_read_back(self, tmp_path):
        results = [
            _make_result(request_id="r1", content="Hello"),
            _make_result(request_id="r2", content="World"),
        ]
        out = tmp_path / "out.jsonl"
        await export_jsonl(results, str(out))

        records = _read_jsonl(out)
        assert len(records) == 2
        assert records[0]["request_id"] == "r1"
        assert records[0]["content"] == "Hello"
        assert records[1]["request_id"] == "r2"
        assert records[1]["content"] == "World"

    @pytest.mark.asyncio
    async def test_each_line_valid_json(self, tmp_path):
        results = [_make_result() for _ in range(5)]
        out = tmp_path / "multi.jsonl"
        await export_jsonl(results, str(out))

        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 5
        for line in lines:
            obj = json.loads(line)
            assert "request_id" in obj

    @pytest.mark.asyncio
    async def test_empty_results_empty_file(self, tmp_path):
        out = tmp_path / "empty.jsonl"
        await export_jsonl([], str(out))
        assert out.read_text(encoding="utf-8") == ""

    @pytest.mark.asyncio
    async def test_include_metadata_true_by_default(self, tmp_path):
        out = tmp_path / "meta.jsonl"
        await export_jsonl([_make_result()], str(out))
        record = _read_jsonl(out)[0]
        assert "from_cache" in record
        assert "cached_at" in record

    @pytest.mark.asyncio
    async def test_exclude_metadata(self, tmp_path):
        out = tmp_path / "nometa.jsonl"
        await export_jsonl([_make_result()], str(out), include_metadata=False)
        record = _read_jsonl(out)[0]
        assert "from_cache" not in record
        assert "cached_at" not in record

    @pytest.mark.asyncio
    async def test_include_raw_response(self, tmp_path):
        out = tmp_path / "raw.jsonl"
        await export_jsonl([_make_result()], str(out), include_raw_response=True)
        record = _read_jsonl(out)[0]
        assert "raw_response" in record
        assert record["raw_response"] == {"id": "resp-1"}

    @pytest.mark.asyncio
    async def test_with_error_field(self, tmp_path):
        err = BatchError(code="timeout", message="Request timed out", retryable=False)
        result = _make_result(error=err)
        out = tmp_path / "err.jsonl"
        await export_jsonl([result], str(out))
        record = _read_jsonl(out)[0]
        assert record["error"]["code"] == "timeout"

    @pytest.mark.asyncio
    async def test_file_is_created(self, tmp_path):
        out = tmp_path / "created.jsonl"
        assert not out.exists()
        await export_jsonl([_make_result()], str(out))
        assert out.exists()


# ── _result_to_row ────────────────────────────────────────────────────────────


class TestResultToRow:
    def test_base_columns_present(self):
        row = _result_to_row(_make_result(), include_metadata=False)
        for col in ["request_id", "job_id", "content", "stop_reason",
                    "input_tokens", "output_tokens", "model", "error"]:
            assert col in row

    def test_no_metadata_columns_when_false(self):
        row = _result_to_row(_make_result(), include_metadata=False)
        assert "from_cache" not in row
        assert "cached_at" not in row

    def test_metadata_columns_when_true(self):
        row = _result_to_row(_make_result(), include_metadata=True)
        assert "from_cache" in row
        assert "cached_at" in row

    def test_error_serialized_as_json_string(self):
        err = BatchError(code="err", message="oops", retryable=False)
        row = _result_to_row(_make_result(error=err), include_metadata=False)
        assert isinstance(row["error"], str)
        parsed = json.loads(row["error"])
        assert parsed["code"] == "err"


# ── export_csv ────────────────────────────────────────────────────────────────


class TestExportCsv:
    @pytest.mark.asyncio
    async def test_write_and_read_back(self, tmp_path):
        results = [
            _make_result(request_id="r1", content="Alpha"),
            _make_result(request_id="r2", content="Beta"),
        ]
        out = tmp_path / "out.csv"
        await export_csv(results, str(out))

        rows = []
        with out.open(encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["request_id"] == "r1"
        assert rows[0]["content"] == "Alpha"
        assert rows[1]["request_id"] == "r2"

    @pytest.mark.asyncio
    async def test_has_header_row(self, tmp_path):
        out = tmp_path / "header.csv"
        await export_csv([_make_result()], str(out))

        with out.open(encoding="utf-8-sig") as fh:
            reader = csv.reader(fh)
            header = next(reader)

        assert "request_id" in header
        assert "content" in header
        assert "model" in header

    @pytest.mark.asyncio
    async def test_metadata_columns_present_by_default(self, tmp_path):
        out = tmp_path / "meta.csv"
        await export_csv([_make_result()], str(out))

        with out.open(encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert "from_cache" in rows[0]
        assert "cached_at" in rows[0]

    @pytest.mark.asyncio
    async def test_metadata_excluded(self, tmp_path):
        out = tmp_path / "nometa.csv"
        await export_csv([_make_result()], str(out), include_metadata=False)

        with out.open(encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert "from_cache" not in rows[0]
        assert "cached_at" not in rows[0]

    @pytest.mark.asyncio
    async def test_empty_results_only_header(self, tmp_path):
        out = tmp_path / "empty.csv"
        await export_csv([], str(out))

        with out.open(encoding="utf-8-sig") as fh:
            reader = csv.reader(fh)
            all_rows = list(reader)

        # Only the header row.
        assert len(all_rows) == 1

    @pytest.mark.asyncio
    async def test_file_created(self, tmp_path):
        out = tmp_path / "new.csv"
        assert not out.exists()
        await export_csv([_make_result()], str(out))
        assert out.exists()

    @pytest.mark.asyncio
    async def test_multiple_rows_correct_count(self, tmp_path):
        results = [_make_result(request_id=f"r{i}") for i in range(10)]
        out = tmp_path / "many.csv"
        await export_csv(results, str(out))

        with out.open(encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 10
