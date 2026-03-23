"""Data models for the relay library.

Provides both frozen dataclasses and Pydantic v2 model variants for all
core types: BatchRequest, BatchConfig, BatchJob, BatchResult, and supporting types.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────────


class JobStatus(str, Enum):
    """Job lifecycle states. Terminal states never transition again."""

    PENDING = "PENDING"
    CACHE_RESOLVING = "CACHE_RESOLVING"
    VALIDATING = "VALIDATING"
    SUBMITTING = "SUBMITTING"
    IN_PROGRESS = "IN_PROGRESS"
    DOWNLOADING = "DOWNLOADING"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        return self in {
            JobStatus.COMPLETED,
            JobStatus.PARTIAL,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }


class RequestStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"


# ── Message type ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Message:
    """A single message in a conversation. Text-only content."""

    role: str  # 'user' | 'assistant' | 'system'
    content: str


# ── Core Dataclasses ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BatchRequest:
    """A single request within a batch job."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[dict[str, str]] = field(default_factory=list)
    system: str | None = None
    model: str | None = None
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float | None = None
    stop_sequences: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass
class BatchConfig:
    """Configuration for a batch job submission."""

    provider: str  # 'anthropic' | 'openai' | 'google' | 'xai'
    model: str
    name: str | None = None  # Human-readable job name for easy lookup
    project: str | None = None
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    use_cache: bool = True
    cache_ttl_override: int | None = None
    output_dir: str | None = None
    on_progress: Callable | None = None
    on_complete: Callable | None = None


@dataclass(frozen=True)
class BatchError:
    """Error information for a failed individual request."""

    code: str
    message: str
    retryable: bool = False


@dataclass(frozen=True)
class BatchResult:
    """Result for a single request within a batch."""

    request_id: str
    job_id: str
    content: str
    stop_reason: str
    input_tokens: int
    output_tokens: int
    model: str
    from_cache: bool
    cached_at: datetime | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)
    error: BatchError | None = None


@dataclass
class BatchJob:
    """A batch job tracked by relay."""

    id: str
    provider_job_id: str
    provider: str
    model: str
    project: str | None
    status: JobStatus
    total_requests: int
    name: str | None = None
    completed_requests: int = 0
    failed_requests: int = 0
    cached_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    actual_cost_usd: float | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    submitted_at: datetime | None = None
    completed_at: datetime | None = None
    tags: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class JobProgress:
    """Progress snapshot for a running job."""

    job_id: str
    status: JobStatus
    total: int
    completed: int
    failed: int
    cached: int
    cost_so_far: float
    elapsed_seconds: float
    eta_seconds: float | None = None


@dataclass(frozen=True)
class CostEstimate:
    """Cost estimation result."""

    total_requests: int
    cache_hits: int
    net_requests: int
    input_tokens: int
    estimated_output_tokens: int
    gross_usd: float
    saved_usd: float
    net_usd: float
    per_provider: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CostSummaryRow:
    """A single row in a cost summary report."""

    group: str  # provider name, model name, or project name
    total_usd: float
    input_tokens: int
    output_tokens: int


# ── Provider-internal types ────────────────────────────────────────────────────


@dataclass
class ProviderStatus:
    """Status information returned by a provider adapter."""

    provider_job_id: str
    status: str  # Provider-native status string
    completed: int = 0
    failed: int = 0
    total: int = 0
    error: str | None = None


# ── Pydantic v2 Variants ──────────────────────────────────────────────────────


class BatchRequestModel(BaseModel):
    """Pydantic v2 variant of BatchRequest for validation and serialization."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[dict[str, str]] = Field(default_factory=list)
    system: str | None = None
    model: str | None = None
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float | None = None
    stop_sequences: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    def to_dataclass(self) -> BatchRequest:
        return BatchRequest(**self.model_dump())


class BatchConfigModel(BaseModel):
    """Pydantic v2 variant of BatchConfig."""

    provider: str
    model: str
    project: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    use_cache: bool = True
    cache_ttl_override: int | None = None
    output_dir: str | None = None


class BatchJobModel(BaseModel):
    """Pydantic v2 variant of BatchJob."""

    id: str
    provider_job_id: str
    provider: str
    model: str
    project: str | None = None
    name: str | None = None
    status: JobStatus
    total_requests: int
    completed_requests: int = 0
    failed_requests: int = 0
    cached_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    actual_cost_usd: float | None = None
    created_at: datetime
    submitted_at: datetime | None = None
    completed_at: datetime | None = None
    tags: list[str] = Field(default_factory=list)
    error: str | None = None


class BatchResultModel(BaseModel):
    """Pydantic v2 variant of BatchResult."""

    request_id: str
    job_id: str
    content: str
    stop_reason: str
    input_tokens: int
    output_tokens: int
    model: str
    from_cache: bool
    cached_at: datetime | None = None
    raw_response: dict[str, Any] = Field(default_factory=dict)
