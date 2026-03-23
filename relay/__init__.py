"""relay — Multi-provider LLM batch prediction library.

Provides a unified, provider-agnostic interface for submitting, managing,
monitoring, and downloading results from large-scale text-only batch
prediction jobs across Anthropic, OpenAI, Google, and XAI.
"""

from relay.client import BatchClient
from relay.fan_out import fan_out
from relay.models import (
    BatchConfig,
    BatchError,
    BatchJob,
    BatchRequest,
    BatchResult,
    CostEstimate,
    CostSummaryRow,
    JobProgress,
    JobStatus,
    Message,
    ProviderStatus,
    RequestStatus,
)

__version__ = "1.0.0"

__all__ = [
    "BatchClient",
    "BatchConfig",
    "BatchError",
    "BatchJob",
    "BatchRequest",
    "BatchResult",
    "CostEstimate",
    "CostSummaryRow",
    "JobProgress",
    "JobStatus",
    "Message",
    "ProviderStatus",
    "RequestStatus",
    "fan_out",
]
