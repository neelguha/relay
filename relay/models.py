"""Data models for Relay."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime

@dataclass
class BatchRequest:
    """Represents a single request in a batch job.
    
    Attributes:
        id: Unique identifier for this request
        model: The model name to use (e.g., "gpt-4")
        system_prompt: The system prompt to use
        prompt: The user prompt/query
        provider_args: Additional provider-specific arguments
    """
    
    id: str
    model: str
    system_prompt: str
    prompt: str
    provider_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchJob:
    """Represents a batch job submitted to the API.
    
    Attributes:
        job_id: Unique identifier for the batch job
        submitted_at: Timestamp when the job was submitted
        status: Current status of the job
        n_requests: Total number of requests in the batch
        completed_requests: Number of requests that have completed
        failed_requests: Number of requests that have failed
    """
    
    job_id: str
    submitted_at: datetime
    status: str
    n_requests: int
    completed_requests: int = 0
    failed_requests: int = 0
