"""Relay - A Python package for batch API calls to commercial LLM APIs."""

__version__ = "0.1.0"

from relay.client import RelayClient
from relay.models import BatchRequest, BatchJob

__all__ = ["RelayClient", "BatchRequest", "BatchJob", "__version__"]
