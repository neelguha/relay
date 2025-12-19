"""Base provider class for LLM batch API providers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pathlib import Path
from relay.models import BatchRequest, BatchJob


class BaseProvider(ABC):
    """Abstract base class for LLM batch API providers.
    
    All provider implementations must inherit from this class and implement
    the abstract methods to support batch operations.
    """
    
    def __init__(
        self,
        api_key: str,
        **kwargs: Any,
    ) -> None:
        """Initialize the provider with API credentials.
        
        Args:
            api_key: API key for authenticating with the provider's API
            **kwargs: Additional provider-specific configuration options
        """
        pass
    
    @abstractmethod
    def submit_batch(
        self,
        requests: List[BatchRequest],
    ) -> BatchJob:
        """Submit a batch of requests to the provider's API.
        
        Args:
            requests: A list of BatchRequest objects to process in the batch
            
        Returns:
            BatchJob object containing the job_id, submitted_at timestamp,
            and initial status
            
        Raises:
            ValueError: If requests list is empty or contains invalid requests
            APIError: If the batch submission fails
        """
        pass
    
    @abstractmethod
    def monitor_batch(
        self,
        job_id: str,
    ) -> BatchJob:
        """Check the progress of a batch job.
        
        Args:
            job_id: The unique identifier of the batch job to monitor
            
        Returns:
            BatchJob object with updated status and metadata
            
        Raises:
            ValueError: If job_id is invalid or not found
            APIError: If the status check fails
        """
        pass
    
    @abstractmethod
    def retrieve_batch_results(
        self,
        job_id: str,
    ) -> List[Dict[str, Any]]:
        """Retrieve the results of a completed batch job.
        
        Args:
            job_id: The unique identifier of the batch job
            
        Returns:
            List of result dictionaries, one per request in the batch
            
        Raises:
            ValueError: If job_id is invalid or not found
            APIError: If the results retrieval fails
        """
        pass
    
    @abstractmethod
    def cancel_batch(
        self,
        job_id: str,
    ) -> bool:
        """Cancel a batch job.
        
        Args:
            job_id: The unique identifier of the batch job to cancel
            
        Returns:
            True if the job was successfully cancelled, False otherwise
            
        Raises:
            ValueError: If job_id is invalid or not found
            APIError: If the cancellation request fails
        """
        pass
