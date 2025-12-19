"""Together AI provider implementation for batch API calls."""

from typing import List, Dict, Any
from pathlib import Path
from relay.providers.base import BaseProvider
from relay.models import BatchRequest, BatchJob


class TogetherProvider(BaseProvider):
    """Provider implementation for Together AI's batch API.
    
    Handles batch submissions, monitoring, and result retrieval for Together AI's
    batch processing endpoints.
    """
    
    def __init__(
        self,
        api_key: str,
        **kwargs: Any,
    ) -> None:
        """Initialize the Together AI provider.
        
        Args:
            api_key: Together AI API key for authentication
            **kwargs: Additional Together AI-specific configuration options
        """
        pass
    
    def submit_batch(
        self,
        requests: List[BatchRequest],
    ) -> BatchJob:
        """Submit a batch of requests to Together AI's batch API.
        
        Submits requests to Together AI's batch endpoint and returns
        a BatchJob with the Together batch ID.
        
        Args:
            requests: A list of BatchRequest objects to process in the batch
            
        Returns:
            BatchJob object containing the Together batch ID, submitted_at timestamp,
            and initial status
            
        Raises:
            ValueError: If requests list is empty or contains invalid requests
            APIError: If the batch submission fails (e.g., authentication error,
                     rate limit exceeded, invalid API key)
        """
        pass
    
    def monitor_batch(
        self,
        job_id: str,
    ) -> BatchJob:
        """Check the progress of a Together AI batch job.
        
        Queries Together AI's batch status endpoint to get current status.
        
        Args:
            job_id: The Together AI batch ID to monitor
            
        Returns:
            BatchJob object with updated status and metadata from Together AI
            
        Raises:
            ValueError: If job_id is invalid or not found
            APIError: If the status check fails
        """
        pass
    
    def retrieve_batch_results(
        self,
        job_id: str,
    ) -> List[Dict[str, Any]]:
        """Retrieve the results of a completed Together AI batch job.
        
        Returns the results of a completed batch job as a list of dictionaries.
        
        Args:
            job_id: The Together AI batch ID
            
        Returns:
            List of result dictionaries, one per request in the batch
            
        Raises:
            ValueError: If job_id is invalid or not found
            APIError: If the results retrieval fails (e.g., batch not completed)
        """
        pass
    
    def cancel_batch(
        self,
        job_id: str,
    ) -> bool:
        """Cancel a Together AI batch job.
        
        Cancels a batch job by calling Together AI's batch cancellation endpoint.
        
        Args:
            job_id: The Together AI batch ID to cancel
            
        Returns:
            True if the job was successfully cancelled, False otherwise
            
        Raises:
            ValueError: If job_id is invalid or not found
            APIError: If the cancellation request fails
        """
        pass
    
    def format_request(
        self,
        request: BatchRequest,
    ) -> Dict[str, Any]:
        """Format a BatchRequest into Together AI's batch API format.
        
        Converts a BatchRequest into the format expected by Together AI's batch API.
        
        Args:
            request: The BatchRequest to format
            
        Returns:
            Dictionary with Together AI API format containing the request data
            in the structure expected by Together AI's batch endpoint
            
        Raises:
            ValueError: If the request cannot be formatted (e.g., missing required fields)
        """
        pass
