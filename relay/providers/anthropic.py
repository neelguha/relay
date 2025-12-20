"""Anthropic provider implementation for batch API calls."""

from typing import List, Dict, Any
from pathlib import Path
from relay.providers.base import BaseProvider
from relay.models import BatchRequest, BatchJob

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from datetime import datetime



class AnthropicProvider(BaseProvider):
    """Provider implementation for Anthropic's batch API.
    
    Handles batch submissions, monitoring, and result retrieval for Anthropic's
    batch processing endpoints.
    """
    
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """Initialize the Anthropic provider.
        
        Args:
            **kwargs: Additional Anthropic-specific configuration options
        """
        self.client = anthropic.Anthropic()
    
    def submit_batch(
        self,
        requests: List[BatchRequest],
    ) -> BatchJob:
        """Submit a batch of requests to Anthropic's batch API.
        
        Submits requests to Anthropic's batch endpoint and returns
        a BatchJob with the Anthropic batch ID.
        
        Args:
            requests: A list of BatchRequest objects to process in the batch
            
        Returns:
            BatchJob object containing the Anthropic batch ID, submitted_at timestamp,
            and initial status
            
        Raises:
            ValueError: If requests list is empty or contains invalid requests
            APIError: If the batch submission fails (e.g., authentication error,
                     rate limit exceeded, invalid API key)
        """
        # Convert the requests to the Anthropic format
        requests_anthropic = []
        for request in requests:
            messages = []
            messages.append({
                "role": "user",
                "content": request.prompt,
            })
            formatted_args = {}
            for key, value in request.provider_args.items():
                if key == "thinking_budget_tokens":
                    formatted_args["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": value,
                    }
                    assert value >= 1024, "Thinking budget tokens must be at least 1024"
                else:
                    formatted_args[key] = value


            requests_anthropic.append(Request(
                custom_id=request.id,
                params=MessageCreateParamsNonStreaming(
                    model=request.model,
                    system=request.system_prompt if request.system_prompt else None,
                    max_tokens=request.max_tokens,
                    messages=messages,
                    **formatted_args
                ),
            ))
        batch_object = self.client.messages.batches.create(
            requests=requests_anthropic,
        )
        return BatchJob(
            job_id=batch_object.id,
            submitted_at=datetime.now(),
            status=batch_object.processing_status,
            n_requests=len(requests),
        )
        
    
    def monitor_batch(
        self,
        job_id: str,
    ) -> BatchJob:
        """Check the progress of an Anthropic batch job.
        
        Queries Anthropic's batch status endpoint to get current status.
        
        Args:
            job_id: The Anthropic batch ID to monitor
            
        Returns:
            BatchJob object with updated status and metadata from Anthropic
            
        Raises:
            ValueError: If job_id is invalid or not found
            APIError: If the status check fails
        """
        batch_object = self.client.messages.batches.retrieve(job_id)

        return BatchJob(
            job_id=batch_object.id,
            submitted_at=batch_object.created_at,
            status=batch_object.processing_status,
            n_requests=batch_object.request_counts.processing,
            completed_requests=batch_object.request_counts.succeeded,
            failed_requests=batch_object.request_counts.errored,
        )
    
    def retrieve_batch_results(
        self,
        job_id: str,
    ) -> List[Dict[str, Any]]:
        """Retrieve the results of a completed Anthropic batch job.
        
        Returns the results of a completed batch job as a list of dictionaries.
        
        Args:
            job_id: The Anthropic batch ID
            
        Returns:
            List of result dictionaries, one per request in the batch
            
        Raises:
            ValueError: If job_id is invalid or not found
            APIError: If the results retrieval fails (e.g., batch not completed)
        """
        batch_object = self.client.messages.batches.retrieve(job_id)
        if batch_object.processing_status != "ended":
            raise ValueError(f"Batch job {job_id} is not completed (status: {batch_object['processing_status']})")
        
        results = []
        for result in self.client.messages.batches.results(job_id):
            results.append(result.to_dict())
        return results
    
    def cancel_batch(
        self,
        job_id: str,
    ) -> bool:
        """Cancel an Anthropic batch job.
        
        Cancels a batch job by calling Anthropic's batch cancellation endpoint.
        
        Args:
            job_id: The Anthropic batch ID to cancel
            
        Returns:
            True if the job was successfully cancelled, False otherwise
            
        Raises:
            ValueError: If job_id is invalid or not found
            APIError: If the cancellation request fails
        """
        self.client.messages.batches.cancel(job_id)
        return True
    
    def format_request(
        self,
        request: BatchRequest,
    ) -> Dict[str, Any]:
        """Format a BatchRequest into Anthropic's batch API format.
        
        Converts a BatchRequest into the format expected by Anthropic's batch API,
        including the message structure with system and user content.
        
        Args:
            request: The BatchRequest to format
            
        Returns:
            Dictionary with Anthropic API format containing the request data
            in the structure expected by Anthropic's batch endpoint (e.g., messages
            array with system and user roles for Claude models)
            
        Raises:
            ValueError: If the request cannot be formatted (e.g., missing required fields)
        """
        pass
