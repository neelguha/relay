"""OpenAI provider implementation for batch API calls."""

from typing import List, Dict, Any
from pathlib import Path
from relay.providers.base import BaseProvider
from relay.models import BatchRequest, BatchJob
from openai import OpenAI
import os
import json
from datetime import datetime

VALID_PROVIDER_ARGS = ["max_output_tokens", "temperature", "reasoning_effort"]


class OpenAIProvider(BaseProvider):
    """Provider implementation for OpenAI's batch API.
    
    Handles batch submissions, monitoring, and result retrieval for OpenAI's
    batch processing endpoints.
    """
    
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenAI provider.
        
        Args:
            **kwargs: Additional OpenAI-specific configuration options
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def submit_batch(
        self,
        requests: List[BatchRequest],
    ) -> BatchJob:
        """Submit a batch of requests to OpenAI's batch API.
        
        Submits requests to OpenAI's Batch API and returns a BatchJob with the OpenAI batch ID. The OpenAI Batch API requires we save the requests to a JSONL file, and then submit that file to the API.

        Right now, this defaults to using the Responses API.
        
        Args:
            requests: A list of BatchRequest objects to process in the batch
            
        Returns:
            BatchJob object containing the OpenAI batch ID, submitted_at timestamp,
            and initial status (e.g., "validating", "in_progress")
            
        Raises:
            ValueError: If requests list is empty or contains invalid requests
            APIError: If the batch submission fails (e.g., authentication error,
                     rate limit exceeded, invalid API key)
        """
        # Save the requests to a JSONL file
        jsonl_file = "requests.jsonl"
        with open(jsonl_file, "w") as f:
            for request in requests:
                request_dict = convert_request_to_dict(request)
                f.write(json.dumps(request_dict) + "\n")
        
        # Submit the file to the API
        batch_input_file = self.client.files.create(
            file=open(jsonl_file, "rb"),
            purpose="batch",
        )
        
        batch_input_file_id = batch_input_file.id
        batch_object = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={}
        )
        return BatchJob(
            job_id=batch_object.id,
            submitted_at=datetime.now(),
            status=batch_object.status,
            n_requests=len(requests),
            completed_requests=batch_object.request_counts.completed,
            failed_requests=batch_object.request_counts.failed,
        )
    
    def monitor_batch(
        self,
        job_id: str,
    ) -> BatchJob:
        """Check the progress of an OpenAI batch job.
        
        Queries OpenAI's /v1/batches/{batch_id} endpoint to get current status.
        Status can be: "validating", "in_progress", "finalizing", "completed",
        "failed", "expired", "cancelling", or "cancelled".
        
        Args:
            job_id: The OpenAI batch ID to monitor
            
        Returns:
            BatchJob object with updated status and metadata from OpenAI
            
        Raises:
            ValueError: If job_id is invalid or not found
            APIError: If the status check fails
        """
        batch = self.client.batches.retrieve(job_id)
        return BatchJob(
            job_id=job_id,
            submitted_at=batch.created_at,
            status=batch.status,
            n_requests=batch.request_counts.total,
            completed_requests=batch.request_counts.completed,
            failed_requests=batch.request_counts.failed,
        )
    
    def retrieve_batch_results(
        self,
        job_id: str,
    ) -> List[Dict[str, Any]]:
        """Retrieve the results of a completed OpenAI batch job.
        
        Returns the results of a completed batch job as a list of dictionaries.
        
        Args:
            job_id: The OpenAI batch ID
            
        Returns:
            List of result dictionaries, one per request in the batch
            
        Raises:
            ValueError: If job_id is invalid or not found, or batch is not completed
            APIError: If the results retrieval fails
        """
        batch = self.client.batches.retrieve(job_id)
        if batch.status != "completed":
            raise ValueError(f"Batch job {job_id} is not completed (status: {batch.status})")
        
        output_file_id = batch.output_file_id
        file_response = self.client.files.content(output_file_id)
        text = file_response.text
        outputs = []  # list of response objects
        for line in text.split("\n"):
            if line.strip():
                response = json.loads(line)
                outputs.append(response)
        return outputs
    
    def cancel_batch(
        self,
        job_id: str,
    ) -> bool:
        """Cancel an OpenAI batch job.
        
        Cancels a batch job by calling OpenAI's /v1/batches/{batch_id}/cancel endpoint.
        Only batches in "validating" or "in_progress" status can be cancelled.
        
        Args:
            job_id: The OpenAI batch ID to cancel
            
        Returns:
            True if the job was successfully cancelled, False otherwise
            
        Raises:
            ValueError: If job_id is invalid or not found
            APIError: If the cancellation request fails
        """
        batch = self.client.batches.cancel(job_id)
    


def convert_request_to_dict(request: BatchRequest) -> Dict[str, Any]:
    """
    Convert a BatchRequest into a dictionary in the format expected by the OpenAI Responses API.
    
    Args:
        request: The BatchRequest to convert
        
    Returns:
        Dictionary with the request data

    {
        "custom_id": "request-1", 
        "method": "POST", 
        "url": "/v1/chat/completions", 
        "body": {
            "instructions": "You are a helpful assistant.",
            "input": "Hello world!",
            "model": "gpt-4o-mini",
        }
    }
    """
    body = {
        "instructions": request.system_prompt,
        "input": request.prompt,
        "model": request.model,
    }
    for key, value in request.provider_args.items():
        if key == "max_tokens":
            body["max_output_tokens"] = value
        elif key == "reasoning_effort":
            body["reasoning"] = {
                "effort": value,
            }
        else:
            body[key] = value

    return {
        "custom_id": request.id,
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }
