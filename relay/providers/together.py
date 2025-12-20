"""Together AI provider implementation for batch API calls."""

from typing import List, Dict, Any
from pathlib import Path
from relay.providers.base import BaseProvider
from relay.models import BatchRequest, BatchJob
from together import Together
import os
import json
from datetime import datetime



class TogetherProvider(BaseProvider):
    """Provider implementation for Together AI's batch API.
    
    Handles batch submissions, monitoring, and result retrieval for Together AI's
    batch processing endpoints.
    """
    
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """Initialize the Together AI provider.
        
        Args:
            **kwargs: Additional Together AI-specific configuration options
        """
        self.client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        self.job_id_to_n_requests = {} # Map of job ID to number of requests, because the Together AI API doesn't return this information in the batch object
    
    def submit_batch(
        self,
        requests: List[BatchRequest],
    ) -> BatchJob:
        """Submit a batch of requests to Together AI's batch API.
        
        Submits requests to Together AI's Batch API and returns a BatchJob with the Together AI batch ID. The Together AI Batch API requires we save the requests to a JSONL file, and then submit that file to the API.

        Right now, this defaults to using the Responses API.
        
        Args:
            requests: A list of BatchRequest objects to process in the batch
            
        Returns:
            BatchJob object containing the Together AI batch ID, submitted_at timestamp,
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
        file_resp = self.client.files.upload(
            file=jsonl_file,
            purpose="batch-api",
            check=False
        )
        
        file_id = file_resp.id
        batch_object = self.client.batches.create_batch(
            file_id,
            endpoint="/v1/chat/completions"
        )
        self.job_id_to_n_requests[batch_object.id] = len(requests)
        return BatchJob(
            job_id=batch_object.id,
            submitted_at=datetime.now(),
            status=batch_object.status,
            n_requests=len(requests),
        )
    
    def monitor_batch(
        self,
        job_id: str,
    ) -> BatchJob:
        """Check the progress of a Together AI batch job.
        
        Queries Together AI's /v1/batches/{batch_id} endpoint to get current status.
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
        batch = self.client.batches.get_batch(job_id)
        return BatchJob(
            job_id=job_id,
            submitted_at=batch.created_at,
            status=batch.status,
            n_requests=self.job_id_to_n_requests[job_id],
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
        batch = self.client.batches.get_batch(job_id)
        if batch.status != "COMPLETED":
            raise ValueError(f"Batch job {job_id} is not completed (status: {batch.status})")


        # Download the output file
        self.client.files.retrieve_content(
            id=batch.output_file_id,
            output="batch_output.jsonl",
        )

        # Load the output file and then delete it
        outputs = []
        with open("batch_output.jsonl", "r") as f:
            for line in f:
                if line.strip():
                    outputs.append(json.loads(line))
        os.remove("batch_output.jsonl")
        return outputs
    
    def cancel_batch(
        self,
        job_id: str,
    ) -> bool:
        """Cancel a Together AI batch job.
        
        Cancels a batch job by calling Together AI's /v1/batches/{batch_id}/cancel endpoint.
        Only batches in "validating" or "in_progress" status can be cancelled.
        
        Args:
            job_id: The Together AI batch ID to cancel
            
        Returns:
            True if the job was successfully cancelled, False otherwise
            
        Raises:
            ValueError: If job_id is invalid or not found
            APIError: If the cancellation request fails
        """
        self.client.batches.cancel_batch(job_id)
        return True
    


def convert_request_to_dict(request: BatchRequest) -> Dict[str, Any]:
    """
    Convert a BatchRequest into a dictionary in the format expected by the Together AI Responses API.
    
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
        "model": request.model,
    }
    # Build the messages array
    messages = []
    if request.system_prompt:
        messages.append({
            "role": "system",
            "content": request.system_prompt,
        })
    messages.append({
        "role": "user",
        "content": request.prompt,
    })
    body["messages"] = messages
    for key, value in request.provider_args.items():
        body[key] = value

    return {
        "custom_id": request.id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }
