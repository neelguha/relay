"""RelayClient - Main client for interacting with batch LLM APIs."""

import json
import os
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from relay.models import BatchRequest, BatchJob
from relay.providers import OpenAIProvider, TogetherProvider, AnthropicProvider


class RelayClient:
    """Client for submitting and managing batch LLM API jobs.
    
    This client wraps different commercial LLM batch APIs into a single
    interface, allowing you to submit batch jobs, monitor their progress,
    and retrieve results. All jobs and results are stored in the specified
    workspace directory.
    
    A workspace directory contains:
    - Job metadata files: {job_id}.json
    - Result files: {job_id}_results.json (when results are retrieved)
    """
    
    def __init__(self, directory: str) -> None:
        """Initialize the RelayClient with a workspace directory.
        
        The directory will store all job metadata and results. If the directory
        already contains jobs, they will be accessible through this client instance.
        
        Args:
            directory: Path to the workspace directory. Will be created if it doesn't exist.
        """
        # Set up workspace directory
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        
        # Cache for provider instances
        self._providers: Dict[str, Any] = {}
    
    def _get_provider(self, provider_name: str) -> Any:
        """Get or create a provider instance.
        
        Args:
            provider_name: Name of the provider (e.g., "openai", "together", "anthropic")
            
        Returns:
            Provider instance
            
        Raises:
            ValueError: If provider_name is not supported
        """
        if provider_name in self._providers:
            return self._providers[provider_name]
        
        if provider_name == "openai":
            provider = OpenAIProvider()
        elif provider_name == "together":
            api_key = os.getenv("TOGETHER_API_KEY")
            if not api_key:
                raise ValueError("TOGETHER_API_KEY environment variable not set")
            provider = TogetherProvider(api_key=api_key)
        elif provider_name == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            provider = AnthropicProvider(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider_name}")
        
        self._providers[provider_name] = provider
        return provider
    
    def _save_job(self, user_job_id: str, provider_job: BatchJob, provider: str, description: str) -> None:
        """Save job metadata to disk.
        
        Args:
            user_job_id: User-provided unique job ID
            provider_job: The BatchJob returned by the provider
            provider: Provider name
            description: Job description
        """
        job_file = self.directory / f"{user_job_id}.json"
        job_data = {
            "job_id": user_job_id,
            "provider_job_id": provider_job.job_id,  # Store provider's job ID for API calls
            "provider": provider,
            "submitted_at": provider_job.submitted_at.isoformat(),
            "status": provider_job.status,
            "n_requests": provider_job.n_requests,
            "completed_requests": provider_job.completed_requests,
            "failed_requests": provider_job.failed_requests,
            "description": description,
        }
        
        with open(job_file, "w") as f:
            json.dump(job_data, f, indent=2)
    
    def _check_job_exists(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Check if a job with the given ID exists and return its metadata if it does.
        
        Args:
            job_id: The job ID to check
            
        Returns:
            Job metadata dictionary if job exists, None otherwise
        """
        job_file = self.directory / f"{job_id}.json"
        if not job_file.exists():
            return None
        
        try:
            with open(job_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def _get_job_file(self, job_id: str) -> Path:
        """Get the path to a job metadata file.
        
        Args:
            job_id: The job ID
            
        Returns:
            Path to the job metadata file
        """
        return self.directory / f"{job_id}.json"
    
    def _get_results_file(self, job_id: str) -> Path:
        """Get the path to a results file.
        
        Args:
            job_id: The job ID
            
        Returns:
            Path to the results file
        """
        return self.directory / f"{job_id}_results.json"
    
    def _is_job_in_progress(self, status: str) -> bool:
        """Check if a job status indicates it's still in progress.
        
        Args:
            status: The job status string
            
        Returns:
            True if job is in progress, False if completed/failed/cancelled
        """
        completed_statuses = {"completed", "failed", "cancelled"}
        return status not in completed_statuses
    
    def _load_job_metadata(self, job_id: str) -> Dict[str, Any]:
        """Load job metadata from disk.
        
        Args:
            job_id: The job ID to load
            
        Returns:
            Dictionary containing job metadata
            
        Raises:
            ValueError: If job_id is not found
        """
        job_file = self._get_job_file(job_id)
        
        if not job_file.exists():
            raise ValueError(f"Job {job_id} not found in workspace")
        
        with open(job_file, "r") as f:
            return json.load(f)
    
    def _save_results(self, job_id: str, results: List[Dict[str, Any]]) -> None:
        """Save results to disk.
        
        Args:
            job_id: The job ID
            results: List of result dictionaries
        """
        results_file = self._get_results_file(job_id)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
    
    def _load_results(self, job_id: str) -> Optional[List[Dict[str, Any]]]:
        """Load results from disk if they exist.
        
        Args:
            job_id: The job ID
            
        Returns:
            List of result dictionaries if file exists, None otherwise
        """
        results_file = self._get_results_file(job_id)
        if not results_file.exists():
            return None
        
        try:
            with open(results_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def submit_batch(
        self,
        requests: List[BatchRequest],
        job_id: str,
        provider: str,
        description: str,
    ) -> BatchJob:
        """Submit a batch of requests to the LLM API.
        
        Submits a list of BatchRequest objects as a single batch job.
        The job is automatically saved to ~/.relay/ as a JSON file.
        
        Args:
            requests: A list of BatchRequest objects to process in the batch
            job_id: A unique identifier for this batch job (user-provided)
            provider: The provider name (e.g., "openai", "together", "anthropic")
            description: A description of the batch job

        Returns:
            BatchJob object containing the job_id, submitted_at timestamp,
            and initial status
            
        Raises:
            ValueError: If requests list is empty, job_id already exists and is in progress,
                        or contains invalid requests
            APIError: If the batch submission fails
        """
        if not requests:
            raise ValueError("Requests list cannot be empty")
        
        if not job_id or not job_id.strip():
            raise ValueError("job_id cannot be empty")
        
        # Check if job with this ID already exists
        existing_job = self._check_job_exists(job_id)
        if existing_job:
            if self._is_job_in_progress(existing_job.get("status", "")):
                raise ValueError(
                    f"Job with ID '{job_id}' already exists and is in progress "
                    f"(status: {existing_job.get('status', 'unknown')}). "
                    "Please use a different job_id or wait for the existing job to complete."
                )
        
        # Get the appropriate provider
        provider_instance = self._get_provider(provider)
        
        # Submit the batch
        provider_job = provider_instance.submit_batch(requests)
        
        # Save job metadata with user-provided job_id
        self._save_job(job_id, provider_job, provider, description)
        
        # Return a BatchJob with the user-provided job_id
        return BatchJob(
            job_id=job_id,
            submitted_at=provider_job.submitted_at,
            status=provider_job.status,
            n_requests=provider_job.n_requests,
        )
    
    def list_jobs(self) -> List[str]:
        """List all job IDs in the workspace.
        
        Returns:
            A list of job IDs (strings) in the workspace
            
        Raises:
            IOError: If the workspace directory cannot be accessed
        """
        try:
            # Only get job metadata files, not results files
            job_files = [f for f in self.directory.glob("*.json") 
                        if f.is_file() and not f.name.endswith("_results.json")]
            job_ids = [f.stem for f in job_files]
            return sorted(job_ids)
        except Exception as e:
            raise IOError(f"Failed to access workspace directory: {str(e)}")
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job metadata for a specific job.
        
        Args:
            job_id: The job ID to retrieve
            
        Returns:
            Dictionary containing job metadata, or None if not found
        """
        return self._check_job_exists(job_id)
    
    def has_results(self, job_id: str) -> bool:
        """Check if results exist for a job.
        
        Args:
            job_id: The job ID to check
            
        Returns:
            True if results file exists, False otherwise
        """
        return self._get_results_file(job_id).exists()
    
    def monitor_batch(
        self,
        job_id: str,
    ) -> BatchJob:
        """Check the progress of a batch job.
        
        Retrieves the current status and metadata for a batch job.
        
        Args:
            job_id: The unique identifier of the batch job to monitor (user-provided ID)
            
        Returns:
            BatchJob object with updated status and metadata
            
        Raises:
            ValueError: If job_id is invalid or not found
            APIError: If the status check fails
        """
        # Load job metadata
        job_metadata = self._load_job_metadata(job_id)
        provider_name = job_metadata["provider"]
        provider_job_id = job_metadata.get("provider_job_id", job_id)  # Use provider's job ID for API calls
        
        # Get the appropriate provider
        provider_instance = self._get_provider(provider_name)
        
        # Monitor the batch using provider's job ID
        provider_job = provider_instance.monitor_batch(provider_job_id)
        
        # Update saved job metadata
        job_metadata["status"] = provider_job.status
        job_metadata["completed_requests"] = provider_job.completed_requests
        job_metadata["failed_requests"] = provider_job.failed_requests
        job_file = self._get_job_file(job_id)
        with open(job_file, "w") as f:
            json.dump(job_metadata, f, indent=2)

        # Return BatchJob with user-provided job_id
        return BatchJob(
            job_id=job_id,
            submitted_at=provider_job.submitted_at,
            status=provider_job.status,
            n_requests=provider_job.n_requests,
            completed_requests=provider_job.completed_requests,
            failed_requests=provider_job.failed_requests,
        )
    
    def retrieve_batch_results(
        self,
        job_id: str,
        force_refresh: bool = False,
    ) -> List[Dict[str, Any]]:
        """Retrieve the results of a completed batch job.
        
        Retrieves the results of a batch job from the provider and saves them
        to the workspace. If results already exist on disk and force_refresh is False,
        returns the cached results.
        
        Args:
            job_id: The unique identifier of the batch job (user-provided ID)
            force_refresh: If True, fetch fresh results even if cached results exist
            
        Returns:
            List of result dictionaries, one per request in the batch
            
        Raises:
            ValueError: If job_id is invalid or not found
            APIError: If the results retrieval fails
        """
        # Check if results already exist
        if not force_refresh:
            cached_results = self._load_results(job_id)
            if cached_results is not None:
                return cached_results
        
        # Load job metadata
        job_metadata = self._load_job_metadata(job_id)
        provider_name = job_metadata["provider"]
        provider_job_id = job_metadata.get("provider_job_id", job_id)  # Use provider's job ID for API calls
        
        # Get the appropriate provider
        provider_instance = self._get_provider(provider_name)
        
        # Retrieve results using provider's job ID
        results = provider_instance.retrieve_batch_results(provider_job_id)
        
        # Save results to disk
        self._save_results(job_id, results)
        
        return results
    
    def get_results(self, job_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get results for a job from disk (does not fetch from API).
        
        Args:
            job_id: The job ID to get results for
            
        Returns:
            List of result dictionaries if results exist, None otherwise
        """
        return self._load_results(job_id)
    
    def cancel_batch(
        self,
        job_id: str,
    ) -> bool:
        """Cancel a batch job.
        
        Attempts to cancel a batch job that is currently in progress.
        
        Args:
            job_id: The unique identifier of the batch job to cancel (user-provided ID)
            
        Returns:
            True if the job was successfully cancelled, False otherwise
            
        Raises:
            ValueError: If job_id is invalid or not found
            APIError: If the cancellation request fails
        """
        # Load job metadata
        job_metadata = self._load_job_metadata(job_id)
        provider_name = job_metadata["provider"]
        provider_job_id = job_metadata.get("provider_job_id", job_id)  # Use provider's job ID for API calls
        
        # Get the appropriate provider
        provider_instance = self._get_provider(provider_name)
        
        # Cancel the batch using provider's job ID
        cancelled = provider_instance.cancel_batch(provider_job_id)
        
        # Update saved job metadata if cancelled
        if cancelled:
            job_metadata["status"] = "cancelled"
            job_file = self._get_job_file(job_id)
            with open(job_file, "w") as f:
                json.dump(job_metadata, f, indent=2)
        
        return cancelled
