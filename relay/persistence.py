"""Persistence layer for saving and loading batch jobs."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from relay.models import BatchJob, BatchJobStatus
from relay.exceptions import RelayError


class JobPersistence:
    """Handles persistence of batch jobs to disk."""

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize job persistence.

        Args:
            storage_path: Path to the storage directory. Defaults to ~/.relay/jobs/
        """
        if storage_path is None:
            home = Path.home()
            self.storage_dir = home / ".relay" / "jobs"
        else:
            self.storage_dir = Path(storage_path)

        # Create storage directory if it doesn't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_job_file(self, job_id: str) -> Path:
        """Get the file path for a job ID."""
        return self.storage_dir / f"{job_id}.json"

    def save_job(self, job: BatchJob) -> None:
        """
        Save a batch job to disk.

        Args:
            job: BatchJob to save

        Raises:
            RelayError: If saving fails
        """
        try:
            job_data = self._job_to_dict(job)
            job_file = self._get_job_file(job.job_id)
            
            with open(job_file, "w") as f:
                json.dump(job_data, f, indent=2)
        except Exception as e:
            raise RelayError(f"Failed to save job {job.job_id}: {str(e)}")

    def load_job(self, job_id: str) -> Optional[BatchJob]:
        """
        Load a batch job from disk.

        Args:
            job_id: The job ID to load

        Returns:
            BatchJob if found, None otherwise

        Raises:
            RelayError: If loading fails
        """
        job_file = self._get_job_file(job_id)
        
        if not job_file.exists():
            return None

        try:
            with open(job_file, "r") as f:
                job_data = json.load(f)
            return self._dict_to_job(job_data)
        except Exception as e:
            raise RelayError(f"Failed to load job {job_id}: {str(e)}")

    def list_jobs(
        self,
        provider: Optional[str] = None,
        status: Optional[BatchJobStatus] = None,
    ) -> List[BatchJob]:
        """
        List all saved jobs, optionally filtered by provider and status.

        Args:
            provider: Filter by provider name
            status: Filter by job status

        Returns:
            List of BatchJob objects

        Raises:
            RelayError: If listing fails
        """
        jobs = []
        
        try:
            for job_file in self.storage_dir.glob("*.json"):
                try:
                    with open(job_file, "r") as f:
                        job_data = json.load(f)
                    job = self._dict_to_job(job_data)
                    
                    # Apply filters
                    if provider and job.provider != provider:
                        continue
                    if status and job.status != status:
                        continue
                    
                    jobs.append(job)
                except Exception:
                    # Skip corrupted files
                    continue
        except Exception as e:
            raise RelayError(f"Failed to list jobs: {str(e)}")

        return jobs

    def delete_job(self, job_id: str) -> bool:
        """
        Delete a saved job from disk.

        Args:
            job_id: The job ID to delete

        Returns:
            True if deleted, False if not found
        """
        job_file = self._get_job_file(job_id)
        
        if job_file.exists():
            job_file.unlink()
            return True
        return False

    def _job_to_dict(self, job: BatchJob) -> dict:
        """Convert a BatchJob to a dictionary for JSON serialization."""
        return {
            "job_id": job.job_id,
            "provider": job.provider,
            "status": job.status.value,
            "total_requests": job.total_requests,
            "completed_requests": job.completed_requests,
            "failed_requests": job.failed_requests,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "provider_job_id": job.provider_job_id,
            "metadata": job.metadata,
        }

    def _dict_to_job(self, data: dict) -> BatchJob:
        """Convert a dictionary to a BatchJob."""
        return BatchJob(
            job_id=data["job_id"],
            provider=data["provider"],
            status=BatchJobStatus(data["status"]),
            total_requests=data["total_requests"],
            completed_requests=data.get("completed_requests", 0),
            failed_requests=data.get("failed_requests", 0),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            provider_job_id=data.get("provider_job_id"),
            metadata=data.get("metadata", {}),
        )

