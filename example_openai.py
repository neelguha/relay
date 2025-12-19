"""Example script demonstrating how to use Relay with OpenAI models.

This script shows how to:
1. Create batch requests with OpenAI models
2. Submit a batch job
3. Monitor the job progress
4. Retrieve and process results
"""

import os
import time
from relay import RelayClient, BatchRequest


def main():
    """Main example function."""
    
    # Initialize the Relay client
    # Make sure OPENAI_API_KEY is set in your environment
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-api-key'")
        return
    
    client = RelayClient(directory="relay_manager")
    
    # Create a list of batch requests
    # Each request represents a single prompt to be processed
    requests = [
        BatchRequest(
            id="req-1",
            model="gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
            prompt="Hello! What is 2+2?",
            provider_args={}
        ),
        BatchRequest(
            id="req-2",
            model="gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
            prompt="What is the capital of France?",
            provider_args={}
        ),
        BatchRequest(
            id="req-3",
            model="gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
            prompt="Explain quantum computing in one sentence.",
            provider_args={}
        ),
        BatchRequest(
            id="req-4",
            model="gpt-4o-mini",
            system_prompt="You are a math tutor.",
            prompt="Solve for x: 3x + 5 = 20",
            provider_args={"temperature": 0.3}  # Lower temperature for more deterministic math
        ),
        BatchRequest(
            id="req-5",
            model="gpt-4o-mini",
            system_prompt="You are a creative writer.",
            prompt="Write a haiku about programming.",
            provider_args={"temperature": 0.9}  # Higher temperature for more creativity
        ),
    ]
    
    print("=" * 60)
    print("Submitting batch job with OpenAI...")
    print(f"Number of requests: {len(requests)}")
    print("=" * 60)
    
    # Submit the batch job
    job = client.submit_batch(
        requests=requests,
        job_id="example-batch-001",  # User-provided unique job ID
        provider="openai",
        description="Example batch job"
    )
    
    print(f"\n✓ Job submitted successfully!")
    print(f"  Job ID: {job.job_id}")
    print(f"  Submitted at: {job.submitted_at}")
    print(f"  Status: {job.status}")
    print(f"  Number of requests: {job.n_requests}")
    
    # List all saved jobs
    print("\n" + "=" * 60)
    print("Listing all saved jobs...")
    print("=" * 60)
    saved_jobs = client.list_jobs()
    print(f"Found {len(saved_jobs)} saved job(s):")
    for job_id in saved_jobs:
        print(f"  - {job_id}")
    
    # Monitor the batch job
    print("\n" + "=" * 60)
    print("Monitoring batch job progress...")
    print("=" * 60)
    
    max_wait_time = 300  # Maximum wait time in seconds (5 minutes)
    check_interval = 10  # Check every 10 seconds
    start_time = time.time()
    
    while True:
        elapsed_time = time.time() - start_time
        
        if elapsed_time > max_wait_time:
            print(f"\n⚠ Maximum wait time ({max_wait_time}s) exceeded")
            print("You can check the job status later using:")
            print(f"  client.monitor_batch('{job.job_id}')")
            break
        
        job_status = client.monitor_batch(job.job_id)
        
        print(f"\n[{elapsed_time:.0f}s] Status: {job_status.status}")
        
        if job_status.status == "completed":
            print("✓ Batch job completed!")
            break
        elif job_status.status == "failed":
            print("✗ Batch job failed!")
            break
        elif job_status.status in ["validating", "in_progress", "finalizing"]:
            print(f"  Job is still processing... (checking again in {check_interval}s)")
            time.sleep(check_interval)
        else:
            print(f"  Unknown status, waiting...")
            time.sleep(check_interval)
    
    # Retrieve the results
    if job_status.status == "completed":
        print("\n" + "=" * 60)
        print("Retrieving batch results...")
        print("=" * 60)
        
        results = client.retrieve_batch_results(
            job_id=job.job_id
        )
        
        print(f"✓ Retrieved {len(results)} results")
        print("\nSample results:")
        for i, result in enumerate(results[:3]):  # Show first 3 results
            print(f"\n  Result {i+1}:")
            print(f"    Custom ID: {result.get('custom_id', 'N/A')}")
            if 'response' in result and 'body' in result['response']:
                body = result['response']['body']
                if 'output' in body:
                    output = body['output']
                    # Truncate long outputs
                    output_str = str(output)[:100] + "..." if len(str(output)) > 100 else str(output)
                    print(f"    Output: {output_str}")
        
        if len(results) > 3:
            print(f"\n  ... and {len(results) - 3} more results")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
