# Relay

Relay is a Python package for batch API calls to commercial LLM APIs. It wraps different commercial LLM batch APIs into a single interface.


**Note:** This is a work in progress. The API is subject to change. Right now, it only supports OpenAI.

## Installation

### From PyPI (when published)

```bash
pip install relay-llm
```

### From Source

```bash
git clone https://github.com/neelguha/relay.git
cd relay
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

To submit a batch job:

```python
from relay import RelayClient, BatchRequest

# Initialize the client with a workspace directory
# All jobs and results will be stored in this directory
client = RelayClient(directory="my_jobs")

# Create batch requests
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
]

# Submit the batch job with a unique job ID
job = client.submit_batch(
    requests=requests,
    job_id="my-batch-001",  # User-provided unique identifier
    provider="openai",
    description="Example batch job"
)
print(f"Job ID: {job.job_id}")
print(f"Job submitted: {job.submitted_at}")
print(f"Status: {job.status}")
print(f"Number of requests: {job.n_requests}")
```

**Note:** Each job must have a unique `job_id`. If you try to submit a job with an ID that already exists and is still in progress, a `ValueError` will be raised.

### Listing Jobs

All jobs are stored in the workspace directory. You can list all jobs with:

```python
jobs = client.list_jobs()
print(f"Found {len(jobs)} job(s):")
for job_id in jobs:
    print(f"  - {job_id}")
```

### Getting Job Information

You can retrieve job metadata without monitoring:

```python
job_info = client.get_job("my-batch-001")
if job_info:
    print(f"Status: {job_info['status']}")
    print(f"Description: {job_info['description']}")
```

### Monitoring Job Progress

You can check on the progress of a job with:

```python
job_status = client.monitor_batch("my-batch-001")
print(f"Status: {job_status.status}")
print(f"Completed: {job_status.completed_requests}/{job_status.n_requests}")
print(f"Failed: {job_status.failed_requests}/{job_status.n_requests}")
```

### Retrieving Results

You can retrieve the results of a completed job. Results are automatically saved to the workspace directory:

```python
results = client.retrieve_batch_results("my-batch-001")
print(f"Retrieved {len(results)} results")

# Process each result
for result in results:
    custom_id = result.get('custom_id')
    # Access the response data based on provider format
    print(f"Request {custom_id}: {result}")
```

The `retrieve_batch_results` method:
- Fetches results from the provider API
- Saves them to `{job_id}_results.json` in the workspace
- Returns a list of dictionaries, one per request in the batch

If results already exist on disk, they are returned from cache. To force a fresh fetch:

```python
results = client.retrieve_batch_results("my-batch-001", force_refresh=True)
```

### Getting Cached Results

You can get results from disk without fetching from the API:

```python
results = client.get_results("my-batch-001")
if results:
    print(f"Found {len(results)} cached results")
else:
    print("No cached results found")
```

### Checking for Results

Check if results exist for a job:

```python
if client.has_results("my-batch-001"):
    print("Results are available")
```

### Cancelling a Job

You can cancel a job that is currently in progress:

```python
cancelled = client.cancel_batch("my-batch-001")
if cancelled:
    print("Job successfully cancelled")
```

## Supported Providers

Relay currently supports the following providers:

- **OpenAI** - Requires `OPENAI_API_KEY` environment variable
- **Together AI** - Requires `TOGETHER_API_KEY` environment variable
- **Anthropic** - Requires `ANTHROPIC_API_KEY` environment variable

## Workspace Directory

Relay uses a workspace directory to store all jobs and results. When you create a `RelayClient`, you specify a directory:

```python
client = RelayClient(directory="my_workspace")
```

The workspace directory structure:

```
my_workspace/
  job-001.json              # Job metadata
  job-001_results.json      # Results (when retrieved)
  job-002.json
  job-002_results.json
  ...
```

**Key benefits:**

- All jobs and results are stored in one place
- You can create a new `RelayClient` with the same directory to access all existing jobs
- Results are cached on disk, so you don't need to re-fetch from the API
- Easy to share or backup a workspace

### Environment Variables

Make sure to set the appropriate API key for your provider:

```bash
export OPENAI_API_KEY='your-api-key'
export TOGETHER_API_KEY='your-api-key'  # For Together AI
export ANTHROPIC_API_KEY='your-api-key'  # For Anthropic
```
