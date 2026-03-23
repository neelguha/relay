# Quick Start

This guide walks through the full lifecycle of a batch job: submit, monitor, and download results.

## Setup

```bash
pip install -e '.[anthropic]'
export ANTHROPIC_API_KEY=your-key-here
```

## 1. Submit a job

### Python

```python
import asyncio
from relay import BatchClient, BatchConfig, BatchRequest

requests = [
    BatchRequest(
        id=f"q{i}",
        messages=[{"role": "user", "content": p}],
        max_tokens=256,
    )
    for i, p in enumerate([
        "What is the capital of France?",
        "Explain photosynthesis briefly.",
        "What causes tides?",
    ])
]

config = BatchConfig(
    provider="anthropic",
    model="claude-haiku-4-5",
    name="my-first-job",          # give it a name so you can find it later
    project="research",
)

async def main():
    async with BatchClient() as client:
        job = await client.submit(requests, config)
        print(f"Submitted: {job.name} ({job.id})")

asyncio.run(main())
```

### CLI

```bash
relay submit requests.jsonl \
  --provider anthropic \
  --model claude-haiku-4-5 \
  --name my-first-job \
  --project research
```

The input JSONL file has one request per line:

```jsonl
{"id": "q0", "messages": [{"role": "user", "content": "What is the capital of France?"}]}
{"id": "q1", "messages": [{"role": "user", "content": "Explain photosynthesis briefly."}]}
{"id": "q2", "messages": [{"role": "user", "content": "What causes tides?"}]}
```

## 2. Monitor

You don't need to save the UUID. Use the job name anywhere you'd use an ID.

### Check a specific job

```bash
relay jobs status my-first-job
```

```
  Job ID               17a46dac-67ca-4c7f-a030-acf2c35772ab
  Name                 my-first-job
  Provider             anthropic
  Model                claude-haiku-4-5
  Status               IN_PROGRESS
  Total requests       3
  Completed            0
  ...
```

### List all jobs

```bash
relay jobs list
relay jobs list --project research
relay jobs list --status COMPLETED
```

### Python

```python
async with BatchClient() as client:
    job = await client.get_job("my-first-job")
    print(f"{job.name}: {job.status.value} ({job.completed_requests}/{job.total_requests})")
```

## 3. Download results

Once the job status is `COMPLETED`, download the results.

### Python

```python
async with BatchClient() as client:
    job = await client.get_job("my-first-job")

    if job.status.is_terminal:
        results = await client.download(job.id)
        for r in results:
            print(f"[{r.request_id}] {r.content[:100]}")
```

### CLI

```bash
relay download my-first-job
```

## 4. Export

Export results to different formats for downstream analysis.

### Python

```python
async with BatchClient() as client:
    await client.export("my-first-job", format="jsonl", path="./results.jsonl")
    await client.export("my-first-job", format="csv", path="./results.csv")
    await client.export("my-first-job", format="parquet", path="./results.parquet")
```

### CLI

```bash
relay export my-first-job --format jsonl --output results.jsonl
relay export my-first-job --format csv --output results.csv
relay export my-first-job --format parquet --output results.parquet
```

## 5. All-in-one: submit + wait + download

If you want to block until results are ready (useful for scripts):

### Python

```python
async with BatchClient() as client:
    job = await client.submit(requests, config)
    results = await client.wait_and_download(job.id)
    # results is a list of BatchResult objects, ready to use
```

### CLI

```bash
relay run requests.jsonl \
  --provider anthropic \
  --model claude-haiku-4-5 \
  --name my-blocking-job \
  --output results.jsonl \
  --watch                    # live progress table
```

## 6. Caching

Relay automatically caches responses. If you resubmit the same prompts with the same model and parameters, cached results are returned instantly with no API call.

```python
# Second run — hits cache, no cost
job2 = await client.submit(requests, config)
# job2.cached_hits == 3, job2.status == COMPLETED immediately
```

Disable caching per-job with `use_cache=False` in `BatchConfig` or `--no-cache` on the CLI.

## 7. Cost estimation

Check what a job will cost before submitting:

```python
estimate = await client.estimate_cost(requests, config)
print(f"${estimate.net_usd:.4f} ({estimate.cache_hits} cache hits save ${estimate.saved_usd:.4f})")
```

```bash
relay estimate requests.jsonl --provider anthropic --model claude-haiku-4-5
```

## Summary

| Step | Python | CLI |
|---|---|---|
| Submit | `client.submit(requests, config)` | `relay submit` / `relay run` |
| Monitor | `client.get_job("name")` | `relay jobs status name` |
| Download | `client.download(job.id)` | `relay download job-id` |
| Export | `client.export(job.id, format, path)` | `relay export job-id --format fmt` |
| Estimate | `client.estimate_cost(requests, config)` | `relay estimate file.jsonl` |
