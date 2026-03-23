# relay

A unified, provider-agnostic Python library for submitting, managing, and downloading results from large-scale LLM batch prediction jobs across Anthropic, OpenAI, Google, and XAI.

## Installation

```bash
# Minimal
pip install relay

# All providers + full feature set
pip install relay[all]

# Individual providers
pip install relay[anthropic]
pip install relay[openai]
pip install relay[google]
pip install relay[xai]
```

## Quickstart

```python
import asyncio
from relay import BatchClient, BatchRequest, BatchConfig, Message

requests = [
    BatchRequest(id=f"req-{i}", messages=[Message(role="user", content=f"Summarize: {text}")])
    for i, text in enumerate(my_texts)
]

config = BatchConfig(provider="anthropic", model="claude-opus-4-5")

async def main():
    async with BatchClient() as client:
        job = await client.submit(requests, config)
        results = await client.wait_and_download(job.id)
    return results

results = asyncio.run(main())
```

## Features

- **Provider-agnostic API** — switch providers with a single config change
- **Resilient job state** — jobs survive process crashes; state is persisted to SQLite
- **Smart caching** — identical `(prompt, model, params)` tuples are never sent twice
- **Cost estimation** — token count and cost estimate before every submission
- **Budget controls** — configurable warn, confirm, and hard-limit thresholds
- **Fan-out** — send the same request to multiple providers simultaneously
- **Multiple export formats** — JSONL, CSV, Parquet, HuggingFace Dataset
- **Terminal dashboard** — live TUI for monitoring jobs, costs, and cache
- **Web dashboard** — lightweight FastAPI UI for remote monitoring
- **Observability** — Prometheus, OpenTelemetry, and JSONL metrics exporters

## Provider Compatibility

| Provider  | Model family      | Batch API | Streaming | Fan-out |
|-----------|-------------------|-----------|-----------|---------|
| Anthropic | Claude 3/4        | Yes       | No        | Yes     |
| OpenAI    | GPT-4o, o-series  | Yes       | No        | Yes     |
| Google    | Gemini 1.5/2.0    | Yes       | No        | Yes     |
| XAI       | Grok              | Yes       | No        | Yes     |

## CLI

```bash
# Submit a batch job
relay submit requests.jsonl --provider anthropic --model claude-opus-4-5

# Submit and wait for results
relay run requests.jsonl --provider openai --model gpt-4o --output results.jsonl

# Dry-run cost estimate
relay estimate requests.jsonl --provider anthropic --model claude-opus-4-5

# List and monitor jobs
relay jobs list
relay jobs status <job-id>

# Export results
relay export <job-id> --format parquet --output results/

# Cache management
relay cache stats
relay cache vacuum

# Spending report
relay costs today
```

## Documentation

See the [spec.md](spec.md) for the full technical specification and API reference.
