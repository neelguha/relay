# relay — Technical Specification & Implementation Guide

**Version:** 1.0.0  
**Providers:** Anthropic · OpenAI · Google · XAI  
**Language:** Python ≥ 3.10  
**Distribution:** PyPI (`relay`)

---

## Table of Contents

1. [Overview & Goals](#1-overview--goals)
2. [Installation & Configuration](#2-installation--configuration)
3. [Core Python API](#3-core-python-api)
4. [Provider Adapters](#4-provider-adapters)
5. [Cache Layer](#5-cache-layer)
6. [Job Persistence & Storage](#6-job-persistence--storage)
7. [Command-Line Interface](#7-command-line-interface)
8. [Terminal Dashboard](#8-terminal-dashboard)
9. [Web Monitoring Dashboard](#9-web-monitoring-dashboard)
10. [Monitoring & Observability](#10-monitoring--observability)
11. [Multi-Provider Patterns](#11-multi-provider-patterns)
12. [Cost Management](#12-cost-management)
13. [Error Handling & Retry](#13-error-handling--retry)
14. [Output Formats & Exporters](#14-output-formats--exporters)
15. [Repository Structure](#15-repository-structure)
16. [Dependency Specification](#16-dependency-specification)
17. [Testing Strategy](#17-testing-strategy)
18. [Complete Usage Examples](#18-complete-usage-examples)
19. [Implementation Instructions for Claude Code](#19-implementation-instructions-for-claude-code)
20. [Appendix A: Provider Pricing Reference](#appendix-a-provider-pricing-reference)
21. [Appendix B: Glossary](#appendix-b-glossary)

---

## 1. Overview & Goals

`relay` is an open-source Python library that provides a unified, provider-agnostic interface for submitting, managing, monitoring, and downloading results from large-scale **text-only** batch prediction jobs across all major LLM providers: Anthropic (Claude), OpenAI (GPT), Google (Gemini), and XAI (Grok).

The primary use case is research, data annotation, and production pipelines where you need to send thousands to millions of text prompts to one or more LLMs, track job state across process restarts, deduplicate repeated inputs via a smart cache layer, control costs via budgets, and inspect results through a live terminal dashboard or a lightweight web UI.

### 1.1 Design Principles

- **Provider-agnostic:** A single API works identically across all providers. Switching models or providers requires only a config change.
- **Resilient by default:** Jobs survive process crashes, network failures, and API timeouts. State is persisted to disk and checkpointed continuously.
- **Zero surprise cost model:** Before submitting any batch, the library estimates token count and cost and can optionally require confirmation above a configurable threshold.
- **Cache-first:** Identical `(prompt, model, params)` tuples are never sent twice unless the cache is explicitly bypassed or expired.
- **Observable:** Every job, request, and result emits structured events that feed the terminal dashboard, the web dashboard, and pluggable exporters.
- **Composable:** The library exposes both a high-level async Python API and a CLI. All internal components (providers, storage backends, cache, exporters) are swappable via a plugin interface.

### 1.2 High-Level Architecture

The library is organized into five main layers:

| Layer | Responsibility |
|---|---|
| **Provider Adapters** | Translate the unified job format into each provider's native batch API calls. Each adapter handles authentication, request chunking, polling, and error mapping. |
| **Job Engine** | Manages the lifecycle of every batch job: creation, submission, polling, retry, completion, and failure. Persists state to a local SQLite database so jobs survive restarts. |
| **Cache Layer** | SQLite-backed content-addressable cache keyed on a hash of `(model, system_prompt, user_prompt, params)`. Results are stored compressed and returned immediately on cache hit. |
| **Monitoring Bus** | Internal event bus that collects metrics (requests/sec, token usage, cost, error rates, latency percentiles) and fans them out to the terminal dashboard, web dashboard, log files, and user-registered callbacks. |
| **Exporters** | Convert completed batch results into structured output formats: JSONL, CSV, Parquet, and HuggingFace Dataset. |

---

## 2. Installation & Configuration

### 2.1 Installation

```bash
# Minimal install
pip install relay

# With all providers and full feature set
pip install relay[all]

# Individual provider extras
pip install relay[anthropic]
pip install relay[openai]
pip install relay[google]
pip install relay[xai]

# Web dashboard extra
pip install relay[dashboard]

# Development install
git clone https://github.com/<org>/relay
cd relay && pip install -e '.[dev,all]'
```

### 2.2 Configuration File

`relay` looks for a configuration file in the following order:
1. Path given to `--config` on the CLI
2. `RELAY_CONFIG` environment variable
3. `~/.relay/config.toml`
4. `pyproject.toml` under `[tool.relay]`

The recommended approach is a project-level `config.toml` at the root of each project.

**`config.toml` — Full Reference:**

```toml
[relay]
db_path = "~/.relay/jobs.db"          # SQLite database for job & cache state
log_level = "INFO"                        # DEBUG | INFO | WARNING | ERROR
log_dir = "~/.relay/logs"              # Rotating log files written here
output_dir = "./relay_results"         # Default directory for exported results

[relay.cost]
warn_threshold_usd = 1.00                # Print warning above this per-batch estimate
require_confirmation_usd = 10.00         # Interactive confirm above this threshold
hard_limit_usd = 100.00                  # Refuse to submit above this (0 = disabled)

[relay.cache]
enabled = true
backend = "sqlite"                       # sqlite | redis | none
redis_url = "redis://localhost:6379/0"   # Only used if backend = redis
ttl_seconds = 2592000                    # 30 days; 0 = never expire
max_size_gb = 5.0                        # Evict LRU entries beyond this
compress = true                          # Zstd-compress stored responses

[relay.retry]
max_attempts = 5
initial_backoff_seconds = 1.0
backoff_multiplier = 2.0
max_backoff_seconds = 60.0
jitter = "full"                          # none | full | equal
retry_on = [429, 500, 502, 503, 504]

[relay.providers.anthropic]
api_key = "${ANTHROPIC_API_KEY}"         # Env-var interpolation supported
default_model = "claude-opus-4-5"
max_concurrent_batches = 5

[relay.providers.openai]
api_key = "${OPENAI_API_KEY}"
default_model = "gpt-4o"
max_concurrent_batches = 10

[relay.providers.google]
api_key = "${GOOGLE_API_KEY}"
project_id = "${GCP_PROJECT_ID}"
default_model = "gemini-2.0-flash"

[relay.providers.xai]
api_key = "${XAI_API_KEY}"
default_model = "grok-3"

[relay.dashboard]
enabled = true
host = "127.0.0.1"
port = 7860
theme = "dark"                           # dark | light
auto_refresh_seconds = 5
```

### 2.3 Secrets Management

API keys should never be hardcoded in config files. `relay` supports three methods, evaluated in this priority order:

1. **Environment variables:** Set `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `XAI_API_KEY` before running.
2. **Config file with env-var interpolation:** Use `${VAR_NAME}` syntax in `config.toml` as shown above.
3. **Custom resolver:** Pass an object implementing the `SecretsResolver` protocol to `BatchClient()` for 1Password / AWS Secrets Manager / HashiCorp Vault integration.

---

## 3. Core Python API

### 3.1 BatchClient

`BatchClient` is the central entry point. It is an async context manager that owns the database connection, cache, provider adapters, and monitoring bus.

```python
from relay import BatchClient, BatchRequest, BatchConfig

async with BatchClient(config='config.toml') as client:
    job = await client.submit(requests, config)
    results = await client.wait_and_download(job.id)
```

**Primary methods:**

| Method | Description |
|---|---|
| `submit(requests, config)` | Validate, estimate cost, check cache, and submit a list of `BatchRequest` objects. Returns a `BatchJob` handle immediately (non-blocking). |
| `get_job(job_id)` | Return the current `BatchJob` snapshot, including status, progress counters, cost so far, and timestamps. |
| `list_jobs(**filters)` | Query the job store with optional filters: provider, model, status, project, tags, date range. Returns a paginated list. |
| `wait(job_id, poll_interval)` | Async generator that yields `JobProgress` snapshots until the job reaches a terminal state. Safe to cancel. |
| `download(job_id)` | Download and parse raw results from the provider API. Returns a list of `BatchResult` objects. Populates cache entries automatically. |
| `wait_and_download(job_id)` | Convenience wrapper combining `wait()` and `download()`. Yields progress events during wait, then returns final results. |
| `cancel(job_id)` | Request cancellation of an in-progress batch. Best-effort: already-processed requests may still be billed. |
| `resubmit_failed(job_id)` | Create a new job containing only the failed requests from a completed job. |
| `export(job_id, format, path)` | Write job results to disk: `jsonl`, `csv`, `parquet`, or `hf_dataset`. |
| `estimate_cost(requests, config)` | Dry-run: tokenize and price requests without submitting. Returns `CostEstimate` with per-provider breakdown. |
| `cache.get(request)` | Directly query the cache for a single request. Returns `BatchResult` or `None`. |
| `cache.invalidate(job_id)` | Delete all cache entries belonging to a job. |

### 3.2 Data Models

All data models are defined as frozen Python dataclasses and also exported as Pydantic v2 models. They are fully serializable to/from JSON and stored in SQLite.

#### BatchRequest

```python
@dataclass(frozen=True)
class BatchRequest:
    id: str                     # Caller-supplied or auto-generated UUID
    messages: list[Message]     # OpenAI-style message list (text content only)
    system: str | None = None   # System prompt (provider-translated)
    model: str | None = None    # Override batch-level model
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float | None = None
    stop_sequences: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)  # Pass-through to provider
    tags: list[str] = field(default_factory=list) # User-defined tags for filtering
```

> **Note:** `messages` content must be plain text strings only. No image, audio, or video content types are supported.

#### BatchConfig

```python
@dataclass
class BatchConfig:
    provider: str                     # 'anthropic' | 'openai' | 'google' | 'xai'
    model: str                        # Model identifier, e.g. 'claude-opus-4-5'
    project: str | None = None        # Logical project name for grouping jobs
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    use_cache: bool = True
    cache_ttl_override: int | None = None  # Override global TTL for this job
    output_dir: str | None = None
    on_progress: Callable | None = None    # Callback for progress events
    on_complete: Callable | None = None    # Callback when job finishes
```

#### BatchJob

```python
@dataclass
class BatchJob:
    id: str                           # relay-internal job UUID
    provider_job_id: str              # Provider-assigned batch ID
    provider: str
    model: str
    project: str | None
    status: JobStatus                 # See status machine below
    total_requests: int
    completed_requests: int
    failed_requests: int
    cached_hits: int                  # Requests served from cache
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    actual_cost_usd: float | None
    created_at: datetime
    submitted_at: datetime | None
    completed_at: datetime | None
    tags: list[str]
    error: str | None
```

#### BatchResult

```python
@dataclass(frozen=True)
class BatchResult:
    request_id: str
    job_id: str
    content: str                      # Raw text response
    stop_reason: str                  # 'end_turn' | 'max_tokens' | etc.
    input_tokens: int
    output_tokens: int
    model: str                        # Actual model used (may differ for routing)
    from_cache: bool
    cached_at: datetime | None
    raw_response: dict                # Full provider response payload
    error: BatchError | None          # Set if this individual request failed
```

### 3.3 Job Status State Machine

Every `BatchJob` moves through the following states. A job in a terminal state will never transition again.

| Status | Terminal? | Description |
|---|---|---|
| `PENDING` | No | Job created locally, not yet submitted to provider. |
| `CACHE_RESOLVING` | No | Cache is being checked. Requests with hits are excluded from the batch. |
| `VALIDATING` | No | Requests are being validated (token limits, format, cost check). |
| `SUBMITTING` | No | Being uploaded to the provider's batch endpoint. |
| `IN_PROGRESS` | No | Provider is processing the batch. Poll loop is active. |
| `DOWNLOADING` | No | Results are being fetched from the provider. |
| `COMPLETED` | **Yes** | All requests completed (some may have per-request errors). |
| `PARTIAL` | **Yes** | Job ended with a mix of successes and hard failures. |
| `FAILED` | **Yes** | Job-level failure (auth error, network failure, etc.). |
| `CANCELLED` | **Yes** | Cancelled by the user. |

---

## 4. Provider Adapters

Each provider has a dedicated adapter module in `relay/providers/`. All adapters implement the `BaseProvider` abstract class.

### 4.1 BaseProvider Interface

```python
class BaseProvider(ABC):
    @abstractmethod
    async def submit_batch(self, requests: list[NativeRequest]) -> str: ...
    # Returns provider_job_id

    @abstractmethod
    async def get_status(self, provider_job_id: str) -> ProviderStatus: ...

    @abstractmethod
    async def download_results(self, provider_job_id: str) -> list[NativeResult]: ...

    @abstractmethod
    async def cancel(self, provider_job_id: str) -> bool: ...

    @abstractmethod
    def estimate_tokens(self, request: BatchRequest) -> tuple[int, int]: ...
    # Returns (estimated_input_tokens, estimated_output_tokens)

    @abstractmethod
    def get_price_per_million(self, model: str) -> tuple[float, float]: ...
    # Returns (input_price_usd, output_price_usd) per million tokens
```

### 4.2 Anthropic Adapter

Uses the **Anthropic Message Batches API** (`/v1/messages/batches`). The adapter:
- Chunks requests into batches of up to 10,000 (the API maximum), auto-splitting if exceeded
- Handles the 24-hour expiry window with configurable polling
- Streams result JSONL files via HTTP range requests to minimise memory usage on large batches
- Maps `messages` (text-only) to the `content` array format

| Feature | Supported | Notes |
|---|---|---|
| Message Batches API | Yes | Native async batch endpoint |
| Streaming results | Yes | JSONL streamed with range requests |
| Token counting | Yes | `anthropic` SDK token counting |
| Cost estimation | Yes | Per-model price table, auto-updated |
| Max requests/batch | 10,000 | Auto-chunked if exceeded |
| Max batch expiry | 24 hours | Polled until completion or expiry |
| Model routing | Yes | Supports `latest` model aliases |

### 4.3 OpenAI Adapter

Uses the **OpenAI Batch API** (`/v1/batches`). Input JSONL files are uploaded via the Files API. Supports standard GPT models and fine-tuned endpoints.

| Feature | Supported | Notes |
|---|---|---|
| Batch API (JSONL) | Yes | Files API for upload/download |
| Fine-tuned models | Yes | Pass `ft:gpt-4o:...` model string |
| Token counting | Yes | `tiktoken` |
| JSON mode | Yes | `response_format` parameter |
| Function calling | Yes | Tools passed through as-is |
| Max requests/batch | 50,000 | OpenAI API limit |

### 4.4 Google Gemini Adapter

Uses the **Gemini Batch Prediction API** through the Vertex AI SDK. Supports both API key auth (Gemini Developer API) and service account auth (Vertex AI). Text-only content only.

| Feature | Supported | Notes |
|---|---|---|
| Gemini Developer API | Yes | API key auth |
| Vertex AI Batch | Yes | Service account auth |
| BigQuery output | Yes | For very large batches |
| Token counting | Yes | `countTokens` API |
| Grounding | Yes | Google Search grounding |
| Max requests/batch | Varies | Quota-dependent |

### 4.5 XAI (Grok) Adapter

Uses the **XAI REST API** which closely mirrors the OpenAI API format. The adapter extends the OpenAI adapter's request formatting logic with XAI-specific auth and endpoints. Until XAI's native batch API is GA, this adapter uses a managed concurrent request pool with intelligent rate limiting.

| Feature | Supported | Notes |
|---|---|---|
| Concurrent pool mode | Yes | Until native batch API is stable |
| Native batch API | Planned | Will switch when GA |
| Token counting | Yes | Via XAI tokenizer endpoint |
| Rate limit handling | Yes | 429 with `Retry-After` respected |
| Max concurrency | 50 | Configurable per-project |

### 4.6 Adding a Custom Provider

Create `relay/providers/myprovider.py`, implement `BaseProvider`, and register it in `relay/providers/__init__.py`. Providers can also be registered from outside the package via Python entry points under the group `relay.providers` in `pyproject.toml`.

---

## 5. Cache Layer

### 5.1 Design

The cache is a content-addressable store keyed on a **SHA-256 hash** of the canonical serialization of:

```
(provider, model, system_prompt, user_messages, temperature, max_tokens, top_p, stop_sequences)
```

The `metadata` and `tags` fields are excluded from the hash so they do not affect caching. The hash input is a canonically sorted JSON string (keys sorted, floats normalized) to guarantee stability across Python versions and platforms.

The default backend is a local SQLite database co-located with the job state database. A Redis backend is available for multi-process or multi-machine setups.

### 5.2 Cache Schema (SQLite)

```sql
CREATE TABLE cache_entries (
    cache_key      TEXT PRIMARY KEY,          -- SHA-256 hex
    provider       TEXT NOT NULL,
    model          TEXT NOT NULL,
    response_json  BLOB NOT NULL,             -- Zstd-compressed JSON
    input_tokens   INTEGER NOT NULL,
    output_tokens  INTEGER NOT NULL,
    created_at     REAL NOT NULL,             -- Unix timestamp
    last_hit_at    REAL NOT NULL,
    hit_count      INTEGER NOT NULL DEFAULT 0,
    expires_at     REAL,                      -- NULL = never expire
    size_bytes     INTEGER NOT NULL
);

CREATE INDEX idx_cache_expires ON cache_entries(expires_at);
CREATE INDEX idx_cache_provider_model ON cache_entries(provider, model);
```

### 5.3 Cache Behaviour

- On every `submit()` call, the engine computes cache keys for all requests before any API call.
- Requests with valid, non-expired cache hits are removed from the batch and their results are returned directly. The job's `cached_hits` counter is updated.
- If **all** requests are cache hits, no API call is made and the job immediately transitions to `COMPLETED`.
- After `download()`, every new result is written to the cache automatically.
- TTL expiry is checked lazily on read and proactively by a background vacuum task that runs every hour.
- Cache size is enforced by LRU eviction: when `max_size_gb` is exceeded, the least-recently-hit entries are deleted.

### 5.4 Cache CLI Commands

```bash
relay cache stats                       # Size, hit rate, entry count
relay cache list --provider anthropic   # List entries with filters
relay cache invalidate <cache_key>      # Delete a specific entry
relay cache invalidate --job <job_id>   # Delete all entries for a job
relay cache vacuum                      # Force TTL expiry sweep + LRU eviction
relay cache clear                       # Delete all cache entries (destructive)
```

---

## 6. Job Persistence & Storage

### 6.1 SQLite Schema

All job state is stored in a single SQLite database file. WAL mode is enabled for concurrent read access. Full DDL is in `relay/db/schema.sql`.

```sql
-- Core job table
CREATE TABLE jobs (
    id                  TEXT PRIMARY KEY,
    provider_job_id     TEXT,
    provider            TEXT NOT NULL,
    model               TEXT NOT NULL,
    project             TEXT,
    description         TEXT,
    status              TEXT NOT NULL,
    total_requests      INTEGER NOT NULL DEFAULT 0,
    completed_requests  INTEGER NOT NULL DEFAULT 0,
    failed_requests     INTEGER NOT NULL DEFAULT 0,
    cached_hits         INTEGER NOT NULL DEFAULT 0,
    input_tokens        INTEGER NOT NULL DEFAULT 0,
    output_tokens       INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd  REAL,
    actual_cost_usd     REAL,
    created_at          REAL NOT NULL,
    submitted_at        REAL,
    completed_at        REAL,
    config_json         TEXT NOT NULL,
    error               TEXT
);

-- Individual request tracking
CREATE TABLE requests (
    id           TEXT PRIMARY KEY,
    job_id       TEXT NOT NULL REFERENCES jobs(id),
    cache_key    TEXT,
    status       TEXT NOT NULL,   -- pending|completed|failed|cached
    payload_json TEXT NOT NULL,   -- Compressed BatchRequest JSON
    error        TEXT
);

-- Results table (populated after download)
CREATE TABLE results (
    request_id    TEXT PRIMARY KEY REFERENCES requests(id),
    job_id        TEXT NOT NULL,
    content       TEXT,
    stop_reason   TEXT,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    from_cache    INTEGER NOT NULL DEFAULT 0,
    response_json TEXT                           -- Full provider payload
);

-- Tags for flexible filtering
CREATE TABLE job_tags (
    job_id TEXT NOT NULL REFERENCES jobs(id),
    tag    TEXT NOT NULL,
    PRIMARY KEY (job_id, tag)
);
```

### 6.2 Checkpointing

During a download, results are committed to the database in batches of 500 records within a single transaction. If the process is killed mid-download, the job engine detects the partial state on restart by comparing `completed_requests` with the row count in `results` and resumes from where it left off.

### 6.3 Job History & Queries

```python
# Python
jobs = await client.list_jobs(
    project='my-annotation-project',
    provider='anthropic',
    status=['COMPLETED', 'PARTIAL'],
    after=datetime(2025, 1, 1),
    tags=['production'],
    order_by='created_at',
    limit=50,
)

# CLI
relay jobs list --project my-annotation-project --status COMPLETED
relay jobs list --provider openai --since '7 days ago' --format table
```

---

## 7. Command-Line Interface

`relay` ships a fully-featured CLI built with **Typer** and **Rich**.

### 7.1 Top-Level Commands

```
Usage: relay [OPTIONS] COMMAND [ARGS]...

  submit      Submit a new batch job from a JSONL file
  status      Show status of one or more jobs
  wait        Wait for a job to complete, streaming live progress
  download    Download results for a completed job
  run         submit + wait + download in one command
  jobs        List, inspect, cancel, resubmit, and tag jobs
  cache       Inspect and manage the response cache
  estimate    Estimate cost and tokens for a JSONL file (no submission)
  dashboard   Launch the web monitoring dashboard
  export      Export results to CSV, Parquet, or HuggingFace Dataset
  config      Validate and print effective configuration
  db          Low-level database utilities (backup, vacuum, migrate)
```

### 7.2 Key Command Details

#### `relay submit`

```bash
relay submit requests.jsonl \
  --provider anthropic \
  --model claude-opus-4-5 \
  --project my-project \
  --tag production \
  --no-cache \
  --output-dir ./results \
  --confirm-above 5.00
```

Input JSONL has one JSON object per line. Each object must have an `id` field and a `messages` array with text-only content. Use `--schema-validate` to validate all records and print a Rich error table for any invalid rows before submitting.

#### `relay run` (most common workflow)

```bash
relay run requests.jsonl \
  --provider openai --model gpt-4o \
  --project analysis-2025 \
  --output results/openai_run1.jsonl \
  --watch                       # Live terminal progress bar
```

During `--watch`, a Rich Live table shows: elapsed time, ETA, requests/min, cost so far, and cache hit %.

#### `relay status`

```bash
relay status <job_id>          # Single job
relay status --all --live      # All active jobs, refreshed every 5s
relay status --project my-project --format json
```

#### `relay estimate`

```bash
relay estimate requests.jsonl --provider anthropic --model claude-opus-4-5

# Output:
# Requests:        12,480
# Cache hits:       3,201  (saved $1.24)
# Net requests:     9,279
# Input tokens:    4.2M
# Output tokens:   ~1.8M  (estimated from max_tokens)
# Estimated cost:  $6.74
```

#### `relay export`

```bash
relay export <job_id> --format parquet --output ./results/job.parquet
relay export <job_id> --format hf_dataset --output ./results/hf/ --push hub/org/dataset
```

---

## 8. Terminal Dashboard

`relay` includes a live-updating terminal dashboard powered by **Textual**. Launch with:

```bash
relay dashboard --tui
```

### 8.1 Dashboard Panels

| Panel | Contents |
|---|---|
| **Active Jobs** | Table of all `IN_PROGRESS` or `SUBMITTING` jobs. Shows job ID, provider, model, project, progress bar (completed/total), current cost, elapsed time, and estimated ETA. |
| **Throughput** | Sparkline charts: requests/sec and tokens/sec over the last 60 seconds, per provider. |
| **Cost Summary** | Running totals: cost today, this week, this month, and all time. Broken down by provider and project. Budget burn-rate indicator. |
| **Cache Stats** | Hit rate (%), money saved today via cache, total cache size, entry count, and last vacuum timestamp. |
| **Recent Jobs** | Last 20 completed jobs with final status, total cost, and duration. Color-coded: green = COMPLETED, orange = PARTIAL, red = FAILED. |
| **Error Log** | Rolling tail of the last 50 error events across all providers: timestamp, job ID, provider, and error message. |

### 8.2 Keyboard Shortcuts

| Key | Action |
|---|---|
| `j` / `k` or `↑` / `↓` | Navigate job list |
| `Enter` | Open job detail view |
| `c` | Cancel selected job (with confirmation) |
| `r` | Resubmit failed requests in selected job |
| `e` | Export selected job results |
| `f` | Filter jobs (opens filter dialog) |
| `?` | Toggle help overlay |
| `q` | Quit dashboard |
| `Tab` | Switch between panels |
| `d` | Toggle dark/light theme |

---

## 9. Web Monitoring Dashboard

A lightweight web dashboard built with **FastAPI** and **HTMX** (no JavaScript build step required). It reads from the same SQLite database as the CLI.

```bash
relay dashboard            # Starts web server on http://127.0.0.1:7860
relay dashboard --port 8080 --host 0.0.0.0 --open
```

### 9.1 Web Dashboard Pages

| Page / Route | Description |
|---|---|
| `/` (Overview) | Summary cards: active jobs, total cost today, cache hit rate, errors in last hour. Three sparkline charts: throughput, cost accumulation, error rate. Auto-refreshes every 5 seconds via HTMX. |
| `/jobs` | Paginated, searchable, filterable table of all jobs. Columns: ID, provider, model, project, status badge, progress, cost, created at, duration. Click any row to open job detail. |
| `/jobs/{id}` | Full job detail: config, timing, token breakdown, per-request result table (paginated). Download results as JSONL or CSV directly from the page. Cancel or resubmit actions with confirmation modal. |
| `/projects` | Aggregated view grouped by project name: total jobs, total cost, tokens, average duration, and success rate per project. |
| `/cache` | Cache statistics and size breakdown by provider/model. Manual invalidation by job ID or cache key. Vacuum trigger button. |
| `/costs` | Cost analytics: bar chart of daily spending by provider (30-day window), cumulative spend chart, top-10 most expensive models, estimated vs actual cost accuracy. |
| `/settings` | View effective configuration (secrets redacted). Refresh provider pricing tables. Force cache vacuum. Trigger database backup. |

### 9.2 Security

The web dashboard is designed for local or internal network use. It does not implement authentication by default. For internet-facing deployments, place it behind a reverse proxy (nginx, Caddy) with basic auth or mTLS. A config option `relay.dashboard.auth_token` enables a simple bearer token check without a full reverse proxy.

---

## 10. Monitoring & Observability

### 10.1 Metrics

| Metric | Type | Description |
|---|---|---|
| `relay.requests.submitted` | Counter | Total requests submitted across all jobs |
| `relay.requests.completed` | Counter | Requests completed successfully |
| `relay.requests.failed` | Counter | Requests that returned an error |
| `relay.requests.cached` | Counter | Requests served from cache |
| `relay.tokens.input` | Counter | Total input tokens consumed |
| `relay.tokens.output` | Counter | Total output tokens generated |
| `relay.cost.estimated_usd` | Gauge | Estimated cost of active jobs |
| `relay.cost.actual_usd` | Counter | Confirmed spend from provider invoices |
| `relay.cache.hit_rate` | Gauge | Rolling 5-minute cache hit rate |
| `relay.cache.size_bytes` | Gauge | Current total cache storage |
| `relay.jobs.active` | Gauge | Number of jobs currently `IN_PROGRESS` |
| `relay.provider.latency_p50_ms` | Gauge | Median API response latency |
| `relay.provider.latency_p99_ms` | Gauge | P99 API response latency |

### 10.2 Metric Exporters

Three built-in exporters, selectable in config:

- **JSON Lines file:** Appends one JSON record per event to a `metrics.jsonl` file. Good for offline analysis with pandas or DuckDB.
- **Prometheus exposition:** Exposes `/metrics` endpoint when the web dashboard is running. Enables Grafana integration.
- **OpenTelemetry:** Emits metrics and traces to any OTLP-compatible backend (Jaeger, Zipkin, Honeycomb, Datadog) when `OTEL_EXPORTER_OTLP_ENDPOINT` is set.

Custom exporters can be registered via the `MetricExporter` ABC.

### 10.3 Logging

Uses Python's standard `logging` module with structured JSON output when `log_format = 'json'` is set. All log records include `job_id`, `provider`, and `request_id` context fields automatically via `contextvars`. Log files are rotated daily with a 30-day retention policy.

### 10.4 Progress Callbacks

```python
def my_progress_handler(progress: JobProgress):
    print(f'{progress.completed}/{progress.total} @ ${progress.cost_so_far:.2f}')

async def my_complete_handler(job: BatchJob, results: list[BatchResult]):
    send_slack_notification(f'Job {job.id} done: ${job.actual_cost_usd:.2f}')

config = BatchConfig(
    provider='anthropic',
    model='claude-opus-4-5',
    on_progress=my_progress_handler,
    on_complete=my_complete_handler,
)
```

---

## 11. Multi-Provider Patterns

### 11.1 Fan-Out: Same Prompts, Multiple Providers

```python
from relay import BatchClient, BatchConfig, fan_out

async with BatchClient() as client:
    results = await fan_out(
        client,
        requests=my_requests,
        configs=[
            BatchConfig(provider='anthropic', model='claude-opus-4-5'),
            BatchConfig(provider='openai',    model='gpt-4o'),
            BatchConfig(provider='google',    model='gemini-2.0-flash'),
            BatchConfig(provider='xai',       model='grok-3'),
        ],
        wait=True,
    )
    # results: dict[str, list[BatchResult]]
    # keys are '<provider>/<model>'
```

### 11.2 Parallel Independent Batches

```python
async with BatchClient() as client:
    jobs = await asyncio.gather(
        client.submit(requests_a, config_anthropic),
        client.submit(requests_b, config_openai),
        client.submit(requests_c, config_google),
    )
    all_results = await asyncio.gather(
        *[client.wait_and_download(j.id) for j in jobs]
    )
```

### 11.3 Sequential Chaining (Two-Stage Pipelines)

```python
async with BatchClient() as client:
    # Stage 1: classification
    stage1_results = await client.wait_and_download(
        (await client.submit(raw_requests, classifier_config)).id
    )

    # Build stage 2 prompts from stage 1 outputs
    stage2_requests = [
        BatchRequest(
            id=f'stage2-{r.request_id}',
            messages=[{'role': 'user', 'content': build_prompt(r.content)}],
        )
        for r in stage1_results if r.error is None
    ]

    # Stage 2: extraction
    stage2_results = await client.wait_and_download(
        (await client.submit(stage2_requests, extractor_config)).id
    )
```

---

## 12. Cost Management

### 12.1 Cost Estimation

Before submitting any batch, `relay` estimates cost using an internal token pricing table (refreshed via `relay db update-prices`). The estimate reports:

- **Gross cost:** Cost if all requests were sent fresh
- **Cache savings:** Cost avoided by serving cache hits
- **Net estimated cost:** Gross minus cache savings
- **Confidence interval:** ±X% depending on output token predictability (`max_tokens` is used as the ceiling)

### 12.2 Budget Controls

| Control | Config Key | Behaviour |
|---|---|---|
| Soft warning | `warn_threshold_usd` | Prints a yellow warning with the estimate and proceeds automatically. |
| Interactive confirm | `require_confirmation_usd` | In CLI/REPL: prompts `"Cost $X. Proceed? [y/N]"`. In non-interactive mode: raises `BudgetConfirmationRequired`. |
| Hard limit | `hard_limit_usd` | Raises `BudgetExceeded` regardless of interactivity. Set to `0` to disable. |

### 12.3 Cost Reporting

```bash
relay costs today
relay costs --since '30 days ago' --group-by project
relay costs --since '30 days ago' --group-by model --format csv > costs.csv
```

```python
summary = await client.get_cost_summary(
    since=datetime(2025, 1, 1),
    group_by='provider',
)
# Returns list of CostSummaryRow(provider, total_usd, input_tokens, output_tokens)
```

---

## 13. Error Handling & Retry

### 13.1 Error Hierarchy

```
RelayError              # Base exception
  ProviderError            # Error from provider API
    RateLimitError         # HTTP 429 — retry-able
    AuthenticationError    # HTTP 401/403 — not retry-able
    ServerError            # HTTP 5xx — retry-able
    BatchExpiredError      # Provider-side expiry before completion
  ValidationError          # Malformed request
  BudgetExceeded           # Hard cost limit reached
  BudgetConfirmationRequired  # Interactive confirmation not possible
  CacheError               # Storage backend failure
  JobNotFound              # Unknown job ID queried
```

### 13.2 Retry Policy

The retry policy is applied at two granularities:

- **Request-level:** Individual requests within a batch that return retryable error codes are retried transparently by the adapter, respecting `Retry-After` headers when present.
- **Job-level:** If a batch job fails at the API level (connection reset, server error on poll), the entire job submission is retried up to `max_attempts` with exponential backoff.

Non-retryable errors (401, 403, 400 validation) are surfaced immediately without retry.

### 13.3 Dead Letter Handling

Requests that exhaust all retry attempts are written to `{output_dir}/{job_id}_dead_letter.jsonl`. Each record includes the original request, the last error response, and a retry count. These can be inspected and resubmitted with:

```bash
relay jobs resubmit-failed <job_id>
```

---

## 14. Output Formats & Exporters

### 14.1 Supported Formats

| Format | Extension | Use Case | Notes |
|---|---|---|---|
| JSON Lines | `.jsonl` | Default; streaming-friendly | One `BatchResult` JSON per line |
| CSV | `.csv` | Spreadsheet tools | Flattened fields; metadata as JSON string column |
| Parquet | `.parquet` | pandas, DuckDB, Spark | Strongly typed; requires `pyarrow` |
| HF Dataset | directory | HuggingFace Transformers | Arrow format; `--push` to push to Hub |
| SQLite | `.db` | Direct SQL queries | Full schema with joined requests + results tables |

### 14.2 Exporter Python API

```python
from relay.exporters import export_job

path = await export_job(
    client,
    job_id='job_abc123',
    format='parquet',
    output_path='./results/job_abc123.parquet',
    include_metadata=True,
    include_raw_response=False,
)

# Stream large exports without loading all results into memory
async for chunk in client.stream_results(job_id='job_abc123', chunk_size=1000):
    process_chunk(chunk)
```

---

## 15. Repository Structure

Create this exact directory layout:

```
relay/
├── pyproject.toml              # PEP 621 metadata, extras, entry points
├── README.md
├── CHANGELOG.md
├── LICENSE                     # MIT
├── Makefile                    # test, lint, type-check, docs targets
│
├── relay/                   # Main package
│   ├── __init__.py             # Public API: BatchClient, BatchRequest, BatchConfig, fan_out
│   ├── client.py               # BatchClient implementation
│   ├── models.py               # All data models (dataclasses + Pydantic v2 variants)
│   ├── exceptions.py           # Error hierarchy
│   ├── config.py               # Config loading, validation, env interpolation
│   ├── fan_out.py              # Multi-provider fan-out helper
│   │
│   ├── providers/              # Provider adapters
│   │   ├── __init__.py         # Provider registry
│   │   ├── base.py             # BaseProvider ABC
│   │   ├── anthropic.py        # Anthropic Message Batches adapter
│   │   ├── openai.py           # OpenAI Batch API adapter
│   │   ├── google.py           # Google Gemini / Vertex AI adapter
│   │   └── xai.py              # XAI Grok adapter (concurrent pool)
│   │
│   ├── cache/                  # Cache layer
│   │   ├── __init__.py
│   │   ├── base.py             # CacheBackend ABC
│   │   ├── sqlite.py           # SQLite backend
│   │   └── redis.py            # Redis backend
│   │
│   ├── db/                     # Job persistence
│   │   ├── __init__.py
│   │   ├── schema.sql          # Full DDL
│   │   ├── migrations/         # Alembic-compatible migration scripts
│   │   ├── store.py            # JobStore: CRUD for jobs/requests/results/tags
│   │   └── prices.py           # Token pricing table + update logic
│   │
│   ├── monitoring/             # Metrics and observability
│   │   ├── __init__.py
│   │   ├── bus.py              # Internal event bus
│   │   ├── metrics.py          # Metric definitions
│   │   └── exporters/
│   │       ├── jsonl.py
│   │       ├── prometheus.py
│   │       └── otel.py
│   │
│   ├── exporters/              # Result format exporters
│   │   ├── __init__.py
│   │   ├── jsonl.py
│   │   ├── csv.py
│   │   ├── parquet.py
│   │   └── hf_dataset.py
│   │
│   ├── cli/                    # Typer CLI
│   │   ├── __init__.py         # Root app
│   │   ├── submit.py           # submit, run commands
│   │   ├── jobs.py             # jobs subcommand group
│   │   ├── cache.py            # cache subcommand group
│   │   ├── costs.py            # costs command
│   │   ├── estimate.py         # estimate command
│   │   └── export.py           # export command
│   │
│   ├── dashboard/              # Textual TUI + FastAPI web dashboard
│   │   ├── tui/
│   │   │   ├── app.py          # Textual Application subclass
│   │   │   └── widgets/        # Individual panel widgets
│   │   └── web/
│   │       ├── app.py          # FastAPI app factory
│   │       ├── routes/         # Route modules per page
│   │       └── templates/      # Jinja2 HTML templates
│   │
│   └── utils/
│       ├── hashing.py          # Cache key computation (canonical JSON + SHA-256)
│       ├── tokenizer.py        # Token counting utilities
│       └── retry.py            # Generic async retry decorator
│
└── tests/
    ├── unit/
    ├── integration/            # Requires provider API keys + pytest-vcr cassettes
    └── fixtures/               # Sample JSONL files, VCR cassettes
```

---

## 16. Dependency Specification

### 16.1 pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "relay"
version = "1.0.0"
requires-python = ">= 3.10"
description = "Multi-provider LLM batch prediction library"
license = {text = "MIT"}
readme = "README.md"

dependencies = [
    "httpx>=0.27",           # Async HTTP client
    "aiofiles>=23.0",        # Async file I/O
    "aiosqlite>=0.20",       # Async SQLite
    "pydantic>=2.0",         # Data validation
    "tomli>=2.0;python_version<'3.11'",
    "rich>=13.0",            # Terminal formatting
    "typer>=0.12",           # CLI
    "zstandard>=0.22",       # Cache compression
]

[project.optional-dependencies]
anthropic = ["anthropic>=0.43"]
openai    = ["openai>=1.50", "tiktoken>=0.7"]
google    = ["google-generativeai>=0.8", "google-cloud-aiplatform>=1.70"]
xai       = ["httpx>=0.27"]      # XAI uses plain HTTPX (same as core dep)
dashboard = ["fastapi>=0.115", "uvicorn[standard]>=0.30",
             "jinja2>=3.1", "textual>=0.80"]
parquet   = ["pyarrow>=17.0"]
redis     = ["redis[hiredis]>=5.0"]
otel      = ["opentelemetry-api>=1.28", "opentelemetry-sdk>=1.28",
             "opentelemetry-exporter-otlp>=1.28"]
hf        = ["datasets>=3.0"]
all       = ["relay[anthropic,openai,google,xai,dashboard,parquet,redis,hf]"]
dev       = ["pytest>=8", "pytest-asyncio>=0.24", "pytest-vcr>=1.0",
             "ruff>=0.7", "mypy>=1.12", "hypothesis>=6.0", "pre-commit>=4.0"]

[project.scripts]
relay = "relay.cli:app"

[project.entry-points."relay.providers"]
# Third-party providers register here
```

---

## 17. Testing Strategy

### 17.1 Unit Tests

Cover all pure logic in isolation: request hashing, cost estimation, retry policy, status transitions, export formatters, config parsing, and cache key generation. Provider adapters are tested against recorded HTTP cassettes (`pytest-vcr`) so no live API keys are required. **Target: 90%+ unit test coverage.**

### 17.2 Integration Tests

Gated behind the `--integration` flag. Require real API keys set as environment variables. Submit small batches (3–5 requests) to each provider, verify the full lifecycle (submit → poll → download → cache → re-submit), and check cost tracking accuracy. Run in CI on a **nightly schedule**, not on every PR, to limit API spend.

```bash
pytest tests/integration --integration  # Runs live tests
```

### 17.3 Property-Based Tests

Hypothesis-based tests verify:

- Any valid `BatchRequest` round-trips through serialization without data loss.
- The cache key function is deterministic: identical inputs always produce the same key.
- The retry policy always terminates and never exceeds `max_attempts`.
- Cost estimation is always non-negative and never exceeds `(input_tokens + max_tokens) * price`.

### 17.4 CI Configuration

```yaml
# .github/workflows/ci.yml
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12', '3.13']
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e '.[dev,all]'
      - run: ruff check . && ruff format --check .
      - run: mypy relay
      - run: pytest tests/unit --cov=relay --cov-report=xml
      - uses: codecov/codecov-action@v4
```

---

## 18. Complete Usage Examples

### 18.1 Simple Research Script

```python
import asyncio
from relay import BatchClient, BatchRequest, BatchConfig

PROMPTS = [
    'Summarize the key findings of climate change research in 2024.',
    'What are the main causes of inflation?',
    # ... thousands more
]

async def main():
    requests = [
        BatchRequest(
            id=f'q{i}',
            messages=[{'role': 'user', 'content': p}],
            max_tokens=512,
        )
        for i, p in enumerate(PROMPTS)
    ]

    config = BatchConfig(
        provider='anthropic',
        model='claude-opus-4-5',
        project='research-2025',
        tags=['pilot'],
    )

    async with BatchClient() as client:
        cost = await client.estimate_cost(requests, config)
        print(f'Estimated cost: ${cost.net_usd:.2f} (cache saves ${cost.saved_usd:.2f})')

        job = await client.submit(requests, config)
        print(f'Submitted job: {job.id}')

        results = await client.wait_and_download(job.id)
        print(f'Got {len(results)} results')

        await client.export(job.id, format='parquet', path='./output.parquet')

asyncio.run(main())
```

### 18.2 Multi-Provider Comparison

```python
from relay import BatchClient, BatchConfig, fan_out
import pandas as pd

async with BatchClient() as client:
    results = await fan_out(
        client,
        requests=eval_requests,
        configs=[
            BatchConfig(provider='anthropic', model='claude-opus-4-5', project='eval'),
            BatchConfig(provider='openai',    model='gpt-4o',          project='eval'),
            BatchConfig(provider='google',    model='gemini-2.0-flash', project='eval'),
        ],
        wait=True,
    )

    # Align results by request ID for side-by-side comparison
    rows = []
    for req_id in [r.id for r in eval_requests]:
        row = {'id': req_id}
        for provider_model, res_list in results.items():
            res = next((r for r in res_list if r.request_id == req_id), None)
            row[provider_model] = res.content if res else None
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv('model_comparison.csv', index=False)
```

### 18.3 CLI One-Liner for Production

```bash
# Submit, wait, download, and export in one command
relay run prompts.jsonl \
  --provider anthropic \
  --model claude-opus-4-5 \
  --project production-2025 \
  --tag v2.1 \
  --output ./results/run_$(date +%Y%m%d).parquet \
  --format parquet \
  --confirm-above 20.00 \
  --watch
```

---

## 19. Implementation Instructions for Claude Code

> **Important:** This section contains specific instructions for the AI coding agent implementing this library. Follow these steps in order.

### 19.1 Implementation Order

1. Set up repository structure exactly as described in [Section 15](#15-repository-structure). Initialize `pyproject.toml` per [Section 16](#16-dependency-specification).
2. Implement `relay/models.py` with all data models from [Section 3.2](#32-data-models). Add full Pydantic v2 model variants alongside the dataclasses.
3. Implement `relay/exceptions.py` with the error hierarchy from [Section 13.1](#131-error-hierarchy).
4. Implement `relay/config.py`: TOML loading, environment variable interpolation (`${VAR}` syntax), default values, and config merging from all sources.
5. Implement `relay/db/`: `schema.sql` first, then `store.py` using `aiosqlite`. Enable WAL mode. Implement all CRUD methods for jobs, requests, results, and tags.
6. Implement `relay/utils/retry.py`: the async retry decorator with full exponential backoff policy.
7. Implement `relay/utils/hashing.py`: deterministic, stable cache key computation. Write property tests immediately after.
8. Implement `relay/cache/`: SQLite backend first, then Redis. Implement LRU eviction and TTL vacuum.
9. Implement `relay/providers/base.py`, then all four provider adapters. Start with `anthropic.py` and `openai.py` as they have the most stable batch APIs. Add comprehensive `pytest-vcr` cassettes for each adapter.
10. Implement `relay/client.py`: the `BatchClient` orchestrating all above components.
11. Implement `relay/monitoring/`: event bus, metrics, and JSONL/Prometheus exporters.
12. Implement `relay/exporters/`: JSONL first (simplest), then CSV, Parquet (`pyarrow`), HF Dataset.
13. Implement `relay/cli/`: full Typer CLI with all commands from [Section 7](#7-command-line-interface). Use Rich for tables and progress bars.
14. Implement `relay/dashboard/tui/`: Textual TUI with panels from [Section 8](#8-terminal-dashboard).
15. Implement `relay/dashboard/web/`: FastAPI + HTMX web dashboard with pages from [Section 9](#9-web-monitoring-dashboard). Use Jinja2 templates. All JS via HTMX CDN only — no build step.
16. Implement `relay/fan_out.py`: the multi-provider fan-out helper from [Section 11.1](#111-fan-out-same-prompts-multiple-providers).
17. Write `tests/unit/` coverage for all components. Aim for 90%.

### 19.2 Key Implementation Notes

- Use **Python 3.10+ syntax**: `match`/`case` for status transitions, union types with `|`, `ParamSpec` for the retry decorator.
- **All I/O must be async.** Never use blocking calls (`open`, `requests`, `sqlite3`) in async code paths. Use `aiofiles` and `aiosqlite` throughout.
- The SQLite database must use `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout=5000` to handle concurrent readers.
- The **cache key hash** must be computed from a canonically sorted JSON serialization (keys sorted recursively, floats normalized to a fixed precision) to guarantee stability across Python versions and platforms.
- **Provider adapters must never raise bare exceptions.** All errors must be mapped to the exception hierarchy in `exceptions.py`.
- All config values must support env-var interpolation. Use a simple `re.sub` pass over the raw TOML string before parsing with `tomllib`.
- The **Textual TUI** must not crash if the database is locked or unavailable. Show a graceful error state panel instead.
- The **web dashboard** must set a `Content-Security-Policy` header. HTMX CDN link: `https://unpkg.com/htmx.org@1.9.12`.
- Token pricing data must be stored in `relay/db/prices.json` and loaded at startup. Ship a static snapshot for the initial version; the `update-prices` command can be implemented as a follow-up.
- The `relay run` CLI command must print a clean **Rich Live** progress table showing: elapsed, ETA, requests/min, cost so far, and cache hit %.
- **Text-only:** All message content handling must accept only `str` values. Reject (with `ValidationError`) any message content that is not a plain string. No image, audio, or tool-result content types.

### 19.3 Environment Variables for Integration Tests

```bash
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
GCP_PROJECT_ID=...
XAI_API_KEY=...
```

Integration tests must check for the presence of required env vars and **skip** (not fail) gracefully if they are absent:

```python
pytest.importorskip  # or use pytest.mark.skipif with os.getenv checks
```

### 19.4 Documentation Requirements

- Every public function, class, and method must have a **Google-style docstring**.
- `README.md` must include: installation, quickstart (10 lines of code), feature overview, provider compatibility table, and a link to full docs.
- Generate API reference docs with `pdoc3`. Include a `make docs` target in the `Makefile`.

---

## Appendix A: Provider Pricing Reference

*Prices as of early 2025. Run `relay db update-prices` to refresh.*

| Provider / Model | Input $/1M tokens | Output $/1M tokens | Batch Discount |
|---|---|---|---|
| `claude-opus-4-5` | $15.00 | $75.00 | 50% via Batch API |
| `claude-sonnet-4-6` | $3.00 | $15.00 | 50% via Batch API |
| `claude-haiku-4-5` | $0.80 | $4.00 | 50% via Batch API |
| `gpt-4o` | $2.50 | $10.00 | 50% via Batch API |
| `gpt-4o-mini` | $0.15 | $0.60 | 50% via Batch API |
| `o3` | $10.00 | $40.00 | 50% via Batch API |
| `gemini-2.0-flash` | $0.075 | $0.30 | None (standard) |
| `gemini-2.5-pro` | $1.25 | $10.00 | None (standard) |
| `grok-3` | $3.00 | $15.00 | TBD |
| `grok-3-mini` | $0.30 | $0.50 | TBD |

---

## Appendix B: Glossary

| Term | Definition |
|---|---|
| **Batch Job** | A single submission of N requests to a provider's batch API endpoint, tracked as a unit by `relay`. |
| **Cache Key** | A SHA-256 hash of the canonical request fields that uniquely identifies a `(prompt, model, params)` tuple for caching purposes. |
| **Provider Adapter** | A class implementing `BaseProvider` that translates `relay`'s generic request format into a specific provider's API format. |
| **Fan-Out** | Submitting identical requests to multiple providers simultaneously for comparison or ensemble purposes. |
| **Dead Letter** | A request that exhausted all retry attempts. Stored in a JSONL sidecar file for later inspection and resubmission. |
| **TTL** | Time-To-Live: the maximum age of a cache entry before it is considered stale and evicted. |
| **LRU Eviction** | Least-Recently-Used eviction: removing the oldest-accessed entries first when the cache exceeds its size limit. |
| **WAL Mode** | Write-Ahead Logging: a SQLite journal mode that allows concurrent reads while a write is in progress. |
| **Fan-Out** | Submitting identical prompts to multiple providers in parallel for comparison purposes. |
