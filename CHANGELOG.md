# Changelog

All notable changes to relay are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
relay adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2025-01-01

### Added

- `BatchClient` async context manager for end-to-end batch job lifecycle management
- Provider adapters for Anthropic (Claude), OpenAI (GPT), Google (Gemini), and XAI (Grok)
- SQLite-backed job persistence — jobs survive process restarts
- Content-addressable response cache (SQLite and Redis backends) with zstd compression
- Pre-submission token count and cost estimation with configurable budget controls
- `fan_out` helper for sending requests to multiple providers simultaneously
- Result exporters: JSONL, CSV, Parquet, HuggingFace Dataset
- CLI (`relay submit`, `relay run`, `relay estimate`, `relay export`, `relay jobs`, `relay cache`, `relay costs`)
- Terminal dashboard (Textual TUI) with live job progress, cost, and cache panels
- Web monitoring dashboard (FastAPI) with overview, jobs, and costs routes
- Monitoring bus with Prometheus, OpenTelemetry, and JSONL exporters
- Configurable retry with exponential backoff and jitter
