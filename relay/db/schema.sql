-- relay database schema
-- WAL mode is enabled at connection time via PRAGMA journal_mode=WAL.

-- Core job table
CREATE TABLE IF NOT EXISTS jobs (
    id                  TEXT PRIMARY KEY,
    provider_job_id     TEXT,
    provider            TEXT NOT NULL,
    model               TEXT NOT NULL,
    project             TEXT,
    name                TEXT,
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
CREATE TABLE IF NOT EXISTS requests (
    id           TEXT PRIMARY KEY,
    job_id       TEXT NOT NULL REFERENCES jobs(id),
    cache_key    TEXT,
    status       TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    error        TEXT
);

-- Results table (populated after download)
CREATE TABLE IF NOT EXISTS results (
    request_id    TEXT PRIMARY KEY REFERENCES requests(id),
    job_id        TEXT NOT NULL,
    content       TEXT,
    stop_reason   TEXT,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    from_cache    INTEGER NOT NULL DEFAULT 0,
    response_json TEXT
);

-- Tags for flexible filtering
CREATE TABLE IF NOT EXISTS job_tags (
    job_id TEXT NOT NULL REFERENCES jobs(id),
    tag    TEXT NOT NULL,
    PRIMARY KEY (job_id, tag)
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_jobs_provider        ON jobs(provider);
CREATE INDEX IF NOT EXISTS idx_jobs_model           ON jobs(model);
CREATE INDEX IF NOT EXISTS idx_jobs_status          ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_project         ON jobs(project);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at      ON jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_requests_job_id      ON requests(job_id);
CREATE INDEX IF NOT EXISTS idx_requests_cache_key   ON requests(cache_key);
CREATE INDEX IF NOT EXISTS idx_results_job_id       ON results(job_id);
CREATE INDEX IF NOT EXISTS idx_job_tags_tag         ON job_tags(tag);
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_project_name ON jobs(project, name) WHERE name IS NOT NULL;
