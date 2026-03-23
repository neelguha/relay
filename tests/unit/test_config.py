"""Unit tests for relay configuration loading."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from relay.config import (
    CacheConfig,
    CostConfig,
    DashboardConfig,
    RelayConfig,
    RetryConfig,
    _build_relay_config,
    _deep_merge,
    _interpolate_env_vars,
    load_config,
    reset_config,
)


# ── Default config values ─────────────────────────────────────────────────────


class TestDefaultConfigValues:
    def setup_method(self):
        reset_config()

    def test_relay_config_defaults(self):
        cfg = RelayConfig()
        assert cfg.db_path == "~/.relay/jobs.db"
        assert cfg.log_level == "INFO"
        assert cfg.log_dir == "~/.relay/logs"
        assert cfg.output_dir == "./relay_results"

    def test_cost_config_defaults(self):
        cfg = CostConfig()
        assert cfg.warn_threshold_usd == 1.00
        assert cfg.require_confirmation_usd == 10.00
        assert cfg.hard_limit_usd == 100.00

    def test_cache_config_defaults(self):
        cfg = CacheConfig()
        assert cfg.enabled is True
        assert cfg.backend == "sqlite"
        assert cfg.redis_url == "redis://localhost:6379/0"
        assert cfg.ttl_seconds == 2_592_000
        assert cfg.max_size_gb == 5.0
        assert cfg.compress is True

    def test_retry_config_defaults(self):
        cfg = RetryConfig()
        assert cfg.max_attempts == 5
        assert cfg.initial_backoff_seconds == 1.0
        assert cfg.backoff_multiplier == 2.0
        assert cfg.max_backoff_seconds == 60.0
        assert cfg.jitter == "full"
        assert 429 in cfg.retry_on

    def test_dashboard_config_defaults(self):
        cfg = DashboardConfig()
        assert cfg.enabled is True
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 7860
        assert cfg.theme == "dark"
        assert cfg.auto_refresh_seconds == 5

    def test_provider_defaults(self):
        cfg = RelayConfig()
        assert cfg.providers.anthropic.default_model == "claude-opus-4-5"
        assert cfg.providers.openai.default_model == "gpt-4o"
        assert cfg.providers.google.default_model == "gemini-2.0-flash"
        assert cfg.providers.xai.default_model == "grok-3"


# ── Environment variable interpolation ───────────────────────────────────────


class TestEnvVarInterpolation:
    def test_substitutes_known_var(self, monkeypatch):
        monkeypatch.setenv("MY_API_KEY", "sk-test-12345")
        result = _interpolate_env_vars('api_key = "${MY_API_KEY}"')
        assert result == 'api_key = "sk-test-12345"'

    def test_leaves_unknown_var_unchanged(self):
        result = _interpolate_env_vars("key = ${DOES_NOT_EXIST_XYZ}")
        assert result == "key = ${DOES_NOT_EXIST_XYZ}"

    def test_multiple_vars(self, monkeypatch):
        monkeypatch.setenv("HOST", "myhost")
        monkeypatch.setenv("PORT", "5432")
        result = _interpolate_env_vars("${HOST}:${PORT}")
        assert result == "myhost:5432"

    def test_no_tokens_unchanged(self):
        raw = "log_level = 'DEBUG'"
        assert _interpolate_env_vars(raw) == raw

    def test_partial_substitution(self, monkeypatch):
        monkeypatch.setenv("KNOWN", "val")
        result = _interpolate_env_vars("${KNOWN} ${UNKNOWN}")
        assert result == "val ${UNKNOWN}"


# ── TOML string loading ───────────────────────────────────────────────────────


class TestConfigFromToml:
    def setup_method(self):
        reset_config()

    def test_load_from_toml_file(self, tmp_path):
        config_file = tmp_path / "relay.toml"
        config_file.write_text(
            """
[relay]
log_level = "DEBUG"
db_path = "/tmp/test.db"

[relay.cost]
hard_limit_usd = 50.0

[relay.cache]
enabled = false
backend = "redis"

[relay.retry]
max_attempts = 3
""",
            encoding="utf-8",
        )
        cfg = load_config(cli_path=str(config_file))
        assert cfg.log_level == "DEBUG"
        assert cfg.db_path == "/tmp/test.db"
        assert cfg.cost.hard_limit_usd == 50.0
        assert cfg.cache.enabled is False
        assert cfg.cache.backend == "redis"
        assert cfg.retry.max_attempts == 3

    def test_load_with_provider_config(self, tmp_path):
        config_file = tmp_path / "relay.toml"
        config_file.write_text(
            """
[relay.providers.anthropic]
api_key = "sk-ant-test"
default_model = "claude-haiku-4-5"
""",
            encoding="utf-8",
        )
        cfg = load_config(cli_path=str(config_file))
        assert cfg.providers.anthropic.api_key == "sk-ant-test"
        assert cfg.providers.anthropic.default_model == "claude-haiku-4-5"
        # Unspecified providers retain defaults.
        assert cfg.providers.openai.default_model == "gpt-4o"

    def test_missing_cli_path_raises(self, tmp_path):
        nonexistent = tmp_path / "no_such_file.toml"
        with pytest.raises(FileNotFoundError):
            load_config(cli_path=str(nonexistent))

    def test_empty_toml_uses_defaults(self, tmp_path):
        config_file = tmp_path / "empty.toml"
        config_file.write_text("[relay]\n", encoding="utf-8")
        cfg = load_config(cli_path=str(config_file))
        # Should fall back to all defaults.
        assert cfg.log_level == "INFO"
        assert cfg.cost.hard_limit_usd == 100.0

    def test_env_var_in_toml(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY", "sk-from-env")
        config_file = tmp_path / "relay.toml"
        config_file.write_text(
            '[relay.providers.openai]\napi_key = "${TEST_API_KEY}"\n',
            encoding="utf-8",
        )
        cfg = load_config(cli_path=str(config_file))
        assert cfg.providers.openai.api_key == "sk-from-env"


# ── _deep_merge ───────────────────────────────────────────────────────────────


class TestDeepMerge:
    def test_override_scalar(self):
        result = _deep_merge({"a": 1}, {"a": 2})
        assert result == {"a": 2}

    def test_nested_dicts_merged(self):
        base = {"a": {"x": 1, "y": 2}}
        override = {"a": {"y": 99, "z": 3}}
        result = _deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 99, "z": 3}}

    def test_new_key_added(self):
        result = _deep_merge({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_does_not_mutate_inputs(self):
        base = {"a": {"x": 1}}
        override = {"a": {"x": 2}}
        _deep_merge(base, override)
        assert base == {"a": {"x": 1}}
        assert override == {"a": {"x": 2}}

    def test_scalar_overrides_dict_when_override_is_scalar(self):
        result = _deep_merge({"a": {"x": 1}}, {"a": "flat"})
        assert result == {"a": "flat"}
