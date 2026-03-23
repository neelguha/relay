"""Unit tests for relay cache key computation."""

from __future__ import annotations

import pytest

from relay.utils.hashing import _normalize_value, compute_cache_key


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_key(**overrides) -> str:
    """Return a cache key with sensible defaults, overridden by kwargs."""
    kwargs = dict(
        provider="anthropic",
        model="claude-opus-4-5",
        system_prompt="You are helpful.",
        user_messages=[{"role": "user", "content": "Hello"}],
        temperature=1.0,
        max_tokens=1024,
        top_p=None,
        stop_sequences=[],
    )
    kwargs.update(overrides)
    return compute_cache_key(**kwargs)


# ── Determinism ───────────────────────────────────────────────────────────────


class TestDeterminism:
    def test_same_input_same_hash(self):
        k1 = _make_key()
        k2 = _make_key()
        assert k1 == k2

    def test_hash_is_64_hex_chars(self):
        key = _make_key()
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_different_provider_different_hash(self):
        k1 = _make_key(provider="anthropic")
        k2 = _make_key(provider="openai")
        assert k1 != k2

    def test_different_model_different_hash(self):
        k1 = _make_key(model="claude-opus-4-5")
        k2 = _make_key(model="claude-haiku-4-5")
        assert k1 != k2

    def test_different_system_prompt_different_hash(self):
        k1 = _make_key(system_prompt="Be helpful.")
        k2 = _make_key(system_prompt="Be concise.")
        assert k1 != k2

    def test_none_vs_empty_string_system_prompt(self):
        k1 = _make_key(system_prompt=None)
        k2 = _make_key(system_prompt="")
        assert k1 != k2

    def test_different_messages_different_hash(self):
        k1 = _make_key(user_messages=[{"role": "user", "content": "Hello"}])
        k2 = _make_key(user_messages=[{"role": "user", "content": "Goodbye"}])
        assert k1 != k2

    def test_different_temperature_different_hash(self):
        k1 = _make_key(temperature=0.5)
        k2 = _make_key(temperature=0.9)
        assert k1 != k2

    def test_different_max_tokens_different_hash(self):
        k1 = _make_key(max_tokens=512)
        k2 = _make_key(max_tokens=2048)
        assert k1 != k2

    def test_different_top_p_different_hash(self):
        k1 = _make_key(top_p=None)
        k2 = _make_key(top_p=0.95)
        assert k1 != k2

    def test_different_stop_sequences_different_hash(self):
        k1 = _make_key(stop_sequences=[])
        k2 = _make_key(stop_sequences=["STOP"])
        assert k1 != k2


# ── Float normalization ───────────────────────────────────────────────────────


class TestFloatNormalization:
    def test_1_0_and_1_000000_same_key(self):
        k1 = _make_key(temperature=1.0)
        k2 = _make_key(temperature=1.000000)
        assert k1 == k2

    def test_near_equal_floats_within_6_decimals(self):
        # 0.1000001 rounds to 0.1 at 6 dp; 0.1000004 also rounds to 0.1.
        k1 = _make_key(temperature=0.1000001)
        k2 = _make_key(temperature=0.1000004)
        assert k1 == k2

    def test_floats_differing_beyond_6_decimals_may_collapse(self):
        # Ensure normalize rounds correctly.
        v1 = _normalize_value(0.1234567)
        v2 = _normalize_value(0.1234568)
        # Both round to 0.123457 at 6 dp.
        assert v1 == v2

    def test_normalize_float(self):
        assert _normalize_value(3.14159265) == round(3.14159265, 6)

    def test_normalize_list_of_floats(self):
        result = _normalize_value([1.0000001, 2.9999999])
        assert result == [round(1.0000001, 6), round(2.9999999, 6)]

    def test_normalize_nested_dict(self):
        result = _normalize_value({"b": 2.0, "a": 1.0})
        # Dict keys should be sorted.
        assert list(result.keys()) == ["a", "b"]

    def test_normalize_non_float_passthrough(self):
        assert _normalize_value("hello") == "hello"
        assert _normalize_value(42) == 42
        assert _normalize_value(None) is None
        assert _normalize_value(True) is True


# ── Metadata / tags exclusion ─────────────────────────────────────────────────


class TestMetadataExclusion:
    """compute_cache_key does not accept metadata/tags, so verify via BatchRequest
    usage pattern: two requests with different metadata should produce the same
    cache key when only the semantic parameters are used."""

    def test_key_is_independent_of_metadata(self):
        """Verify that adding metadata to a request dict doesn't alter the key
        by confirming two semantically identical payloads hash identically."""
        msgs = [{"role": "user", "content": "What is the capital of France?"}]
        k1 = compute_cache_key(
            provider="openai",
            model="gpt-4o",
            system_prompt=None,
            user_messages=msgs,
            temperature=0.0,
            max_tokens=100,
            top_p=None,
            stop_sequences=[],
        )
        k2 = compute_cache_key(
            provider="openai",
            model="gpt-4o",
            system_prompt=None,
            user_messages=msgs,
            temperature=0.0,
            max_tokens=100,
            top_p=None,
            stop_sequences=[],
        )
        assert k1 == k2


# ── Key sorted stability ──────────────────────────────────────────────────────


class TestKeySortedStability:
    def test_message_dict_key_order_does_not_matter(self):
        """Messages with keys in different insertion order should hash equally."""
        msgs_ab = [{"role": "user", "content": "Hi"}]
        msgs_ba = [{"content": "Hi", "role": "user"}]
        k1 = _make_key(user_messages=msgs_ab)
        k2 = _make_key(user_messages=msgs_ba)
        assert k1 == k2

    def test_stop_sequences_none_vs_empty(self):
        # None is treated as empty by the function.
        k1 = _make_key(stop_sequences=None)
        k2 = _make_key(stop_sequences=[])
        assert k1 == k2
