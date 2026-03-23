"""Unit tests for the pricing table and cost estimation utilities."""

from __future__ import annotations

import pytest

from relay.db.prices import (
    estimate_cost_usd,
    get_batch_discount_pct,
    get_batch_price_per_million,
    get_model_info,
    get_price_per_million,
    list_models,
)


# ── get_price_per_million ─────────────────────────────────────────────────────


class TestGetPricePerMillion:
    @pytest.mark.parametrize("model, expected_input, expected_output", [
        ("claude-opus-4-5", 15.0, 75.0),
        ("claude-sonnet-4-6", 3.0, 15.0),
        ("claude-haiku-4-5", 0.80, 4.0),
        ("gpt-4o", 2.50, 10.0),
        ("gpt-4o-mini", 0.15, 0.60),
    ])
    def test_known_models(self, model, expected_input, expected_output):
        inp, out = get_price_per_million(model)
        assert inp == pytest.approx(expected_input)
        assert out == pytest.approx(expected_output)

    def test_unknown_model_raises_key_error(self):
        with pytest.raises(KeyError):
            get_price_per_million("totally-fake-model-xyz")

    def test_returns_tuple_of_two_floats(self):
        inp, out = get_price_per_million("gpt-4o")
        assert isinstance(inp, float)
        assert isinstance(out, float)

    def test_prices_are_positive(self):
        for model in list_models():
            inp, out = get_price_per_million(model)
            assert inp >= 0
            assert out >= 0


# ── get_batch_price_per_million ───────────────────────────────────────────────


class TestGetBatchPricePerMillion:
    def test_50_pct_discount_halves_price(self):
        std_inp, std_out = get_price_per_million("gpt-4o")
        batch_inp, batch_out = get_batch_price_per_million("gpt-4o")
        assert batch_inp == pytest.approx(std_inp * 0.5)
        assert batch_out == pytest.approx(std_out * 0.5)

    def test_batch_price_le_standard_price(self):
        for model in list_models():
            std_inp, std_out = get_price_per_million(model)
            batch_inp, batch_out = get_batch_price_per_million(model)
            assert batch_inp <= std_inp
            assert batch_out <= std_out

    def test_unknown_model_raises_key_error(self):
        with pytest.raises(KeyError):
            get_batch_price_per_million("no-such-model")


# ── get_batch_discount_pct ────────────────────────────────────────────────────


class TestGetBatchDiscountPct:
    def test_known_discount(self):
        pct = get_batch_discount_pct("claude-opus-4-5")
        assert pct == pytest.approx(50.0)

    def test_discount_in_range(self):
        for model in list_models():
            pct = get_batch_discount_pct(model)
            assert 0.0 <= pct <= 100.0

    def test_unknown_model_raises_key_error(self):
        with pytest.raises(KeyError):
            get_batch_discount_pct("unknown-model")


# ── estimate_cost_usd ─────────────────────────────────────────────────────────


class TestEstimateCostUsd:
    def test_zero_tokens_zero_cost(self):
        cost = estimate_cost_usd("gpt-4o", input_tokens=0, output_tokens=0)
        assert cost == 0.0

    def test_one_million_input_tokens_equals_batch_price(self):
        # With 1M input tokens and 0 output, cost == batch input price per million.
        batch_inp, _ = get_batch_price_per_million("gpt-4o")
        cost = estimate_cost_usd("gpt-4o", input_tokens=1_000_000, output_tokens=0)
        assert cost == pytest.approx(batch_inp)

    def test_one_million_output_tokens_equals_batch_price(self):
        _, batch_out = get_batch_price_per_million("gpt-4o")
        cost = estimate_cost_usd("gpt-4o", input_tokens=0, output_tokens=1_000_000)
        assert cost == pytest.approx(batch_out)

    def test_non_batch_pricing(self):
        std_inp, std_out = get_price_per_million("gpt-4o")
        cost = estimate_cost_usd(
            "gpt-4o",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            use_batch_pricing=False,
        )
        assert cost == pytest.approx(std_inp + std_out)

    def test_batch_pricing_cheaper_than_standard(self):
        std_cost = estimate_cost_usd(
            "claude-opus-4-5",
            input_tokens=500_000,
            output_tokens=250_000,
            use_batch_pricing=False,
        )
        batch_cost = estimate_cost_usd(
            "claude-opus-4-5",
            input_tokens=500_000,
            output_tokens=250_000,
            use_batch_pricing=True,
        )
        assert batch_cost < std_cost

    def test_unknown_model_raises_key_error(self):
        with pytest.raises(KeyError):
            estimate_cost_usd("nonexistent-model", input_tokens=100, output_tokens=100)

    def test_cost_is_non_negative(self):
        for model in list_models():
            cost = estimate_cost_usd(model, input_tokens=1000, output_tokens=500)
            assert cost >= 0.0

    def test_concrete_calculation_gpt4o(self):
        # gpt-4o batch: input=$1.25/M, output=$5.00/M (50% off 2.50/10.0).
        # 100k input + 50k output:
        # = (100_000 * 1.25 + 50_000 * 5.0) / 1_000_000
        # = (125_000 + 250_000) / 1_000_000
        # = 0.375
        cost = estimate_cost_usd("gpt-4o", input_tokens=100_000, output_tokens=50_000)
        assert cost == pytest.approx(0.375, rel=1e-6)


# ── list_models ───────────────────────────────────────────────────────────────


class TestListModels:
    def test_returns_sorted_list(self):
        models = list_models()
        assert models == sorted(models)

    def test_contains_known_models(self):
        models = list_models()
        assert "claude-opus-4-5" in models
        assert "gpt-4o" in models

    def test_non_empty(self):
        assert len(list_models()) > 0


# ── get_model_info ────────────────────────────────────────────────────────────


class TestGetModelInfo:
    def test_returns_dict_with_required_keys(self):
        info = get_model_info("gpt-4o")
        assert "provider" in info
        assert "input_per_million" in info
        assert "output_per_million" in info

    def test_returns_copy(self):
        info1 = get_model_info("gpt-4o")
        info1["hacked"] = True
        info2 = get_model_info("gpt-4o")
        assert "hacked" not in info2

    def test_unknown_model_raises_key_error(self):
        with pytest.raises(KeyError):
            get_model_info("ghost-model")
