import numpy as np
import pytest

from router.costs import estimate_run, frame_cost, local_cost, wall_clock_hours


def test_sonnet_list_price(prices):
    # 1,000 in x $3/M + 500 out x $15/M = $0.003 + $0.0075
    assert prices.query_cost("claude-sonnet-4.6", 1000, 500) == pytest.approx(0.0105)


def test_batch_halves_the_bill(prices):
    assert prices.query_cost("claude-sonnet-4.6", 1000, 500, batch=True) == pytest.approx(0.00525)


def test_cache_reads_bill_at_a_tenth(prices):
    # 200 fresh x $3 + 800 cached x $0.30 + 500 out x $15, per million
    assert prices.query_cost("claude-sonnet-4.6", 1000, 500, cached_in=800) == pytest.approx(0.00834)


def test_local_model_is_free_under_m1(prices):
    assert prices.query_cost("qwen3.5-4b", 10_000, 10_000) == 0.0


def test_vectorised_matches_scalar(prices):
    n_in, n_out = np.array([10, 200, 3000]), np.array([5, 50, 700])
    got = prices.query_cost("claude-haiku-4.5", n_in, n_out)
    want = [prices.query_cost("claude-haiku-4.5", a, b) for a, b in zip(n_in, n_out)]
    assert got == pytest.approx(want)


def test_unknown_model_error_says_where_to_fix_it(prices):
    with pytest.raises(KeyError, match="configs/models.yaml"):
        prices.query_cost("gpt-9000", 1, 1)


@pytest.mark.parametrize("kwargs", [dict(n_in=-1, n_out=1), dict(n_in=10, n_out=1, cached_in=11)])
def test_rejects_impossible_token_counts(prices, kwargs):
    with pytest.raises(ValueError):
        prices.query_cost("claude-sonnet-4.6", **kwargs)


def test_local_cost_m2_one_hour_of_hardware():
    assert local_cost(3600, usd_per_hour=1.0, tokens_per_sec=1.0) == pytest.approx(1.0)


def test_wall_clock_for_the_track_b_gsm8k_run():
    tokens = 1500 * 3 * 250
    assert wall_clock_hours(tokens, 37) == pytest.approx(tokens / 37 / 3600)
    assert wall_clock_hours(tokens, 37, batch_speedup=4) == pytest.approx(tokens / 37 / 3600 / 4)


def test_estimate_run_matches_hand_calculation(prices):
    e = estimate_run(prices, "claude-sonnet-4.6", n_items=1000, k=1, mean_in=120, mean_out=250, batch=True)
    assert e.tokens_in == 120_000 and e.tokens_out == 250_000
    assert e.usd == pytest.approx((120_000 * 3 + 250_000 * 15) / 1e6 * 0.5)  # $2.055
    assert e.hours is None


def test_estimate_run_reports_hours_for_local_models(prices):
    e = estimate_run(prices, "qwen3.5-4b", n_items=1500, k=3, mean_in=120, mean_out=250)
    assert e.usd == 0.0
    assert e.hours == pytest.approx(1500 * 3 * 250 / prices["qwen3.5-4b"].tokens_per_sec / 3600)


def test_reference_models_are_priced(prices):
    assert not prices[prices.reference_frontier].is_local
    assert prices[prices.reference_local].is_local


def test_frame_cost_prices_only_the_frontier_arm(prices, syn):
    cost = frame_cost(syn.scores, prices)
    small = syn.scores["model"] == syn.config.small_model
    assert (cost[small] == 0).all()
    assert (cost[~small] > 0).all()
