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


def test_price_as_borrows_the_reference_price(prices):
    gpt4 = prices["gpt-4-1106-preview"]
    sonnet = prices["claude-sonnet-4.6"]
    assert (gpt4.input, gpt4.output) == (sonnet.input, sonnet.output)
    assert prices["mistralai/mixtral-8x7b-chat"].is_local


def test_price_as_must_point_at_a_priced_model(tmp_path):
    from router.costs import PriceTable
    bad = tmp_path / "m.yaml"
    bad.write_text("as_of: x\nbatch_discount: 0.5\ncache_read_multiplier: 0.1\nreference_frontier: a\n"
                   "reference_local: a\nmodels:\n  a: {price_as: nope}\n")
    with pytest.raises(KeyError, match="price_as"):
        PriceTable.load(bad)


# ---- T05 / T24: measured vs estimated speeds, cost model M2 -------------------------------------

def test_measured_flag_marks_only_timed_models(prices):
    assert prices["qwen3.5-4b"].measured and prices["llama-3.2-3b"].measured
    assert not prices["phi-4-mini"].measured


def test_hardware_hourly_rate_is_amortisation_plus_energy():
    from router.costs import Hardware
    hw = Hardware("m", price_usd=876, lifetime_years=1, hours_per_day=24 / 10, watts=1000, usd_per_kwh=0.5)
    # 876 / (365 * 2.4) = $1/h amortised, + 1 kW * $0.5/kWh
    assert hw.usd_per_hour == pytest.approx(1.5)


def test_local_query_cost_adds_prefill_when_known():
    from router.costs import Hardware, local_query_cost
    hw = Hardware("m", price_usd=876, lifetime_years=1, hours_per_day=2.4, watts=0, usd_per_kwh=0,
                  prefill_tokens_per_sec=10)
    # 3600 out tokens at 1 tok/s = 1 h = $1; 36,000 in tokens at 10 tok/s = 1 h = $1
    assert local_query_cost(36_000, 3600, hardware=hw, tokens_per_sec=1) == pytest.approx(2.0)


def test_crossover_is_where_local_and_api_cost_the_same(prices):
    from router.costs import crossover_tokens_per_sec, local_query_cost
    hw = prices.hardware["m2_16gb"]
    x = crossover_tokens_per_sec(prices, "claude-haiku-4.5", hardware=hw, n_in=120, n_out=250)
    api = prices.query_cost("claude-haiku-4.5", 120, 250)
    assert local_query_cost(120, 250, hardware=hw, tokens_per_sec=x) == pytest.approx(api)
    assert local_query_cost(120, 250, hardware=hw, tokens_per_sec=2 * x) < api


def test_pair_costs_m1_and_m2(prices):
    import pandas as pd
    from router.costs import pair_costs
    scores = pd.DataFrame({
        "qid": ["a", "b", "a", "b"],
        "model": ["mistralai/mixtral-8x7b-chat"] * 2 + ["gpt-4-1106-preview"] * 2,
        "score": [1.0, 0.0, 1.0, 1.0], "grader": "x", "grader_version": "1",
        "mean_n_in": [100.0, 200.0, 100.0, 200.0], "mean_n_out": [50.0, 10.0, 60.0, 20.0],
    })
    m1 = pair_costs(prices, scores, small="mistralai/mixtral-8x7b-chat", large="gpt-4-1106-preview")
    assert (m1["c_S"] == 0).all()
    assert m1.loc["a", "c_L"] == pytest.approx((100 * 3 + 60 * 15) / 1e6)  # priced as Sonnet 4.6
    batch = pair_costs(prices, scores, small="mistralai/mixtral-8x7b-chat", large="gpt-4-1106-preview",
                       frontier_batch=True)
    assert batch["c_L"].to_numpy() == pytest.approx(m1["c_L"].to_numpy() * 0.5)
    m2 = pair_costs(prices, scores, small="mistralai/mixtral-8x7b-chat", large="gpt-4-1106-preview",
                    cost_model="M2")
    assert (m2["c_S"] > 0).all() and (m2["c_L"] == m1["c_L"]).all()
