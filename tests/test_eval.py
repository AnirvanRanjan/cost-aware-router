"""Unit tests for router evaluation, PGR, CPT, APGR, and the ablation ladder (DESIGN.md § 3.3, § 4.3)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from router.eval import (
    RoutingCurve,
    RoutingMetrics,
    build_ablation_table,
    compute_binary_auc,
    compute_pgr,
    compute_routing_curve,
    evaluate_router_config,
)


def test_pgr_mathematical_properties():
    q_S, q_L = 0.40, 0.90
    assert compute_pgr(0.40, q_S, q_L) == pytest.approx(0.0)
    assert compute_pgr(0.90, q_S, q_L) == pytest.approx(1.0)
    assert compute_pgr(0.40 + 0.95 * 0.50, q_S, q_L) == pytest.approx(0.95)

    # Zero headroom does not divide by zero
    assert compute_pgr(0.50, 0.50, 0.50) == 0.0


def test_random_router_baseline_metrics():
    rng = np.random.default_rng(42)
    n = 2000
    s_S = rng.binomial(1, 0.35, size=n).astype(float)
    s_L = rng.binomial(1, 0.85, size=n).astype(float)
    random_scores = rng.uniform(0.0, 1.0, size=n)

    m = evaluate_router_config("Random router", random_scores, s_S, s_L)
    assert m.auc == pytest.approx(0.50, abs=0.03)
    assert m.spearman_rho == pytest.approx(0.00, abs=0.04)
    assert m.apgr == pytest.approx(0.50, abs=0.03)
    assert m.cpt_95 == pytest.approx(0.95, abs=0.03)
    assert m.savings_95 == pytest.approx(0.05, abs=0.03)


def test_oracle_router_ceiling_metrics():
    rng = np.random.default_rng(123)
    n = 1000
    s_S = rng.binomial(1, 0.40, size=n).astype(float)
    s_L = rng.binomial(1, 0.80, size=n).astype(float)
    oracle_scores = s_L - s_S

    m = evaluate_router_config("Oracle router", oracle_scores, s_S, s_L)
    assert m.auc == pytest.approx(1.0, abs=1e-5)
    assert m.spearman_rho == pytest.approx(1.0, abs=1e-5)
    assert m.apgr > 0.80
    assert m.cpt_95 < 0.60
    assert m.savings_95 > 0.40


def test_cost_normalized_gain_v2_savings():
    # Construct a dataset where queries have equal quality gap but unequal cost
    # Short query: gap = 1.0, cost = 1.0
    # Long query:  gap = 1.0, cost = 10.0
    n = 100
    s_S = np.zeros(n)
    s_L = np.ones(n)
    gap = s_L - s_S

    costs_S = np.zeros(n)
    costs_L = np.array([1.0] * 50 + [10.0] * 50)

    # v1 (raw gap): all queries tied with score 1.0
    scores_v1 = gap
    curve_v1 = compute_routing_curve(scores_v1, s_S, s_L, costs_S=costs_S, costs_L=costs_L)

    # v2 (cost-normalized gain: G = gap / delta_cost):
    # Short queries receive gain 1.0 / 1.0 = 1.0; long queries receive 1.0 / 10.0 = 0.10
    scores_v2 = gap / (costs_L - costs_S)
    curve_v2 = compute_routing_curve(scores_v2, s_S, s_L, costs_S=costs_S, costs_L=costs_L)

    # Cost-normalized rule routes cheap queries first, dramatically boosting savings at 95% PGR
    assert curve_v2.savings_95 > curve_v1.savings_95


def test_build_ablation_table():
    metrics = [
        RoutingMetrics("Random router", 0.50, 0.00, 0.95, 0.05, 0.50, 0.85, 0.40, 0.90),
        RoutingMetrics("F0 heuristics + LogReg", 0.62, 0.25, 0.80, 0.20, 0.65, 0.87, 0.40, 0.90),
        RoutingMetrics("F2 + F0 (headline)", 0.78, 0.60, 0.45, 0.55, 0.82, 0.89, 0.40, 0.90),
        RoutingMetrics("Oracle router", 1.00, 1.00, 0.30, 0.70, 0.95, 0.90, 0.40, 0.90),
    ]
    df = build_ablation_table(metrics)
    assert len(df) == 4
    expected_cols = ["Config", "AUC", "rho(delta_hat, delta)", "CPT(95%)", "Savings @ PGR .95", "APGR"]
    assert list(df.columns) == expected_cols
    assert df.loc[0, "Config"] == "Random router"
    assert df.loc[3, "Config"] == "Oracle router"
