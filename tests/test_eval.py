"""Unit tests for router evaluation, PGR, CPT, APGR, and the ablation ladder (DESIGN.md § 3.3, § 4.3)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from router.eval import (
    LOSOSummary,
    RoutingCurve,
    RoutingMetrics,
    build_ablation_table,
    compute_binary_auc,
    compute_pgr,
    compute_routing_curve,
    evaluate_router_config,
    loso_subject_eval,
    run_real_loso,
    within_subject_eval,
    WithinSubjectSummary,
    RealLOSOSummary,
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


# ~~~ T26: leave-one-subject-out on MMLU ~~~

SPECIFIC_SUBJECTS = ("anatomy", "moral_scenarios", "world_religions")


def _mmlu_like(seed=7, n_per=200):
    """Subjects with very different difficulty, a router that is decent within a subject."""
    rng = np.random.default_rng(seed)
    subject_offsets = {"moral_scenarios": 0.0, "anatomy": 0.5, "world_religions": 1.0}
    subjects, s_S, s_L, scores = [], [], [], []
    for subj, offset in subject_offsets.items():
        ss = rng.binomial(1, np.clip(0.55 - offset * 0.4, 0.05, 0.9), n_per).astype(float)
        sl = rng.binomial(1, np.clip(0.92 - offset * 0.2, 0.1, 0.99), n_per).astype(float)
        scores.append((sl - 0.2) - offset + rng.normal(0, 0.1, n_per))
        s_S.append(ss)
        s_L.append(sl)
        subjects += [subj] * n_per
    return np.concatenate(scores), np.concatenate(s_S), np.concatenate(s_L), np.array(subjects, dtype=object)


def test_loso_subject_eval_folds_and_gaps():
    scores, s_S, s_L, subjects = _mmlu_like()
    summary = loso_subject_eval("mmlu router", subjects, scores, s_S, s_L)
    assert isinstance(summary, LOSOSummary)
    assert list(summary.subject_table["subject"]) == list(SPECIFIC_SUBJECTS)
    assert summary.subject_table["n"].sum() == len(scores)

    # pooled row must equal the same rows evaluated as one fold
    pooled = evaluate_router_config("mmlu router", scores, s_S, s_L)
    assert summary.pooled.savings_95 == pytest.approx(pooled.savings_95)
    assert summary.pooled.cpt_95 == pytest.approx(pooled.cpt_95)

    # gaps are pooled minus LOSO mean, with the worst held-out subject identified
    mean_cpt = summary.subject_table["cpt_95"].mean()
    mean_sav = summary.subject_table["savings_95"].mean()
    assert summary.gaps["cpt_gap"] == pytest.approx(pooled.cpt_95 - mean_cpt)
    assert summary.gaps["savings_gap"] == pytest.approx(pooled.savings_95 - mean_sav)
    assert summary.gaps["worst_subject"] == summary.subject_table.loc[
        summary.subject_table["savings_95"].idxmin(), "subject"]


def test_loso_true_gap_router_transfers_between_subjects():
    # The oracle (true gap) needs no subject knowledge, so LOSO ~= grouped for it.
    scores, s_S, s_L, subjects = _mmlu_like()
    summary = loso_subject_eval("oracle", subjects, s_L - s_S, s_S, s_L)
    assert summary.gaps["savings_gap"] < 0.05
    assert summary.gaps["cpt_gap"] < 0.05


def test_loso_subject_only_router_has_no_within_subject_signal():
    # Scores that are pure subject offsets: fine between subjects, useless inside one. The LOSO
    # table catches what a pooled AUC would partly flatter.
    scores, s_S, s_L, subjects = _mmlu_like()
    off = {s: i for i, s in enumerate(SPECIFIC_SUBJECTS)}
    bad = np.array([off[s] for s in subjects], dtype=float)
    summary = loso_subject_eval("subject-only router", subjects, bad, s_S, s_L)
    within_auc = [evaluate_router_config(s, bad[subjects == s], s_S[subjects == s], s_L[subjects == s]).auc
                  for s in summary.subject_table["subject"]]
    assert max(within_auc) <= 0.55


def test_loso_input_validation_and_min_subject_n():
    scores, s_S, s_L, subjects = _mmlu_like()
    with pytest.raises(ValueError):
        loso_subject_eval("x", subjects[:-1], scores, s_S, s_L)
    bad_scores = scores.copy()
    bad_scores[0] = np.nan
    with pytest.raises(ValueError):
        loso_subject_eval("x", subjects, bad_scores, s_S, s_L)
    with pytest.raises(ValueError):
        loso_subject_eval("x", subjects, scores, s_S, s_L, min_subject_n=len(scores) + 1)


def test_loso_rows_without_subject_are_pooled_as_a_fold():
    scores, s_S, s_L, subjects = _mmlu_like()
    subjects2 = subjects.copy()
    subjects2[subjects2 == "anatomy"] = None      # e.g. gsm8k rows alongside the MMLU rows
    summary = loso_subject_eval("mix", subjects2, scores, s_S, s_L)
    pooled_auc = evaluate_router_config("mix", scores, s_S, s_L).auc
    assert summary.pooled.auc == pytest.approx(pooled_auc)
    assert "(no subject)" in set(summary.subject_table["subject"])

def test_within_subject_eval_and_alias():
    scores, s_S, s_L, subjects = _mmlu_like()
    summary = within_subject_eval("mmlu router", subjects, scores, s_S, s_L)
    assert isinstance(summary, WithinSubjectSummary)
    assert isinstance(summary, LOSOSummary)
    assert list(summary.subject_table["subject"]) == list(SPECIFIC_SUBJECTS)


def test_real_loso_retraining_and_gaps():
    rng = np.random.default_rng(42)
    n_samples = 300
    subjects = np.array([f"subj_{i % 12}" for i in range(n_samples)], dtype=object)
    y_S = rng.binomial(1, 0.4, n_samples).astype(float)
    y_L = rng.binomial(1, 0.8, n_samples).astype(float)
    gap = y_L - y_S
    X = rng.standard_normal((n_samples, 16)).astype(np.float32)
    X[:, 0] += (gap > 0).astype(float) * 2.0
    tr_idx = np.arange(0, 200)
    te_idx = np.arange(200, 300)
    res = run_real_loso(
        X_train=X[tr_idx],
        y_train_S=y_S[tr_idx],
        y_train_L=y_L[tr_idx],
        subjects_train=subjects[tr_idx],
        X_test=X[te_idx],
        y_test_S=y_S[te_idx],
        y_test_L=y_L[te_idx],
        subjects_test=subjects[te_idx],
        n_folds=4,
        seed=42,
    )
    assert isinstance(res, RealLOSOSummary)
    assert len(res.fold_table) > 0
    assert "cpt_gap" in res.gaps
    assert "savings_gap" in res.gaps
    assert "worst_fold" in res.gaps
    assert res.pooled_retrained.auc > 0.5
