#!/usr/bin/env python3
"""T60: Investigate seed variance and produce seed-free WSR bounds on Track B cascade.

Usage:
    python scripts/17_t60_seed_free_wsr.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import schema
from router import threshold as th
from router.costs import PriceTable, pair_costs

HEADLINE = ("mmlu", "gsm8k", "arc-challenge", "winogrande")


def compute_seed_free_wsr(
    scores_ca: np.ndarray,
    s_S_ca: np.ndarray,
    s_L_ca: np.ndarray,
    taus_eval: np.ndarray,
    M: int = 50,
    rho: float = 0.95,
    delta: float = 0.05,
    chunk: int = 128,
) -> np.ndarray:
    """Compute seed-free p-values by averaging terminal e-values over M permutations.

    Averages the terminal wealth K_n = prod_{i=1}^n (1 - nu_i (loss_i - 0.5)) over M
    independent permutations. Since each terminal wealth is an e-value (E[K_n] <= 1 under H0),
    their arithmetic mean E_bar is also a valid e-value. By Markov's inequality:
        p = min(1, 1 / E_bar)
    is an unconditionally valid, finite-sample p-value free of single-permutation seed noise.
    """
    a, b = th._margin_columns(s_S_ca, s_L_ca, rho)
    n = len(scores_ca)
    t_steps = np.arange(1, n + 1, dtype=float)

    E_sum = np.zeros(len(taus_eval), dtype=float)
    for m in range(M):
        perm = np.random.default_rng(m).permutation(n)
        sc, a_p, b_p = scores_ca[perm], a[perm], b[perm]
        for start in range(0, len(taus_eval), chunk):
            tt = taus_eval[start:start + chunk, None]
            e = np.where(sc[None, :] > tt, a_p[None, :], b_p[None, :])
            loss = (rho - e) / (2.0 * rho)
            mu = (np.cumsum(loss, axis=1) + 0.5) / (1.0 + t_steps)
            s2 = (np.cumsum((loss - mu) ** 2, axis=1) + 0.25) / (1.0 + t_steps)
            s2 = np.concatenate([np.full((len(tt), 1), 0.25), s2[:, :-1]], axis=1)
            nu = np.minimum(np.sqrt(2.0 * np.log(1.0 / delta) / (n * s2)), 1.0)
            log_k_n = np.sum(np.log1p(-nu * (loss - 0.5)), axis=1)
            E_sum[start:start + chunk] += np.exp(log_k_n)

    E_mean = E_sum / float(M)
    p_values = np.clip(1.0 / np.maximum(E_mean, 1e-12), 0.0, 1.0)
    return p_values


def main():
    print("Loading Track B data...")
    corpus = schema.read("artifacts/trackb/corpus.parquet", schema.CORPUS).set_index("qid")
    scores = schema.read("artifacts/trackb/scores_hard.parquet", schema.SCORES)
    preds = schema.read("artifacts/trackb/preview_predictions.parquet", schema.PREDICTIONS)
    wide = scores.pivot(index="qid", columns="model", values="score")
    table = PriceTable.load()
    m1 = pair_costs(table, scores, small="qwen3.5-4b", large="gpt-4-1106-preview")

    f = preds[preds["router"] == "cascade_conf"].copy()
    f = f.join(corpus[["split", "source"]], on="qid")
    f["s_S"] = wide.loc[f["qid"], "qwen3.5-4b"].to_numpy()
    f["s_L"] = wide.loc[f["qid"], "gpt-4-1106-preview"].to_numpy()
    f["c_L"] = m1.loc[f["qid"], "c_L"].to_numpy()
    f["v2"] = f["delta_hat"] / f["c_hat_L"].clip(lower=1e-9)
    f = f[f["source"].isin(HEADLINE)].copy()

    tr = f[f["split"] == "train"]
    ca = f[f["split"] == "cal"]
    te = f[f["split"] == "test"]

    print("Evaluating seed variance across seeds 0-19 on standard n=1000 train grid...")
    taus_grid = th.threshold_grid(tr["v2"].to_numpy(), n=1000)
    scores_ca, s_S_ca, s_L_ca = ca["v2"].to_numpy(), ca["s_S"].to_numpy(), ca["s_L"].to_numpy()
    sw = th.evaluate_thresholds(scores_ca, s_S_ca, s_L_ca, np.r_[taus_grid, -np.inf])
    taus_eval = sw.tau

    single_seed_savings = []
    for s in range(20):
        p_single = th.wsr_pvalues(scores_ca, s_S_ca, s_L_ca, taus_eval, rho=0.95, delta=0.05, seed=s)
        p_single = np.where(sw.k == sw.n, 0.0, p_single)
        passed = p_single <= 0.05
        fails = np.flatnonzero(~passed)
        last = (fails[0] - 1) if len(fails) else len(passed) - 1
        tau_s = sw.tau[last]
        ev_s = th.evaluate_at_tau(te["v2"].to_numpy(), te["s_S"].to_numpy(), te["s_L"].to_numpy(), tau_s, c_L=te["c_L"].to_numpy(), B=200, seed=0)
        single_seed_savings.append(ev_s["savings"])

    print(f"Single-seed savings range (seeds 0-19): {min(single_seed_savings):.1%} to {max(single_seed_savings):.1%}")

    print("\nComputing seed-free terminal WSR with M=50 permutations...")
    p_mean = compute_seed_free_wsr(scores_ca, s_S_ca, s_L_ca, taus_eval, M=50, rho=0.95, delta=0.05)
    p_mean = np.where(sw.k == sw.n, 0.0, p_mean)
    passed_m = p_mean <= 0.05
    fails_m = np.flatnonzero(~passed_m)
    last_m = (fails_m[0] - 1) if len(fails_m) else len(passed_m) - 1
    tau_star_m = sw.tau[last_m]
    cal_alpha_m = sw.k[last_m] / sw.n

    ev_final = th.evaluate_at_tau(te["v2"].to_numpy(), te["s_S"].to_numpy(), te["s_L"].to_numpy(), tau_star_m, c_L=te["c_L"].to_numpy(), B=2000, seed=0)
    print(f"Seed-Free WSR tau*: {tau_star_m:.4f}")
    print(f"Calibration escalation: {cal_alpha_m:.1%}")
    print(f"Test escalation: {ev_final['alpha']:.1%} [{ev_final['ci']['alpha'][0]:.1%}, {ev_final['ci']['alpha'][1]:.1%}]")
    print(f"Test PGR: {ev_final['pgr']:.1%} [{ev_final['ci']['pgr'][0]:.1%}, {ev_final['ci']['pgr'][1]:.1%}]")
    print(f"Certified Test M1 Savings: {ev_final['savings']:.1%} [{ev_final['ci']['savings'][0]:.1%}, {ev_final['ci']['savings'][1]:.1%}]")


if __name__ == "__main__":
    main()

