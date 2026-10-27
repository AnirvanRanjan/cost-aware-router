"""Evaluation metrics for LLM routing (DESIGN.md § 3.3, § 4.3, briefs/B.md T16, T20).

Provides:
  * PGR (Performance Guarantee Ratio): (Q(theta) - Q_S) / (Q_L - Q_S)
  * CPT(95%): Minimum fraction of frontier calls to achieve PGR >= 0.95
  * APGR: Area under the PGR-frontier fraction curve (in [0, 1])
  * Savings @ PGR .95: Fraction of frontier cost saved at 95% PGR
  * AUC: Classification AUC for identifying questions where Large wins
  * Spearman rho: Rank correlation between routing score and true gap
  * Ablation table builder matching DESIGN.md § 4.3
  * T26: leave-one-subject-out (LOSO) evaluation per MMLU subject, grouped vs LOSO gap
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from router.models import spearman_rho

EVAL_VERSION = "eval-1.0"


@dataclass(frozen=True)
class RoutingMetrics:
    """Evaluation summary for a single router configuration."""
    name: str
    auc: float
    spearman_rho: float
    cpt_95: float
    savings_95: float
    apgr: float
    mean_q_at_95: float
    q_S: float
    q_L: float


@dataclass(frozen=True)
class RoutingCurve:
    """Pareto routing curve across frontier call fraction gamma in [0, 1]."""
    gammas: np.ndarray      # fraction sent to Large (0 to 1)
    thresholds: np.ndarray  # score thresholds corresponding to gammas
    qualities: np.ndarray   # Q(gamma)
    pgrs: np.ndarray        # PGR(gamma)
    costs: np.ndarray       # C(gamma)
    cpt_95: float           # minimum gamma where PGR >= 0.95
    savings_95: float       # 1 - C(cpt_95) / C_L
    apgr: float             # trapezoidal integral of PGR over gamma


def compute_pgr(q: float | np.ndarray, q_S: float, q_L: float) -> float | np.ndarray:
    """Performance Guarantee Ratio: (Q - Q_S) / (Q_L - Q_S)."""
    headroom = q_L - q_S
    if abs(headroom) < 1e-9:
        return 0.0 if np.isscalar(q) else np.zeros_like(q)
    return (q - q_S) / headroom


def compute_routing_curve(
    routing_scores: Sequence[float] | np.ndarray,
    s_S: Sequence[float] | np.ndarray,
    s_L: Sequence[float] | np.ndarray,
    costs_S: Sequence[float] | np.ndarray | None = None,
    costs_L: Sequence[float] | np.ndarray | None = None,
    n_points: int = 201,
) -> RoutingCurve:
    """Compute empirical PGR, quality, and cost curves across frontier fraction gamma in [0, 1].

    Sorts items by routing_scores descending. For fraction gamma in [0, 1], the top gamma
    fraction is routed to Large and the rest to Small.
    """
    scores = np.asarray(routing_scores, dtype=float)
    y_S = np.asarray(s_S, dtype=float)
    y_L = np.asarray(s_L, dtype=float)
    n = len(scores)

    if n == 0:
        empty = np.zeros(n_points)
        return RoutingCurve(
            gammas=empty,
            thresholds=empty,
            qualities=empty,
            pgrs=empty,
            costs=empty,
            cpt_95=1.0,
            savings_95=0.0,
            apgr=0.0,
        )

    if costs_S is None:
        c_S = np.zeros(n, dtype=float)
    else:
        c_S = np.asarray(costs_S, dtype=float)

    if costs_L is None:
        c_L = np.ones(n, dtype=float)
    else:
        c_L = np.asarray(costs_L, dtype=float)

    q_S = float(np.mean(y_S))
    q_L = float(np.mean(y_L))
    mean_cost_L = float(np.mean(c_L)) if np.mean(c_L) > 0 else 1.0

    # Sort descending by routing score (break ties randomly but deterministically)
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-1e-12, 1e-12, size=n)
    order = np.argsort(-(scores + jitter))

    y_S_sorted = y_S[order]
    y_L_sorted = y_L[order]
    c_S_sorted = c_S[order]
    c_L_sorted = c_L[order]
    scores_sorted = scores[order]

    # Precompute cumulative sums for O(n) exact evaluation
    # When top k items go to L:
    # Q(k) = (sum_{i=1}^k y_L[i] + sum_{i=k+1}^n y_S[i]) / n
    cum_L = np.cumsum(y_L_sorted)
    cum_S = np.cumsum(y_S_sorted)
    sum_S_total = cum_S[-1]

    cum_cost_L = np.cumsum(c_L_sorted)
    cum_cost_S = np.cumsum(c_S_sorted)
    sum_cost_S_total = cum_cost_S[-1]

    gammas = np.linspace(0.0, 1.0, n_points)
    qualities = np.zeros(n_points, dtype=float)
    costs = np.zeros(n_points, dtype=float)
    thresholds = np.zeros(n_points, dtype=float)

    for idx, g in enumerate(gammas):
        k = int(round(g * n))
        k = np.clip(k, 0, n)
        if k == 0:
            q = q_S
            c = float(np.mean(c_S))
            thresh = float(scores_sorted[0]) + 1e-4
        elif k == n:
            q = q_L
            c = mean_cost_L
            thresh = float(scores_sorted[-1]) - 1e-4
        else:
            q = float((cum_L[k - 1] + (sum_S_total - cum_S[k - 1])) / n)
            c = float((cum_cost_L[k - 1] + (sum_cost_S_total - cum_cost_S[k - 1])) / n)
            thresh = float(scores_sorted[k - 1])

        qualities[idx] = q
        costs[idx] = c
        thresholds[idx] = thresh

    pgrs = np.asarray(compute_pgr(qualities, q_S, q_L), dtype=float)
    # Ensure boundary consistency: PGR(0)=0, PGR(1)=1
    pgrs[0] = 0.0
    pgrs[-1] = 1.0

    # CPT(95%): minimum gamma where PGR >= 0.95
    pgr_95_mask = pgrs >= 0.95
    if pgr_95_mask.any():
        cpt_idx = int(np.argmax(pgr_95_mask))
        cpt_95 = float(gammas[cpt_idx])
        cost_at_95 = float(costs[cpt_idx])
    else:
        cpt_95 = 1.0
        cost_at_95 = mean_cost_L

    savings_95 = float(np.clip(1.0 - (cost_at_95 / mean_cost_L), 0.0, 1.0))
    if hasattr(np, "trapezoid"):
        apgr = float(np.trapezoid(pgrs, gammas))
    else:
        apgr = float(np.trapz(pgrs, gammas))

    return RoutingCurve(
        gammas=gammas,
        thresholds=thresholds,
        qualities=qualities,
        pgrs=pgrs,
        costs=costs,
        cpt_95=cpt_95,
        savings_95=savings_95,
        apgr=apgr,
    )


def compute_binary_auc(
    routing_scores: Sequence[float] | np.ndarray,
    s_S: Sequence[float] | np.ndarray,
    s_L: Sequence[float] | np.ndarray,
) -> float:
    """Compute ROC-AUC of predicting whether Large strictly outperforms Small (s_L > s_S)."""
    scores = np.asarray(routing_scores, dtype=float)
    gap = np.asarray(s_L, dtype=float) - np.asarray(s_S, dtype=float)

    # Large strictly wins: gap > 0
    y_bin = (gap > 0).astype(int)
    n_pos = int(y_bin.sum())
    n_neg = len(y_bin) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.50

    try:
        return float(roc_auc_score(y_bin, scores))
    except ValueError:
        return 0.50


def evaluate_router_config(
    name: str,
    routing_scores: Sequence[float] | np.ndarray,
    s_S: Sequence[float] | np.ndarray,
    s_L: Sequence[float] | np.ndarray,
    costs_S: Sequence[float] | np.ndarray | None = None,
    costs_L: Sequence[float] | np.ndarray | None = None,
) -> RoutingMetrics:
    """Evaluate a single router configuration against all DESIGN.md § 4.3 metrics."""
    scores = np.asarray(routing_scores, dtype=float)
    y_S = np.asarray(s_S, dtype=float)
    y_L = np.asarray(s_L, dtype=float)
    gap = y_L - y_S

    auc = compute_binary_auc(scores, y_S, y_L)
    rho = spearman_rho(scores, gap)
    curve = compute_routing_curve(scores, y_S, y_L, costs_S=costs_S, costs_L=costs_L)

    return RoutingMetrics(
        name=name,
        auc=auc,
        spearman_rho=rho,
        cpt_95=curve.cpt_95,
        savings_95=curve.savings_95,
        apgr=curve.apgr,
        mean_q_at_95=float(curve.qualities[curve.pgrs >= 0.95][0]) if (curve.pgrs >= 0.95).any() else float(np.mean(y_L)),
        q_S=float(np.mean(y_S)),
        q_L=float(np.mean(y_L)),
    )


# ~~ T26: leave-one-subject-out (LOSO) on MMLU ~~
# MMLU questions carry a subject label (router/data.py splits the "mmlu-<subject>" template
# name). A router trained on all subjects can look good on the pooled test split while quietly
# memorising subject-level difficulty (MMLU subjects differ a lot: moral_scenarios vs anatomy).
# Leave-one-subject-out makes that visible: each subject is scored as its own held-out fold,
# exactly as evaluate_router_config would score it, and the gap between pooled ("grouped")
# metrics and the mean over held-out subjects is what T26 asks to report.


@dataclass(frozen=True)
class LOSOSummary:
    """Leave-one-subject-out evaluation result (T26).

    subject_table : one RoutingMetrics row per subject, plus the subject name and n
    pooled        : metrics on all subjects together (the grouped-split view)
    gaps          : pooled (grouped) minus LOSO mean; positive cpt/savings gaps mean the
                    pooled number is flattered by between-subject ranking the router could
                    not generalise to a fresh subject. `worst` names the subject with the
                    largest savings shortfall vs pooled.
    """

    subject_table: pd.DataFrame
    pooled: RoutingMetrics
    gaps: dict


def loso_subject_eval(
    name: str,
    subjects: Sequence[str | None],
    routing_scores: Sequence[float] | np.ndarray,
    s_S: Sequence[float] | np.ndarray,
    s_L: Sequence[float] | np.ndarray,
    costs_S: Sequence[float] | np.ndarray | None = None,
    costs_L: Sequence[float] | np.ndarray | None = None,
    min_subject_n: int = 1,
) -> LOSOSummary:
    """Evaluate a router per subject (e.g. each MMLU subject held out) vs pooled (T26).

    Same scoring per fold as the pooled run: for every subject, `evaluate_router_config` on
    that subject's rows alone. Rows whose subject is None (other sources) enter the pooled
    row but are folded into a "(no subject)" bucket only if any such rows exist.
    """
    subjects = np.asarray([str(s) if s is not None else "(no subject)" for s in subjects], dtype=object)
    scores = np.asarray(routing_scores, dtype=float)
    y_S = np.asarray(s_S, dtype=float)
    y_L = np.asarray(s_L, dtype=float)
    if not len(subjects) == len(scores) == len(y_S) == len(y_L):
        raise ValueError("subjects, routing_scores, s_S and s_L must have the same length")

    if np.isnan(scores).any():
        raise ValueError("routing_scores contain NaN")

    rows: list[dict] = []
    folded: dict[str, np.ndarray] = {}
    for subj in pd.unique(subjects):
        folded[str(subj)] = np.flatnonzero(subjects == subj)

    for subj, idx in sorted(folded.items()):
        if len(idx) < min_subject_n:
            continue
        metrics = evaluate_router_config(
            str(subj), scores[idx], y_S[idx], y_L[idx], costs_S=None if costs_S is None else np.asarray(costs_S, dtype=float)[idx],
            costs_L=None if costs_L is None else np.asarray(costs_L, dtype=float)[idx],
        )
        rows.append({
            "subject": str(subj),
            "n": len(idx),
            "auc": metrics.auc,
            "rho": metrics.spearman_rho,
            "cpt_95": metrics.cpt_95,
            "savings_95": metrics.savings_95,
            "apgr": metrics.apgr,
            "q_S": metrics.q_S,
            "q_L": metrics.q_L,
        })
    if not rows:
        raise ValueError("no fold had at least min_subject_n rows; nothing to report")

    pooled_metrics = evaluate_router_config(name, scores, y_S, y_L, costs_S=costs_S, costs_L=costs_L)
    table = pd.DataFrame(rows).sort_values("subject", kind="stable").reset_index(drop=True)
    mean = table[["cpt_95", "savings_95", "apgr"]].mean(numeric_only=True)
    worst = int(table["savings_95"].idxmin()) if len(table) >= 1 else None  # first min on ties
    gaps = {
        "cpt_gap": float(pooled_metrics.cpt_95 - mean["cpt_95"]),
        "savings_gap": float(pooled_metrics.savings_95 - mean["savings_95"]),
        "apgr_gap": float(pooled_metrics.apgr - mean["apgr"]),
        "worst_subject": str(table.loc[worst, "subject"]),
        "worst_savings": float(table.loc[worst, "savings_95"]),
    }
    return LOSOSummary(subject_table=table, pooled=pooled_metrics, gaps=gaps)


def build_ablation_table(
    metrics_list: Sequence[RoutingMetrics],
) -> pd.DataFrame:
    """Construct the ablation table DataFrame according to DESIGN.md § 4.3 format."""
    rows = []
    for m in metrics_list:
        rows.append({
            "Config": m.name,
            "AUC": round(m.auc, 4),
            "rho(delta_hat, delta)": round(m.spearman_rho, 4),
            "CPT(95%)": round(m.cpt_95, 4),
            "Savings @ PGR .95": f"{m.savings_95 * 100:.1f}%",
            "APGR": round(m.apgr, 4),
        })
    return pd.DataFrame(rows)
