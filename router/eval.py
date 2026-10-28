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
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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


# ~~ T26: within-subject evaluation and real leave-one-subject-out (LOSO) on MMLU ~~
# MMLU questions carry a subject label (router/data.py splits the "mmlu-<subject>" template
# name). A router trained on all subjects can look good on the pooled test split while quietly
# memorising subject-level difficulty (MMLU subjects differ a lot: moral_scenarios vs anatomy).
#
# (1) Within-subject evaluation: scores an existing router per subject on its own held-out test rows
#     to check within-subject discrimination.
# (2) Real LOSO: retrains the router (e.g. LogisticRegression on F2) with each subject fold's train
#     rows completely removed from training, then evaluates on that fold's test rows.


@dataclass(frozen=True)
class WithinSubjectSummary:
    """Within-subject evaluation result on existing router predictions (T26).

    subject_table : one RoutingMetrics row per subject, plus the subject name and n
    pooled        : metrics on all subjects together (the grouped-split view)
    gaps          : pooled (grouped) minus mean over subjects; positive cpt/savings gaps mean
                    the pooled number is flattered by between-subject ranking.
    """

    subject_table: pd.DataFrame
    pooled: RoutingMetrics
    gaps: dict


# Backwards compatibility alias
LOSOSummary = WithinSubjectSummary


def within_subject_eval(
    name: str,
    subjects: Sequence[str | None],
    routing_scores: Sequence[float] | np.ndarray,
    s_S: Sequence[float] | np.ndarray,
    s_L: Sequence[float] | np.ndarray,
    costs_S: Sequence[float] | np.ndarray | None = None,
    costs_L: Sequence[float] | np.ndarray | None = None,
    min_subject_n: int = 1,
) -> WithinSubjectSummary:
    """Evaluate an existing router within each subject vs pooled.

    For every subject, runs `evaluate_router_config` on that subject's test rows alone.
    Rows whose subject is None enter the pooled row but are folded into "(no subject)"
    only if any such rows exist.
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
    return WithinSubjectSummary(subject_table=table, pooled=pooled_metrics, gaps=gaps)


# Backwards compatibility alias
loso_subject_eval = within_subject_eval


@dataclass(frozen=True)
class RealLOSOSummary:
    """True Leave-One-Subject-Out evaluation result with model retraining (T26).

    fold_table       : one row per subject fold with held-out test metrics, subjects in fold, and test n
    loso_mean        : average metrics across the held-out folds
    pooled_retrained : metrics of the baseline model trained on all train rows and tested on pooled test
    loso_test_metrics: metrics scored from LOSO out-of-fold predictions on all test rows together
    pooled_fold_mean : average metrics of the pooled baseline model evaluated per fold
    gaps             : comparison gaps (both fold-average like-with-like and pooled out-of-fold)
    """

    fold_table: pd.DataFrame
    loso_mean: dict[str, float]
    pooled_retrained: RoutingMetrics
    gaps: dict[str, Any]
    loso_test_metrics: RoutingMetrics | None = None
    pooled_fold_mean: dict[str, float] | None = None


def run_real_loso(
    X_train: np.ndarray,
    y_train_S: np.ndarray,
    y_train_L: np.ndarray,
    subjects_train: Sequence[str | None],
    X_test: np.ndarray,
    y_test_S: np.ndarray,
    y_test_L: np.ndarray,
    subjects_test: Sequence[str | None],
    costs_test_S: Sequence[float] | np.ndarray | None = None,
    costs_test_L: Sequence[float] | np.ndarray | None = None,
    n_folds: int = 10,
    seed: int = 42,
    C: float = 1.0,
    max_iter: int = 1000,
) -> RealLOSOSummary:
    """True Leave-One-Subject-Out (LOSO) cross-validation with retrained Logistic Regression.

    Subjects are grouped into ~n_folds folds. For each fold:
      1. All train rows belonging to the fold's subjects are removed from training.
      2. A LogisticRegression classifier is trained on the remaining training rows.
      3. The model predicts routing scores on the held-out test rows for that fold's subjects.
      4. Metrics are computed for each held-out fold, and the true LOSO gap is reported.
    """
    X_tr = np.asarray(X_train, dtype=np.float32)
    y_tr_S = np.asarray(y_train_S, dtype=float)
    y_tr_L = np.asarray(y_train_L, dtype=float)
    s_tr = np.asarray([str(s) if s is not None else "(no subject)" for s in subjects_train], dtype=object)

    X_te = np.asarray(X_test, dtype=np.float32)
    y_te_S = np.asarray(y_test_S, dtype=float)
    y_te_L = np.asarray(y_test_L, dtype=float)
    s_te = np.asarray([str(s) if s is not None else "(no subject)" for s in subjects_test], dtype=object)

    if not len(X_tr) == len(y_tr_S) == len(y_tr_L) == len(s_tr):
        raise ValueError("X_train, y_train_S, y_train_L and subjects_train must have same length")
    if not len(X_te) == len(y_te_S) == len(y_te_L) == len(s_te):
        raise ValueError("X_test, y_test_S, y_test_L and subjects_test must have same length")

    # Distinct subjects across both splits
    all_subjects = sorted(list(set(s_tr) | set(s_te)))
    rng = np.random.default_rng(seed)
    shuffled_subjects = list(rng.permutation(all_subjects))

    # Partition subjects into ~n_folds groups
    k = min(n_folds, len(shuffled_subjects))
    if k < 1:
        raise ValueError("No subjects found to create folds")

    subject_folds: list[list[str]] = [[] for _ in range(k)]
    for i, subj in enumerate(shuffled_subjects):
        subject_folds[i % k].append(subj)

    fold_rows: list[dict[str, Any]] = []
    loso_test_scores = np.zeros(len(X_te), dtype=float)
    scored_test_mask = np.zeros(len(X_te), dtype=bool)

    costs_S_arr = None if costs_test_S is None else np.asarray(costs_test_S, dtype=float)
    costs_L_arr = None if costs_test_L is None else np.asarray(costs_test_L, dtype=float)

    # Train pooled baseline model on all training data first
    y_tr_all_bin = ((y_tr_L - y_tr_S) > 0.0).astype(int)
    clf_pooled = LogisticRegression(C=C, max_iter=max_iter, random_state=seed, solver="lbfgs")
    clf_pooled.fit(X_tr, y_tr_all_bin)
    pooled_test_scores = clf_pooled.predict_proba(X_te)[:, 1]

    pooled_fold_rows: list[dict[str, float]] = []

    for fold_idx, fold_subjects in enumerate(subject_folds):
        fold_set = set(fold_subjects)
        # Train on all train rows whose subject is NOT in fold_set
        tr_mask = np.array([subj not in fold_set for subj in s_tr], dtype=bool)
        te_mask = np.array([subj in fold_set for subj in s_te], dtype=bool)

        if not tr_mask.any():
            raise RuntimeError(f"Fold {fold_idx}: no training rows remaining when holding out {fold_subjects}")
        if not te_mask.any():
            continue

        y_tr_bin = ((y_tr_L[tr_mask] - y_tr_S[tr_mask]) > 0.0).astype(int)
        # Handle degenerate cases where all classes are identical
        if len(np.unique(y_tr_bin)) < 2:
            scores_te = np.full(te_mask.sum(), float(y_tr_bin[0]))
        else:
            clf = LogisticRegression(C=C, max_iter=max_iter, random_state=seed, solver="lbfgs")
            clf.fit(X_tr[tr_mask], y_tr_bin)
            scores_te = clf.predict_proba(X_te[te_mask])[:, 1]

        loso_test_scores[te_mask] = scores_te
        scored_test_mask[te_mask] = True

        m = evaluate_router_config(
            f"fold_{fold_idx}",
            scores_te,
            y_te_S[te_mask],
            y_te_L[te_mask],
            costs_S=None if costs_S_arr is None else costs_S_arr[te_mask],
            costs_L=None if costs_L_arr is None else costs_L_arr[te_mask],
        )

        # Evaluate pooled model on the exact same fold test rows for like-with-like fold comparison
        m_pooled_fold = evaluate_router_config(
            f"pooled_fold_{fold_idx}",
            pooled_test_scores[te_mask],
            y_te_S[te_mask],
            y_te_L[te_mask],
            costs_S=None if costs_S_arr is None else costs_S_arr[te_mask],
            costs_L=None if costs_L_arr is None else costs_L_arr[te_mask],
        )
        pooled_fold_rows.append({
            "auc": m_pooled_fold.auc,
            "rho": m_pooled_fold.spearman_rho,
            "cpt_95": m_pooled_fold.cpt_95,
            "savings_95": m_pooled_fold.savings_95,
            "apgr": m_pooled_fold.apgr,
        })

        fold_rows.append({
            "fold": fold_idx,
            "subjects": ", ".join(fold_subjects[:3]) + (f" (+{len(fold_subjects)-3} more)" if len(fold_subjects) > 3 else ""),
            "n_subjects": len(fold_subjects),
            "n_test": int(te_mask.sum()),
            "auc": m.auc,
            "rho": m.spearman_rho,
            "cpt_95": m.cpt_95,
            "savings_95": m.savings_95,
            "apgr": m.apgr,
            "pooled_fold_auc": m_pooled_fold.auc,
            "pooled_fold_savings_95": m_pooled_fold.savings_95,
        })

    if not fold_rows:
        raise ValueError("No test folds contained rows to evaluate")

    fold_df = pd.DataFrame(fold_rows)

    pooled_metrics = evaluate_router_config(
        "pooled_f2_logreg",
        pooled_test_scores,
        y_te_S,
        y_te_L,
        costs_S=costs_S_arr,
        costs_L=costs_L_arr,
    )

    # Score LOSO out-of-fold predictions on all test rows together
    loso_test_metrics = evaluate_router_config(
        "loso_out_of_fold",
        loso_test_scores,
        y_te_S,
        y_te_L,
        costs_S=costs_S_arr,
        costs_L=costs_L_arr,
    )

    loso_mean = {
        "auc": float(fold_df["auc"].mean()),
        "rho": float(fold_df["rho"].mean()),
        "cpt_95": float(fold_df["cpt_95"].mean()),
        "savings_95": float(fold_df["savings_95"].mean()),
        "apgr": float(fold_df["apgr"].mean()),
    }

    pooled_fold_df = pd.DataFrame(pooled_fold_rows)
    pooled_fold_mean = {
        "auc": float(pooled_fold_df["auc"].mean()),
        "rho": float(pooled_fold_df["rho"].mean()),
        "cpt_95": float(pooled_fold_df["cpt_95"].mean()),
        "savings_95": float(pooled_fold_df["savings_95"].mean()),
        "apgr": float(pooled_fold_df["apgr"].mean()),
    }

    worst_idx = int(fold_df["savings_95"].idxmin())
    gaps = {
        # Comparison like-with-like (1): fold averages (pooled_fold_mean - loso_mean)
        "fold_auc_gap": float(pooled_fold_mean["auc"] - loso_mean["auc"]),
        "fold_savings_gap": float(pooled_fold_mean["savings_95"] - loso_mean["savings_95"]),
        "fold_cpt_gap": float(pooled_fold_mean["cpt_95"] - loso_mean["cpt_95"]),
        "fold_apgr_gap": float(pooled_fold_mean["apgr"] - loso_mean["apgr"]),
        # Comparison like-with-like (2): all test rows scored together (pooled_retrained - loso_test_metrics)
        "oof_auc_gap": float(pooled_metrics.auc - loso_test_metrics.auc),
        "oof_savings_gap": float(pooled_metrics.savings_95 - loso_test_metrics.savings_95),
        "oof_cpt_gap": float(pooled_metrics.cpt_95 - loso_test_metrics.cpt_95),
        "oof_apgr_gap": float(pooled_metrics.apgr - loso_test_metrics.apgr),
        # Legacy gaps for backwards compatibility:
        "cpt_gap": float(pooled_metrics.cpt_95 - loso_mean["cpt_95"]),
        "savings_gap": float(pooled_metrics.savings_95 - loso_mean["savings_95"]),
        "apgr_gap": float(pooled_metrics.apgr - loso_mean["apgr"]),
        "worst_fold": int(fold_df.loc[worst_idx, "fold"]),
        "worst_fold_subjects": str(fold_df.loc[worst_idx, "subjects"]),
        "worst_savings": float(fold_df.loc[worst_idx, "savings_95"]),
    }

    return RealLOSOSummary(
        fold_table=fold_df,
        loso_mean=loso_mean,
        pooled_retrained=pooled_metrics,
        gaps=gaps,
        loso_test_metrics=loso_test_metrics,
        pooled_fold_mean=pooled_fold_mean,
    )


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
