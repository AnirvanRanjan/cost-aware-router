"""Router models and baselines (DESIGN.md §1.3, §2, briefs/B.md T13 & T14).

Baselines predict the relative performance gap Δ = s_L − s_S.
Judged by how well they rank which questions are hard (Spearman ρ(Δ̂, Δ)),
not by accuracy — the downstream Pareto threshold sweep consumes the ranking.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.linear_model import LogisticRegression, Ridge

from router import schema

VERSION = "models-1.0"


def spearman_rho(pred: Sequence[float] | np.ndarray, target: Sequence[float] | np.ndarray) -> float:
    """Compute Spearman rank correlation coefficient ρ(pred, target).

    Handles constant arrays or non-finite values by returning 0.0.
    """
    p = np.asarray(pred, dtype=np.float64).ravel()
    t = np.asarray(target, dtype=np.float64).ravel()
    if len(p) != len(t) or len(p) < 2:
        return 0.0
    if np.all(p == p[0]) or np.all(t == t[0]):
        return 0.0

    rank_p = np.argsort(np.argsort(p))
    rank_t = np.argsort(np.argsort(t))
    corr = np.corrcoef(rank_p, rank_t)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


def z_score_normalize(X: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalize features, avoiding division by zero for constant columns."""
    X = np.asarray(X, dtype=np.float32)
    if mean is None:
        mean = np.mean(X, axis=0, keepdims=True)
    if std is None:
        std = np.std(X, axis=0, keepdims=True)
    std_safe = np.where(std < 1e-8, 1.0, std)
    normed = (X - mean) / std_safe
    return normed, mean, std


def fuse_features(f_embed: np.ndarray, f_heuristics: np.ndarray, z_score: bool = True) -> np.ndarray:
    """Concatenate F0 heuristics onto F2 embeddings [emb ‖ z-scored heuristics] (DESIGN.md §1.3)."""
    f_embed = np.asarray(f_embed, dtype=np.float32)
    f_heur = np.asarray(f_heuristics, dtype=np.float32)
    if z_score:
        f_heur, _, _ = z_score_normalize(f_heur)
    return np.concatenate([f_embed, f_heur], axis=1)


class BaseRouter:
    """Abstract base class for all routers and baselines."""

    def __init__(self, name: str, seed: int = 42):
        self.name = name
        self.seed = seed
        self.is_fitted: bool = False

    def fit(self, X: Any, y_gap: np.ndarray, s_S: np.ndarray | None = None, s_L: np.ndarray | None = None) -> BaseRouter:
        raise NotImplementedError

    def predict_gap(self, X: Any) -> np.ndarray:
        """Predict the relative gap Δ̂ = ŝ_L − ŝ_S in [-1, 1]."""
        raise NotImplementedError

    def evaluate_spearman(self, X: Any, y_gap: np.ndarray) -> float:
        """Compute ranking quality metric: Spearman ρ(Δ̂, Δ)."""
        pred = self.predict_gap(X)
        return spearman_rho(pred, y_gap)

    def to_predictions_df(
        self,
        X: Any,
        qids: Sequence[str],
        s_hat_S: np.ndarray | None = None,
        s_hat_L: np.ndarray | None = None,
        c_hat_L: np.ndarray | Sequence[float] | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Construct a DataFrame satisfying the schema.PREDICTIONS contract."""
        delta_hat = np.clip(self.predict_gap(X), -1.0, 1.0).astype(float)
        n = len(qids)
        seed_val = self.seed if seed is None else int(seed)

        if s_hat_S is None:
            # Consistent reconstruction of individual marginals when only gap is predicted:
            # delta = s_L - s_S; s_L in [0, 1], s_S in [0, 1]
            s_hat_S = np.clip(0.5 - 0.5 * delta_hat, 0.0, 1.0)
        else:
            s_hat_S = np.clip(np.asarray(s_hat_S, dtype=float), 0.0, 1.0)

        if s_hat_L is None:
            s_hat_L = np.clip(s_hat_S + delta_hat, 0.0, 1.0)
        else:
            s_hat_L = np.clip(np.asarray(s_hat_L, dtype=float), 0.0, 1.0)

        if c_hat_L is None:
            cost_col = np.zeros(n, dtype=float)
        else:
            cost_col = np.asarray(c_hat_L, dtype=float)

        df = pd.DataFrame({
            "qid": [str(q) for q in qids],
            "router": [self.name] * n,
            "seed": [seed_val] * n,
            "s_hat_S": s_hat_S,
            "s_hat_L": s_hat_L,
            "delta_hat": delta_hat,
            "c_hat_L": cost_col,
        })
        schema.validate(df, schema.PREDICTIONS)
        return df


class LogRegRouter(BaseRouter):
    """Logistic regression router for F0 and F1 features (DESIGN.md §1.3).

    Trained on binary win labels (s_L > s_S) to predict escalation probability,
    which acts as a monotonic ranking score for the difficulty gap.
    """

    def __init__(
        self,
        name: str = "f0_logreg",
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        max_iter: int = 1000,
        seed: int = 42,
    ):
        super().__init__(name=name, seed=seed)
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.model: LogisticRegression | None = None

    def fit(self, X: Any, y_gap: np.ndarray, s_S: np.ndarray | None = None, s_L: np.ndarray | None = None) -> LogRegRouter:
        y_gap = np.asarray(y_gap, dtype=np.float32)
        # Binary target: 1 if Large strictly outperformed Small, else 0
        y_bin = (y_gap > 0.0).astype(int)

        self.model = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.seed,
        )
        self.model.fit(X, y_bin)
        self.is_fitted = True
        return self

    def predict_gap(self, X: Any) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("LogRegRouter must be fitted before predicting.")
        # Probability of Large winning: P(y=1) in [0, 1] mapped to [-1, 1] scale
        probs = self.model.predict_proba(X)[:, 1]
        delta_hat = 2.0 * probs - 1.0
        return np.clip(delta_hat, -1.0, 1.0)


class XGBoostRouter(BaseRouter):
    """XGBoost gradient-boosted decision forest for F0 + F2 features (DESIGN.md §2.1).

    Fits continuous regression directly on the quality gap Δ = s_L − s_S.
    """

    def __init__(
        self,
        name: str = "xgboost_f0_f2",
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        seed: int = 42,
    ):
        super().__init__(name=name, seed=seed)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.model: Any = None

    def fit(self, X: Any, y_gap: np.ndarray, s_S: np.ndarray | None = None, s_L: np.ndarray | None = None) -> XGBoostRouter:
        import xgboost as xgb

        y_gap = np.asarray(y_gap, dtype=np.float32)
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.seed,
            objective="reg:squarederror",
        )
        self.model.fit(X, y_gap)
        self.is_fitted = True
        return self

    def predict_gap(self, X: Any) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("XGBoostRouter must be fitted before predicting.")
        preds = self.model.predict(X)
        return np.clip(np.asarray(preds, dtype=float), -1.0, 1.0)
