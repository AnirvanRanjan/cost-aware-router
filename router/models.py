"""Router models and baselines (DESIGN.md §1.3, §2, briefs/B.md T13 & T14).

Baselines predict the relative performance gap Δ = s_L − s_S.
Judged by how well they rank which questions are hard (Spearman ρ(Δ̂, Δ)),
not by accuracy — the downstream Pareto threshold sweep consumes the ranking.
"""
from __future__ import annotations

from typing import Any, Sequence
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.linear_model import LogisticRegression, Ridge

from router import schema

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    F = None

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


class TwoHeadMLP(nn.Module if HAS_TORCH else object):
    """Two-head MLP router architecture (DESIGN.md § 2.1, briefs/B.md T14).

    Input: [F2 ‖ F0] fused representation (typically 768 + 23 = 791 dimensions).
    Trunk: Linear(input_dim, 256) -> GELU -> Dropout(0.1).
    Head S: Linear(256, 1) -> predicts small model logit.
    Head L: Linear(256, 1) -> predicts frontier model logit.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for TwoHeadMLP. Install via 'pip install torch'.")
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_S = nn.Linear(hidden_dim, 1)
        self.head_L = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        logit_S = self.head_S(h).squeeze(-1)
        logit_L = self.head_L(h).squeeze(-1)
        return logit_S, logit_L

    def predict_proba(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logit_S, logit_L = self.forward(x)
        return torch.sigmoid(logit_S), torch.sigmoid(logit_L)


def two_head_loss(
    logit_S: torch.Tensor,
    logit_L: torch.Tensor,
    target_S: torch.Tensor,
    target_L: torch.Tensor,
    gap_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Loss for the two-head router (DESIGN.md § 2.1, briefs/B.md T14):

    L = BCE(s_hat_S, s_S) + BCE(s_hat_L, s_L) + 0.5 * MSE(s_hat_L - s_hat_S, s_L - s_S)
    """
    bce_S = F.binary_cross_entropy_with_logits(logit_S, target_S.float())
    bce_L = F.binary_cross_entropy_with_logits(logit_L, target_L.float())
    s_hat_S = torch.sigmoid(logit_S)
    s_hat_L = torch.sigmoid(logit_L)
    delta_hat = s_hat_L - s_hat_S
    delta_target = (target_L - target_S).float()
    mse_gap = F.mse_loss(delta_hat, delta_target)
    total_loss = bce_S + bce_L + gap_weight * mse_gap
    return total_loss, {
        "bce_S": float(bce_S.item()),
        "bce_L": float(bce_L.item()),
        "mse_gap": float(mse_gap.item()),
        "total": float(total_loss.item()),
    }


class TwoHeadRouter(BaseRouter):
    """Two-head MLP router (DESIGN.md § 2.1, briefs/B.md T14).

    Jointly predicts small model quality s_S and frontier model quality s_L.
    Trained with AdamW on fused features [F2 ‖ F0].
    Early stopping carves a grouped 85/15 sub-split strictly out of the training split;
    the calibration split is reserved for calibration and cutoff selection and never touched.
    """

    def __init__(
        self,
        name: str = "two_head_mlp",
        input_dim: int | None = None,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        gap_weight: float = 0.5,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        batch_size: int = 64,
        epochs: int = 60,
        patience: int = 8,
        val_fraction: float = 0.15,
        seed: int = 42,
        device: str = "cpu",
    ):
        super().__init__(name=name, seed=seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.gap_weight = gap_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.device = device
        self.model: Any = None
        self.scaler_mean: np.ndarray | None = None
        self.scaler_std: np.ndarray | None = None
        self.training_history: list[dict[str, float]] = []
        self.best_epoch: int = 0
        self.calibrator: Any = None

    def fit(
        self,
        X: Any,
        s_S: np.ndarray,
        s_L: np.ndarray,
        group_ids: np.ndarray | Sequence[str] | None = None,
        verbose: bool = False,
    ) -> TwoHeadRouter:
        """Fit the two-head router using early stopping on a grouped 85/15 sub-split.

        Anti-leakage:
          Only train-split data should be passed to fit(). When group_ids is provided,
          router.splits.grouped_split carves an 85/15 sub-split ensuring no group_id crosses
          train_sub and val_sub.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for TwoHeadRouter. Install via 'pip install torch'.")

        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from router.splits import assert_no_leakage, grouped_split

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if scipy.sparse.issparse(X):
            X_arr = X.toarray().astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        s_S = np.asarray(s_S, dtype=np.float32)
        s_L = np.asarray(s_L, dtype=np.float32)
        n = len(X_arr)

        # 1. Carve grouped 85/15 validation split strictly out of train (anti-leakage requirement)
        if group_ids is not None:
            g_arr = np.asarray(group_ids)
            sub_splits = grouped_split(
                g_arr,
                fractions={"train_sub": 1.0 - self.val_fraction, "val_sub": self.val_fraction},
                seed=self.seed,
            )
            df_leak_check = pd.DataFrame({"group_id": g_arr, "split": sub_splits})
            assert_no_leakage(df_leak_check)
            train_mask = sub_splits == "train_sub"
            val_mask = sub_splits == "val_sub"
        else:
            rng = np.random.default_rng(self.seed)
            perm = rng.permutation(n)
            n_val = max(1, int(n * self.val_fraction))
            val_idx = set(perm[:n_val])
            train_mask = np.array([i not in val_idx for i in range(n)])
            val_mask = ~train_mask

        # 2. Fit normalization on train_sub only
        self.scaler_mean = np.mean(X_arr[train_mask], axis=0, dtype=np.float32)
        self.scaler_std = np.std(X_arr[train_mask], axis=0, dtype=np.float32)
        self.scaler_std[self.scaler_std < 1e-6] = 1.0

        X_train = (X_arr[train_mask] - self.scaler_mean) / self.scaler_std
        X_val = (X_arr[val_mask] - self.scaler_mean) / self.scaler_std

        y_S_train, y_L_train = s_S[train_mask], s_L[train_mask]
        y_S_val, y_L_val = s_S[val_mask], s_L[val_mask]
        gap_val = y_L_val - y_S_val

        # 3. Model setup
        self.input_dim = X_train.shape[1]
        self.model = TwoHeadMLP(input_dim=self.input_dim, hidden_dim=self.hidden_dim, dropout=self.dropout)
        dev = torch.device(self.device if torch.cuda.is_available() and self.device == "cuda" else "cpu")
        self.model.to(dev)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        train_ds = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_S_train),
            torch.from_numpy(y_L_train),
        )
        batch_sz = min(self.batch_size, len(X_train)) if len(X_train) > 0 else self.batch_size
        train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=True)

        X_val_t = torch.from_numpy(X_val).to(dev)
        y_S_val_t = torch.from_numpy(y_S_val).to(dev)
        y_L_val_t = torch.from_numpy(y_L_val).to(dev)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        self.training_history = []
        self.best_epoch = 0

        # 4. Training loop with early stopping
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss_sum = 0.0
            n_batches = 0

            for bx, by_S, by_L in train_loader:
                bx = bx.to(dev)
                by_S = by_S.to(dev)
                by_L = by_L.to(dev)

                optimizer.zero_grad()
                logit_S, logit_L = self.model(bx)
                loss, _ = two_head_loss(logit_S, logit_L, by_S, by_L, gap_weight=self.gap_weight)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item()
                n_batches += 1

            train_loss_avg = train_loss_sum / max(1, n_batches)

            # Evaluate on val_sub
            self.model.eval()
            with torch.no_grad():
                val_logit_S, val_logit_L = self.model(X_val_t)
                val_loss, loss_parts = two_head_loss(
                    val_logit_S, val_logit_L, y_S_val_t, y_L_val_t, gap_weight=self.gap_weight
                )
                val_s_hat_S = torch.sigmoid(val_logit_S).cpu().numpy()
                val_s_hat_L = torch.sigmoid(val_logit_L).cpu().numpy()
                val_delta_hat = val_s_hat_L - val_s_hat_S
                val_rho = spearman_rho(val_delta_hat, gap_val)

            rec = {
                "epoch": epoch,
                "train_loss": train_loss_avg,
                "val_loss": val_loss.item(),
                "val_rho": val_rho,
                **{f"val_{k}": v for k, v in loss_parts.items()},
            }
            self.training_history.append(rec)

            if verbose:
                print(f"Epoch {epoch:02d}: train_loss={train_loss_avg:.4f}, val_loss={val_loss.item():.4f}, val_rho={val_rho:.4f}")

            if val_loss.item() < best_val_loss - 1e-4:
                best_val_loss = val_loss.item()
                best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        self.is_fitted = True
        return self

    def calibrate(
        self,
        X_cal: Any,
        s_S_cal: np.ndarray,
        s_L_cal: np.ndarray,
    ) -> TwoHeadRouter:
        """Fit isotonic calibration on calibration split (DESIGN.md § 2.3, briefs/B.md T15)."""
        from router.calibrate import TwoHeadCalibrator

        s_hat_S, s_hat_L = self.predict_proba(X_cal, calibrated=False)
        self.calibrator = TwoHeadCalibrator()
        self.calibrator.fit(s_hat_S, s_hat_L, s_S_cal, s_L_cal)
        return self

    def predict_proba(self, X: Any, calibrated: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Predict small model probability s_hat_S and frontier probability s_hat_L."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("TwoHeadRouter must be fitted before predicting.")

        import torch

        if scipy.sparse.issparse(X):
            X_arr = X.toarray().astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        X_norm = (X_arr - self.scaler_mean) / self.scaler_std
        dev = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X_norm).to(dev)
            s_hat_S, s_hat_L = self.model.predict_proba(X_t)
            s_S_np = s_hat_S.cpu().numpy()
            s_L_np = s_hat_L.cpu().numpy()

        if calibrated and self.calibrator is not None:
            s_S_np, s_L_np = self.calibrator.calibrate(s_S_np, s_L_np)
        return s_S_np, s_L_np

    def predict_gap(self, X: Any, calibrated: bool = False) -> np.ndarray:
        """Predict relative gap delta_hat = s_hat_L - s_hat_S in [-1, 1]."""
        s_hat_S, s_hat_L = self.predict_proba(X, calibrated=calibrated)
        delta = s_hat_L - s_hat_S
        return np.clip(delta, -1.0, 1.0)

    def to_predictions_df(
        self,
        X: Any,
        qids: Sequence[str],
        seed: int | None = None,
        c_hat_L: np.ndarray | Sequence[float] | None = None,
        calibrated: bool = False,
    ) -> pd.DataFrame:
        """Construct DataFrame satisfying schema.PREDICTIONS contract."""
        s_hat_S, s_hat_L = self.predict_proba(X, calibrated=calibrated)
        return super().to_predictions_df(
            X=X,
            qids=qids,
            s_hat_S=s_hat_S,
            s_hat_L=s_hat_L,
            c_hat_L=c_hat_L,
            seed=seed if seed is not None else self.seed,
        )

    def retrain_small_head(
        self,
        X: Any,
        s_S_new: np.ndarray,
        group_ids: np.ndarray | Sequence[str] | None = None,
        epochs: int = 40,
        lr: float = 1e-3,
        patience: int = 8,
        verbose: bool = False,
    ) -> TwoHeadRouter:
        """T36 deliverable: retrain ONLY the small-model head on new small-model answers (e.g. Qwen on Track B).

        Freezes trunk and frontier head w_L so frontier representations remain completely intact.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Must fit TwoHeadRouter before retraining small head.")

        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from router.splits import assert_no_leakage, grouped_split

        if scipy.sparse.issparse(X):
            X_arr = X.toarray().astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        s_S_new = np.asarray(s_S_new, dtype=np.float32)
        n = len(X_arr)

        if group_ids is not None:
            g_arr = np.asarray(group_ids)
            sub_splits = grouped_split(
                g_arr,
                fractions={"train_sub": 1.0 - self.val_fraction, "val_sub": self.val_fraction},
                seed=self.seed,
            )
            df_leak_check = pd.DataFrame({"group_id": g_arr, "split": sub_splits})
            assert_no_leakage(df_leak_check)
            train_mask = sub_splits == "train_sub"
            val_mask = sub_splits == "val_sub"
        else:
            rng = np.random.default_rng(self.seed)
            perm = rng.permutation(n)
            n_val = max(1, int(n * self.val_fraction))
            val_idx = set(perm[:n_val])
            train_mask = np.array([i not in val_idx for i in range(n)])
            val_mask = ~train_mask

        X_norm = (X_arr - self.scaler_mean) / self.scaler_std
        X_train = X_norm[train_mask]
        X_val = X_norm[val_mask]
        y_train = s_S_new[train_mask]
        y_val = s_S_new[val_mask]

        # Freeze trunk and head_L
        for param in self.model.trunk.parameters():
            param.requires_grad = False
        for param in self.model.head_L.parameters():
            param.requires_grad = False
        for param in self.model.head_S.parameters():
            param.requires_grad = True

        dev = next(self.model.parameters()).device
        optimizer = torch.optim.AdamW(self.model.head_S.parameters(), lr=lr, weight_decay=self.weight_decay)

        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        batch_sz = min(self.batch_size, len(X_train)) if len(X_train) > 0 else self.batch_size
        train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=True)

        X_val_t = torch.from_numpy(X_val).to(dev)
        y_val_t = torch.from_numpy(y_val).to(dev)

        best_val_loss = float("inf")
        best_head_state = None
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            self.model.head_S.train()
            train_loss_sum = 0.0
            n_batches = 0

            for bx, by_S in train_loader:
                bx = bx.to(dev)
                by_S = by_S.to(dev)

                optimizer.zero_grad()
                with torch.no_grad():
                    h = self.model.trunk(bx)
                logit_S = self.model.head_S(h).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logit_S, by_S)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item()
                n_batches += 1

            # Validate
            self.model.head_S.eval()
            with torch.no_grad():
                h_val = self.model.trunk(X_val_t)
                val_logit_S = self.model.head_S(h_val).squeeze(-1)
                val_loss = F.binary_cross_entropy_with_logits(val_logit_S, y_val_t)

            if val_loss.item() < best_val_loss - 1e-4:
                best_val_loss = val_loss.item()
                best_head_state = copy.deepcopy(self.model.head_S.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_head_state is not None:
            self.model.head_S.load_state_dict(best_head_state)

        # Unfreeze trunk and head_L
        for param in self.model.trunk.parameters():
            param.requires_grad = True
        for param in self.model.head_L.parameters():
            param.requires_grad = True

        self.model.eval()
        return self

    def save(self, path: str | Path) -> Path:
        """Save fitted router weights and normalization statistics."""
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "seed": self.seed,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "gap_weight": self.gap_weight,
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "state_dict": self.model.state_dict() if self.model else None,
            "calibrator": self.calibrator,
            "is_fitted": self.is_fitted,
            "best_epoch": self.best_epoch,
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> TwoHeadRouter:
        """Load fitted router from saved file."""
        import torch

        payload = torch.load(path, map_location=map_location, weights_only=False)
        router = cls(
            name=payload["name"],
            input_dim=payload["input_dim"],
            hidden_dim=payload["hidden_dim"],
            dropout=payload["dropout"],
            gap_weight=payload["gap_weight"],
            seed=payload["seed"],
            device=map_location,
        )
        router.scaler_mean = payload["scaler_mean"]
        router.scaler_std = payload["scaler_std"]
        router.calibrator = payload.get("calibrator", None)
        router.is_fitted = payload["is_fitted"]
        router.best_epoch = payload.get("best_epoch", 0)
        if payload["state_dict"] is not None:
            router.model = TwoHeadMLP(
                input_dim=payload["input_dim"],
                hidden_dim=payload["hidden_dim"],
                dropout=payload["dropout"],
            )
            router.model.load_state_dict(payload["state_dict"])
            router.model.eval()
        return router


def train_two_head_multi_seed(
    X_fused: np.ndarray,
    s_S: np.ndarray,
    s_L: np.ndarray,
    group_ids: np.ndarray,
    split: np.ndarray,
    qids: Sequence[str],
    seeds: Sequence[int] = (42, 43, 44, 45, 46),
    hidden_dim: int = 256,
    dropout: float = 0.1,
    gap_weight: float = 0.5,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    patience: int = 7,
    eval_splits: Sequence[str] = ("cal", "test"),
    c_hat_L: np.ndarray | None = None,
    calibrate_on_cal: bool = False,
) -> tuple[list[TwoHeadRouter], pd.DataFrame]:
    """Train TwoHeadRouter across 5 seeds and generate schema.PREDICTIONS table (briefs/B.md T14 & T15).

    Anti-leakage:
      The training split (split == 'train') is the ONLY data fed to router.fit().
      Within router.fit(), an 85/15 sub-split is carved via grouped_split for early stopping.
      When calibrate_on_cal=True, isotonic calibration is fitted on the calibration split ('cal').
      The evaluation splits (e.g. 'cal' and 'test') are only evaluated during inference.
    """
    split_arr = np.asarray(split)
    train_mask = split_arr == "train"
    eval_mask = np.isin(split_arr, eval_splits)
    cal_mask = split_arr == "cal"

    X_train = X_fused[train_mask]
    s_S_train = s_S[train_mask]
    s_L_train = s_L[train_mask]
    g_train = np.asarray(group_ids)[train_mask]

    X_cal = X_fused[cal_mask] if cal_mask.any() else None
    s_S_cal = s_S[cal_mask] if cal_mask.any() else None
    s_L_cal = s_L[cal_mask] if cal_mask.any() else None

    X_eval = X_fused[eval_mask]
    eval_qids = [q for q, m in zip(qids, eval_mask) if m]
    eval_c_hat_L = c_hat_L[eval_mask] if c_hat_L is not None else None

    models: list[TwoHeadRouter] = []
    pred_frames: list[pd.DataFrame] = []

    for seed in seeds:
        router = TwoHeadRouter(
            name="two_head_mlp",
            hidden_dim=hidden_dim,
            dropout=dropout,
            gap_weight=gap_weight,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            seed=seed,
        )
        router.fit(X_train, s_S_train, s_L_train, group_ids=g_train)
        if calibrate_on_cal and X_cal is not None:
            router.calibrate(X_cal, s_S_cal, s_L_cal)
        models.append(router)

        pred_df = router.to_predictions_df(
            X_eval,
            qids=eval_qids,
            seed=seed,
            c_hat_L=eval_c_hat_L,
            calibrated=calibrate_on_cal,
        )
        pred_frames.append(pred_df)

    combined_preds = pd.concat(pred_frames, ignore_index=True)
    schema.validate(combined_preds, schema.PREDICTIONS)
    return models, combined_preds
