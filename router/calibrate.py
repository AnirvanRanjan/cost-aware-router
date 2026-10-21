"""Isotonic calibration and reliability diagrams for the two-head router (DESIGN.md § 2.3, briefs/B.md T15).

Fits isotonic regression on the calibration split mapping s_hat_M -> E[s_M | s_hat_M] for each head independently.
Provides:
  * 15-bin Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
  * IsotonicCalibrator (per-head) and TwoHeadCalibrator (both heads)
  * Publication-quality reliability diagrams saved to notes/
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless CI/servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from router import schema

CALIBRATE_VERSION = "calibrate-1.0"


@dataclass(frozen=True)
class CalibrationResult:
    """Statistics across calibration bins."""
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    bin_confs: np.ndarray
    bin_accs: np.ndarray
    bin_counts: np.ndarray
    bin_weights: np.ndarray
    ece: float
    mce: float
    n_samples: int
    n_bins: int


def compute_calibration_bins(
    y_pred: Sequence[float] | np.ndarray,
    y_true: Sequence[float] | np.ndarray,
    n_bins: int = 15,
) -> CalibrationResult:
    """Bin predictions into equal-width bins in [0, 1] and compute accuracy, confidence, and ECE.

    Formula (15-bin ECE):
      ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
      MCE = max_{m: |B_m| > 0} |acc(B_m) - conf(B_m)|
    """
    preds = np.clip(np.asarray(y_pred, dtype=float), 0.0, 1.0)
    trues = np.clip(np.asarray(y_true, dtype=float), 0.0, 1.0)

    if len(preds) != len(trues):
        raise ValueError(f"Length mismatch: len(y_pred)={len(preds)} vs len(y_true)={len(trues)}")
    n = len(preds)
    if n == 0:
        empty = np.zeros(n_bins)
        return CalibrationResult(
            bin_edges=np.linspace(0.0, 1.0, n_bins + 1),
            bin_centers=np.linspace(0.5 / n_bins, 1.0 - 0.5 / n_bins, n_bins),
            bin_confs=empty,
            bin_accs=empty,
            bin_counts=empty.astype(int),
            bin_weights=empty,
            ece=0.0,
            mce=0.0,
            n_samples=0,
            n_bins=n_bins,
        )

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Digitize into bins 0 .. n_bins - 1
    # For pred == 1.0, clamp into bin n_bins - 1
    bin_indices = np.digitize(preds, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_confs = np.zeros(n_bins, dtype=float)
    bin_accs = np.zeros(n_bins, dtype=float)
    bin_counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = bin_indices == b
        count = int(mask.sum())
        bin_counts[b] = count
        if count > 0:
            bin_confs[b] = float(np.mean(preds[mask]))
            bin_accs[b] = float(np.mean(trues[mask]))
        else:
            bin_confs[b] = bin_centers[b]
            bin_accs[b] = bin_centers[b]

    bin_weights = bin_counts / n
    gaps = np.abs(bin_accs - bin_confs)
    ece = float(np.sum(bin_weights * gaps))

    occupied = bin_counts > 0
    mce = float(np.max(gaps[occupied])) if occupied.any() else 0.0

    return CalibrationResult(
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        bin_confs=bin_confs,
        bin_accs=bin_accs,
        bin_counts=bin_counts,
        bin_weights=bin_weights,
        ece=ece,
        mce=mce,
        n_samples=n,
        n_bins=n_bins,
    )


def expected_calibration_error(
    y_pred: Sequence[float] | np.ndarray,
    y_true: Sequence[float] | np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute 15-bin Expected Calibration Error (ECE) as a scalar float in [0, 1]."""
    res = compute_calibration_bins(y_pred, y_true, n_bins=n_bins)
    return res.ece


class IsotonicCalibrator:
    """Single-head non-parametric isotonic regression calibrator (DESIGN.md § 2.3).

    Maps uncalibrated probability s_hat in [0, 1] to monotonically non-decreasing
    calibrated expectation E[s | s_hat] in [0, 1].
    """

    def __init__(self, y_min: float = 0.0, y_max: float = 1.0):
        self.y_min = y_min
        self.y_max = y_max
        self.model: IsotonicRegression | None = None
        self.is_fitted: bool = False

    def fit(self, y_pred: Sequence[float] | np.ndarray, y_true: Sequence[float] | np.ndarray) -> IsotonicCalibrator:
        """Fit isotonic regression on calibration data."""
        x = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, 1.0)
        y = np.clip(np.asarray(y_true, dtype=np.float64), self.y_min, self.y_max)

        self.model = IsotonicRegression(
            y_min=self.y_min,
            y_max=self.y_max,
            out_of_bounds="clip",
            increasing=True,
        )
        self.model.fit(x, y)
        self.is_fitted = True
        return self

    def predict(self, y_pred: Sequence[float] | np.ndarray) -> np.ndarray:
        """Predict calibrated probabilities."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("IsotonicCalibrator must be fitted before predict.")
        x = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, 1.0)
        calibrated = self.model.predict(x)
        return np.clip(np.asarray(calibrated, dtype=float), self.y_min, self.y_max)

    def evaluate(
        self,
        y_pred: Sequence[float] | np.ndarray,
        y_true: Sequence[float] | np.ndarray,
        n_bins: int = 15,
    ) -> tuple[CalibrationResult, CalibrationResult]:
        """Compute calibration metrics before and after applying this calibrator."""
        before = compute_calibration_bins(y_pred, y_true, n_bins=n_bins)
        y_cal = self.predict(y_pred)
        after = compute_calibration_bins(y_cal, y_true, n_bins=n_bins)
        return before, after


class TwoHeadCalibrator:
    """Pair of isotonic calibrators for the small and frontier heads (DESIGN.md § 2.3)."""

    def __init__(self, y_min: float = 0.0, y_max: float = 1.0):
        self.calibrator_S = IsotonicCalibrator(y_min=y_min, y_max=y_max)
        self.calibrator_L = IsotonicCalibrator(y_min=y_min, y_max=y_max)
        self.is_fitted: bool = False

    def fit(
        self,
        s_hat_S: Sequence[float] | np.ndarray,
        s_hat_L: Sequence[float] | np.ndarray,
        s_S_true: Sequence[float] | np.ndarray,
        s_L_true: Sequence[float] | np.ndarray,
    ) -> TwoHeadCalibrator:
        """Fit both calibrators independently on the calibration split."""
        self.calibrator_S.fit(s_hat_S, s_S_true)
        self.calibrator_L.fit(s_hat_L, s_L_true)
        self.is_fitted = True
        return self

    def calibrate(
        self,
        s_hat_S: Sequence[float] | np.ndarray,
        s_hat_L: Sequence[float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return calibrated (s_hat_S, s_hat_L)."""
        if not self.is_fitted:
            raise RuntimeError("TwoHeadCalibrator must be fitted before calibrate.")
        cal_S = self.calibrator_S.predict(s_hat_S)
        cal_L = self.calibrator_L.predict(s_hat_L)
        return cal_S, cal_L

    def calibrate_predictions_df(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """Calibrate a DataFrame conforming to schema.PREDICTIONS."""
        out = df if inplace else df.copy()
        cal_S, cal_L = self.calibrate(out["s_hat_S"].to_numpy(), out["s_hat_L"].to_numpy())
        out["s_hat_S"] = cal_S
        out["s_hat_L"] = cal_L
        out["delta_hat"] = np.clip(cal_L - cal_S, -1.0, 1.0)
        schema.validate(out, schema.PREDICTIONS)
        return out

    def save(self, path: str | Path) -> Path:
        """Save fitted calibrators to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, path: str | Path) -> TwoHeadCalibrator:
        """Load fitted calibrators from file."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, TwoHeadCalibrator):
            raise TypeError(f"Loaded object is {type(obj)}, expected TwoHeadCalibrator")
        return obj


def plot_reliability_diagrams(
    res_S_before: CalibrationResult,
    res_S_after: CalibrationResult,
    res_L_before: CalibrationResult,
    res_L_after: CalibrationResult,
    out_path: str | Path | None = None,
    small_name: str = "Small Model (S)",
    large_name: str = "Frontier Model (L)",
) -> plt.Figure:
    """Generate publication-ready 2x2 grid of reliability diagrams.

    Shows empirical accuracy vs predicted confidence with 15 bins, ideal calibration diagonal,
    ECE reduction metrics, and sample frequency distributions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), dpi=300)

    panels = [
        (axes[0, 0], res_S_before, f"{small_name} - Raw", "#2b5c8f"),
        (axes[0, 1], res_S_after, f"{small_name} - Calibrated (Isotonic)", "#287271"),
        (axes[1, 0], res_L_before, f"{large_name} - Raw", "#b56576"),
        (axes[1, 1], res_L_after, f"{large_name} - Calibrated (Isotonic)", "#6d597a"),
    ]

    for ax, res, title, color in panels:
        # Diagonal perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect Calibration")

        # Bin bars
        occupied = res.bin_counts > 0
        centers = res.bin_centers[occupied]
        accs = res.bin_accs[occupied]
        confs = res.bin_confs[occupied]
        width = 1.0 / res.n_bins * 0.85

        # Bar chart of empirical accuracy
        ax.bar(
            centers,
            accs,
            width=width,
            color=color,
            alpha=0.65,
            edgecolor="black",
            linewidth=0.8,
            label="Empirical Accuracy",
        )

        # Gap between accuracy and confidence
        gap = accs - confs
        for c, conf, acc, g in zip(centers, confs, accs, gap):
            if abs(g) > 0.005:
                ax.plot([c, c], [conf, acc], color="red", alpha=0.8, linewidth=1.5)

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("Mean Predicted Confidence")
        ax.set_ylabel("Empirical Accuracy")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)

        # Metric text box
        textstr = f"ECE: {res.ece * 100:.2f}%\nMCE: {res.mce * 100:.2f}%\n(N={res.n_samples:,})"
        props = dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#ccc")
        ax.text(0.05, 0.92, textstr, transform=ax.transAxes, fontsize=9, verticalalignment="top", bbox=props)
        ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved reliability diagrams to {out_path}")

    return fig
