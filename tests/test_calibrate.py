"""Unit tests for isotonic calibration and reliability diagrams (DESIGN.md § 2.3, briefs/B.md T15)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from router import schema
from router.calibrate import (
    CalibrationResult,
    IsotonicCalibrator,
    TwoHeadCalibrator,
    compute_calibration_bins,
    expected_calibration_error,
    plot_reliability_diagrams,
)
from router.features import heuristics
from router.models import TwoHeadRouter, fuse_features


def test_ece_perfect_and_uncalibrated_math():
    # 1. Perfectly calibrated probabilities
    # 100 samples with pred=0.2 and exactly 20% 1s
    # 100 samples with pred=0.8 and exactly 80% 1s
    y_pred = np.array([0.2] * 100 + [0.8] * 100)
    y_true = np.array([1.0] * 20 + [0.0] * 80 + [1.0] * 80 + [0.0] * 20)

    res = compute_calibration_bins(y_pred, y_true, n_bins=15)
    assert res.ece == pytest.approx(0.0, abs=1e-6)
    assert res.mce == pytest.approx(0.0, abs=1e-6)
    assert expected_calibration_error(y_pred, y_true, n_bins=15) == pytest.approx(0.0, abs=1e-6)

    # 2. Completely miscalibrated (predict 1.0 everywhere, but true accuracy is 0.0)
    y_bad_pred = np.ones(100)
    y_bad_true = np.zeros(100)
    bad_ece = expected_calibration_error(y_bad_pred, y_bad_true, n_bins=15)
    assert bad_ece == pytest.approx(1.0, abs=1e-6)

    # 3. Empty input edge case
    empty_res = compute_calibration_bins([], [], n_bins=15)
    assert empty_res.ece == 0.0
    assert empty_res.n_samples == 0


def test_isotonic_calibrator_monotonicity_and_ece_reduction():
    rng = np.random.default_rng(42)
    n = 1000

    # True latent probability
    z = rng.uniform(0.0, 1.0, size=n)
    y_true = rng.binomial(1, z).astype(float)

    # Distorted/uncalibrated probabilities: severe over-confidence (p -> p^0.3)
    y_uncal = np.clip(z ** 0.3, 0.0, 1.0)
    ece_before = expected_calibration_error(y_uncal, y_true, n_bins=15)
    assert ece_before > 0.10, f"Expected high uncalibrated ECE, got {ece_before:.3f}"

    calibrator = IsotonicCalibrator()
    calibrator.fit(y_uncal, y_true)

    y_cal = calibrator.predict(y_uncal)
    ece_after = expected_calibration_error(y_cal, y_true, n_bins=15)

    # Isotonic calibration must significantly reduce ECE
    assert ece_after < ece_before * 0.35, f"Expected ECE reduction, got before={ece_before:.3f}, after={ece_after:.3f}"

    # Output probabilities must stay in [0, 1]
    assert (y_cal >= 0.0).all() and (y_cal <= 1.0).all()

    # Monotonicity test: sorting by input preserves sorted order of output
    order = np.argsort(y_uncal)
    diffs = np.diff(y_cal[order])
    assert (diffs >= -1e-6).all(), "Isotonic calibration must be monotonically non-decreasing"


def test_two_head_calibrator_and_predictions_dataframe():
    rng = np.random.default_rng(101)
    n = 300

    s_S_true = rng.binomial(1, 0.4, size=n).astype(float)
    s_L_true = rng.binomial(1, 0.8, size=n).astype(float)

    # Uncalibrated predictions
    s_hat_S = np.clip(s_S_true * 0.5 + 0.3, 0.0, 1.0)
    s_hat_L = np.clip(s_L_true * 0.4 + 0.5, 0.0, 1.0)

    calibrator = TwoHeadCalibrator()
    calibrator.fit(s_hat_S, s_hat_L, s_S_true, s_L_true)

    cal_S, cal_L = calibrator.calibrate(s_hat_S, s_hat_L)
    assert len(cal_S) == n and len(cal_L) == n

    # Construct schema.PREDICTIONS test frame
    raw_df = pd.DataFrame({
        "qid": [f"q_{i}" for i in range(n)],
        "router": ["two_head_mlp"] * n,
        "seed": [42] * n,
        "s_hat_S": s_hat_S,
        "s_hat_L": s_hat_L,
        "delta_hat": s_hat_L - s_hat_S,
        "c_hat_L": np.zeros(n),
    })
    schema.validate(raw_df, schema.PREDICTIONS)

    cal_df = calibrator.calibrate_predictions_df(raw_df)
    schema.validate(cal_df, schema.PREDICTIONS)

    assert np.allclose(cal_df["s_hat_S"], cal_S)
    assert np.allclose(cal_df["s_hat_L"], cal_L)
    assert np.allclose(cal_df["delta_hat"], cal_L - cal_S)


def test_two_head_calibrator_save_and_load(tmp_path: Path):
    rng = np.random.default_rng(7)
    x = rng.uniform(0, 1, size=50)
    y = rng.uniform(0, 1, size=50)

    calibrator = TwoHeadCalibrator()
    calibrator.fit(x, x, y, y)

    path = tmp_path / "calibrator.pkl"
    calibrator.save(path)
    assert path.exists()

    loaded = TwoHeadCalibrator.load(path)
    test_x = np.linspace(0.1, 0.9, 9)
    out_orig_S, out_orig_L = calibrator.calibrate(test_x, test_x)
    out_load_S, out_load_L = loaded.calibrate(test_x, test_x)

    assert np.allclose(out_orig_S, out_load_S)
    assert np.allclose(out_orig_L, out_load_L)


def test_reliability_diagrams_plotting(tmp_path: Path):
    rng = np.random.default_rng(99)
    y_pred = rng.uniform(0.1, 0.9, size=200)
    y_true = rng.binomial(1, y_pred).astype(float)

    cal = IsotonicCalibrator().fit(y_pred, y_true)
    before_S, after_S = cal.evaluate(y_pred, y_true, n_bins=15)
    before_L, after_L = cal.evaluate(y_pred, y_true, n_bins=15)

    plot_file = tmp_path / "test_reliability.png"
    fig = plot_reliability_diagrams(
        before_S, after_S, before_L, after_L,
        out_path=plot_file,
        small_name="Mixtral",
        large_name="GPT-4",
    )
    assert plot_file.exists()
    assert plot_file.stat().st_size > 5000  # Non-trivial image written


def test_two_head_router_calibrate_integration(syn):
    pytest.importorskip("torch")

    c = syn.corpus.set_index("qid")
    wide = syn.scores.pivot(index="qid", columns="model", values="score")
    s_S = wide[syn.config.small_model].loc[c.index].to_numpy()
    s_L = wide[syn.config.large_model].loc[c.index].to_numpy()
    gap = s_L - s_S

    X_f0, _ = heuristics(c["prompt"].tolist())
    rng = np.random.default_rng(42)
    fake_f2 = rng.normal(size=(len(c), 32)).astype(np.float32)
    for i in range(len(c)):
        fake_f2[i, 0] += float(gap[i]) * 1.5

    X_fused = fuse_features(fake_f2, X_f0, z_score=False)

    train_mask = (c["split"] == "train").to_numpy()
    cal_mask = (c["split"] == "cal").to_numpy()
    test_mask = (c["split"] == "test").to_numpy()

    router = TwoHeadRouter(hidden_dim=32, epochs=15, batch_size=64, seed=42)
    router.fit(
        X_fused[train_mask],
        s_S=s_S[train_mask],
        s_L=s_L[train_mask],
        group_ids=c["group_id"].to_numpy()[train_mask],
    )

    # 1. Uncalibrated predictions on test
    s_uncal_S, s_uncal_L = router.predict_proba(X_fused[test_mask], calibrated=False)
    ece_uncal_S = expected_calibration_error(s_uncal_S, s_S[test_mask])

    # 2. Fit isotonic calibration strictly on cal split (no test data seen!)
    router.calibrate(
        X_fused[cal_mask],
        s_S_cal=s_S[cal_mask],
        s_L_cal=s_L[cal_mask],
    )
    assert router.calibrator is not None

    # 3. Calibrated predictions on test
    s_cal_S, s_cal_L = router.predict_proba(X_fused[test_mask], calibrated=True)
    assert (s_cal_S >= 0.0).all() and (s_cal_S <= 1.0).all()
    assert (s_cal_L >= 0.0).all() and (s_cal_L <= 1.0).all()

    # Verify that calibrated to_predictions_df satisfies contract
    test_qids = c.index[test_mask].tolist()
    pred_df = router.to_predictions_df(X_fused[test_mask], qids=test_qids, calibrated=True)
    schema.validate(pred_df, schema.PREDICTIONS)
