"""Unit tests for baseline routers and ranking metrics (DESIGN.md §1.3, briefs/B.md T13)."""
import numpy as np
import pytest

from router import schema
from router.features import TfidfFeatureExtractor, heuristics
from router.models import (
    LogRegRouter,
    XGBoostRouter,
    fuse_features,
    spearman_rho,
    z_score_normalize,
)


def test_spearman_rho_properties():
    # Perfect positive rank correlation
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert spearman_rho(a, b) == pytest.approx(1.0)

    # Perfect inverse rank correlation
    c = [50.0, 40.0, 30.0, 20.0, 10.0]
    assert spearman_rho(a, c) == pytest.approx(-1.0)

    # Constant input returns 0.0 without throwing warning/error
    assert spearman_rho([1.0, 1.0, 1.0], [2.0, 3.0, 4.0]) == 0.0


def test_fuse_features_and_z_score():
    f_emb = np.ones((5, 16), dtype=np.float32)
    f_heur = np.arange(5 * 4, dtype=np.float32).reshape(5, 4)

    fused = fuse_features(f_emb, f_heur, z_score=True)
    assert fused.shape == (5, 20)
    # Check that heuristic columns have mean approx 0
    heur_part = fused[:, 16:]
    assert np.allclose(np.mean(heur_part, axis=0), 0.0, atol=1e-5)


def test_f0_logreg_baseline_on_synthetic(syn):
    c = syn.corpus.set_index("qid")
    wide = syn.scores.pivot(index="qid", columns="model", values="score")
    s_S = wide[syn.config.small_model].loc[c.index].to_numpy()
    s_L = wide[syn.config.large_model].loc[c.index].to_numpy()
    gap = s_L - s_S

    X_f0, _ = heuristics(c["prompt"].tolist())

    train_mask = (c["split"] == "train").to_numpy()
    test_mask = (c["split"] == "test").to_numpy()

    router = LogRegRouter(name="f0_logreg")
    router.fit(X_f0[train_mask], gap[train_mask], s_S=s_S[train_mask], s_L=s_L[train_mask])

    rho_test = router.evaluate_spearman(X_f0[test_mask], gap[test_mask])
    # Ranking metric: Spearman rho should capture signal on synthetic data
    assert rho_test > 0.15, f"Expected Spearman rho > 0.15 on test, got {rho_test:.3f}"

    # Verify predictions table satisfies schema.PREDICTIONS contract
    test_qids = c.index[test_mask].tolist()
    pred_df = router.to_predictions_df(X_f0[test_mask], qids=test_qids, seed=42)
    schema.validate(pred_df, schema.PREDICTIONS)
    assert len(pred_df) == len(test_qids)
    assert (pred_df["router"] == "f0_logreg").all()


def test_f1_logreg_baseline_on_synthetic(syn):
    c = syn.corpus.set_index("qid")
    wide = syn.scores.pivot(index="qid", columns="model", values="score")
    s_S = wide[syn.config.small_model].loc[c.index].to_numpy()
    s_L = wide[syn.config.large_model].loc[c.index].to_numpy()
    gap = s_L - s_S

    train_mask = (c["split"] == "train").to_numpy()
    test_mask = (c["split"] == "test").to_numpy()

    tfidf = TfidfFeatureExtractor(min_df=1, max_word_features=2000, max_char_features=2000)
    X_train = tfidf.fit_transform(c.loc[train_mask, "prompt"].tolist())
    X_test = tfidf.transform(c.loc[test_mask, "prompt"].tolist())

    router = LogRegRouter(name="f1_logreg")
    router.fit(X_train, gap[train_mask])

    rho_test = router.evaluate_spearman(X_test, gap[test_mask])
    assert rho_test > 0.10

    test_qids = c.index[test_mask].tolist()
    pred_df = router.to_predictions_df(X_test, qids=test_qids)
    schema.validate(pred_df, schema.PREDICTIONS)


def test_xgboost_f0_f2_baseline_on_synthetic(syn):
    c = syn.corpus.set_index("qid")
    wide = syn.scores.pivot(index="qid", columns="model", values="score")
    s_S = wide[syn.config.small_model].loc[c.index].to_numpy()
    s_L = wide[syn.config.large_model].loc[c.index].to_numpy()
    gap = s_L - s_S

    X_f0, _ = heuristics(c["prompt"].tolist())

    # Mock F2 embeddings (normalized 32-dim)
    rng = np.random.default_rng(42)
    fake_f2 = rng.normal(size=(len(c), 32)).astype(np.float32)
    fake_f2 = fake_f2 / np.linalg.norm(fake_f2, axis=1, keepdims=True)

    X_fused = fuse_features(fake_f2, X_f0, z_score=True)

    train_mask = (c["split"] == "train").to_numpy()
    test_mask = (c["split"] == "test").to_numpy()

    router = XGBoostRouter(name="xgboost_f0_f2", n_estimators=60, max_depth=3)
    router.fit(X_fused[train_mask], gap[train_mask])

    rho_test = router.evaluate_spearman(X_fused[test_mask], gap[test_mask])
    assert rho_test > 0.20, f"Expected XGBoost Spearman rho > 0.20 on test, got {rho_test:.3f}"

    test_qids = c.index[test_mask].tolist()
    pred_df = router.to_predictions_df(X_fused[test_mask], qids=test_qids)
    schema.validate(pred_df, schema.PREDICTIONS)
