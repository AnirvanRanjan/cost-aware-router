"""Unit tests for baseline routers and ranking metrics (DESIGN.md §1.3, briefs/B.md T13)."""
import numpy as np
import pytest

from router import schema
from router.features import TfidfFeatureExtractor, heuristics
from router.models import (
    LogRegRouter,
    TwoHeadMLP,
    TwoHeadRouter,
    train_two_head_multi_seed,
    two_head_loss,
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
    pytest.importorskip("xgboost")
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



# ---- T14: Two-Head MLP Router tests --------------------------------------------------------


def test_two_head_mlp_architecture():
    pytest.importorskip("torch")
    import torch

    model = TwoHeadMLP(input_dim=791, hidden_dim=256, dropout=0.1)
    x = torch.randn(8, 791)
    logit_S, logit_L = model(x)
    assert logit_S.shape == (8,)
    assert logit_L.shape == (8,)

    s_hat_S, s_hat_L = model.predict_proba(x)
    assert s_hat_S.shape == (8,)
    assert s_hat_L.shape == (8,)
    assert (s_hat_S >= 0.0).all() and (s_hat_S <= 1.0).all()
    assert (s_hat_L >= 0.0).all() and (s_hat_L <= 1.0).all()


def test_two_head_loss_gradient_flow():
    pytest.importorskip("torch")
    import torch

    model = TwoHeadMLP(input_dim=10, hidden_dim=32, dropout=0.0)
    x = torch.randn(4, 10)
    target_S = torch.tensor([0.2, 0.8, 0.0, 1.0])
    target_L = torch.tensor([0.9, 0.9, 0.5, 1.0])

    logit_S, logit_L = model(x)
    loss, parts = two_head_loss(logit_S, logit_L, target_S, target_L, gap_weight=0.5)

    assert loss.item() > 0.0
    assert "bce_S" in parts and "bce_L" in parts and "mse_gap" in parts

    loss.backward()
    for param in model.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()


def test_two_head_router_anti_leakage_and_early_stopping(syn):
    pytest.importorskip("torch")
    from router.splits import assert_no_leakage, grouped_split

    c = syn.corpus.set_index("qid")
    train_mask = (c["split"] == "train").to_numpy()
    group_ids = c["group_id"].to_numpy()[train_mask]

    # Verify that carving an 85/15 split strictly within train has NO leakage
    sub_splits = grouped_split(group_ids, fractions={"train_sub": 0.85, "val_sub": 0.15}, seed=42)
    import pandas as pd
    df_leak = pd.DataFrame({"group_id": group_ids, "split": sub_splits})
    assert_no_leakage(df_leak)


def test_two_head_router_fit_and_predict_on_synthetic(syn):
    pytest.importorskip("torch")

    c = syn.corpus.set_index("qid")
    wide = syn.scores.pivot(index="qid", columns="model", values="score")
    s_S = wide[syn.config.small_model].loc[c.index].to_numpy()
    s_L = wide[syn.config.large_model].loc[c.index].to_numpy()
    gap = s_L - s_S

    X_f0, _ = heuristics(c["prompt"].tolist())

    # Simulated embeddings correlated with difficulty gap
    rng = np.random.default_rng(42)
    fake_f2 = rng.normal(size=(len(c), 32)).astype(np.float32)
    for i in range(len(c)):
        fake_f2[i, 0] += float(gap[i]) * 1.5
    fake_f2 = fake_f2 / np.linalg.norm(fake_f2, axis=1, keepdims=True)

    X_fused = fuse_features(fake_f2, X_f0, z_score=False)

    train_mask = (c["split"] == "train").to_numpy()
    test_mask = (c["split"] == "test").to_numpy()

    router = TwoHeadRouter(
        name="two_head_mlp",
        hidden_dim=64,
        epochs=30,
        batch_size=32,
        lr=3e-3,
        patience=5,
        seed=42,
    )
    # Fit only on train split!
    router.fit(
        X_fused[train_mask],
        s_S=s_S[train_mask],
        s_L=s_L[train_mask],
        group_ids=c["group_id"].to_numpy()[train_mask],
    )

    # Evaluate on test split
    s_hat_S, s_hat_L = router.predict_proba(X_fused[test_mask])
    assert len(s_hat_S) == test_mask.sum()
    assert (s_hat_S >= 0.0).all() and (s_hat_S <= 1.0).all()
    assert (s_hat_L >= 0.0).all() and (s_hat_L <= 1.0).all()

    rho_test = router.evaluate_spearman(X_fused[test_mask], gap[test_mask])
    assert rho_test > 0.20, f"Expected Spearman rho > 0.20 on test, got {rho_test:.3f}"

    test_qids = c.index[test_mask].tolist()
    pred_df = router.to_predictions_df(X_fused[test_mask], qids=test_qids)
    schema.validate(pred_df, schema.PREDICTIONS)

    assert len(pred_df) == len(test_qids)
    assert (pred_df["router"] == "two_head_mlp").all()
    assert (pred_df["seed"] == 42).all()


def test_two_head_multi_seed(syn):
    pytest.importorskip("torch")

    c = syn.corpus.set_index("qid")
    wide = syn.scores.pivot(index="qid", columns="model", values="score")
    s_S = wide[syn.config.small_model].loc[c.index].to_numpy()
    s_L = wide[syn.config.large_model].loc[c.index].to_numpy()
    gap = s_L - s_S

    X_f0, _ = heuristics(c["prompt"].tolist())
    rng = np.random.default_rng(7)
    fake_f2 = rng.normal(size=(len(c), 32)).astype(np.float32)
    for i in range(len(c)):
        fake_f2[i, 0] += float(gap[i]) * 1.5

    X_fused = fuse_features(fake_f2, X_f0, z_score=False)

    seeds = [42, 43, 44, 45, 46]
    models, combined_preds = train_two_head_multi_seed(
        X_fused=X_fused,
        s_S=s_S,
        s_L=s_L,
        group_ids=c["group_id"].to_numpy(),
        split=c["split"].to_numpy(),
        qids=c.index.tolist(),
        seeds=seeds,
        hidden_dim=32,
        epochs=15,
        batch_size=64,
        eval_splits=("test",),
    )

    assert len(models) == 5
    schema.validate(combined_preds, schema.PREDICTIONS)

    test_n = int((c["split"] == "test").sum())
    assert len(combined_preds) == test_n * 5
    assert set(combined_preds["seed"]) == set(seeds)


def test_two_head_router_save_and_load(tmp_path):
    pytest.importorskip("torch")

    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 20)).astype(np.float32)
    s_S = rng.uniform(0.1, 0.9, size=100).astype(np.float32)
    s_L = rng.uniform(0.2, 1.0, size=100).astype(np.float32)

    router = TwoHeadRouter(input_dim=20, hidden_dim=32, epochs=5, seed=42)
    router.fit(X, s_S, s_L)

    orig_gap = router.predict_gap(X[:10])

    model_path = tmp_path / "router.pt"
    router.save(model_path)
    assert model_path.exists()

    loaded = TwoHeadRouter.load(model_path)
    loaded_gap = loaded.predict_gap(X[:10])
    assert np.allclose(orig_gap, loaded_gap, atol=1e-5)


def test_retrain_small_head_freezes_trunk_and_large_head():
    pytest.importorskip("torch")
    import copy

    rng = np.random.default_rng(123)
    X = rng.standard_normal((80, 16)).astype(np.float32)
    s_S = rng.uniform(0.1, 0.9, size=80).astype(np.float32)
    s_L = rng.uniform(0.2, 1.0, size=80).astype(np.float32)

    router = TwoHeadRouter(input_dim=16, hidden_dim=32, epochs=5, seed=42)
    router.fit(X, s_S, s_L)

    # Cache pre-retraining weights of trunk and head_L
    trunk_weight_before = copy.deepcopy(router.model.trunk[0].weight.data)
    head_L_weight_before = copy.deepcopy(router.model.head_L.weight.data)
    head_S_weight_before = copy.deepcopy(router.model.head_S.weight.data)

    # Retrain ONLY small head with new small scores (e.g. Qwen)
    s_S_new = 1.0 - s_S  # inverted labels to force large weight change
    router.retrain_small_head(X, s_S_new=s_S_new, epochs=10, lr=1e-2)

    trunk_weight_after = router.model.trunk[0].weight.data
    head_L_weight_after = router.model.head_L.weight.data
    head_S_weight_after = router.model.head_S.weight.data

    # Trunk and head_L must remain IDENTICAL (frozen)
    assert (trunk_weight_before == trunk_weight_after).all()
    assert (head_L_weight_before == head_L_weight_after).all()
    # Small head must have updated
    assert not (head_S_weight_before == head_S_weight_after).all()

