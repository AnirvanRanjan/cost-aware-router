#!/usr/bin/env python3
"""Retrain only the small-model head of TwoHeadRouter on new small-model scores (T36 prep).

    # Test with Yi-34B scores standing in for Qwen:
    python scripts/18_retrain_small_head.py --new-small zero-one-ai/Yi-34B-Chat

    # When Track B arrives:
    python scripts/18_retrain_small_head.py --scores artifacts/trackb/scores.parquet \\
                                           --corpus artifacts/trackb/corpus.parquet \\
                                           --new-small qwen3.5-4b

Modular architecture (briefs/B.md T36):
Freezes the shared representation trunk (Linear 791->256, GELU) and the frontier head (w_L)
so all learned frontier quality representations remain bitwise identical. Only head_S (Linear 256->1)
is retrained on the new small model's performance labels.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import schema
from router.features import heuristics
from router.models import (
    TwoHeadRouter,
    fuse_features,
    spearman_rho,
)


def load_fused_features(
    corpus: pd.DataFrame,
    features_path: Path,
) -> tuple[np.ndarray, list[str]]:
    """Load F2 embeddings and concatenate with F0 heuristics."""
    qids = corpus["qid"].astype(str).tolist()
    prompts = corpus["prompt"].tolist()

    # F0 heuristics
    x0, _ = heuristics(prompts)
    x0 = np.nan_to_num(np.asarray(x0, dtype=np.float32))

    # F2 embeddings
    feat_df = schema.read(features_path, schema.FEATURES)
    f2_map = dict(zip(feat_df["qid"].astype(str), feat_df["vector"]))
    f2_list = [f2_map[q] for q in qids]
    f2_arr = np.asarray(f2_list, dtype=np.float32)

    X_fused = fuse_features(f2_arr, x0, z_score=True)
    return X_fused, qids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("artifacts/trackb/corpus.parquet"))
    ap.add_argument("--scores", type=Path, default=Path("artifacts/trackb/scores_hard.parquet"))
    ap.add_argument("--features", type=Path, default=Path("artifacts/features/features.parquet"))
    ap.add_argument("--base-corpus", type=Path, default=Path("artifacts/routerbench/corpus.parquet"))
    ap.add_argument("--base-scores", type=Path, default=Path("artifacts/routerbench/scores.parquet"))
    ap.add_argument("--router-in", type=Path, default=Path("artifacts/models/two_head_mixtral.pt"), help="Path to pre-trained TwoHeadRouter .pt file")
    ap.add_argument("--base-small", default="mistralai/mixtral-8x7b-chat", help="Original small model")
    ap.add_argument("--new-small", default="qwen3.5-4b", help="New small model to retrain head_S on")
    ap.add_argument("--large", default="gpt-4-1106-preview", help="Frontier model")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("artifacts/models/two_head_qwen.pt"))
    ap.add_argument("--predictions-out", type=Path, default=Path("artifacts/trackb/qwen_query_predictions.parquet"))
    args = ap.parse_args()

    import torch

    # Step 1: Obtain or train base router on Track A (Mixtral vs GPT-4)
    if args.router_in is not None and args.router_in.exists():
        print(f"Loading pre-trained base router from {args.router_in}...")
        router = TwoHeadRouter.load(args.router_in)
    else:
        print(f"No pre-trained router found at {args.router_in}. Fitting base router on {args.base_small} vs {args.large} from {args.base_scores}...")
        b_corpus = schema.read(args.base_corpus, schema.CORPUS)
        b_scores = schema.read(args.base_scores, schema.SCORES)
        b_wide = b_scores.pivot(index="qid", columns="model", values="score")
        if args.base_small not in b_wide.columns or args.large not in b_wide.columns:
            sys.exit(f"Base models not found in {args.base_scores}.")
        
        b_valid = b_corpus["qid"].isin(b_wide.index)
        b_c = b_corpus[b_valid].copy().reset_index(drop=True)
        b_qids = b_c["qid"].tolist()
        b_split = b_c["split"].to_numpy()
        b_group_ids = b_c["group_id"].to_numpy()
        b_X_fused, _ = load_fused_features(b_c, args.features)
        
        b_s_S = b_wide.loc[b_qids, args.base_small].to_numpy(dtype=float)
        b_s_L = b_wide.loc[b_qids, args.large].to_numpy(dtype=float)
        b_tr = b_split == "train"
        b_ca = b_split == "cal"

        router = TwoHeadRouter(name="two_head_mlp", epochs=20, seed=args.seed)
        router.fit(b_X_fused[b_tr], b_s_S[b_tr], b_s_L[b_tr], group_ids=b_group_ids[b_tr], verbose=False)
        router.calibrate(b_X_fused[b_ca], b_s_S[b_ca], b_s_L[b_ca])
        if args.router_in:
            args.router_in.parent.mkdir(parents=True, exist_ok=True)
            router.save(args.router_in)
            print(f"Saved base router to {args.router_in}")

    # Step 2: Load Track B data for the new small model
    corpus = schema.read(args.corpus, schema.CORPUS)
    scores = schema.read(args.scores, schema.SCORES)

    wide = scores.pivot(index="qid", columns="model", values="score")
    if args.new_small not in wide.columns:
        sys.exit(f"Model '{args.new_small}' not found in scores file. Available: {list(wide.columns)}")
    if args.large not in wide.columns:
        sys.exit(f"Model '{args.large}' not found in scores file. Available: {list(wide.columns)}")

    valid_qids = corpus["qid"].isin(wide.index)
    c = corpus[valid_qids].copy().reset_index(drop=True)
    qids = c["qid"].tolist()
    split = c["split"].to_numpy()
    group_ids = c["group_id"].to_numpy()

    print(f"Loading features for {len(c):,} Track B questions...")
    X_fused, _ = load_fused_features(c, args.features)

    s_new_S = wide.loc[qids, args.new_small].to_numpy(dtype=float)
    s_L = wide.loc[qids, args.large].to_numpy(dtype=float)

    tr = split == "train"
    ca = split == "cal"
    te = split == "test"

    # Record weights before retraining for verification
    trunk_before = copy.deepcopy(router.model.trunk.state_dict())
    head_L_before = copy.deepcopy(router.model.head_L.state_dict())
    head_S_before = copy.deepcopy(router.model.head_S.state_dict())

    # Step 3: Retrain ONLY head_S on new small model scores
    print(f"\nRetraining head_S on '{args.new_small}' labels (freezing trunk and head_L)...")
    router.retrain_small_head(
        X=X_fused[tr],
        s_S_new=s_new_S[tr],
        group_ids=group_ids[tr],
        epochs=args.epochs,
        lr=args.lr,
        verbose=False,
    )

    # Step 3: Verify trunk and head_L weights are completely unchanged
    trunk_after = router.model.trunk.state_dict()
    head_L_after = router.model.head_L.state_dict()
    head_S_after = router.model.head_S.state_dict()

    for k in trunk_before:
        if not torch.equal(trunk_before[k], trunk_after[k]):
            raise RuntimeError(f"Trunk parameter '{k}' was modified during retrain_small_head!")

    for k in head_L_before:
        if not torch.equal(head_L_before[k], head_L_after[k]):
            raise RuntimeError(f"Frontier head parameter '{k}' was modified during retrain_small_head!")

    s_differs = False
    for k in head_S_before:
        if not torch.equal(head_S_before[k], head_S_after[k]):
            s_differs = True
            break
    if not s_differs:
        print("Warning: head_S weights did not change (check learning rate or data).")
    else:
        print("SUCCESS: Trunk and head_L weights bitwise verified intact. head_S successfully updated.")

    # Step 4: Re-calibrate on the calibration split
    print(f"Calibrating updated router on calibration split (N={ca.sum():,})...")
    router.calibrate(X_fused[ca], s_new_S[ca], s_L[ca])

    # Step 5: Evaluate on the untouched test split
    gap_test = s_L[te] - s_new_S[te]
    pred_S, pred_L = router.predict_proba(X_fused[te], calibrated=True)
    pred_gap = pred_L - pred_S
    rho_test = spearman_rho(pred_gap, gap_test)

    # Evaluate classification AUC: does router predict when Large outperforms New Small?
    from sklearn.metrics import roc_auc_score
    y_test_bin = (gap_test > 0).astype(int)
    auc_test = roc_auc_score(y_test_bin, pred_gap) if len(np.unique(y_test_bin)) > 1 else 0.5

    report = {
        "status": "success",
        "pair": {"small": args.new_small, "large": args.large},
        "retrain_target": "head_S",
        "trunk_frozen": True,
        "head_L_frozen": True,
        "test_n": int(te.sum()),
        "test_spearman_rho": float(rho_test),
        "test_roc_auc": float(auc_test),
        "mean_s_hat_S": float(pred_S.mean()),
        "mean_s_true_S": float(s_new_S[te].mean()),
        "mean_s_hat_L": float(pred_L.mean()),
        "mean_s_true_L": float(s_L[te].mean()),
    }
    print("\nRetraining Report on Test Split:")
    # Step 6: Save predictions table if requested
    if args.predictions_out:
        from router.costs import PriceTable
        from router.data import approx_tokens

        prices = PriceTable.load()
        n_in = np.array([approx_tokens(p) for p in c["prompt"]], dtype=float)
        mean_out = float(scores[scores["model"] == args.large]["mean_n_out"].mean())
        c_hat_L = np.asarray(prices.query_cost(args.large, n_in, np.full(len(c), mean_out)), float)
        pred_df = router.to_predictions_df(X_fused, qids=qids, c_hat_L=c_hat_L, calibrated=True)
        schema.write(pred_df, schema.PREDICTIONS, args.predictions_out)
        print(f"Wrote predictions to {args.predictions_out}")

    # Step 7: Save retrained router
    args.out.parent.mkdir(parents=True, exist_ok=True)
    router.save(args.out)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved retrained router to {args.out}")


if __name__ == "__main__":
    main()
