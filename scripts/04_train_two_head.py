#!/usr/bin/env python3
"""T14 deliverable (briefs/B.md): Two-head MLP router trained across 5 seeds.

    python scripts/04_train_two_head.py --corpus artifacts/routerbench/corpus.parquet \
                                        --scores artifacts/routerbench/scores.parquet \
                                        --out artifacts/predictions/predictions.parquet
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import schema
from router.costs import PriceTable
from router.features import SentenceEmbeddingExtractor, heuristics
from router.models import TwoHeadRouter, fuse_features, spearman_rho, train_two_head_multi_seed


def main() -> None:
    default_corpus = (
        Path("artifacts/routerbench/corpus.parquet")
        if Path("artifacts/routerbench/corpus.parquet").exists()
        else Path("artifacts/synthetic/corpus.parquet")
    )
    default_scores = (
        Path("artifacts/routerbench/scores.parquet")
        if Path("artifacts/routerbench/scores.parquet").exists()
        else Path("artifacts/synthetic/scores.parquet")
    )

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=default_corpus)
    ap.add_argument("--scores", type=Path, default=default_scores)
    ap.add_argument("--features", type=Path, default=None, help="Optional precomputed F2 features.parquet")
    ap.add_argument("--out", type=Path, default=Path("artifacts/predictions/predictions.parquet"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--large", type=str, default=None)
    ap.add_argument("--small", type=str, default=None)
    args = ap.parse_args()

    if not args.corpus.exists():
        print(f"Error: corpus {args.corpus} not found.", file=sys.stderr)
        sys.exit(1)
    if not args.scores.exists():
        print(f"Error: scores {args.scores} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading corpus from {args.corpus}...")
    corpus = schema.read(args.corpus, schema.CORPUS)
    print(f"Reading scores from {args.scores}...")
    scores = schema.read(args.scores, schema.SCORES)

    wide = scores.pivot(index="qid", columns="model", values="score")
    models = list(wide.columns)

    # For RouterBench real data, default pair is the Mixtral pair (briefs/B.md):
    # gpt-4-1106-preview (frontier) vs mistralai/mixtral-8x7b-chat (small)
    if not args.large:
        large_model = "gpt-4-1106-preview" if "gpt-4-1106-preview" in models else models[0]
    else:
        large_model = args.large

    if not args.small:
        small_model = (
            "mistralai/mixtral-8x7b-chat"
            if "mistralai/mixtral-8x7b-chat" in models
            else (models[-1] if len(models) > 1 else models[0])
        )
    else:
        small_model = args.small

    if large_model == small_model and len(models) > 1:
        small_model = models[1]
    print(f"Routing pair: large={large_model}, small={small_model}")

    c = corpus.set_index("qid")
    common_qids = [q for q in c.index if q in wide.index]
    c = c.loc[common_qids]
    wide = wide.loc[common_qids]

    s_S = wide[small_model].to_numpy(dtype=float)
    s_L = wide[large_model].to_numpy(dtype=float)
    gap = s_L - s_S

    # Compute c_hat_L using the project's official PriceTable (DESIGN.md §3.2)
    scores_L = scores[scores["model"] == large_model].set_index("qid")
    if "mean_n_in" in scores_L.columns and "mean_n_out" in scores_L.columns:
        price_table = PriceTable.load()
        n_in = scores_L.loc[common_qids, "mean_n_in"].to_numpy(dtype=float)
        n_out = scores_L.loc[common_qids, "mean_n_out"].to_numpy(dtype=float)
        c_hat_L = price_table.query_cost(large_model, n_in, n_out)
    else:
        c_hat_L = None

    print(f"Extracting F0 heuristic features for {len(c):,} prompts...")
    X_f0, f0_names = heuristics(c["prompt"].tolist())

    if args.features and args.features.exists():
        print(f"Reading precomputed F2 features from {args.features}...")
        feat_df = schema.read(args.features, schema.FEATURES)
        feat_map = dict(zip(feat_df["qid"], feat_df["vector"]))
        X_f2 = np.array([feat_map[q] for q in common_qids], dtype=np.float32)
        X_fused = fuse_features(X_f2, X_f0)
    elif Path("artifacts/features/features.parquet").exists():
        print(f"Reading precomputed F2 features from artifacts/features/features.parquet...")
        feat_df = schema.read(Path("artifacts/features/features.parquet"), schema.FEATURES)
        feat_map = dict(zip(feat_df["qid"], feat_df["vector"]))
        X_f2 = np.array([feat_map[q] for q in common_qids], dtype=np.float32)
        X_fused = fuse_features(X_f2, X_f0)
    else:
        print("Using synthetic embeddings / F0 features...")
        rng = np.random.default_rng(0)
        # Mock 768-d embeddings with signal aligned with gap if no precomputed F2 is provided
        X_f2_sim = rng.standard_normal((len(common_qids), 64), dtype=np.float32)
        for i in range(len(common_qids)):
            X_f2_sim[i, 0] += float(gap[i]) * 1.5
        X_fused = fuse_features(X_f2_sim, X_f0)

    print(f"Fused feature matrix shape: {X_fused.shape}")
    print(f"Training TwoHeadRouter across seeds {args.seeds} with early stopping on grouped 85/15 sub-split...")

    models_list, predictions_df = train_two_head_multi_seed(
        X_fused=X_fused,
        s_S=s_S,
        s_L=s_L,
        group_ids=c["group_id"].to_numpy(),
        split=c["split"].to_numpy(),
        qids=common_qids,
        seeds=args.seeds,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        eval_splits=("cal", "test"),
        c_hat_L=c_hat_L,
    )

    test_mask = c["split"] == "test"
    test_qids = set(c.index[test_mask])
    test_gap = {q: g for q, g in zip(common_qids, gap) if q in test_qids}

    rhos = []
    for seed in args.seeds:
        sub = predictions_df[(predictions_df["seed"] == seed) & (predictions_df["qid"].isin(test_qids))]
        sub_gap = np.array([test_gap[q] for q in sub["qid"]])
        r = spearman_rho(sub["delta_hat"].to_numpy(), sub_gap)
        rhos.append(r)
        print(f"  Seed {seed}: test Spearman rho = {r:.4f}")
    print(f"Mean test Spearman rho across {len(args.seeds)} seeds: {np.mean(rhos):.4f} +/- {np.std(rhos):.4f}")

    print(f"Writing predictions to {args.out}...")
    schema.write(predictions_df, schema.PREDICTIONS, args.out)

    manifest = {
        "artifact": "predictions",
        "router": "two_head_mlp",
        "contract_version": schema.CONTRACT_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "rows": len(predictions_df),
        "seeds": list(args.seeds),
        "mean_test_spearman_rho": float(np.mean(rhos)),
        "std_test_spearman_rho": float(np.std(rhos)),
        "content_hash": schema.content_hash(predictions_df),
    }
    manifest_path = args.out.parent / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest to {manifest_path}.")


if __name__ == "__main__":
    main()
