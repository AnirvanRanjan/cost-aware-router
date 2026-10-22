#!/usr/bin/env python3
"""Interim router predictions on REAL RouterBench data that need no embeddings (F0 and F1 only).

Lets lane C build the threshold / Pareto / guarantee pipeline (T21-T25) on real numbers before F2
embeddings exist. Uses lane B's own classes (router/features.py, router/models.py,
router/calibrate.py). Writes PREDICTIONS rows for every split (train, cal and test) so tau can be
chosen on cal and reported on test.

    python scripts/07_baseline_predictions.py                       # Mixtral vs GPT-4 (D-08 dev pair)
    python scripts/07_baseline_predictions.py --small zero-one-ai/Yi-34B-Chat

Routers written (column `router`):
  f0_logreg      23 heuristic features + logistic regression on 1{s_L > s_S}
  f1_logreg      TF-IDF word + char n-grams + logistic regression
  f0f1_twohead   two-head MLP on [F0 | SVD-256 of F1], isotonic-calibrated on cal

c_hat_L: predicted frontier cost per query = prompt tokens (len/4) at the frontier input price plus
the frontier's MEAN output length on train at the output price. Output length is not known when
routing, so the mean is the honest estimate (DESIGN.md §2.2).

INTERIM: splits are provisional until the embedding dedup pass freezes them (T04). Re-run after.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import schema  # noqa: E402
from router.costs import PriceTable  # noqa: E402
from router.data import approx_tokens  # noqa: E402
from router.features import TfidfFeatureExtractor, heuristics  # noqa: E402
from router.models import LogRegRouter, TwoHeadRouter  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("artifacts/routerbench/corpus.parquet"))
    ap.add_argument("--scores", type=Path, default=Path("artifacts/routerbench/scores.parquet"))
    ap.add_argument("--small", default="mistralai/mixtral-8x7b-chat")
    ap.add_argument("--large", default="gpt-4-1106-preview")
    ap.add_argument("--out", type=Path, default=Path("artifacts/predictions/baselines.parquet"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--svd-dim", type=int, default=256)
    ap.add_argument("--skip-mlp", action="store_true", help="logistic baselines only (no torch needed)")
    args = ap.parse_args()

    t0 = time.time()
    corpus = schema.read(args.corpus, schema.CORPUS)
    scores = schema.read(args.scores, schema.SCORES)
    wide = scores.pivot(index="qid", columns="model", values="score")
    for m in (args.small, args.large):
        if m not in wide.columns:
            sys.exit(f"model {m!r} not in scores; have {list(wide.columns)}")
    c = corpus.set_index("qid").loc[lambda d: d.index.isin(wide.index)]
    wide = wide.loc[c.index]
    s_S = wide[args.small].to_numpy(float)
    s_L = wide[args.large].to_numpy(float)
    gap = s_L - s_S
    split = c["split"].to_numpy()
    tr, ca = split == "train", split == "cal"
    qids = c.index.to_numpy()
    print(f"pair: S={args.small}  L={args.large}  | train {tr.sum():,}  cal {ca.sum():,}  test {(split == 'test').sum():,}")

    # predicted frontier cost (DESIGN.md §2.2)
    prices = PriceTable.load()
    large_rows = scores[scores["model"] == args.large].set_index("qid").loc[qids]
    mean_out = float(large_rows.loc[tr, "mean_n_out"].mean())
    n_in = np.array([approx_tokens(p) for p in c["prompt"]], dtype=float)
    c_hat_L = np.asarray(prices.query_cost(args.large, n_in, np.full(len(c), mean_out)), dtype=float)

    prompts = c["prompt"].tolist()
    frames = []

    print("F0 heuristics ...")
    X0, _ = heuristics(prompts)
    X0 = np.nan_to_num(np.asarray(X0, dtype=np.float64))
    mu, sd = X0[tr].mean(0), X0[tr].std(0) + 1e-8
    X0z = (X0 - mu) / sd
    r0 = LogRegRouter(name="f0_logreg", seed=args.seed).fit(X0z[tr], gap[tr])
    frames.append(r0.to_predictions_df(X0z, qids, c_hat_L=c_hat_L))

    print("F1 TF-IDF (fit on train only) ...")
    tfidf = TfidfFeatureExtractor(min_df=2, max_word_features=3000, max_char_features=3000)
    tfidf.fit([p for p, t in zip(prompts, tr) if t])
    X1 = tfidf.transform(prompts)
    r1 = LogRegRouter(name="f1_logreg", seed=args.seed).fit(X1[tr], gap[tr])
    frames.append(r1.to_predictions_df(X1, qids, c_hat_L=c_hat_L))

    info = {"pair": {"small": args.small, "large": args.large}, "seed": args.seed,
            "mean_frontier_out_tokens_train": mean_out, "interim": True}
    if not args.skip_mlp:
        from sklearn.decomposition import TruncatedSVD
        print(f"F0 | SVD-{args.svd_dim}(F1) two-head MLP ...")
        svd = TruncatedSVD(n_components=args.svd_dim, random_state=args.seed).fit(X1[tr])
        Xf = np.hstack([X0z, svd.transform(X1)]).astype(np.float32)
        r2 = TwoHeadRouter(name="f0f1_twohead", epochs=args.epochs, seed=args.seed)
        r2.fit(Xf[tr], s_S[tr], s_L[tr], group_ids=c["group_id"].to_numpy()[tr])
        r2.calibrate(Xf[ca], s_S[ca], s_L[ca])
        frames.append(r2.to_predictions_df(Xf, qids, c_hat_L=c_hat_L, calibrated=True))
        info["mlp_best_epoch"] = int(r2.best_epoch)
        info["svd_explained_variance"] = float(svd.explained_variance_ratio_.sum())

    preds = pd.concat(frames, ignore_index=True)
    schema.write(preds, schema.PREDICTIONS, args.out)
    info["routers"] = sorted(preds["router"].unique())
    info["seconds"] = round(time.time() - t0, 1)
    args.out.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(f"wrote {len(preds):,} rows -> {args.out}  ({info['seconds']} s)")


if __name__ == "__main__":
    main()
