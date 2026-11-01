#!/usr/bin/env python3
"""PREVIEW of the headline pair, Qwen3.5-4B vs GPT-4, on Track B (frozen splits; test touched once).

    python scripts/16_trackb_preview.py
    python scripts/08_threshold_report.py --predictions artifacts/trackb/preview_predictions.parquet \
        --corpus artifacts/trackb/corpus.parquet --scores artifacts/trackb/scores_hard.parquet \
        --small qwen3.5-4b --out-md notes/trackb_preview_report.md --out-json artifacts/reports/trackb_preview.json

Quality is what a user would actually get:
  * Qwen, GSM8K: the majority answer of its 3 samples (ties -> first sample), graded by exact match;
  * Qwen, MMLU / ARC / Winogrande: its highest-probability option, right or wrong (not the soft P(gold));
  * GPT-4: RouterBench's stored score.
Writes artifacts/trackb/scores_hard.parquet with those 0/1 scores.

Routers (fitted on TRAIN; gradient-boosted trees on the gap, as in scripts/12_cascade_explore.py):
  query_f2         F2 sentence embeddings of the question only (needs artifacts/features/features.parquet)
  cascade_conf     Qwen's own signals only: GSM8K sample agreement (3/3, 2/3, 1/3), mean token
                   log-probability, answer length; MCQ chosen-option probability and margin
  cascade_conf+f2  both

This is lane B's T36 / D-12 territory, built only to see the numbers early. Lane B owns the final router.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import graders, schema  # noqa: E402
from router.costs import PriceTable  # noqa: E402
from router.data import approx_tokens  # noqa: E402

HEADLINE = ("mmlu", "gsm8k", "arc-challenge", "winogrande")
SMALL, LARGE = "qwen3.5-4b", "gpt-4-1106-preview"


def gsm8k_signals(responses: pd.DataFrame, corpus: pd.DataFrame) -> pd.DataFrame:
    """Per question: majority-answer correctness and Qwen's confidence signals."""
    c = corpus.set_index("qid")
    rows = []
    for qid, g in responses.groupby("qid"):
        q, gold = c.at[qid, "prompt"], float(c.at[qid, "reference"])
        nums = [graders.extract_number(t or "", q) for t in g.sort_values("sample_idx")["text"]]
        valid = [n for n in nums if n is not None]
        counts = Counter(round(n, 6) for n in valid)
        top, top_n = (counts.most_common(1)[0] if counts else (None, 0))
        maj = top if top_n > 1 else (nums[0] if nums else None)
        rows.append({"qid": qid,
                     "s_hard": float(maj is not None and abs(maj - gold) < 1e-6),
                     "agree": top_n / len(nums), "n_distinct": len(counts),
                     "lp_mean": float(g["mean_logprob"].mean()), "lp_min": float(g["mean_logprob"].min()),
                     "len": float(np.log1p(g["n_out"].mean())), "p_chosen": np.nan, "margin": np.nan})
    return pd.DataFrame(rows)


def mcq_signals(options: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for qid, correct, p, lps in zip(options["qid"], options["correct"], options["p_chosen"], options["logprobs"]):
        lp = np.array(sorted(json.loads(lps).values(), reverse=True))
        pr = np.exp(lp - lp.max())
        pr = pr / pr.sum()
        margin = float(pr[0] - pr[1]) if len(pr) > 1 else 1.0
        rows.append({"qid": qid, "s_hard": float(correct), "agree": np.nan, "n_distinct": np.nan,
                     "lp_mean": np.nan, "lp_min": np.nan, "len": np.nan, "p_chosen": float(p), "margin": margin})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("artifacts/trackb"))
    ap.add_argument("--features", type=Path, default=Path("artifacts/features/features.parquet"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingRegressor

    corpus = schema.read(args.dir / "corpus.parquet", schema.CORPUS)
    rb = schema.read(args.dir / "scores.parquet", schema.SCORES)
    resp = schema.read(args.dir / "qwen_responses.parquet", schema.RESPONSES)
    opts = pd.read_parquet(args.dir / "qwen_mcq_options.parquet")

    sig = pd.concat([gsm8k_signals(resp, corpus), mcq_signals(opts)], ignore_index=True)
    d = corpus.merge(sig, on="qid", how="inner")
    d = d[d["source"].isin(HEADLINE)].reset_index(drop=True)
    big = rb[rb["model"] == LARGE].set_index("qid")
    d["s_L"] = big.loc[d["qid"], "score"].to_numpy()

    # hard scores file for 08_threshold_report
    small_rb = rb[rb["model"] == SMALL].set_index("qid")
    hard = pd.concat([
        pd.DataFrame({"qid": d["qid"], "model": SMALL, "score": d["s_hard"], "grader": "trackb_hard",
                      "grader_version": graders.VERSION,
                      "mean_n_in": small_rb.loc[d["qid"], "mean_n_in"].to_numpy(),
                      "mean_n_out": small_rb.loc[d["qid"], "mean_n_out"].to_numpy()}),
        big.loc[d["qid"]].reset_index()[schema.SCORES.names],
    ], ignore_index=True)
    schema.write(hard, schema.SCORES, args.dir / "scores_hard.parquet")

    print("Hard quality per dataset (what the user gets):")
    print(d.groupby("source")[["s_hard", "s_L"]].mean().rename(columns={"s_hard": "Qwen", "s_L": "GPT-4"})
          .assign(gap=lambda t: t["GPT-4"] - t["Qwen"]).round(3).to_string())

    src = pd.get_dummies(d["source"]).reindex(columns=list(HEADLINE), fill_value=0).astype(float).to_numpy()
    conf = d[["agree", "n_distinct", "lp_mean", "lp_min", "len", "p_chosen", "margin"]].to_numpy(float)
    X = {"cascade_conf": np.column_stack([conf, src])}
    if args.features.exists():
        f = schema.read(args.features, schema.FEATURES)
        fm = dict(zip(f["qid"], f["vector"]))
        F2 = np.array([fm[q] for q in d["qid"]], dtype=np.float32)
        X["query_f2"] = np.column_stack([F2, src])
        X["cascade_conf+f2"] = np.column_stack([conf, F2, src])
    gap = d["s_L"].to_numpy(float) - d["s_hard"].to_numpy(float)
    tr = (d["split"] == "train").to_numpy()

    prices = PriceTable.load()
    n_in = np.array([approx_tokens(p) for p in d["prompt"]], dtype=float)
    c_hat_L = np.asarray(prices.query_cost(LARGE, n_in, np.full(len(d), float(big["mean_n_out"].mean()))), float)

    frames = []
    for name, M in X.items():
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                              l2_regularization=1.0, random_state=args.seed)
        model.fit(M[tr], gap[tr])
        dh = np.clip(model.predict(M), -1, 1)
        frames.append(pd.DataFrame({"qid": d["qid"], "router": name, "seed": args.seed,
                                    "s_hat_S": np.clip(0.5 - dh / 2, 0, 1), "s_hat_L": np.clip(0.5 + dh / 2, 0, 1),
                                    "delta_hat": dh, "c_hat_L": c_hat_L}))
        print(f"fitted {name}")
    preds = pd.concat(frames, ignore_index=True)
    schema.write(preds, schema.PREDICTIONS, args.dir / "preview_predictions.parquet")
    (args.dir / "preview_predictions.json").write_text(json.dumps({"interim": True, "preview": True,
                                                                   "pair": {"small": SMALL, "large": LARGE}}))
    te = (d["split"] == "test").to_numpy()
    g = d[te & (d["source"] == "gsm8k")]
    print(f"GSM8K test: Qwen right {g.loc[g.agree == 1, 's_hard'].mean():.3f} when all 3 samples agree "
          f"({(g.agree == 1).mean():.0%} of questions), {g.loc[g.agree < 1, 's_hard'].mean():.3f} otherwise")
    m = d[te & d["source"].isin(("mmlu", "arc-challenge", "winogrande"))]
    hi = m["p_chosen"] >= 0.9
    print(f"MCQ test: Qwen right {m.loc[hi, 's_hard'].mean():.3f} when its option probability >= 0.9 "
          f"({hi.mean():.0%} of questions), {m.loc[~hi, 's_hard'].mean():.3f} otherwise")


if __name__ == "__main__":
    main()
