#!/usr/bin/env python3
"""Build the router bundle the demo server loads (T41 interface; interim F0 + TF-IDF version).

    python scripts/11_build_bundle.py                                  # Mixtral labels (dev pair)
    python scripts/11_build_bundle.py --scores artifacts/trackb/scores.parquet --small qwen3.5-4b

Trains the two-head router on TRAIN, calibrates both heads on CAL, picks tau* on CAL with the WSR
fixed-sequence rule (grid from TRAIN scores), and writes artifacts/bundle/router.pkl.
Then checks the acceptance test of T41: the bundle's decisions equal the evaluation code's on
1,000 test queries.

INTERIM until lane B ships F2: no embeddings, and until Track B exists the small-model head is
trained on Mixtral, not on the Qwen model the demo actually runs. Lane B replaces the featurizer
and retrains the small head on Qwen (TwoHeadRouter.retrain_small_head) with the same file format.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import schema  # noqa: E402
from router import threshold as th  # noqa: E402
from router.bundle import CascadeFeaturizer, F0F1Featurizer, F0F2Featurizer, RouterBundle  # noqa: E402
from router.cascade import CascadeRouter  # noqa: E402
from router.costs import PriceTable  # noqa: E402
from router.data import approx_tokens  # noqa: E402
from router import graders  # noqa: E402
from router.features import TfidfFeatureExtractor, heuristics  # noqa: E402
from router.models import TwoHeadRouter  # noqa: E402

HEADLINE = ("mmlu", "gsm8k", "arc-challenge", "winogrande")


def gsm8k_signals(responses: pd.DataFrame, corpus: pd.DataFrame, n_samples: int = 1) -> pd.DataFrame:
    """Per question: correctness and Qwen confidence signals."""
    c = corpus.set_index("qid")
    rows = []
    for qid, g in responses.groupby("qid"):
        q, gold = c.at[qid, "prompt"], float(c.at[qid, "reference"])
        g_sorted = g.sort_values("sample_idx")
        s0 = g_sorted.iloc[0]
        text0 = s0["text"] or ""
        num = graders.extract_number(text0, q)
        s_hard = float(num is not None and abs(num - gold) < 1e-6)
        rows.append({
            "qid": qid,
            "s_hard": s_hard,
            "resp_text": text0,
            "agree": 1.0,
            "n_distinct": 1.0,
            "lp_mean": float(s0["mean_logprob"]),
            "lp_min": float(s0["mean_logprob"]),
            "len": float(np.log1p(s0["n_out"])),
            "p_chosen": np.nan,
            "margin": np.nan,
        })
    return pd.DataFrame(rows)


def mcq_signals(options: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for qid, correct, p, lps in zip(options["qid"], options["correct"], options["p_chosen"], options["logprobs"]):
        lp = np.array(sorted(json.loads(lps).values(), reverse=True))
        pr = np.exp(lp - lp.max())
        pr = pr / pr.sum()
        margin = float(pr[0] - pr[1]) if len(pr) > 1 else 1.0
        rows.append({
            "qid": qid,
            "s_hard": float(correct),
            "resp_text": "",
            "agree": np.nan,
            "n_distinct": np.nan,
            "lp_mean": np.nan,
            "lp_min": np.nan,
            "len": np.nan,
            "p_chosen": float(p),
            "margin": margin,
        })
    return pd.DataFrame(rows)


def build_cascade_bundle(args: argparse.Namespace) -> None:
    corpus = schema.read(args.dir / "corpus.parquet", schema.CORPUS)
    hard_path = args.dir / "scores_hard.parquet"
    rb_path = hard_path if hard_path.exists() else (args.dir / "scores.parquet")
    rb = schema.read(rb_path, schema.SCORES)
    resp = schema.read(args.dir / "qwen_responses.parquet", schema.RESPONSES)
    opts = pd.read_parquet(args.dir / "qwen_mcq_options.parquet")

    sig1 = pd.concat([gsm8k_signals(resp, corpus, n_samples=1), mcq_signals(opts)], ignore_index=True)
    d1 = corpus.merge(sig1, on="qid", how="inner")
    d1 = d1[d1["source"].isin(HEADLINE)].reset_index(drop=True)
    big = rb[rb["model"] == args.large].set_index("qid")
    d1["s_L"] = big.loc[d1["qid"], "score"].to_numpy()
    d1["s_S"] = d1["s_hard"].to_numpy()
    d1["gap"] = d1["s_L"] - d1["s_S"]

    split = d1["split"].to_numpy()
    tr = split == "train"
    ca = split == "cal"
    te = split == "test"

    feat = CascadeFeaturizer(headline_sources=HEADLINE)
    texts = d1["prompt"].tolist()
    resps = d1["resp_text"].tolist()
    sources = d1["source"].tolist()
    p_chosen = d1["p_chosen"].tolist()
    mean_lp = d1["lp_mean"].tolist()

    X = feat.transform(
        texts=texts,
        responses=resps,
        sources=sources,
        p_chosen_list=p_chosen,
        mean_logprobs_list=mean_lp,
    )

    router = CascadeRouter(name="cascade_1sample", max_iter=300, learning_rate=0.05, seed=args.seed)
    router.fit(X[tr], d1["gap"].to_numpy()[tr])

    prices = PriceTable.load()
    p = prices[args.large] if args.large in prices.models else prices[prices.reference_frontier]
    mean_out = float(big.loc[d1["qid"][tr]]["mean_n_out"].mean())
    bundle = RouterBundle(
        featurizer=feat,
        router=router,
        tau=0.0,
        rule=args.rule,
        frontier_price=(p.input, p.output),
        mean_out_tokens=mean_out,
        _is_cascade=True,
    )

    pr = bundle.predict(
        texts,
        responses=resps,
        sources=sources,
        p_chosen_list=p_chosen,
        mean_logprobs_list=mean_lp,
    )

    sel = th.select_tau(
        pr["score"][ca],
        d1["s_S"].to_numpy()[ca],
        d1["s_L"].to_numpy()[ca],
        taus=th.threshold_grid(pr["score"][tr]),
        rho=args.rho,
        delta=args.delta,
        method="wsr",
    )
    bundle.tau = sel.tau_star

    test = th.evaluate_at_tau(
        pr["score"][te],
        d1["s_S"].to_numpy()[te],
        d1["s_L"].to_numpy()[te],
        sel.tau_star,
        B=500,
    )
    bundle.meta = {
        "pair": {"small": args.small, "large": args.large},
        "rule": args.rule,
        "rho": args.rho,
        "delta": args.delta,
        "certified": sel.certified,
        "cal": sel.cal,
        "test": {k: test[k] for k in ("alpha", "pgr", "savings")},
        "featurizer": feat.name,
        "is_cascade": True,
        "sources": list(HEADLINE),
    }
    bundle.save(args.out)

    # T41 acceptance: reproduce evaluation decisions on 1,000 test queries
    loaded = RouterBundle.load(args.out)
    idx = np.flatnonzero(te)[:1000]
    got = np.array([
        loaded.decide(
            text=texts[i],
            response=resps[i],
            source=sources[i],
            confidence=p_chosen[i] if not np.isnan(p_chosen[i]) else None,
            mean_logprob=mean_lp[i] if not np.isnan(mean_lp[i]) else None,
        ).route == "frontier"
        for i in idx
    ])
    want = pr["score"][idx] > sel.tau_star
    agree = float((got == want).mean())
    bundle.meta["t41_agreement_1000"] = agree
    bundle.save(args.out)
    print(json.dumps(bundle.meta, indent=2, default=float))
    if agree < 1.0:
        sys.exit(f"T41 check failed: bundle agrees with evaluation on only {agree:.1%} of 1,000 test queries")
    print(f"wrote {args.out}  (T41 check: {agree:.0%} agreement on {len(idx)} test queries)")


def build_query_bundle(args: argparse.Namespace) -> None:
    from sklearn.decomposition import TruncatedSVD

    corpus = schema.read(args.corpus, schema.CORPUS)
    scores = schema.read(args.scores, schema.SCORES)
    wide = scores.pivot(index="qid", columns="model", values="score")
    c = corpus.set_index("qid").loc[lambda d: d.index.isin(wide.index)]
    c = c[c["source"].isin(HEADLINE)]
    wide = wide.loc[c.index]
    s_S, s_L = wide[args.small].to_numpy(float), wide[args.large].to_numpy(float)
    split = c["split"].to_numpy()
    tr, ca, te = split == "train", split == "cal", split == "test"
    texts = c["prompt"].tolist()

    x0, _ = heuristics(texts)
    x0 = np.nan_to_num(np.asarray(x0, dtype=np.float64))
    mean, std = x0[tr].mean(0), x0[tr].std(0) + 1e-8

    if args.featurizer == "f0f2":
        if not args.features.exists():
            sys.exit(f"Features file {args.features} not found for f0f2 featurizer.")
        f_df = schema.read(args.features, schema.FEATURES)
        f2_map = dict(zip(f_df["qid"].astype(str), f_df["vector"]))
        f2_arr = np.asarray([f2_map[str(q)] for q in c.index], dtype=np.float32)
        x0_norm = ((x0 - mean) / std).astype(np.float32)
        X = np.concatenate([f2_arr, x0_norm], axis=1).astype(np.float32)
        feat = F0F2Featurizer(mean=mean, std=std)
    else:
        tfidf = TfidfFeatureExtractor(min_df=2, max_word_features=3000, max_char_features=3000)
        tfidf.fit([t for t, m in zip(texts, tr) if m])
        svd = TruncatedSVD(n_components=args.svd_dim, random_state=args.seed).fit(
            tfidf.transform([t for t, m in zip(texts, tr) if m])
        )
        feat = F0F1Featurizer(mean, std, tfidf, svd)
        X = feat.transform(texts)

    router = TwoHeadRouter(name=f"bundle_{args.featurizer}", epochs=args.epochs, seed=args.seed)
    router.fit(X[tr], s_S[tr], s_L[tr], group_ids=c["group_id"].to_numpy()[tr])
    router.calibrate(X[ca], s_S[ca], s_L[ca])

    prices = PriceTable.load()
    p = prices[args.large] if args.large in prices.models else prices[prices.reference_frontier]
    mean_out = float(scores[(scores["model"] == args.large) & scores["qid"].isin(c.index[tr])]["mean_n_out"].mean())
    bundle = RouterBundle(
        featurizer=feat,
        router=router,
        tau=0.0,
        rule=args.rule,
        frontier_price=(p.input, p.output),
        mean_out_tokens=mean_out,
        _is_cascade=False,
    )
    pr = bundle.predict(texts)
    sel = th.select_tau(
        pr["score"][ca],
        s_S[ca],
        s_L[ca],
        taus=th.threshold_grid(pr["score"][tr]),
        rho=args.rho,
        delta=args.delta,
        method="wsr",
    )
    bundle.tau = sel.tau_star
    test = th.evaluate_at_tau(pr["score"][te], s_S[te], s_L[te], sel.tau_star, B=500)
    bundle.meta = {
        "pair": {"small": args.small, "large": args.large},
        "rule": args.rule,
        "rho": args.rho,
        "delta": args.delta,
        "certified": sel.certified,
        "cal": sel.cal,
        "test": {k: test[k] for k in ("alpha", "pgr", "savings")},
        "featurizer": feat.name,
        "is_cascade": False,
        "sources": list(HEADLINE),
    }
    bundle.save(args.out)

    loaded = RouterBundle.load(args.out)
    idx = np.flatnonzero(te)[:1000]
    got = np.array([loaded.decide(texts[i]).route == "frontier" for i in idx])
    want = pr["score"][idx] > sel.tau_star
    agree = float((got == want).mean())
    bundle.meta["t41_agreement_1000"] = agree
    bundle.save(args.out)
    print(json.dumps(bundle.meta, indent=2, default=float))
    if agree < 1.0:
        sys.exit(f"T41 check failed: bundle agrees with evaluation on only {agree:.1%} of 1,000 test queries")
    print(f"wrote {args.out}  (T41 check: {agree:.0%} agreement on {len(idx)} test queries)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("cascade", "query"), default="cascade")
    ap.add_argument("--dir", type=Path, default=Path("artifacts/trackb"))
    ap.add_argument("--corpus", type=Path, default=Path("artifacts/routerbench/corpus.parquet"))
    ap.add_argument("--scores", type=Path, default=Path("artifacts/routerbench/scores.parquet"))
    ap.add_argument("--features", type=Path, default=Path("artifacts/features/features.parquet"))
    ap.add_argument("--featurizer", choices=("f0f2", "f0f1"), default="f0f2")
    ap.add_argument("--small", default="qwen3.5-4b")
    ap.add_argument("--large", default="gpt-4-1106-preview")
    ap.add_argument("--rule", choices=("v1", "v2"), default="v2")
    ap.add_argument("--rho", type=float, default=0.95)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--svd-dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("artifacts/bundle/router.pkl"))
    args = ap.parse_args()

    if args.mode == "cascade":
        build_cascade_bundle(args)
    else:
        build_query_bundle(args)


if __name__ == "__main__":
    main()

