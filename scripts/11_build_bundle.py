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
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import schema  # noqa: E402
from router import threshold as th  # noqa: E402
from router.bundle import F0F1Featurizer, RouterBundle  # noqa: E402
from router.costs import PriceTable  # noqa: E402
from router.features import TfidfFeatureExtractor, heuristics  # noqa: E402
from router.models import TwoHeadRouter  # noqa: E402

HEADLINE = ("mmlu", "gsm8k", "arc-challenge", "winogrande")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("artifacts/routerbench/corpus.parquet"))
    ap.add_argument("--scores", type=Path, default=Path("artifacts/routerbench/scores.parquet"))
    ap.add_argument("--small", default="mistralai/mixtral-8x7b-chat")
    ap.add_argument("--large", default="gpt-4-1106-preview")
    ap.add_argument("--rule", choices=("v1", "v2"), default="v1")
    ap.add_argument("--rho", type=float, default=0.95)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--svd-dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("artifacts/bundle/router.pkl"))
    args = ap.parse_args()

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
    tfidf = TfidfFeatureExtractor(min_df=2, max_word_features=3000, max_char_features=3000)
    tfidf.fit([t for t, m in zip(texts, tr) if m])
    svd = TruncatedSVD(n_components=args.svd_dim, random_state=args.seed).fit(tfidf.transform([t for t, m in zip(texts, tr) if m]))
    feat = F0F1Featurizer(mean, std, tfidf, svd)
    X = feat.transform(texts)

    router = TwoHeadRouter(name="bundle_f0f1", epochs=args.epochs, seed=args.seed)
    router.fit(X[tr], s_S[tr], s_L[tr], group_ids=c["group_id"].to_numpy()[tr])
    router.calibrate(X[ca], s_S[ca], s_L[ca])

    prices = PriceTable.load()
    p = prices[args.large] if args.large in prices.models else prices[prices.reference_frontier]
    mean_out = float(scores[(scores["model"] == args.large) & scores["qid"].isin(c.index[tr])]["mean_n_out"].mean())
    bundle = RouterBundle(featurizer=feat, router=router, tau=0.0, rule=args.rule,
                          frontier_price=(p.input, p.output), mean_out_tokens=mean_out)
    pr = bundle.predict(texts)
    sel = th.select_tau(pr["score"][ca], s_S[ca], s_L[ca], taus=th.threshold_grid(pr["score"][tr]),
                        rho=args.rho, delta=args.delta, method="wsr")
    bundle.tau = sel.tau_star
    test = th.evaluate_at_tau(pr["score"][te], s_S[te], s_L[te], sel.tau_star, B=500)
    bundle.meta = {"pair": {"small": args.small, "large": args.large}, "rule": args.rule, "rho": args.rho,
                   "delta": args.delta, "certified": sel.certified, "cal": sel.cal,
                   "test": {k: test[k] for k in ("alpha", "pgr", "savings")}, "featurizer": feat.name,
                   "interim": True, "sources": list(HEADLINE)}
    bundle.save(args.out)

    # T41 acceptance: the saved bundle reproduces the evaluation code's decisions on 1,000 test queries
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


if __name__ == "__main__":
    main()
