#!/usr/bin/env python3
"""Week-2 deliverable: write the synthetic corpus and scores that unblock lanes B, C and D.

    python scripts/00_make_synthetic.py                 # 2,000 queries -> artifacts/synthetic/
    python scripts/00_make_synthetic.py --n 5000 --seed 11
"""
import argparse
import dataclasses
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import schema  # noqa: E402
from router.splits import assert_no_leakage  # noqa: E402
from router.synthetic import GENERATOR_VERSION, SyntheticConfig, make_synthetic  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("artifacts/synthetic"))
    args = ap.parse_args()

    data = make_synthetic(SyntheticConfig(n=args.n, seed=args.seed))
    assert_no_leakage(data.corpus)

    out = args.out
    schema.write(data.corpus, schema.CORPUS, out / "corpus.parquet")
    schema.write(data.scores, schema.SCORES, out / "scores.parquet")
    data.truth.to_parquet(out / "truth.parquet", index=False)  # not a contract: tests and oracle only

    manifest = {
        "artifact": "synthetic",
        "warning": "Synthetic data with a planted signal. Never report a number computed on it.",
        "generator": GENERATOR_VERSION,
        "contract_version": schema.CONTRACT_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config": dataclasses.asdict(data.config),
        "rows": {"corpus": len(data.corpus), "scores": len(data.scores)},
        "content_hash": {"corpus": schema.content_hash(data.corpus), "scores": schema.content_hash(data.scores)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    wide = data.scores.pivot(index="qid", columns="model", values="score")
    by_src = data.corpus.set_index("qid")["source"]
    small, large = data.config.small_model, data.config.large_model
    print(f"Wrote {out}/  corpus={len(data.corpus)}  scores={len(data.scores)}  hash={manifest['content_hash']['corpus']}")
    print(f"\n{'source':<8}{'n':>6}{'Q_S':>8}{'Q_L':>8}{'headroom':>10}")
    for src, qids in by_src.groupby(by_src).groups.items():
        q_s, q_l = wide.loc[qids, small].mean(), wide.loc[qids, large].mean()
        print(f"{src:<8}{len(qids):>6}{q_s:>8.3f}{q_l:>8.3f}{q_l - q_s:>10.3f}")
    print("\nsplit sizes:", data.corpus["split"].value_counts().to_dict())
    n_dup_groups = int((data.corpus["group_id"].value_counts() > 1).sum())
    print(f"near-duplicate groups: {n_dup_groups} (all within one split)")


if __name__ == "__main__":
    main()
