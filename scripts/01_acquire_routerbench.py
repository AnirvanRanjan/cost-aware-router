#!/usr/bin/env python3
"""Load RouterBench into the project's contracts: artifacts/routerbench/{corpus,scores}.parquet.

    python scripts/01_acquire_routerbench.py
    python scripts/01_acquire_routerbench.py --keep-unverified-gsm8k   # only to inspect; never report it

Needs two files (paths in configs/datasets.yaml):
  * routerbench_0shot.pkl from the official withmartian/routerbench repo. Pickle files run code
    when loaded, so never load one from anywhere else.
  * the GSM8K train answers parquet from openai/gsm8k, to re-grade GSM8K. Without it, GSM8K is
    left out and the script says so.
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import schema  # noqa: E402
from router.data import LOADER_VERSION, headroom_table, load_gsm8k_gold, load_routerbench  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/datasets.yaml"))
    ap.add_argument("--experiment", type=Path, default=Path("configs/experiment.yaml"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/routerbench"))
    ap.add_argument("--keep-unverified-gsm8k", action="store_true",
                    help="load GSM8K with RouterBench's rubric scores when the answers file is missing")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())["routerbench"]
    exp = yaml.safe_load(args.experiment.read_text())
    pkl, gold_path = Path(cfg["file"]), Path(cfg["gsm8k_train"])
    if not pkl.exists():
        sys.exit(f"Not found: {pkl}\nDownload routerbench_0shot.pkl first (see STEP1.md).")

    gold = None
    if gold_path.exists():
        gold = load_gsm8k_gold(pd.read_parquet(gold_path))
        print(f"GSM8K answers: {len(gold):,} from {gold_path}")
    else:
        print(f"GSM8K answers not found at {gold_path} — the URL is in {args.config}.")

    res = load_routerbench(
        pd.read_pickle(pkl), large=cfg["large"], small=list(cfg["small"]), gsm8k_gold=gold,
        keep_unverified_gsm8k=args.keep_unverified_gsm8k,
        fractions=exp["split"]["fractions"], seed=int(exp["seed"]), source_file=pkl.name,
    )
    out = args.out
    schema.write(res.corpus, schema.CORPUS, out / "corpus.parquet")
    schema.write(res.scores, schema.SCORES, out / "scores.parquet")
    manifest = {
        "artifact": "routerbench",
        "loader": LOADER_VERSION,
        "contract_version": schema.CONTRACT_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "inputs": {"routerbench": str(pkl), "gsm8k_train": str(gold_path) if gold else None},
        "seed": int(exp["seed"]),
        "report": res.report,
        "warnings": res.warnings,
        "content_hash": {"corpus": schema.content_hash(res.corpus), "scores": schema.content_hash(res.scores)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nWrote {out}/  corpus={len(res.corpus):,}  scores={len(res.scores):,}  "
          f"hash={manifest['content_hash']['corpus']}")
    print("Dropped tasks:", sum(res.report["dropped_tasks"].values()), "rows from",
          len(res.report["dropped_tasks"]), "custom tasks")
    print("Splits:", res.report["splits"], "| exact-duplicate groups:", res.report["exact_duplicate_groups"])
    for w in res.warnings:
        print("WARNING:", w)
    table = headroom_table(res.corpus, res.scores, cfg["large"], list(cfg["small"]))
    print("\nMean score per dataset (headroom = large − small; the kill-check needs ≥ 0.05):")
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
