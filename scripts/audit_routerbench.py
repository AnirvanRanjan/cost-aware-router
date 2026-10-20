#!/usr/bin/env python3
"""Answer the five RouterBench audit questions and write notes/routerbench_audit.md.

    python scripts/audit_routerbench.py
    python scripts/audit_routerbench.py --pkl path/to/routerbench_0shot.pkl

Download the file first (about 100 MB; skip routerbench_raw.pkl, which is 1.2 GB):
    hf download withmartian/routerbench routerbench_0shot.pkl \\
        --repo-type dataset --local-dir artifacts/raw/routerbench

Pickle files run code when loaded. Only use the file from the official withmartian/routerbench repo.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.routerbench_audit import run, to_markdown  # noqa: E402

DEFAULT_PKL = Path("artifacts/raw/routerbench/routerbench_0shot.pkl")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", type=Path, default=DEFAULT_PKL)
    ap.add_argument("--out", type=Path, default=Path("notes/routerbench_audit.md"))
    args = ap.parse_args()
    if not args.pkl.exists():
        sys.exit(f"Not found: {args.pkl}\nDownload it first — see the top of this script.")
    df = pd.read_pickle(args.pkl)
    audit = run(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(to_markdown(audit, args.pkl.name))
    print(f"Wrote {args.out}: {audit.shape[0]:,} rows, {len(audit.models)} models.")
    for w in audit.warnings:
        print("WARNING:", w)


if __name__ == "__main__":
    main()
