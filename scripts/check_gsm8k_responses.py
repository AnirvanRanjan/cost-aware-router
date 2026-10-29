#!/usr/bin/env python3
"""Grade a GSM8K responses file and show a few answers, to check a Qwen run before trusting it.

    python scripts/check_gsm8k_responses.py artifacts/responses/qwen3.5-4b.parquet
    python scripts/check_gsm8k_responses.py artifacts/trackb/qwen_gsm8k_testcal.parquet --show 3

Prints accuracy (numeric exact match, the same grader Track B uses), how often no number could be
extracted, how often the "The answer is N." line is present, answer length, and the end of a few answers.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import graders, schema  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("responses", type=Path)
    ap.add_argument("--corpus", type=Path, default=Path("artifacts/routerbench/corpus.parquet"))
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    r = pd.read_parquet(args.responses)
    c = schema.read(args.corpus, schema.CORPUS).set_index("qid")
    r = r[r["qid"].isin(c.index[c["source"] == "gsm8k"])].copy()
    if r.empty:
        sys.exit("no GSM8K rows in this file")
    gold = c.loc[r["qid"], "reference"].astype(float).to_numpy()
    q = c.loc[r["qid"], "prompt"].to_numpy()
    text = r["text"].fillna("")
    r["correct"] = [graders.gsm8k_exact_match(t, g, qq) for t, g, qq in zip(text, gold, q)]
    r["no_number"] = [graders.extract_number(t, qq) is None for t, qq in zip(text, q)]
    r["marker"] = text.str.contains("answer is", case=False)
    print(f"{len(r):,} answers to {r['qid'].nunique():,} questions, model {r['model'].iloc[0]}")
    print(f"  accuracy            {r['correct'].mean():.3f}")
    print(f"  no number found     {r['no_number'].mean():.3f}")
    print(f"  'The answer is' line {r['marker'].mean():.3f}")
    print(f"  errors (no text)    {r['text'].isna().mean():.3f}")
    print(f"  output tokens       median {r['n_out'].median():.0f}, max {r['n_out'].max():.0f}")
    if "sample_idx" in r and r["sample_idx"].max() > 0:
        per_q = r.groupby("qid")["correct"].agg(["mean", "count"])
        agree = r.groupby("qid")["text"].apply(lambda s: len({graders.extract_number(t) for t in s.fillna("")}) == 1)
        print(f"  samples per question {per_q['count'].median():.0f}; all samples agree on {agree.mean():.3f}")
    for _, row in r.head(args.show).iterrows():
        print(f"\n--- {row['qid']}  gold {c.at[row['qid'], 'reference']}  correct={row['correct']}")
        print("..." + str(row["text"])[-300:])


if __name__ == "__main__":
    main()
