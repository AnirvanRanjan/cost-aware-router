#!/usr/bin/env python3
"""T32 validation smoke test (briefs/B.md):
Verify option scorer on 20 real MMLU questions on Apple Silicon / Mac.

Criterion:
1. Accuracy is clearly above chance (> 0.25).
2. No single letter is chosen more than ~40% of the time (guards against collapse or <think> failure).

Usage on Mac:
    python scripts/check_mmlu_smoke.py --model-path mlx-community/Qwen2.5-3B-Instruct-4bit
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.graders import parse_options
from router.mcq_mlx import MLXOptionScorer


def run_mmlu_smoke_test(
    scorer: MLXOptionScorer,
    corpus_path: Path,
    n_questions: int = 20,
    seed: int = 42,
) -> dict:
    df = pd.read_parquet(corpus_path)
    mmlu = df[df["source"].str.startswith("mmlu") & df["reference"].notna()].copy()
    sample = mmlu.sample(n=min(n_questions, len(mmlu)), random_state=seed)

    preds: list[str] = []
    golds: list[str] = []
    correct: int = 0

    print(f"\nEvaluating {len(sample)} real MMLU questions...")
    print("-" * 75)
    print(f"{'QID':<22} {'Gold':<6} {'Pred':<6} {'P(Gold)':<10} {'Status':<8} {'Options'}")
    print("-" * 75)

    for _, row in sample.iterrows():
        qid = str(row["qid"])
        prompt = str(row["prompt"])
        gold = str(row["reference"]).strip().upper()
        options = parse_options(prompt)
        if not options:
            options = ["A", "B", "C", "D"]

        lps = scorer.option_logprobs(prompt, options, enable_thinking=False)
        if not lps:
            pred = "A"
            p_gold = 0.0
        else:
            pred = max(lps, key=lps.get)
            # Softmax probability for gold
            import math
            max_lp = max(lps.values())
            denom = sum(math.exp(v - max_lp) for v in lps.values())
            p_gold = math.exp(lps.get(gold, -100.0) - max_lp) / denom if denom > 0 else 0.0

        is_corr = (pred == gold)
        if is_corr:
            correct += 1

        preds.append(pred)
        golds.append(gold)
        print(f"{qid:<22} {gold:<6} {pred:<6} {p_gold:<10.3f} {'CORRECT' if is_corr else 'WRONG':<8} {options}")

    accuracy = correct / len(sample)
    letter_counts = collections.Counter(preds)
    max_freq = max(letter_counts.values()) / len(sample)

    print("-" * 75)
    print(f"Total Questions : {len(sample)}")
    print(f"Accuracy        : {accuracy:.1%} (target > 25.0%)")
    print(f"Letter Counts   : {dict(letter_counts)}")
    print(f"Max Letter Freq : {max_freq:.1%} (target <= 40.0%)")

    pass_acc = accuracy > 0.25
    pass_dist = max_freq <= 0.40
    all_passed = pass_acc and pass_dist

    print("\nResult:")
    print(f"  [{'PASS' if pass_acc else 'FAIL'}] Accuracy > 25%: {accuracy:.1%}")
    print(f"  [{'PASS' if pass_dist else 'FAIL'}] No letter > 40%: {max_freq:.1%}")
    print(f"  Overall: {'PASSED (T32 verification complete)' if all_passed else 'FAILED'}\n")

    return {
        "accuracy": accuracy,
        "letter_counts": dict(letter_counts),
        "max_freq": max_freq,
        "passed": all_passed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", type=str, default="mlx-community/Qwen2.5-3B-Instruct-4bit")
    ap.add_argument("--corpus", type=Path, default=Path("artifacts/routerbench/corpus.parquet"))
    ap.add_argument("-n", "--n-questions", type=int, default=20)
    args = ap.parse_args()

    if not args.corpus.exists():
        sys.exit(f"Corpus not found at {args.corpus}. Run verify_data.py first.")

    print(f"Loading model {args.model_path} via MLX...")
    scorer = MLXOptionScorer.from_pretrained(args.model_path)
    res = run_mmlu_smoke_test(scorer, args.corpus, n_questions=args.n_questions)
    if not res["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
