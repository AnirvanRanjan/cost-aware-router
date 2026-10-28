#!/usr/bin/env python3
"""Track B, multiple choice: score Qwen on MMLU, ARC and Winogrande by option probability (Mac only).

    python scripts/14_score_mcq.py                              # all three datasets, all splits
    python scripts/14_score_mcq.py --sources mmlu --limit 50    # quick check first

One forward pass per question with lane B's scorer (router/mcq_mlx.py, thinking mode off): read the
probability of each option letter, keep P(gold) as the score (graders.mcq_score). Only questions
with a gold letter are scored (T07; the rest have no known answer).

Resumable: every result is appended to artifacts/trackb/mcq_progress.jsonl as it is computed;
re-running skips questions already there. When finished (or any time) it writes
  artifacts/trackb/qwen_mcq_scores.parquet    SCORES contract, for scripts/10_build_trackb.py --mcq
  artifacts/trackb/qwen_mcq_options.parquet   every option's log-probability, the chosen letter and
                                              its probability (Qwen's confidence, for D-12)
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

from router import graders, schema  # noqa: E402

MCQ_SOURCES = ("mmlu", "arc-challenge", "winogrande")


def load_done(path: Path) -> dict[str, dict]:
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    done[r["qid"]] = r
                except (ValueError, KeyError):
                    continue                     # a half-written last line from a crash
    return done


def score_corpus(corpus: pd.DataFrame, scorer, progress_path: Path, *, every: int = 25,
                 log=print) -> dict[str, dict]:
    """Score every row not yet in `progress_path`; append each result as it is computed."""
    done = load_done(progress_path)
    todo = corpus[~corpus["qid"].isin(done.keys())]
    log(f"{len(done):,} already scored, {len(todo):,} to go")
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if progress_path.exists() and progress_path.stat().st_size and not progress_path.read_bytes().endswith(b"\n"):
        with progress_path.open("a") as fh:          # close off a line half-written before a crash
            fh.write("\n")
    with progress_path.open("a") as fh:
        for i, (qid, prompt, gold) in enumerate(zip(todo["qid"], todo["prompt"], todo["reference"]), 1):
            labels = graders.parse_options(prompt)
            if not labels:
                rec = {"qid": qid, "error": "no options found"}
            else:
                lps = scorer.option_logprobs(prompt, labels)
                chosen = max(lps, key=lps.get)
                probs = np.exp(np.array(list(lps.values())) - max(lps.values()))
                probs = probs / probs.sum()
                rec = {"qid": qid, "logprobs": lps, "chosen": chosen, "p_chosen": float(probs.max()),
                       "score": graders.mcq_score(lps, gold), "correct": float(chosen == str(gold).strip().upper()),
                       "n_in": len(str(prompt)) // 4}
            fh.write(json.dumps(rec) + "\n")
            done[qid] = rec
            if i % every == 0 or i == len(todo):
                fh.flush()
                rate = i / max(time.time() - t0, 1e-9)
                log(f"  {i:,}/{len(todo):,}  ({rate:.1f} q/s, about {(len(todo) - i) / max(rate, 1e-9) / 60:.0f} min left)")
    return done


def write_outputs(done: dict[str, dict], model: str, out_dir: Path) -> tuple[Path, Path]:
    ok = [r for r in done.values() if "score" in r]
    scores = pd.DataFrame({"qid": [r["qid"] for r in ok], "model": model,
                           "score": [r["score"] for r in ok], "grader": "mcq_logprob",
                           "grader_version": graders.VERSION,
                           "mean_n_in": [float(r["n_in"]) for r in ok], "mean_n_out": 1.0})
    options = pd.DataFrame({"qid": [r["qid"] for r in ok], "model": model,
                            "chosen": [r["chosen"] for r in ok], "p_chosen": [r["p_chosen"] for r in ok],
                            "correct": [r["correct"] for r in ok],
                            "logprobs": [json.dumps(r["logprobs"]) for r in ok]})
    s_path = schema.write(scores, schema.SCORES, out_dir / "qwen_mcq_scores.parquet")
    o_path = out_dir / "qwen_mcq_options.parquet"
    options.to_parquet(o_path, index=False)
    return s_path, o_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("artifacts/routerbench/corpus.parquet"))
    ap.add_argument("--model-path", default="mlx-community/Qwen3.5-4B-MLX-4bit")
    ap.add_argument("--model-name", default="qwen3.5-4b", help="name written to the scores file")
    ap.add_argument("--sources", nargs="*", default=list(MCQ_SOURCES))
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/trackb"))
    args = ap.parse_args()

    corpus = schema.read(args.corpus, schema.CORPUS)
    corpus = corpus[corpus["source"].isin(args.sources) & corpus["reference"].notna()]
    if args.splits:
        corpus = corpus[corpus["split"].isin(args.splits)]
    corpus = corpus.sort_values("qid")
    if args.limit:
        corpus = corpus.head(args.limit)
    print(f"{len(corpus):,} questions with a gold letter in {args.sources}")

    from router.mcq_mlx import MLXOptionScorer
    scorer = MLXOptionScorer.from_pretrained(args.model_path)
    done = score_corpus(corpus, scorer, args.out_dir / "mcq_progress.jsonl")
    s_path, o_path = write_outputs(done, args.model_name, args.out_dir)
    ok = [r for r in done.values() if "score" in r]
    if ok:
        acc = np.mean([r["correct"] for r in ok])
        top = pd.Series([r["chosen"] for r in ok]).value_counts(normalize=True)
        print(f"accuracy {acc:.3f} on {len(ok):,}; most-chosen letter {top.index[0]} at {top.iloc[0]:.0%} "
              f"(above ~40% suggests a formatting bias)")
    print(f"wrote {s_path} and {o_path}")


if __name__ == "__main__":
    main()
