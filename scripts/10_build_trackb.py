#!/usr/bin/env python3
"""Build Track B: Qwen's scores beside RouterBench's GPT-4 scores, same questions, same splits (T35).

    python scripts/10_build_trackb.py
    python scripts/10_build_trackb.py --responses artifacts/trackb/qwen_responses.parquet \\
                                      --mcq artifacts/trackb/qwen_mcq_scores.parquet

Inputs from lane D's runs on the M2 (either or both; partial runs are fine):
  * --responses  RESPONSES contract, GSM8K free-text answers (k samples per question)
  * --mcq        SCORES contract, MMLU / ARC / Winogrande scores from the mcq_logprob grader
Writes artifacts/trackb/{corpus,scores}.parquet and manifest.json, and prints coverage and the
headroom kill-check.
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import schema  # noqa: E402
from router.trackb import TRACKB_VERSION, build_trackb  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--routerbench", type=Path, default=Path("artifacts/routerbench"))
    ap.add_argument("--responses", type=Path, default=Path("artifacts/trackb/qwen_responses.parquet"))
    ap.add_argument("--mcq", type=Path, default=Path("artifacts/trackb/qwen_mcq_scores.parquet"))
    ap.add_argument("--small", default="qwen3.5-4b")
    ap.add_argument("--large", default="gpt-4-1106-preview")
    ap.add_argument("--out", type=Path, default=Path("artifacts/trackb"))
    args = ap.parse_args()

    corpus = schema.read(args.routerbench / "corpus.parquet", schema.CORPUS)
    rb = schema.read(args.routerbench / "scores.parquet", schema.SCORES)
    responses = schema.read(args.responses, schema.RESPONSES) if args.responses.exists() else None
    mcq = schema.read(args.mcq, schema.SCORES) if args.mcq.exists() else None
    if responses is None and mcq is None:
        sys.exit(f"No Qwen results yet: expected {args.responses} and/or {args.mcq} (lane D, T33/T34).")

    tb = build_trackb(corpus, rb, small=args.small, large=args.large, responses=responses, mcq_scores=mcq)
    schema.write(tb.corpus, schema.CORPUS, args.out / "corpus.parquet")
    schema.write(tb.scores, schema.SCORES, args.out / "scores.parquet")
    rb_manifest = args.routerbench / "manifest.json"
    manifest = {
        "artifact": "trackb", "builder": TRACKB_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "inputs": {"routerbench_corpus_hash": json.loads(rb_manifest.read_text())["content_hash"]["corpus"]
                   if rb_manifest.exists() else None,
                   "responses": str(args.responses) if responses is not None else None,
                   "mcq": str(args.mcq) if mcq is not None else None},
        "report": tb.report, "warnings": tb.warnings,
        "content_hash": {"corpus": schema.content_hash(tb.corpus), "scores": schema.content_hash(tb.scores)},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Wrote {args.out}/  questions={len(tb.corpus):,}  hash={manifest['content_hash']['corpus']}")
    print(f"\n{'dataset':<15}{'aligned':>9}{'of':>8}{'Q_S':>8}{'Q_L':>8}{'gap':>8}")
    for s, cov in tb.report["coverage"].items():
        h = tb.report["headroom"].get(s)
        nums = f"{h['Q_S']:>8.3f}{h['Q_L']:>8.3f}{h['gap']:>8.3f}" if h else f"{'—':>8}{'—':>8}{'—':>8}"
        print(f"{s:<15}{cov['aligned']:>9,}{cov['eligible']:>8,}{nums}")
    for w in tb.warnings:
        print("WARNING:", w)


if __name__ == "__main__":
    main()
