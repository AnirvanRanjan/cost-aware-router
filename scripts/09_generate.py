#!/usr/bin/env python3
"""Generate answers for corpus questions with any client; cached and resumable (T30, lane D).

    # GSM8K Track B on the M2: Qwen, 3 samples, test + cal first
    python scripts/09_generate.py --client qwen3.5-4b --sources gsm8k --splits cal test --k 3 --temperature 0.7

    # check a key and the model id (T40), then a 5-question smoke test
    python scripts/09_generate.py --list-models gemini-3-flash
    python scripts/09_generate.py --client gemini-3-flash --sources gsm8k --limit 5

    # pipeline dry run without any model
    python scripts/09_generate.py --client fake --limit 50

Writes artifacts/responses/<client>.parquet (RESPONSES contract). Stop it any time: re-running the
same command continues from the cache and makes no repeat calls. --offline refuses to call a model.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import schema  # noqa: E402
from router.generate import GenerationCache, GenParams, RunStats, load_client, run_generation  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--client", default="fake")
    ap.add_argument("--list-models", metavar="CLIENT", help="print the model ids an API client serves, then exit")
    ap.add_argument("--corpus", type=Path, default=Path("artifacts/routerbench/corpus.parquet"))
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=1, help="samples per question (forced to 1 at temperature 0)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", type=Path, default=Path("artifacts/cache/generations.sqlite"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    if args.list_models:
        for m in load_client(args.list_models).list_models():
            print(m)
        return

    corpus = schema.read(args.corpus, schema.CORPUS)
    if args.sources:
        corpus = corpus[corpus["source"].isin(args.sources)]
    if args.splits:
        corpus = corpus[corpus["split"].isin(args.splits)]
    corpus = corpus.sort_values("qid")
    if args.limit:
        corpus = corpus.head(args.limit)

    client = load_client(args.client)
    params = GenParams(max_tokens=args.max_tokens, temperature=args.temperature, seed=args.seed)
    stats = RunStats()
    last = [0.0]

    def progress(done, total):
        if time.time() - last[0] > 5 or done == total:
            last[0] = time.time()
            print(f"  {done:,}/{total:,}", flush=True)

    print(f"{args.client}: {len(corpus):,} questions x k={args.k if args.temperature > 0 else 1}")
    resp = run_generation(corpus, client, GenerationCache(args.cache), params, k=args.k,
                          offline=args.offline, progress=progress, stats=stats)
    out = args.out or Path("artifacts/responses") / f"{args.client.replace('/', '_')}.parquet"
    schema.write(resp, schema.RESPONSES, out)
    print(f"cache hits {stats.cache_hits:,}, model calls {stats.model_calls:,}, errors {stats.errors:,}, "
          f"{stats.seconds:.0f} s -> {out}")
    if stats.errors:
        print(f"  first errors: {stats.errors_sample}  (not cached; re-run to retry)")


if __name__ == "__main__":
    main()
