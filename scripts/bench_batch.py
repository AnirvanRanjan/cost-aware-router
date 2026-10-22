#!/usr/bin/env python3
"""T31: measure MLX batched-decoding speed-up on the M2 (lane D). Mac only.

    python scripts/bench_batch.py --client qwen3.5-4b --batches 1 2 4 8 --n 16

Times the same GSM8K prompts at each batch size (greedy, fixed max tokens, no cache) and prints
output tokens per second and the speed-up over batch 1. Put the best measured speed-up in
configs/models.yaml (and max_batch in configs/clients.yaml) and commit notes/batch_speedup.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import schema  # noqa: E402
from router.generate import GenParams, build_prompt, load_client  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--client", default="qwen3.5-4b")
    ap.add_argument("--corpus", type=Path, default=Path("artifacts/routerbench/corpus.parquet"))
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--n", type=int, default=16, help="prompts per batch size")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", type=Path, default=Path("notes/batch_speedup.json"))
    args = ap.parse_args()

    corpus = schema.read(args.corpus, schema.CORPUS)
    prompts = [build_prompt("gsm8k", p) for p in corpus[corpus["source"] == "gsm8k"].sort_values("qid")["prompt"].head(args.n)]
    params = GenParams(max_tokens=args.max_tokens)
    results = []
    for b in args.batches:
        client = load_client(args.client, max_batch=b)
        client.generate_batch(prompts[:1], params, [0])                  # warm-up: load weights, compile
        t0 = time.perf_counter()
        n_out = 0
        for i in range(0, len(prompts), b):
            chunk = prompts[i:i + b]
            n_out += sum(g.n_out for g in client.generate_batch(chunk, params, [0] * len(chunk)))
        secs = time.perf_counter() - t0
        results.append({"batch": b, "seconds": round(secs, 2), "out_tokens": n_out, "tok_per_s": round(n_out / secs, 1)})
        print(f"batch {b:>2}: {n_out:,} tokens in {secs:.1f} s = {n_out / secs:.1f} tok/s", flush=True)
    base = results[0]["tok_per_s"]
    for r in results:
        r["speedup"] = round(r["tok_per_s"] / base, 2)
    best = max(results, key=lambda r: r["speedup"])
    print(f"best: batch {best['batch']} at {best['speedup']}x")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"client": args.client, "machine": platform.platform(), "max_tokens": args.max_tokens,
                                    "n_prompts": len(prompts), "results": results,
                                    "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
