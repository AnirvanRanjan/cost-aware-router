#!/usr/bin/env python3
"""Phase-0 gate: print projected spend and wall-clock BEFORE spending anything (DESIGN.md §4.2).

    python scripts/estimate_spend.py
    python scripts/estimate_spend.py --gsm8k 1000 --k-frontier 1 --batch-speedup 4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.costs import PriceTable, crossover_tokens_per_sec, estimate_run, local_query_cost  # noqa: E402

FRONTIER = [
    ("claude-sonnet-4.6", False), ("claude-sonnet-4.6", True), ("claude-haiku-4.5", True),
    ("gpt-5.6-terra", True), ("gpt-5-nano", True), ("gemini-3-flash", False),
]
LOCAL = ["llama-3.2-3b", "qwen3.5-4b", "phi-4-mini", "gemma-3-4b", "qwen3.5-9b"]
GSM8K_TOKENS = (120, 250)   # mean input, mean output per sample
MMLU_TOKENS = (200, 5)


def _usd(x: float) -> str:
    return f"${x:,.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gsm8k", type=int, default=1500, help="GSM8K items in Track B")
    ap.add_argument("--mmlu", type=int, default=2000, help="MMLU items in Track B")
    ap.add_argument("--k-small", type=int, default=3, help="samples per item from the local model")
    ap.add_argument("--k-frontier", type=int, default=3, help="samples per item from the frontier model")
    ap.add_argument("--batch-speedup", type=float, default=3.5, help="MLX batched-decoding throughput multiplier")
    args = ap.parse_args()

    t = PriceTable.load()
    print(f"Prices as of {t.as_of}.  Track B: GSM8K {args.gsm8k} items, MMLU {args.mmlu} items.\n")

    print(f"FRONTIER ARM (k={args.k_frontier} on GSM8K, k=1 on MMLU): dollars you would pay")
    print(f"  {'model':<20}{'mode':<7}{'GSM8K':>9}{'MMLU':>9}{'total':>9}")
    for model, batch in FRONTIER:
        g = estimate_run(t, model, n_items=args.gsm8k, k=args.k_frontier, mean_in=GSM8K_TOKENS[0],
                         mean_out=GSM8K_TOKENS[1], batch=batch)
        m = estimate_run(t, model, n_items=args.mmlu, k=1, mean_in=MMLU_TOKENS[0], mean_out=MMLU_TOKENS[1], batch=batch)
        note = "   <- free tier: $0 actually charged" if t[model].free_tier else ""
        mode = "batch" if batch else "list"
        print(f"  {model:<20}{mode:<7}{_usd(g.usd):>9}{_usd(m.usd):>9}{_usd(g.usd + m.usd):>9}{note}")

    print(f"\nSMALL ARM (k={args.k_small} on GSM8K): local wall-clock. MMLU is logprob scoring and takes minutes.")
    print(f"  {'model':<20}{'tok/s':>7}  {'source':<9}{'batch 1':>10}{f'batch x{args.batch_speedup:g}':>12}")
    for model in LOCAL:
        one = estimate_run(t, model, n_items=args.gsm8k, k=args.k_small, mean_in=GSM8K_TOKENS[0], mean_out=GSM8K_TOKENS[1])
        fast = estimate_run(t, model, n_items=args.gsm8k, k=args.k_small, mean_in=GSM8K_TOKENS[0],
                            mean_out=GSM8K_TOKENS[1], batch_speedup=args.batch_speedup)
        src = "measured" if t[model].measured else "ESTIMATE"
        print(f"  {model:<20}{t[model].tokens_per_sec:>7g}  {src:<9}{one.hours:>8.1f} h{fast.hours:>10.1f} h")
    n_meas = sum(t[m].measured for m in LOCAL)
    print(f"\n  'measured' = timed on the team's M2 (configs/models.yaml); 'ESTIMATE' = not timed yet "
          f"({n_meas}/{len(LOCAL)} measured).")
    print(f"  The batch x{args.batch_speedup:g} column is an ASSUMED speed-up until lane D measures it (T31).")

    if t.hardware:
        hw_name, hw = next(iter(t.hardware.items()))
        print(f"\nCOST MODEL M2 ({hw_name}, all inputs ASSUMED, see configs/models.yaml): "
              f"${hw.usd_per_hour:.3f} per hour of use")
        n_in, n_out = GSM8K_TOKENS
        print(f"  per GSM8K query ({n_in} in / {n_out} out):")
        for model in LOCAL:
            usd = local_query_cost(n_in, n_out, hardware=hw, tokens_per_sec=t[model].tokens_per_sec)
            print(f"    {model:<20}${usd * 1000:.3f} per 1k queries")
        for api in ("claude-haiku-4.5", "gemini-3-flash", "gpt-5-nano"):
            x = crossover_tokens_per_sec(t, api, hardware=hw, n_in=n_in, n_out=n_out)
            print(f"  local beats {api:<17} on price above {x:,.1f} tok/s")


if __name__ == "__main__":
    main()
