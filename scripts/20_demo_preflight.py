#!/usr/bin/env python3
"""T48/T49: one command to run on the M2 before recording the video or starting a dry run.

    python scripts/20_demo_preflight.py              # offline demo (what the video and dry runs use)
    python scripts/20_demo_preflight.py --fake       # no MLX: checks the plumbing only

Checks, in order, and stops at the first failure:
  1. scikit-learn is the pinned 1.8.0 (another version fits a different router);
  2. the bundle is the D-15 router: cascade_t61_3sample, rule v2, tau* 8.2187, T41 check 1.0;
  3. MLX's GPU cache is capped (MLX_CACHE_LIMIT_GB, default 2) so memory stays bounded;
  4. every demo-set question answers OFFLINE, through the same code as `serve_demo.py --offline`,
     with no error, and goes where configs/demo_set.yaml expects (local / frontier / any).
Prints a table and PASS / FAIL. Exit code 0 only if everything passes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.bundle import RouterBundle
from router.costs import PriceTable
from router.generate import (
    FakeClient,
    GenerationCache,
    GenParams,
    load_client,
    set_mlx_cache_limit,
)
from router.serve import Demo

PINNED_SKLEARN = "1.8.0"
HEADLINE_TAU = 8.2187


def fail(msg: str) -> None:
    print(f"\nFAIL: {msg}")
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, default=Path("artifacts/bundle/router.pkl"))
    ap.add_argument("--demo-set", type=Path, default=Path("configs/demo_set.yaml"))
    ap.add_argument("--cache", type=Path, default=Path("artifacts/cache/demo.sqlite"))
    ap.add_argument("--small", default="qwen3.5-4b")
    ap.add_argument("--large", default="gemini-3-flash")
    ap.add_argument("--max-tokens", type=int, default=512, help="must match serve_demo.py")
    ap.add_argument("--fake", action="store_true", help="canned local answers, no MLX (plumbing check only)")
    args = ap.parse_args()

    # 1. scikit-learn version
    import sklearn

    print(f"[1] scikit-learn {sklearn.__version__}", end="  ")
    if sklearn.__version__ != PINNED_SKLEARN:
        fail(f"scikit-learn {sklearn.__version__}; run `pip install -r requirements.txt` (pins {PINNED_SKLEARN})")
    print("ok")

    # 2. the bundle
    if not args.bundle.exists():
        fail(f"no bundle at {args.bundle}: python scripts/11_build_bundle.py --mode cascade_t61 --samples 3")
    bundle = RouterBundle.load(args.bundle)
    feat, t41 = bundle.meta.get("featurizer"), bundle.meta.get("t41_agreement_1000")
    print(f"[2] bundle {feat}, rule {bundle.rule}, tau* {bundle.tau:.4f}, T41 {t41}", end="  ")
    if feat != "cascade_t61_3sample" or bundle.rule != "v2":
        fail("not the D-15 bundle; rebuild with --mode cascade_t61 --samples 3")
    if abs(bundle.tau - HEADLINE_TAU) > 5e-4:
        fail(f"tau* {bundle.tau:.4f} is not the headline {HEADLINE_TAU}; check scikit-learn and rebuild")
    if t41 != 1.0:
        fail("T41 agreement is not 1.0")
    print("ok")

    # 3. MLX memory cap
    if args.fake:
        print("[3] MLX cache cap: skipped (--fake)")
        small = FakeClient(args.small)
    else:
        small = load_client(args.small)
        small._load()                                   # loads the model and applies the cap
        print(f"[3] MLX cache cap {set_mlx_cache_limit()} GB (MLX_CACHE_LIMIT_GB)  ok")

    # 4. every demo question offline, through the server's own code path
    prices = PriceTable.load()
    ref = prices[prices.reference_frontier]
    cache = args.cache if not args.fake else Path("artifacts/cache/demo_fake.sqlite")
    demo = Demo(bundle, small, FakeClient(args.large), GenerationCache(cache), params=GenParams(max_tokens=args.max_tokens),
                offline=not args.fake, frontier_price=(ref.input, ref.output))
    items = yaml.safe_load(args.demo_set.read_text())["queries"]
    print(f"[4] {len(items)} demo questions, {'fake frontier' if args.fake else 'offline'}, cache {cache}\n")
    print(f"{'id':<9}{'expect':<10}{'got':<10}{'ms':>7}  {'signal':<34}status")
    bad = 0
    for it in items:
        t0 = time.perf_counter()
        r = demo.chat(it["query"])
        ms = (time.perf_counter() - t0) * 1000
        route = r.get("route", {}).get("route", "?")
        signal = f"{r.get('route', {}).get('kind', '')}: {r.get('route', {}).get('signal', '')}"
        expect = it.get("expect", "any")
        problem = ("ERROR " + r["error"][:40]) if "error" in r else ("" if expect in ("any", route) else "WRONG ROUTE")
        if not args.fake:                       # canned answers cannot match the plan; only errors count
            bad += bool(problem)
        else:
            bad += problem.startswith("ERROR")
        print(f"{it['id']:<9}{expect:<10}{route:<10}{ms:>7.0f}  {signal[:33]:<34}{problem or 'ok'}")

    t = demo.totals.as_dict()
    print(f"\nsavings counter after the run: {t['local']} local / {t['frontier']} frontier, "
          f"saved {100 * t['saved_share']:.1f}% vs always-frontier (demo traffic, not a certified figure)")
    if bad:
        fail(f"{bad} demo question(s) errored or went the wrong way. Fix before recording "
             "(swap from configs/demo_candidates.yaml after checking them on this machine)")
    print("\nPASS: ready to record / present.  Start the demo with:  python scripts/serve_demo.py --offline")


if __name__ == "__main__":
    main()
