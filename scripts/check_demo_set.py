#!/usr/bin/env python3
"""T45: check every planned demo query against the router bundle BEFORE the demo.

    python scripts/check_demo_set.py                         # artifacts/bundle/router.pkl, Qwen on this Mac
    python scripts/check_demo_set.py --bundle other.pkl
    python scripts/check_demo_set.py --fake                  # no MLX: canned answers, checks the plumbing only

Reads configs/demo_set.yaml (each query has `expect: local | frontier | any`), routes every query
exactly as the live server does, and prints a table. Exits non-zero if any planned decision would
surprise us on stage. Re-run after every new bundle (new features, new tau*).

Cascade bundles (D-14, D-15) decide on Qwen's own answer, so Qwen answers each query first, through
the same code path as `serve.py`: option probabilities for multiple choice, k sampled answers for
arithmetic (D-15), and a label on question types the router was not certified on. Answers are cached in the demo's cache
(artifacts/cache/demo.sqlite), so the check also warms the cache for the live demo; the frontier
model is never called.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.bundle import RouterBundle  # noqa: E402
from router.generate import FakeClient, GenerationCache, GenParams, load_client  # noqa: E402
from router.serve import Demo  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, default=Path("artifacts/bundle/router.pkl"))
    ap.add_argument("--demo-set", type=Path, default=Path("configs/demo_set.yaml"))
    ap.add_argument("--small", default="qwen3.5-4b")
    ap.add_argument("--max-tokens", type=int, default=512, help="must match serve_demo.py")
    ap.add_argument("--raw-prompt", action="store_true", help="send the query as typed (no 'answer briefly' template)")
    ap.add_argument("--cache", type=Path, default=None,
                    help="answer cache (default: artifacts/cache/demo.sqlite, or demo_fake.sqlite with --fake)")
    ap.add_argument("--fake", action="store_true", help="canned local answers (no MLX); for testing the script")
    ap.add_argument("--out-md", type=Path, default=None, help="also write the table here (e.g. notes/demo_set_check.md)")
    args = ap.parse_args()

    if not args.bundle.exists():
        sys.exit(f"no bundle at {args.bundle}: run scripts/11_build_bundle.py --mode cascade_t61 first")
    bundle = RouterBundle.load(args.bundle)
    items = yaml.safe_load(args.demo_set.read_text())["queries"]

    small = FakeClient(args.small) if args.fake else (load_client(args.small) if bundle.is_cascade else None)
    cache_path = args.cache or Path("artifacts/cache/demo_fake.sqlite" if args.fake else "artifacts/cache/demo.sqlite")
    params = GenParams(max_tokens=args.max_tokens)
    demo = Demo(bundle, small, FakeClient("frontier-not-called"), GenerationCache(cache_path), params=params,
                prompt_style=None if args.raw_prompt else "demo")

    kind = "cascade (Qwen answers first)" if bundle.is_cascade else "query-only"
    lines = [f"tau* = {bundle.tau:.3f}, rule {bundle.rule}, {kind}, {bundle.meta.get('featurizer', '')}"
             + (" [FAKE answers]" if args.fake else ""), "",
             f"{'id':<10}{'expect':<10}{'got':<10}{'delta_hat':>10}  {'signal':<34}query  ->  local answer"]
    bad = 0
    for it in items:
        q = it["query"]
        if bundle.is_cascade:                      # exactly what the live server does, minus the frontier call
            with demo.lock:
                d, ans, extra = demo._answer_local_first(q)
            d = {**d.as_dict(), **extra}
        else:
            d, ans, extra = demo.route(q), "", {}
        ans = " ".join((ans or "").split())
        sig = (f"{extra.get('kind', '')}: {extra.get('signal', '')}" if extra else "")
        sig += " [uncertified]" if extra.get("certified") is False else ""
        expect = it.get("expect", "any")
        ok = expect in ("any", d["route"])
        bad += not ok
        lines.append(f"{it['id']:<10}{expect:<10}{d['route']:<10}{d['delta_hat']:>10.3f}  {sig[:33]:<34}"
                     f"{q[:40]}" + (f"  ->  {ans[-50:]}" if ans else "") + ("" if ok else "   <-- MISMATCH"))
    lines += ["", f"{len(items) - bad}/{len(items)} as planned"]
    print("\n".join(lines))
    if args.out_md:
        args.out_md.write_text("# Demo set check (T45)\n\n*`python scripts/check_demo_set.py` on "
                               f"`{args.bundle}`.*\n\n```\n" + "\n".join(lines) + "\n```\n")
        print(f"wrote {args.out_md}")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
