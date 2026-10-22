#!/usr/bin/env python3
"""T45: check every planned demo query against the router bundle BEFORE the demo.

    python scripts/check_demo_set.py                         # artifacts/bundle/router.pkl
    python scripts/check_demo_set.py --bundle other.pkl

Reads configs/demo_set.yaml (each query has `expect: local | frontier | any`), routes every query
with the bundle, and prints a table. Exits non-zero if any planned decision would surprise us on
stage. Re-run after every new bundle (new features, Qwen-trained head, new tau*).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.bundle import RouterBundle  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, default=Path("artifacts/bundle/router.pkl"))
    ap.add_argument("--demo-set", type=Path, default=Path("configs/demo_set.yaml"))
    args = ap.parse_args()

    if not args.bundle.exists():
        sys.exit(f"no bundle at {args.bundle}: run scripts/11_build_bundle.py first")
    bundle = RouterBundle.load(args.bundle)
    items = yaml.safe_load(args.demo_set.read_text())["queries"]
    bad = 0
    print(f"tau* = {bundle.tau:.3f}, rule {bundle.rule}\n")
    print(f"{'id':<10}{'expect':<10}{'got':<10}{'delta_hat':>10}  query")
    for it in items:
        d = bundle.decide(it["query"])
        ok = it.get("expect", "any") in ("any", d.route)
        bad += not ok
        mark = "" if ok else "   <-- MISMATCH"
        print(f"{it['id']:<10}{it.get('expect', 'any'):<10}{d.route:<10}{d.delta_hat:>10.3f}  "
              f"{it['query'][:60].replace(chr(10), ' ')}{mark}")
    print(f"\n{len(items) - bad}/{len(items)} as planned")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
