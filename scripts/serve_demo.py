#!/usr/bin/env python3
"""Run the live demo (T42/T43/T46). Open http://127.0.0.1:8000 in Safari or Chrome.

    python scripts/serve_demo.py                         # Qwen on this Mac + Gemini free tier
    python scripts/serve_demo.py --offline               # Wi-Fi off: cached frontier answers only
    python scripts/serve_demo.py --fake                  # no models, no bundle: check the UI anywhere

Needs: pip install -r requirements-ml.txt (fastapi, uvicorn, mlx-lm on the Mac), a router bundle
from scripts/11_build_bundle.py, and GEMINI_API_KEY in .env unless --offline or --fake.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.bundle import ConstantRouter, LengthFeaturizer, RouterBundle  # noqa: E402
from router.costs import PriceTable  # noqa: E402
from router.generate import FakeClient, GenerationCache, GenParams, load_client  # noqa: E402
from router.serve import Demo, create_app  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, default=Path("artifacts/bundle/router.pkl"))
    ap.add_argument("--small", default="qwen3.5-4b")
    ap.add_argument("--large", default="gemini-3-flash")
    ap.add_argument("--price-as", default=None, help="price the frontier like this models.yaml model "
                                                     "(default: the Rev A reference, claude-sonnet-4.6)")
    ap.add_argument("--cache", type=Path, default=Path("artifacts/cache/demo.sqlite"))
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    import uvicorn

    prices = PriceTable.load()
    ref = prices[args.price_as or prices.reference_frontier]
    if args.fake:
        bundle = RouterBundle(LengthFeaturizer(), ConstantRouter(0.7, 0.9), tau=0.1, meta={"interim": True})
        small, large = FakeClient(args.small), FakeClient(args.large)
    else:
        if not args.bundle.exists():
            sys.exit(f"no router bundle at {args.bundle}: run scripts/11_build_bundle.py first (or use --fake)")
        bundle = RouterBundle.load(args.bundle)
        small = load_client(args.small)
        large = FakeClient(args.large) if args.offline else load_client(args.large)
    demo = Demo(bundle, small, large, GenerationCache(args.cache), params=GenParams(max_tokens=args.max_tokens),
                offline=args.offline, frontier_price=(ref.input, ref.output))
    print(f"router tau*={bundle.tau:.3f} rule {bundle.rule} | local {small.name} | frontier {large.name}"
          f"{' (offline)' if args.offline else ''} | priced as {ref.name}")
    uvicorn.run(create_app(demo), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
