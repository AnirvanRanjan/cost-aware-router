#!/usr/bin/env python3
"""Check that your local data is the same version as everyone else's.

    python scripts/verify_data.py                        # artifacts/routerbench
    python scripts/verify_data.py --dir artifacts/synthetic

Recomputes each file's content fingerprint and compares it with manifest.json. Get the data with:
    gh release download data-<fingerprint> -R AnirvanRanjan/cost-aware-router -D artifacts/routerbench
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import schema  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("artifacts/routerbench"))
    args = ap.parse_args()
    mpath = args.dir / "manifest.json"
    if not mpath.exists():
        sys.exit(f"No manifest at {mpath}. Download the data first (see the top of this script).")
    want = json.loads(mpath.read_text()).get("content_hash", {})
    ok = True
    for name, contract in (("corpus", schema.CORPUS), ("scores", schema.SCORES)):
        path = args.dir / f"{name}.parquet"
        if name not in want:
            continue
        if not path.exists():
            print(f"MISSING  {path}")
            ok = False
            continue
        got = schema.content_hash(schema.read(path, contract))
        match = got == want[name]
        ok &= match
        print(f"{'OK      ' if match else 'MISMATCH'} {name:<7} {got}" + ("" if match else f"  (manifest says {want[name]})"))
    if not ok:
        sys.exit("Your data differs from the manifest. Re-download it; don't regenerate it by hand.")
    print("Same data as everyone else.")


if __name__ == "__main__":
    main()
