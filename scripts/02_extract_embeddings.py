#!/usr/bin/env python3
"""T12 deliverable (briefs/B.md): extract BAAI/bge-base-en-v1.5 embeddings for all prompts.

    python scripts/02_extract_embeddings.py --corpus artifacts/routerbench/corpus.parquet --out artifacts/features/features.parquet
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import schema
from router.features import SentenceEmbeddingExtractor, build_f2_features


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("artifacts/routerbench/corpus.parquet"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/features/features.parquet"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--model", type=str, default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    if not args.corpus.exists():
        print(f"Error: corpus file {args.corpus} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading corpus from {args.corpus}...")
    corpus = schema.read(args.corpus, schema.CORPUS)
    print(f"Loaded {len(corpus):,} queries.")

    extractor = SentenceEmbeddingExtractor(
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"Encoding prompts with {args.model} on device={extractor.device}...")
    df_features, report = build_f2_features(corpus, extractor=extractor, output_path=args.out)

    print(f"Saved {len(df_features):,} F2 feature vectors to {args.out}.")
    print(f"Metadata: dim={report['dim']}, truncated (>512 tokens)={report['n_truncated']}")

    manifest = {
        "artifact": "features_f2",
        "model": args.model,
        "contract_version": schema.CONTRACT_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "rows": len(df_features),
        "dim": report["dim"],
        "n_truncated": report["n_truncated"],
        "content_hash": schema.content_hash(df_features),
    }
    manifest_path = args.out.parent / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest to {manifest_path}.")


if __name__ == "__main__":
    main()
