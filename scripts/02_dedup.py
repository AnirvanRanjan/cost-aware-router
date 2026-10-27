#!/usr/bin/env python3
"""Group near-duplicate questions and re-split, so no near-copy crosses train / cal / test (T04).

    python scripts/02_dedup.py                                   # lexical pass (MinHash)
    python scripts/02_dedup.py --embeddings artifacts/features/features.parquet
                                                                 # + semantic pass, once F2 exists (T12)

Rewrites artifacts/routerbench/corpus.parquet (new group_id and split), writes dedup_pairs.parquet
and notes/dedup_report.md, and records the step in manifest.json. Scores are untouched.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import schema  # noqa: E402
from router.dedup import DEDUP_VERSION, dedup_corpus  # noqa: E402


def report_md(res, corpus: pd.DataFrame, examples: int = 4) -> str:
    r = res.report
    lines = ["# Dedup report", "",
             f"_`scripts/02_dedup.py`, {DEDUP_VERSION}. Embedding pass: "
             f"{'yes' if r['embedding_pass'] else 'not yet (run again after T12)'}._", "",
             f"- Near-duplicate pairs found: **{len(res.pairs):,}**",
             f"- Groups merged away: **{r['groups_merged_away']:,}**; rows now sharing a group: "
             f"**{r['rows_in_multi_item_groups']:,}**; largest group: **{r['largest_group']}**",
             f"- Rows whose split changed: **{r['rows_whose_split_changed']:,}**",
             f"- Splits: {r['splits']}", "", "## Pairs by dataset", "",
             "| dataset / method | pairs |", "|---|---|"]
    lines += [f"| {k} | {v:,} |" for k, v in sorted(r["pairs_by_source_and_method"].items())] or ["| — | 0 |"]
    text = corpus.set_index("qid")["prompt"]
    lines += ["", "## How to read this", "",
              "Pairs are grouped so both items land in the same split. On RouterBench, almost all MMLU pairs are",
              "different questions on the same reading passage, and almost all Winogrande pairs are the dataset's",
              "deliberate twin sentences (one word changed, opposite answer). Neither is a copy, but a router that",
              "saw one in training would have an unfair head start on the other, so grouping them is correct.",
              "", "## Examples (lowest similarity first: the borderline calls worth checking)", ""]
    for src, grp in res.pairs.groupby("source"):
        lines += [f"### {src}", ""]
        for row in grp.sort_values("similarity").head(examples).itertuples():
            a, b = (text[row.qid_a][:160].replace("\n", " "), text[row.qid_b][:160].replace("\n", " "))
            lines += [f"- **{row.similarity:.2f}** ({row.method}) `{row.qid_a}` / `{row.qid_b}`",
                      f"  - {a}", f"  - {b}"]
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("artifacts/routerbench"))
    ap.add_argument("--embeddings", type=Path, help="features.parquet holding feature_set F2")
    ap.add_argument("--experiment", type=Path, default=Path("configs/experiment.yaml"))
    ap.add_argument("--report", type=Path, default=Path("notes/dedup_report.md"))
    args = ap.parse_args()

    exp = yaml.safe_load(args.experiment.read_text())
    corpus = schema.read(args.dir / "corpus.parquet", schema.CORPUS)
    mpath = args.dir / "manifest.json"
    if mpath.exists() and "dedup" in json.loads(mpath.read_text()):
        print("WARNING: this corpus was already deduplicated, and existing groups are kept, so earlier merges "
              "stick. For a clean pass, run scripts/01_acquire_routerbench.py first.")
    emb = None
    if args.embeddings:
        f = schema.read(args.embeddings, schema.FEATURES)
        f = f[f["feature_set"] == "F2"]
        emb = {q: np.asarray(v, dtype=np.float32) for q, v in zip(f["qid"], f["vector"])}
        print(f"Embeddings: {len(emb):,} F2 vectors")

    res = dedup_corpus(corpus, embeddings=emb, fractions=exp["split"]["fractions"], seed=int(exp["seed"]))
    schema.write(res.corpus, schema.CORPUS, args.dir / "corpus.parquet")
    res.pairs.to_parquet(args.dir / "dedup_pairs.parquet", index=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_md(res, res.corpus))

    mpath = args.dir / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    manifest["dedup"] = res.report
    manifest.setdefault("content_hash", {})["corpus"] = schema.content_hash(res.corpus)
    mpath.write_text(json.dumps(manifest, indent=2))

    r = res.report
    print(f"Pairs: {len(res.pairs):,}  {r['pairs_by_source_and_method']}")
    print(f"Groups merged away: {r['groups_merged_away']:,}  largest group: {r['largest_group']}  "
          f"split changed for {r['rows_whose_split_changed']:,} rows")
    print(f"Splits: {r['splits']}  corpus hash: {manifest['content_hash']['corpus']}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
