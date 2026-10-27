#!/usr/bin/env python3
"""T26: leave-one-subject-out (LOSO) evaluation on MMLU, grouped vs LOSO gap.

    python scripts/13_loso_mmlu.py --corpus artifacts/routerbench/corpus.parquet \
                                   --scores artifacts/routerbench/scores.parquet \
                                   --predictions artifacts/predictions/all_interim.parquet

For every router and decision rule (v1: delta_hat, v2: delta_hat per predicted dollar) in the
predictions file, scores the MMLU test rows of each subject as its own held-out fold
(`router.eval.loso_subject_eval`) and reports, per subject, AUC / CPT(95%) / savings@PGR .95 /
APGR against the pooled (grouped-split) number. The headline output is the LOSO gap:
        pooled - mean(per-subject),
per metric, plus the worst subject. If subjects degrade the router, that shows up here and not in
the pooled table alone (DESIGN.md §"Report both a random grouped split and a LOSO split").

Writes notes/loso_report.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import schema  # noqa: E402
from router.costs import pair_costs, PriceTable  # noqa: E402
from router.eval import loso_subject_eval  # noqa: E402

RULES = ("v1", "v2")


def load_frame(pred_path, corpus_path, scores_path, small, large):
    """Same frame as 08_threshold_report's loader, plus the corpus `subject` column.

    One row per (qid, router): split, source, subject, s_S, s_L, v1/v2 scores, M1 costs.
    """
    preds = schema.read(pred_path, schema.PREDICTIONS)
    preds = preds.groupby(["qid", "router"], as_index=False)[["s_hat_S", "s_hat_L", "delta_hat", "c_hat_L"]].mean()
    corpus = schema.read(corpus_path, schema.CORPUS).set_index("qid")
    scores = schema.read(scores_path, schema.SCORES)
    wide = scores.pivot(index="qid", columns="model", values="score")
    m1 = pair_costs(PriceTable.load(), scores, small=small, large=large)
    f = preds.join(corpus[["split", "source", "subject"]], on="qid")
    f["s_S"] = wide.loc[f["qid"], small].to_numpy()
    f["s_L"] = wide.loc[f["qid"], large].to_numpy()
    f["c_L"] = m1.loc[f["qid"], "c_L"].to_numpy()
    f["v1"] = f["delta_hat"]
    f["v2"] = f["delta_hat"] / f["c_hat_L"].clip(lower=1e-9)
    return f



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("artifacts/routerbench/corpus.parquet"))
    ap.add_argument("--scores", type=Path, default=Path("artifacts/routerbench/scores.parquet"))
    ap.add_argument("--predictions", type=Path, default=Path("artifacts/predictions/all_interim.parquet"))
    ap.add_argument("--small", type=str, default="mistralai/mixtral-8x7b-chat")
    ap.add_argument("--large", type=str, default="gpt-4-1106-preview")
    ap.add_argument("--source", type=str, default="mmlu")
    ap.add_argument("--out-md", type=Path, default=Path("notes/loso_report.md"))
    args = ap.parse_args()

    for p, what in ((args.corpus, "corpus"), (args.scores, "scores"), (args.predictions, "predictions")):
        if not p.exists():
            print(f"Error: missing {what} {p}", file=sys.stderr)
            sys.exit(1)

    df = load_frame(args.predictions, args.corpus, args.scores, small=args.small, large=args.large)
    mmlu = df[(df["source"] == args.source) & df["subject"].notna()]
    if mmlu.empty:
        print(f"Error: no {args.source} rows in the predictions frame.", file=sys.stderr)
        sys.exit(1)
    n_subjects = mmlu[mmlu["subject"] != "(no subject)"]["subject"].nunique()
    print(f"{args.source}: {len(mmlu):,} rows, {n_subjects} subjects")

    lines = [
        f"# Leave-one-subject-out on {args.source} (T26)",
        "",
        f"*Generated {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M UTC} by `scripts/13_loso_mmlu.py` "
        f"from `{args.predictions}`.*",
        "",
        f"Pair: S = `{args.small}`, L = `{args.large}`. Test rows only. Each subject is scored as its "
        "**own held-out fold**; `LOSO` = mean over folds, `Grouped` = the same rows pooled as one "
        "test set. Positive gaps mean the pooled number flatters the router. **Interim** like "
        "`notes/threshold_report.md`: re-run after the frozen splits + F2.",
        "",
    ]
    for router in sorted(mmlu["router"].unique()):
        r = mmlu[mmlu["router"] == router]
        for rule in RULES:
            summary = loso_subject_eval(f"{router} ({rule})",
                                        r["subject"].to_numpy(), r[rule].to_numpy(),
                                        r["s_S"].to_numpy(), r["s_L"].to_numpy())
            pooled = summary.pooled
            g = summary.gaps
            lines += [
                f"## {router} ({rule})",
                "",
                f"| {args.source} subject | n | AUC | CPT(95%) | Savings@PGR .95 | APGR |",
                "|---|--:|:--:|:--:|:--:|:--:|",
            ]
            for _, row in summary.subject_table.iterrows():
                lines.append(f"| {row['subject']} | {int(row['n']):,} | {row['auc']:.4f} | "
                             f"{row['cpt_95']:.4f} | {row['savings_95']:.1%} | {row['apgr']:.4f} |")
            lines += [
                f"| **LOSO mean** | {int(summary.subject_table['n'].sum()):,} | — | "
                f"{summary.subject_table['cpt_95'].mean():.4f} | "
                f"{summary.subject_table['savings_95'].mean():.1%} | "
                f"{summary.subject_table['apgr'].mean():.4f} |",
                f"| **Grouped (pooled)** | {int(len(r)):,} | — | {pooled.cpt_95:.4f} | "
                f"{pooled.savings_95:.1%} | {pooled.apgr:.4f} |",
                "",
                f"**Gaps (grouped - LOSO):** CPT {g['cpt_gap']:+.4f}, savings "
                f"{g['savings_gap']:+.1%}, APGR {g['apgr_gap']:+.4f}. Worst subject: "
                f"**{g['worst_subject']}** (savings {g['worst_savings']:.1%}).",
                "",
            ]

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
