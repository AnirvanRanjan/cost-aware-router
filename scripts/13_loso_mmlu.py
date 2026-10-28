#!/usr/bin/env python3
"""T26: leave-one-subject-out (LOSO) evaluation on MMLU:
1. Within-subject evaluation: scores existing router predictions per subject.
2. True LOSO evaluation: groups subjects into ~10 folds, retrains LogisticRegression
   on F2 with each subject fold's train rows removed, and scores that fold's test rows.

    python scripts/13_loso_mmlu.py --corpus artifacts/routerbench/corpus.parquet \\
                                   --scores artifacts/routerbench/scores.parquet \\
                                   --predictions artifacts/predictions/all_interim.parquet \\
                                   --features artifacts/features/features.parquet

Writes notes/loso_report.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import schema  # noqa: E402
from router.costs import pair_costs, PriceTable  # noqa: E402
from router.eval import within_subject_eval, run_real_loso  # noqa: E402

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

    f = preds[preds["qid"].isin(corpus.index) & preds["qid"].isin(wide.index) & preds["qid"].isin(m1.index)].copy()
    if f.empty:
        return pd.DataFrame(columns=["qid", "router", "split", "source", "subject", "s_S", "s_L", "c_L", "v1", "v2"])

    f = f.join(corpus[["split", "source", "subject"]], on="qid")
    f["s_S"] = wide.loc[f["qid"], small].to_numpy()
    f["s_L"] = wide.loc[f["qid"], large].to_numpy()
    f["c_L"] = m1.loc[f["qid"], "c_L"].to_numpy()
    f["v1"] = f["delta_hat"]
    f["v2"] = f["delta_hat"] / f["c_hat_L"].clip(lower=1e-9)
    return f


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_corpus = Path("artifacts/routerbench/corpus.parquet") if Path("artifacts/routerbench/corpus.parquet").exists() else Path("artifacts/synthetic/corpus.parquet")
    default_scores = Path("artifacts/routerbench/scores.parquet") if Path("artifacts/routerbench/scores.parquet").exists() else Path("artifacts/synthetic/scores.parquet")
    default_small = "mistralai/mixtral-8x7b-chat" if Path("artifacts/routerbench/corpus.parquet").exists() else "qwen3.5-4b"
    default_large = "gpt-4-1106-preview" if Path("artifacts/routerbench/corpus.parquet").exists() else "claude-sonnet-4.6"
    ap.add_argument("--corpus", type=Path, default=default_corpus)
    ap.add_argument("--scores", type=Path, default=default_scores)
    ap.add_argument("--predictions", type=Path, default=Path("artifacts/predictions/all_interim.parquet"))
    ap.add_argument("--features", type=Path, default=None)
    ap.add_argument("--small", type=str, default=default_small)
    ap.add_argument("--large", type=str, default=default_large)
    ap.add_argument("--source", type=str, default="mmlu")
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-md", type=Path, default=Path("notes/loso_report.md"))
    args = ap.parse_args()

    for p, what in ((args.corpus, "corpus"), (args.scores, "scores")):
        if not p.exists():
            print(f"Error: missing {what} {p}", file=sys.stderr)
            sys.exit(1)

    has_preds = args.predictions.exists()

    corpus = schema.read(args.corpus, schema.CORPUS)
    scores = schema.read(args.scores, schema.SCORES)
    wide = scores.pivot(index="qid", columns="model", values="score")
    m1 = pair_costs(PriceTable.load(), scores, small=args.small, large=args.large)

    lines = [
        f"# Subject Evaluation & Leave-One-Subject-Out on {args.source} (T26)",
        "",
        f"*Generated {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M UTC} by `scripts/13_loso_mmlu.py`.*",
        "",
        f"Pair: S = `{args.small}`, L = `{args.large}`.",
        "",
    ]

    # --- Part 1: Within-subject evaluation on existing predictions ---
    if has_preds:
        df = load_frame(args.predictions, args.corpus, args.scores, small=args.small, large=args.large)
        mmlu_preds = df[(df["source"] == args.source) & df["subject"].notna() & (df["split"] == "test")]
        if not mmlu_preds.empty:
            lines += [
                "## 1. Within-Subject Evaluation (Existing Router Predictions)",
                "",
                "Scores existing test-split predictions per subject. Note: These routers were trained on all "
                "subjects, so this measures within-subject ranking performance rather than generalization to "
                "unseen subjects.",
                "",
            ]
            for router in sorted(mmlu_preds["router"].unique()):
                r = mmlu_preds[mmlu_preds["router"] == router]
                for rule in RULES:
                    summary = within_subject_eval(
                        f"{router} ({rule})",
                        r["subject"].to_numpy(),
                        r[rule].to_numpy(),
                        r["s_S"].to_numpy(),
                        r["s_L"].to_numpy(),
                    )
                    pooled = summary.pooled
                    g = summary.gaps
                    lines += [
                        f"### {router} ({rule})",
                        "",
                        f"| {args.source} subject | n | AUC | CPT(95%) | Savings@PGR .95 | APGR |",
                        "|---|--:|:--:|:--:|:--:|:--:|",
                    ]
                    for _, row in summary.subject_table.iterrows():
                        lines.append(
                            f"| {row['subject']} | {int(row['n']):,} | {row['auc']:.4f} | "
                            f"{row['cpt_95']:.4f} | {row['savings_95']:.1%} | {row['apgr']:.4f} |"
                        )
                    lines += [
                        f"| **Subject mean** | {int(summary.subject_table['n'].sum()):,} | — | "
                        f"{summary.subject_table['cpt_95'].mean():.4f} | "
                        f"{summary.subject_table['savings_95'].mean():.1%} | "
                        f"{summary.subject_table['apgr'].mean():.4f} |",
                        f"| **Grouped (pooled)** | {int(len(r)):,} | — | {pooled.cpt_95:.4f} | "
                        f"{pooled.savings_95:.1%} | {pooled.apgr:.4f} |",
                        "",
                        f"**Gaps (pooled - subject mean):** CPT {g['cpt_gap']:+.4f}, savings "
                        f"{g['savings_gap']:+.1%}, APGR {g['apgr_gap']:+.4f}. Worst subject: "
                        f"**{g['worst_subject']}** (savings {g['worst_savings']:.1%}).",
                        "",
                    ]

    # --- Part 2: True Leave-One-Subject-Out (LOSO) with Retrained LogReg on F2 ---
    feat_path = args.features
    if feat_path is None:
        default_feat = Path("artifacts/features/features.parquet")
        if default_feat.exists():
            feat_path = default_feat

    c_source = corpus[(corpus["source"] == args.source) & corpus["subject"].notna()].copy()
    c_source = c_source[c_source["qid"].isin(wide.index)]

    if not c_source.empty:
        c_tr = c_source[c_source["split"] == "train"]
        c_te = c_source[c_source["split"] == "test"]

        if not c_tr.empty and not c_te.empty:
            qids_tr = c_tr["qid"].tolist()
            qids_te = c_te["qid"].tolist()

            if feat_path and feat_path.exists():
                feat_df = schema.read(feat_path, schema.FEATURES)
                feat_map = dict(zip(feat_df["qid"], feat_df["vector"]))
                X_tr = np.array([feat_map[q] for q in qids_tr], dtype=np.float32)
                X_te = np.array([feat_map[q] for q in qids_te], dtype=np.float32)
            else:
                # Synthetic or lightweight representation fallback: word/char ngram TF-IDF
                from router.features import TfidfFeatureExtractor
                tfidf = TfidfFeatureExtractor(min_df=1, max_word_features=500, max_char_features=500)
                X_tr = tfidf.fit_transform(c_tr["prompt"].tolist()).toarray().astype(np.float32)
                X_te = tfidf.transform(c_te["prompt"].tolist()).toarray().astype(np.float32)

            y_tr_S = wide.loc[qids_tr, args.small].to_numpy(dtype=float)
            y_tr_L = wide.loc[qids_tr, args.large].to_numpy(dtype=float)
            y_te_S = wide.loc[qids_te, args.small].to_numpy(dtype=float)
            y_te_L = wide.loc[qids_te, args.large].to_numpy(dtype=float)

            costs_te_L = m1.loc[qids_te, "c_L"].to_numpy(dtype=float) if "c_L" in m1.columns else None

            loso_summary = run_real_loso(
                X_train=X_tr,
                y_train_S=y_tr_S,
                y_train_L=y_tr_L,
                subjects_train=c_tr["subject"].tolist(),
                X_test=X_te,
                y_test_S=y_te_S,
                y_test_L=y_te_L,
                subjects_test=c_te["subject"].tolist(),
                costs_test_L=costs_te_L,
                n_folds=args.n_folds,
                seed=args.seed,
            )

            p_ret = loso_summary.pooled_retrained
            l_mean = loso_summary.loso_mean
            l_oof = loso_summary.loso_test_metrics
            p_fold_mean = loso_summary.pooled_fold_mean
            gaps = loso_summary.gaps

            lines += [
                "## 2. True Leave-One-Subject-Out (LOSO) Evaluation (Retrained on F2 / LogReg)",
                "",
                f"Subjects grouped into {len(loso_summary.fold_table)} held-out folds. For each fold, "
                "the router is retrained with that fold's train rows removed, then evaluated on the fold's test rows.",
                "",
                "### Fold-by-Fold Results",
                "",
                "| Fold | Subjects Sample | Subjects Count | Test n | LOSO AUC | Pooled Fold AUC | LOSO Savings@PGR .95 | Pooled Fold Savings |",
                "|---|---|--:|--:|:--:|:--:|:--:|:--:|",
            ]
            for _, r_fold in loso_summary.fold_table.iterrows():
                lines.append(
                    f"| {int(r_fold['fold'])} | {r_fold['subjects']} | {int(r_fold['n_subjects'])} | "
                    f"{int(r_fold['n_test']):,} | {r_fold['auc']:.4f} | {r_fold.get('pooled_fold_auc', np.nan):.4f} | "
                    f"{r_fold['savings_95']:.1%} | {r_fold.get('pooled_fold_savings_95', np.nan):.1%} |"
                )
            lines += [
                "",
                "### Like-With-Like Comparisons",
                "",
                "To compare like with like, we examine both: (1) averaging metrics across the exact same folds, "
                "and (2) scoring predictions across all test rows together.",
                "",
                "| Evaluation Scheme | Comparison Basis | Test n | AUC | CPT(95%) | Savings@PGR .95 | APGR |",
                "|---|---|--:|:--:|:--:|:--:|:--:|",
            ]
            if p_fold_mean is not None:
                lines.append(
                    f"| **Pooled Baseline (Fold Mean)** | Same folds averaged | {int(loso_summary.fold_table['n_test'].sum()):,} | "
                    f"{p_fold_mean['auc']:.4f} | {p_fold_mean['cpt_95']:.4f} | {p_fold_mean['savings_95']:.1%} | {p_fold_mean['apgr']:.4f} |"
                )
            lines.append(
                f"| **LOSO Retrained (Fold Mean)** | Same folds averaged | {int(loso_summary.fold_table['n_test'].sum()):,} | "
                f"{l_mean['auc']:.4f} | {l_mean['cpt_95']:.4f} | {l_mean['savings_95']:.1%} | {l_mean['apgr']:.4f} |"
            )
            lines.append(
                f"| **Pooled Baseline (All Test Rows)** | All test rows scored | {len(qids_te):,} | "
                f"{p_ret.auc:.4f} | {p_ret.cpt_95:.4f} | {p_ret.savings_95:.1%} | {p_ret.apgr:.4f} |"
            )
            if l_oof is not None:
                lines.append(
                    f"| **LOSO Out-of-Fold (All Test Rows)** | All test rows scored | {len(qids_te):,} | "
                    f"{l_oof.auc:.4f} | {l_oof.cpt_95:.4f} | {l_oof.savings_95:.1%} | {l_oof.apgr:.4f} |"
                )
            lines += [
                "",
                f"**Like-with-like Generalization Gaps:**",
                f"- **Fold-level average gap (Pooled Fold Mean - LOSO Fold Mean):** AUC {gaps.get('fold_auc_gap', 0.0):+.4f}, "
                f"Savings {gaps.get('fold_savings_gap', 0.0):+.1%}, CPT {gaps.get('fold_cpt_gap', 0.0):+.4f}.",
                f"- **All test rows gap (Pooled Retrained - LOSO Out-of-Fold):** AUC {gaps.get('oof_auc_gap', 0.0):+.4f}, "
                f"Savings {gaps.get('oof_savings_gap', 0.0):+.1%}, CPT {gaps.get('oof_cpt_gap', 0.0):+.4f}.",
                f"- Worst fold: **Fold {gaps['worst_fold']}** ({gaps['worst_fold_subjects']}, savings {gaps['worst_savings']:.1%}).",
                "",
            ]

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
