#!/usr/bin/env python3
"""Choose tau* on cal, report once on test, with guarantees and both cost models (T22-T24).

    python scripts/08_threshold_report.py --predictions artifacts/predictions/baselines.parquet

For every router in the predictions file and both decision rules (v1: delta_hat, v2: delta_hat per
predicted dollar), this:
  1. builds the threshold grid from the TRAIN split's scores (fixed before cal is looked at),
  2. picks tau* on CAL by fixed-sequence Learn-then-Test with each bound (wsr, bootstrap,
     hoeffding) and the naive point estimate,
  3. applies each frozen tau* to TEST once: frontier-call rate, PGR and savings with paired-
     bootstrap 95% CIs, under cost model M1 (list price), against a Batch-API always-frontier
     baseline, and under M2 (local hardware time),
  4. adds the exact test-split CPT(95%), envelope CPT and APGR, with random and oracle references.

Headline sources (D-06): MMLU, GSM8K, ARC, Winogrande. HellaSwag is reported on its own.
Writes notes/threshold_report.md and artifacts/reports/threshold_report.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import schema  # noqa: E402
from router import threshold as th  # noqa: E402
from router.costs import PriceTable, pair_costs  # noqa: E402

HEADLINE = ("mmlu", "gsm8k", "arc-challenge", "winogrande")
METHODS = ("wsr", "bootstrap", "hoeffding", "naive")


def load_frame(pred_path, corpus_path, scores_path, small, large, cost_model="M1"):
    """One row per (qid, router): split, source, s_S, s_L, scores for v1/v2, c_S, c_L (M1, batch, M2)."""
    preds = schema.read(pred_path, schema.PREDICTIONS)
    preds = preds.groupby(["qid", "router"], as_index=False)[["s_hat_S", "s_hat_L", "delta_hat", "c_hat_L"]].mean()
    corpus = schema.read(corpus_path, schema.CORPUS).set_index("qid")
    scores = schema.read(scores_path, schema.SCORES)
    wide = scores.pivot(index="qid", columns="model", values="score")
    table = PriceTable.load()
    m1 = pair_costs(table, scores, small=small, large=large)
    batch = pair_costs(table, scores, small=small, large=large, frontier_batch=True)
    m2 = pair_costs(table, scores, small=small, large=large, cost_model="M2")
    f = preds.join(corpus[["split", "source"]], on="qid")
    f["s_S"] = wide.loc[f["qid"], small].to_numpy()
    f["s_L"] = wide.loc[f["qid"], large].to_numpy()
    f["c_L"] = m1.loc[f["qid"], "c_L"].to_numpy()
    f["c_L_batch"] = batch.loc[f["qid"], "c_L"].to_numpy()
    f["c_S_m2"] = m2.loc[f["qid"], "c_S"].to_numpy()
    f["v1"] = f["delta_hat"]
    f["v2"] = f["delta_hat"] / f["c_hat_L"].clip(lower=1e-9)
    return f, table


def report_one(df: pd.DataFrame, rule: str, rho: float, delta: float, B: int, seed: int) -> dict:
    tr, ca, te = (df[df["split"] == s] for s in ("train", "cal", "test"))
    if len(tr):
        grid, grid_from = th.threshold_grid(tr[rule].to_numpy(), n=1000), "train"
    else:   # predictions file has no train rows: fall back to cal's own breakpoints (say so in the report)
        grid, grid_from = th.breakpoints(ca[rule].to_numpy()), "cal (no train predictions; data-dependent grid)"
    out = {"rule": rule, "n_cal": len(ca), "n_test": len(te), "grid_from": grid_from, "selections": {}}
    for method in METHODS:
        sel = th.select_tau(ca[rule].to_numpy(), ca["s_S"].to_numpy(), ca["s_L"].to_numpy(), taus=grid,
                            rho=rho, delta=delta, method=method, B=B, seed=seed)
        test = {}
        for name, cS, cL in (("M1", None, "c_L"), ("M2", "c_S_m2", "c_L")):
            test[name] = th.evaluate_at_tau(te[rule].to_numpy(), te["s_S"].to_numpy(), te["s_L"].to_numpy(),
                                            sel.tau_star, c_S=None if cS is None else te[cS].to_numpy(),
                                            c_L=te[cL].to_numpy(), B=B, seed=seed)
        # vs a Batch-API always-frontier baseline: router pays list price for what it escalates
        to_L = te[rule].to_numpy() > sel.tau_star
        router_cost = np.where(to_L, te["c_L"], 0.0).mean()
        test["savings_vs_batch"] = float(1 - router_cost / te["c_L_batch"].mean())
        out["selections"][method] = {"tau_star": sel.tau_star, "certified": sel.certified,
                                     "cal": sel.cal, "naive_cal": sel.naive, "test": test}
    sw = th.sweep(te[rule].to_numpy(), te["s_S"].to_numpy(), te["s_L"].to_numpy(), c_L=te["c_L"].to_numpy())
    out["test_curve"] = {"cpt95": sw.cpt(rho), "envelope_cpt95": th.envelope_cpt(sw, rho), "apgr": sw.apgr(),
                         "savings_at_cpt95": sw.savings(rho), "q_S": sw.q_S, "q_L": sw.q_L}
    return out


def references(df: pd.DataFrame, rho: float) -> dict:
    te = df[df["split"] == "test"].drop_duplicates("qid")
    ref = th.reference_curves(te["s_S"].to_numpy(), te["s_L"].to_numpy(), c_L=te["c_L"].to_numpy())
    return {k: {"cpt95": v.cpt(rho), "apgr": v.apgr()} for k, v in ref.items()}


def per_source(df: pd.DataFrame, rule: str, tau: float) -> list[dict]:
    rows = []
    te = df[df["split"] == "test"]
    for src, g in te.groupby("source"):
        to_L = g[rule].to_numpy() > tau
        s_r = np.where(to_L, g["s_L"], g["s_S"])
        qs, ql = g["s_S"].mean(), g["s_L"].mean()
        rows.append({"source": src, "n": len(g), "q_S": qs, "q_L": ql, "alpha": to_L.mean(),
                     "pgr": (s_r.mean() - qs) / (ql - qs) if ql - qs > 1e-9 else float("nan"),
                     "savings_m1": 1 - np.where(to_L, g["c_L"], 0).mean() / g["c_L"].mean()})
    return rows


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def ci(c):
    return f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"


def to_markdown(rep: dict) -> str:
    L = []
    L.append("# Threshold selection and guarantees (T22-T24)\n")
    L.append(f"*Generated {rep['generated']} by `scripts/08_threshold_report.py` from `{rep['predictions']}`.*\n")
    L.append(f"Pair: S = `{rep['small']}`, L = `{rep['large']}`. Target PGR ≥ {rep['rho']}, δ = {rep['delta']}. "
             f"tau* chosen on **cal** only; every test number below is one application of a frozen tau*.\n")
    if rep.get("interim"):
        L.append("> **INTERIM.** F0/F1 routers only (no embeddings yet) and provisional splits (the embedding "
                 "dedup pass has not run). The pipeline is final; the numbers are not. Re-run after T04 + F2.\n")
    L.append(f"References on the headline test set: random CPT(95%) = {pct(rep['refs']['random']['cpt95'])}, "
             f"APGR {rep['refs']['random']['apgr']:.3f}; oracle CPT(95%) = {pct(rep['refs']['oracle']['cpt95'])}, "
             f"APGR {rep['refs']['oracle']['apgr']:.3f}.\n")
    L.append("## Headline: tau* from cal, applied once to test (MMLU, GSM8K, ARC, Winogrande)\n")
    L.append("| Router | Rule | Bound | Certified | α cal | α test | PGR test [95% CI] | Savings M1 [CI] | vs Batch | Savings M2 |")
    L.append("|---|---|---|:-:|--:|--:|--:|--:|--:|--:|")
    for r in rep["routers"]:
        for rule in r["rules"]:
            for m, s in rule["selections"].items():
                t1 = s["test"]["M1"]
                flag = "" if rule.get("grid_from", "train") == "train" else " †"
                L.append(f"| {r['router']}{flag} | {rule['rule']} | {m} | {'yes' if s['certified'] else 'no'} | "
                         f"{pct(s['cal']['alpha'])} | {pct(t1['alpha'])} | {pct(t1['pgr'])} {ci(t1['ci']['pgr'])} | "
                         f"{pct(t1['savings'])} {ci(t1['ci']['savings'])} | {pct(s['test']['savings_vs_batch'])} | "
                         f"{pct(s['test']['M2']['savings'])} |")
    if any(rr.get("grid_from", "train") != "train" for r in rep["routers"] for rr in r["rules"]):
        L.append("\n† No train-split predictions in the file, so the threshold grid came from cal itself. "
                 "Learn-then-Test is then only approximately valid; write train predictions to fix it.")
    L.append("\n## Test-split curves (no tau selection; for comparing routers)\n")
    L.append("| Router | Rule | CPT(95%) | Envelope CPT(95%) | APGR | Savings at CPT |")
    L.append("|---|---|--:|--:|--:|--:|")
    for r in rep["routers"]:
        for rule in r["rules"]:
            c = rule["test_curve"]
            L.append(f"| {r['router']} | {rule['rule']} | {pct(c['cpt95'])} | {pct(c['envelope_cpt95'])} | "
                     f"{c['apgr']:.3f} | {pct(c['savings_at_cpt95'])} |")
    best = rep["best"]
    L.append(f"\n## Per source at the best certified tau* ({best['router']}, {best['rule']}, {best['method']})\n")
    L.append("| Source | n | Q_S | Q_L | α | PGR | Savings M1 |")
    L.append("|---|--:|--:|--:|--:|--:|--:|")
    for row in best["per_source"]:
        L.append(f"| {row['source']} | {row['n']:,} | {row['q_S']:.3f} | {row['q_L']:.3f} | {pct(row['alpha'])} | "
                 f"{pct(row['pgr'])} | {pct(row['savings_m1'])} |")
    L.append("\nHellaSwag (reported separately, D-06) and the small sets (MBPP, MT-Bench) are not in the "
             "headline pool. A single tau* is chosen on the pooled headline sources, so per-source PGR "
             "varies around the pooled 95%.\n")
    L.append("## How to read this\n")
    L.append("- **Certified = no** means the bound could not certify any threshold that sends less than "
             "everything to L at δ = 0.05: the honest answer is then always-frontier.")
    L.append("- **naive** picks the cheapest threshold whose *point estimate* meets the target on cal. It is "
             "shown to measure the optimism gap; it carries no guarantee.")
    L.append("- **wsr** is finite-sample valid (Waudby-Smith & Ramdas; the bound RCPS uses). **bootstrap** "
             "is DESIGN.md's paired bootstrap, asymptotic only. **hoeffding** is valid and loose.")
    L.append("- **vs Batch**: savings against always-frontier at Batch-API prices (50% off), the fair "
             "baseline DESIGN.md §3.2 asks for. **M2** prices local generation as hardware time "
             "(assumptions in configs/models.yaml).")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, default=Path("artifacts/predictions/baselines.parquet"))
    ap.add_argument("--corpus", type=Path, default=Path("artifacts/routerbench/corpus.parquet"))
    ap.add_argument("--scores", type=Path, default=Path("artifacts/routerbench/scores.parquet"))
    ap.add_argument("--small", default="mistralai/mixtral-8x7b-chat")
    ap.add_argument("--large", default="gpt-4-1106-preview")
    ap.add_argument("--rho", type=float, default=0.95)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--B", type=int, default=1000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-md", type=Path, default=Path("notes/threshold_report.md"))
    ap.add_argument("--out-json", type=Path, default=Path("artifacts/reports/threshold_report.json"))
    args = ap.parse_args()

    f, _ = load_frame(args.predictions, args.corpus, args.scores, args.small, args.large)
    head = f[f["source"].isin(HEADLINE)]
    meta_path = args.predictions.with_suffix(".json")
    rep = {"generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "predictions": str(args.predictions), "small": args.small, "large": args.large,
           "rho": args.rho, "delta": args.delta,
           "interim": json.loads(meta_path.read_text()).get("interim", False) if meta_path.exists() else False,
           "refs": references(head, args.rho), "routers": []}
    best = None
    for router, g in head.groupby("router"):
        print(f"{router} ...", flush=True)
        r = {"router": router, "rules": [report_one(g, rule, args.rho, args.delta, args.B, args.seed)
                                         for rule in ("v1", "v2")]}
        rep["routers"].append(r)
        for rule in r["rules"]:
            s = rule["selections"]["wsr"]
            if s["certified"] and (best is None or s["test"]["M1"]["savings"] > best[0]):
                best = (s["test"]["M1"]["savings"], router, rule["rule"], s["tau_star"])
    if best is None:   # nothing certified: show the naive choice of the first router so the table exists
        r0 = rep["routers"][0]
        best = (None, r0["router"], "v1", r0["rules"][0]["selections"]["naive"]["tau_star"])
        method = "naive"
    else:
        method = "wsr"
    g = f[(f["router"] == best[1]) & ~f["source"].isin(())]
    rep["best"] = {"router": best[1], "rule": best[2], "method": method, "tau_star": best[3],
                   "per_source": per_source(g, best[2], best[3])}

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(rep, indent=2, default=float))
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(to_markdown(rep))
    print(f"wrote {args.out_md} and {args.out_json}")


if __name__ == "__main__":
    main()
