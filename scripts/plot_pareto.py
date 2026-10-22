#!/usr/bin/env python3
"""Pareto figure, panels A-D (DESIGN.md §3.5; TASKS.md T25).

    python scripts/plot_pareto.py --predictions artifacts/predictions/baselines.parquet \
                                  --router f0f1_twohead --out notes/pareto.png

  A  mean cost ($ per 1k queries, log x) vs PGR: router v1, router v2, random, oracle; PGR = .95
     dashed; the frozen tau* from cal marked, with its savings multiple.
  B  frontier-call rate alpha vs PGR (the CPT plot), with the concave envelope and a paired-
     bootstrap 95% band.
  C  one small panel per headline source, alpha vs PGR, random and oracle on each.
  D  distribution of the router score, split by whether S actually got the question right.

All curves are on the TEST split. tau* comes from scripts/08_threshold_report.py's JSON (chosen on
cal) when present, otherwise it is chosen here on cal with the same WSR fixed-sequence rule.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import threshold as th  # noqa: E402

_spec = importlib.util.spec_from_file_location("report", Path(__file__).with_name("08_threshold_report.py"))
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)

THEMES = {
    "light": {"bg": "#ffffff", "fg": "#1f2328", "grid": "#d0d7de", "muted": "#8c959f",
              "v1": "#0969da", "v2": "#cf222e", "oracle": "#1f2328", "band": "#0969da", "s_ok": "#1a7f37"},
    "dark": {"bg": "#0d1117", "fg": "#e6edf3", "grid": "#30363d", "muted": "#8b949e",
             "v1": "#58a6ff", "v2": "#ff7b72", "oracle": "#e6edf3", "band": "#58a6ff", "s_ok": "#3fb950"},
}


def style(ax, t):
    ax.set_facecolor(t["bg"])
    for s in ax.spines.values():
        s.set_color(t["grid"])
    ax.tick_params(colors=t["fg"], labelsize=8)
    ax.xaxis.label.set_color(t["fg"])
    ax.yaxis.label.set_color(t["fg"])
    ax.title.set_color(t["fg"])
    ax.grid(True, color=t["grid"], linewidth=0.5, alpha=0.7)


def bootstrap_band(score, s_S, s_L, alphas, B=200, seed=0):
    """Paired-bootstrap 2.5/97.5% band of PGR at fixed alphas (queries resampled with both arms)."""
    rng = np.random.default_rng(seed)
    n = len(score)
    reps = np.empty((B, len(alphas)))
    for b in range(B):
        i = rng.integers(0, n, n)
        sw = th.sweep(score[i], s_S[i], s_L[i])
        o = np.argsort(sw.alpha)
        reps[b] = np.interp(alphas, sw.alpha[o], sw.pgr[o])
    return np.nanquantile(reps, 0.025, axis=0), np.nanquantile(reps, 0.975, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, default=Path("artifacts/predictions/baselines.parquet"))
    ap.add_argument("--corpus", type=Path, default=Path("artifacts/routerbench/corpus.parquet"))
    ap.add_argument("--scores", type=Path, default=Path("artifacts/routerbench/scores.parquet"))
    ap.add_argument("--small", default="mistralai/mixtral-8x7b-chat")
    ap.add_argument("--large", default="gpt-4-1106-preview")
    ap.add_argument("--router", default=None, help="router name in the predictions file (default: first)")
    ap.add_argument("--report-json", type=Path, default=Path("artifacts/reports/threshold_report.json"))
    ap.add_argument("--rho", type=float, default=0.95)
    ap.add_argument("--theme", choices=sorted(THEMES), default="light")
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("notes/pareto.png"))
    args = ap.parse_args()
    t = THEMES[args.theme]

    f, _ = report.load_frame(args.predictions, args.corpus, args.scores, args.small, args.large)
    router = args.router or sorted(f["router"].unique())[0]
    g = f[(f["router"] == router) & f["source"].isin(report.HEADLINE)]
    te, ca = g[g["split"] == "test"], g[g["split"] == "cal"]
    tr = g[g["split"] == "train"]
    sS, sL, cL = te["s_S"].to_numpy(), te["s_L"].to_numpy(), te["c_L"].to_numpy()

    # tau* per rule, chosen on cal
    taus = {}
    rep = json.loads(args.report_json.read_text()) if args.report_json.exists() else None
    for rule in ("v1", "v2"):
        tau = None
        if rep:
            for r in rep["routers"]:
                if r["router"] == router:
                    for rr in r["rules"]:
                        if rr["rule"] == rule:
                            tau = rr["selections"]["wsr"]["tau_star"]
        if tau is None:
            sel = th.select_tau(ca[rule].to_numpy(), ca["s_S"].to_numpy(), ca["s_L"].to_numpy(),
                                taus=th.threshold_grid(tr[rule].to_numpy()), rho=args.rho, method="wsr")
            tau = sel.tau_star
        taus[rule] = tau

    sweeps = {rule: th.sweep(te[rule].to_numpy(), sS, sL, c_L=cL) for rule in ("v1", "v2")}
    refs = th.reference_curves(sS, sL, c_L=cL)

    fig = plt.figure(figsize=(12, 9), dpi=200, facecolor=t["bg"])
    gs = fig.add_gridspec(3, 4, height_ratios=[1.25, 0.8, 0.9], hspace=0.55, wspace=0.35)
    axA, axB = fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[0, 2:])
    axC = [fig.add_subplot(gs[1, i]) for i in range(4)]
    axD = fig.add_subplot(gs[2, :])
    for ax in (axA, axB, *axC, axD):
        style(ax, t)

    # ---- Panel A: cost vs PGR
    k = 1000.0
    for name, sw, col, ls in (("router v1", sweeps["v1"], t["v1"], "-"), ("router v2 (per $)", sweeps["v2"], t["v2"], "-"),
                              ("random", refs["random"], t["muted"], "--"), ("oracle", refs["oracle"], t["oracle"], ":")):
        o = np.argsort(sw.cost)
        axA.plot(np.maximum(sw.cost[o] * k, 1e-3 * cL.mean() * k), sw.pgr[o], ls, color=col, lw=1.6, label=name)
    axA.axhline(args.rho, color=t["v2"], ls="--", lw=0.8, alpha=0.6)
    c_all = cL.mean() * k
    rep_v1 = th.evaluate_at_tau(te["v1"].to_numpy(), sS, sL, taus["v1"], c_L=cL, B=1)
    axA.scatter([rep_v1["cost"] * k], [rep_v1["pgr"]], color=t["v1"], zorder=5, s=30)
    mult = c_all / (rep_v1["cost"] * k) if rep_v1["cost"] > 0 else float("inf")
    axA.annotate(f"tau* (cal): PGR {rep_v1['pgr']:.3f}, {mult:.2f}x cheaper",
                 (rep_v1["cost"] * k, rep_v1["pgr"]), textcoords="offset points", xytext=(-150, -28),
                 color=t["fg"], fontsize=8, arrowprops={"arrowstyle": "-", "color": t["muted"]})
    axA.set_xscale("log")
    axA.set_xlim(1e-2 * c_all, 1.3 * c_all)
    axA.set_xlabel("mean cost, $ per 1,000 queries (M1, log)")
    axA.set_ylabel("PGR")
    axA.set_title("A  cost vs quality recovered", loc="left", fontsize=10, color=t["fg"])
    axA.set_ylim(-0.05, 1.05)
    leg = axA.legend(fontsize=7.5, loc="lower right", facecolor=t["bg"], edgecolor=t["grid"])
    for txt in leg.get_texts():
        txt.set_color(t["fg"])

    # ---- Panel B: alpha vs PGR with envelope and band
    alphas = np.linspace(0, 1, 101)
    lo, hi = bootstrap_band(te["v1"].to_numpy(), sS, sL, alphas, B=args.B)
    axB.fill_between(alphas, lo, hi, color=t["band"], alpha=0.18, lw=0, label="95% band (v1)")
    sw = sweeps["v1"]
    o = np.argsort(sw.alpha)
    axB.plot(sw.alpha[o], sw.pgr[o], color=t["v1"], lw=1.4, label=f"router v1  CPT {sw.cpt(args.rho):.0%}")
    idx = th.upper_envelope(sw.alpha, sw.pgr)
    axB.plot(sw.alpha[idx], sw.pgr[idx], color=t["v1"], lw=0.9, ls="-.", alpha=0.8,
             label=f"envelope  CPT {th.envelope_cpt(sw, args.rho):.0%}")
    for name, ref, col, ls in (("random", refs["random"], t["muted"], "--"), ("oracle", refs["oracle"], t["oracle"], ":")):
        o = np.argsort(ref.alpha)
        axB.plot(ref.alpha[o], ref.pgr[o], ls, color=col, lw=1.2, label=f"{name}  CPT {ref.cpt(args.rho):.0%}")
    axB.axhline(args.rho, color=t["v2"], ls="--", lw=0.8, alpha=0.6)
    axB.axvline(rep_v1["alpha"], color=t["v1"], lw=0.8, ls=":")
    axB.set_xlabel("alpha: share of queries sent to the frontier")
    axB.set_ylabel("PGR")
    axB.set_title("B  CPT plot", loc="left", fontsize=10, color=t["fg"])
    axB.set_xlim(0, 1)
    axB.set_ylim(-0.05, 1.05)
    leg = axB.legend(fontsize=7, loc="lower right", facecolor=t["bg"], edgecolor=t["grid"])
    for txt in leg.get_texts():
        txt.set_color(t["fg"])

    # ---- Panel C: per source
    for ax, src in zip(axC, report.HEADLINE):
        s = te[te["source"] == src]
        if s.empty or (s["s_L"] - s["s_S"]).abs().sum() == 0:
            ax.set_title(f"C  {src}: no data", loc="left", fontsize=8, color=t["fg"])
            continue
        a, b = s["s_S"].to_numpy(), s["s_L"].to_numpy()
        sw = th.sweep(s["v1"].to_numpy(), a, b)
        ref = th.reference_curves(a, b)
        for curve, col, ls in ((sw, t["v1"], "-"), (ref["random"], t["muted"], "--"), (ref["oracle"], t["oracle"], ":")):
            o = np.argsort(curve.alpha)
            ax.plot(curve.alpha[o], curve.pgr[o], ls, color=col, lw=1.1)
        ax.axhline(args.rho, color=t["v2"], ls="--", lw=0.6, alpha=0.6)
        ax.set_title(f"C  {src} (n={len(s):,}, gap {b.mean() - a.mean():.2f})", loc="left", fontsize=8, color=t["fg"])
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel("alpha", fontsize=7)

    # ---- Panel D: score distribution by whether S was right
    sc = te["v1"].to_numpy()
    bins = np.linspace(sc.min(), sc.max(), 50)
    axD.hist(sc[sS >= 0.5], bins=bins, alpha=0.6, color=t["s_ok"], label="S right", density=True)
    axD.hist(sc[sS < 0.5], bins=bins, alpha=0.6, color=t["v2"], label="S wrong", density=True)
    axD.axvline(taus["v1"], color=t["fg"], ls=":", lw=1, label="tau* (v1, from cal)")
    axD.set_xlabel("router score delta_hat (v1)")
    axD.set_ylabel("density")
    axD.set_title("D  does the router separate S's successes from its failures?", loc="left", fontsize=10, color=t["fg"])
    leg = axD.legend(fontsize=8, facecolor=t["bg"], edgecolor=t["grid"])
    for txt in leg.get_texts():
        txt.set_color(t["fg"])

    fig.suptitle(f"{router}: S = {args.small.split('/')[-1]}, L = {args.large}  (test split, headline sources)",
                 color=t["fg"], fontsize=11, y=0.995)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor=t["bg"], bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
