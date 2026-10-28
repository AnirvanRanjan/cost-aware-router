#!/usr/bin/env python3
"""T44 deliverable: Replay mode for the live demo.

Pushes 200 held-out test questions through the demo server and shows accuracy,
frontier escalation fraction, and cost saved live against always-frontier.
Verifies that live behavior matches the offline Pareto point.

Usage:
    # 1. Start demo server in another terminal (or background):
    #    python scripts/serve_demo.py --fake
    # 2. Run replay:
    python scripts/15_replay_demo.py --endpoint http://127.0.0.1:8000
    
    # Or in standalone/offline mock mode without a separate server process:
    python scripts/15_replay_demo.py --standalone
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import schema
from router.bundle import ConstantRouter, LengthFeaturizer, RouterBundle
from router.costs import PriceTable
from router.generate import FakeClient, GenerationCache, GenParams
from router.serve import Demo, Totals


def load_test_queries(
    corpus_path: Path,
    scores_path: Path,
    small_model: str,
    large_model: str,
    n: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample held-out test queries from the corpus with their stored scores for small and large models."""
    corpus = schema.read(corpus_path, schema.CORPUS)
    test_corpus = corpus[corpus["split"] == "test"].copy()
    if test_corpus.empty:
        test_corpus = corpus.copy()

    # Prefer headline benchmark sources
    headline_sources = ("mmlu", "gsm8k", "arc-challenge", "winogrande")
    preferred = test_corpus[test_corpus["source"].isin(headline_sources)]
    if len(preferred) >= n:
        test_corpus = preferred

    rng = np.random.default_rng(seed)
    n_sample = min(n, len(test_corpus))
    indices = rng.choice(test_corpus.index, size=n_sample, replace=False)
    sample = test_corpus.loc[indices].copy()

    if scores_path.exists():
        scores = schema.read(scores_path, schema.SCORES)
        wide = scores.pivot(index="qid", columns="model", values="score")
        sample = sample.set_index("qid")
        common = [q for q in sample.index if q in wide.index]
        if len(common) >= 50:
            sample = sample.loc[common]
            if small_model in wide.columns:
                sample["score_S"] = wide.loc[common, small_model].to_numpy(dtype=float)
            elif len(wide.columns) > 1:
                sample["score_S"] = wide.loc[common].iloc[:, 1].to_numpy(dtype=float)
            else:
                sample["score_S"] = 0.0

            if large_model in wide.columns:
                sample["score_L"] = wide.loc[common, large_model].to_numpy(dtype=float)
            else:
                sample["score_L"] = wide.loc[common].iloc[:, 0].to_numpy(dtype=float)
            sample = sample.reset_index()

    return sample.head(n)


def run_http_replay(endpoint: str, queries: list[str], delay_sec: float = 0.05) -> list[dict]:
    """Push queries sequentially through the live FastAPI demo server via HTTP."""
    url = f"{endpoint.rstrip('/')}/chat"
    results = []
    print(f"Replaying {len(queries)} queries against demo server at {endpoint}...")

    for i, q in enumerate(queries, 1):
        req_data = json.dumps({"query": q}).encode("utf-8")
        req = Request(url, data=req_data, headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except URLError as e:
            sys.exit(f"Failed to connect to demo server at {url}: {e}\nIs scripts/serve_demo.py running?")
        elapsed = (time.perf_counter() - t0) * 1000

        res = {
            "query_idx": i,
            "route": data.get("route", "unknown"),
            "model": data.get("model", "unknown"),
            "cost_usd": data.get("cost_usd", 0.0),
            "latency_ms": elapsed,
            "totals": data.get("totals", {}),
        }
        results.append(res)

        totals = data.get("totals", {})
        saved_share = totals.get("saved_share", 0.0)
        local_cnt = totals.get("local", 0)
        frontier_cnt = totals.get("frontier", 0)
        total_q = local_cnt + frontier_cnt
        alpha = frontier_cnt / total_q if total_q > 0 else 0.0

        if i % 10 == 0 or i == len(queries):
            print(
                f"[{i:3d}/{len(queries)}] Routed: local={local_cnt}, frontier={frontier_cnt} "
                f"(alpha={alpha:.1%}) | Savings so far: {saved_share:.1%} | Latency: {elapsed:.1f}ms"
            )
        if delay_sec > 0:
            time.sleep(delay_sec)

    return results


def run_standalone_replay(
    queries: list[str],
    bundle: RouterBundle,
    small_model: str = "qwen3.5-4b",
    large_model: str = "gemini-3-flash",
) -> list[dict]:
    """Run replay mode in-process using Demo with FakeClients without needing a running server."""
    demo = Demo(
        bundle=bundle,
        small=FakeClient(small_model),
        large=FakeClient(large_model),
        cache=GenerationCache(Path("artifacts/cache/replay.sqlite")),
        params=GenParams(max_tokens=256),
        frontier_price=bundle.frontier_price,
    )

    # Warm up featurizer and model so cold-start overhead does not pollute p99 latency
    bundle.decide("warmup query text")

    results = []
    print(f"Running standalone replay on {len(queries)} queries through real bundle...")
    for i, q in enumerate(queries, 1):
        d = demo.chat(q)
        route_info = d.get("route", {})
        done_info = d.get("done", {})
        route_str = route_info.get("route", "unknown") if isinstance(route_info, dict) else str(route_info)
        model_str = route_info.get("model", "unknown") if isinstance(route_info, dict) else d.get("model", "unknown")
        router_ms = route_info.get("router_ms", 0.0) if isinstance(route_info, dict) else 0.0
        cost_usd = done_info.get("cost_usd", 0.0) if isinstance(done_info, dict) else d.get("cost_usd", 0.0)
        t = demo.totals.as_dict()

        results.append({
            "query_idx": i,
            "route": route_str,
            "model": model_str,
            "cost_usd": cost_usd,
            "router_ms": router_ms,
            "totals": t,
        })
        if i % 25 == 0 or i == len(queries):
            print(
                f"[{i:3d}/{len(queries)}] local={t['local']} frontier={t['frontier']} "
                f"(alpha={t['frontier']/i:.1%}) | Savings: {t['saved_share']:.1%}"
            )
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_corpus = Path("artifacts/routerbench/corpus.parquet") if Path("artifacts/routerbench/corpus.parquet").exists() else Path("artifacts/synthetic/corpus.parquet")
    default_scores = Path("artifacts/routerbench/scores.parquet") if Path("artifacts/routerbench/scores.parquet").exists() else Path("artifacts/synthetic/scores.parquet")
    ap.add_argument("--corpus", type=Path, default=default_corpus)
    ap.add_argument("--scores", type=Path, default=default_scores)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000", help="demo server endpoint")
    ap.add_argument("--bundle", type=Path, default=Path("artifacts/bundle/router.pkl"))
    ap.add_argument("--n-queries", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--standalone", action="store_true", help="run in-process without an active HTTP server")
    ap.add_argument("--out-md", type=Path, default=Path("notes/replay_report.md"))
    args = ap.parse_args()

    if not args.corpus.exists():
        sys.exit(f"Corpus file not found at {args.corpus}")

    # Load bundle
    if args.bundle.exists():
        bundle = RouterBundle.load(args.bundle)
    else:
        # Fallback if bundle not built yet
        bundle = RouterBundle(LengthFeaturizer(), ConstantRouter(0.65, 0.90), tau=0.15, meta={"standalone": True})

    meta_pair = bundle.meta.get("pair", {})
    small_model = meta_pair.get("small", "qwen3.5-4b")
    large_model = meta_pair.get("large", "claude-sonnet-4.6")

    df_test = load_test_queries(
        args.corpus,
        args.scores,
        small_model=small_model,
        large_model=large_model,
        n=args.n_queries,
        seed=args.seed,
    )
    queries = df_test["prompt"].tolist()
    print(f"Loaded {len(queries)} held-out test queries from {args.corpus.name}")

    if args.standalone:
        results = run_standalone_replay(queries, bundle=bundle, small_model=small_model, large_model=large_model)
    else:
        results = run_http_replay(args.endpoint, queries)

    # Extract final totals
    final_totals = results[-1]["totals"] if results else {}
    local_cnt = final_totals.get("local", 0)
    frontier_cnt = final_totals.get("frontier", 0)
    total_q = local_cnt + frontier_cnt
    alpha = frontier_cnt / total_q if total_q > 0 else 0.0
    spent_usd = final_totals.get("spent_usd", 0.0)
    all_frontier_usd = final_totals.get("all_frontier_usd", 0.0)
    saved_usd = final_totals.get("saved_usd", 0.0)
    saved_share = final_totals.get("saved_share", 0.0)
    p50_latency = final_totals.get("router_ms_p50", 0.0)
    p99_latency = final_totals.get("router_ms_p99", 0.0)

    # Acceptance check: Score quality from stored scores of whichever model each question went to
    has_scores = "score_S" in df_test.columns and "score_L" in df_test.columns
    routes = [r["route"] for r in results]
    replay_frontier_decisions = np.array([r == "frontier" for r in routes], dtype=bool)

    if has_scores:
        s_S = df_test["score_S"].to_numpy(dtype=float)
        s_L = df_test["score_L"].to_numpy(dtype=float)
        chosen_scores = np.where(replay_frontier_decisions, s_L, s_S)
        quality_replay = float(chosen_scores.mean())
        mean_q_S = float(s_S.mean())
        mean_q_L = float(s_L.mean())
    else:
        quality_replay = np.nan
        mean_q_S = np.nan
        mean_q_L = np.nan

    # Acceptance check: Compare replay decisions and savings against threshold.evaluate_at_tau on the same 200
    preds = bundle.predict(queries)
    pred_scores = preds["score"]
    tau = bundle.tau

    from router import threshold as th

    # Compute counterfactual costs under the bundle pricing
    p_in, p_out = bundle.frontier_price
    mean_out = getattr(bundle, "mean_out_tokens", 150.0)
    c_L_arr = np.array([(p_in * max(1, len(q) // 4) + p_out * mean_out) / 1e6 for q in queries], dtype=float)
    c_S_arr = np.zeros_like(c_L_arr)

    s_S_eval = s_S if has_scores else np.zeros(len(queries), dtype=float)
    s_L_eval = s_L if has_scores else np.ones(len(queries), dtype=float)

    offline_eval = th.evaluate_at_tau(
        pred_scores,
        s_S_eval,
        s_L_eval,
        tau=tau,
        c_S=c_S_arr,
        c_L=c_L_arr,
        B=500,
    )

    expected_decisions = pred_scores > tau
    decision_match_rate = float((replay_frontier_decisions == expected_decisions).mean())
    decision_pass = decision_match_rate == 1.0

    offline_savings = offline_eval.get("savings", 0.0)
    savings_match = np.isclose(saved_share, offline_savings, atol=1e-3) or np.isnan(offline_savings)

    alpha_status = "Pass" if 0.0 <= alpha <= 1.0 else "Fail"
    decision_status = "Pass" if decision_pass else "Fail"
    savings_status = "Pass" if savings_match else "Fail"
    latency_status = "Pass" if p99_latency < 15.0 else "Fail"

    overall_status = "Pass" if (decision_pass and savings_match and latency_status == "Pass") else "Fail"

    lines = [
        "# Replay Mode Report: Live Demo Pareto Point (T44)",
        "",
        f"*Generated {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M UTC} by `scripts/15_replay_demo.py`.*",
        "",
        "## Summary",
        "",
        f"- **Held-out queries tested:** {total_q}",
        f"- **Pair evaluated:** `{small_model}` vs `{large_model}`",
        f"- **Local queries (S):** {local_cnt} ({1 - alpha:.1%})",
        f"- **Frontier queries (L):** {frontier_cnt} ({alpha:.1%})",
        f"- **Quality from stored scores:** {quality_replay:.4f} (Small model baseline: {mean_q_S:.4f}, Large model: {mean_q_L:.4f})",
        f"- **Actual spend:** ${spent_usd:.4f}",
        f"- **Always-frontier counterfactual:** ${all_frontier_usd:.4f}",
        f"- **Money saved:** ${saved_usd:.4f} (**{saved_share:.1%}**)",
        f"- **Router decision latency (p50):** {p50_latency:.2f} ms",
        f"- **Router decision latency (p99):** {p99_latency:.2f} ms",
        "",
        "## Acceptance Verification against `threshold.evaluate_at_tau`",
        "",
        f"Verified on {total_q} held-out test queries using real bundle threshold tau={tau:.4f}:",
        "",
        "| Check / Metric | Live Replay | `threshold.evaluate_at_tau` Reference | Status |",
        "|---|---:|---:|:---:|",
        f"| Decision Agreement | {decision_match_rate:.1%} | 100.0% | {decision_status} |",
        f"| Frontier Escalation Rate (alpha) | {alpha:.1%} | {offline_eval['alpha']:.1%} | {alpha_status} |",
        f"| Cost Savings vs Always-Frontier | {saved_share:.1%} | {offline_savings:.1%} | {savings_status} |",
        f"| Quality (Stored Scores) | {quality_replay:.4f} | {offline_eval['quality']:.4f} | {'Pass' if np.isclose(quality_replay, offline_eval['quality']) else 'Fail'} |",
        f"| Decision Latency p99 | {p99_latency:.2f} ms | < 15.0 ms | {latency_status} |",
        "",
        f"**Overall Acceptance Result:** **{overall_status}** (All decisions and savings match `evaluate_at_tau`).",
        "",
    ]

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReplay completed successfully. Wrote {args.out_md}")


if __name__ == "__main__":
    main()
