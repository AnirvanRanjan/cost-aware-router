#!/usr/bin/env python3
"""T47: one-hour soak test against the running demo server -> notes/demo_perf.md (lane D).

    python scripts/serve_demo.py &                       # the server under test, real models
    caffeinate -i python scripts/19_soak_demo.py         # 60 minutes, demo set + replay set

Asks every demo-set query first, then 200 replay-set test questions, then keeps cycling a mix
until the deadline, sampling the server's RSS, its phys_footprint (`footprint -p`, else
`vmmap --summary`) and /stats on a background thread. The run also completes T46: every frontier
answer it sees is written to artifacts/cache/demo.sqlite under the same keys
`serve_demo.py --offline` looks up, so after this the demo runs with the Wi-Fi off.

Reports cold start (first query, model load included), first-token and total latency split into
cache hits and live calls, router latency from /stats, spend and savings computed from the
answered queries only (failed calls are excluded from both), phys_footprint and RSS over time,
and any errors or throttling.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def server_pid(port: int) -> int | None:
    try:
        out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                             capture_output=True, text=True, timeout=10).stdout.split()
        return int(out[0]) if out else None
    except Exception:                                          # noqa: BLE001
        return None


def rss_kb(pid: int) -> int | None:
    try:
        out = subprocess.run(["ps", "-p", str(pid), "-o", "rss="], capture_output=True, text=True,
                             timeout=10).stdout.strip()
        return int(out) if out else None
    except Exception:                                          # noqa: BLE001
        return None


_FP_UNITS = {"KB": 1 / 1024, "MB": 1.0, "GB": 1024.0, "K": 1 / 1024**2, "M": 1 / 1024, "G": 1.0}


def phys_footprint_mb(pid: int) -> tuple[float | None, str | None]:
    """The server process's real footprint in MB: `footprint -p <pid>`, else `vmmap --summary`.

    `footprint` prints "Footprint: 1361 KB" / "... MB" / "... GB" depending on size, and
    `vmmap --summary` prints "Physical footprint: 1233K" -- both are the phys_footprint number
    that includes MLX's IOAccelerator allocations, which RSS misses. Returns (MB, tool).
    """
    try:
        out = subprocess.run(["footprint", "-p", str(pid)], capture_output=True, text=True,
                             timeout=20).stdout
        m = re.search(r"Footprint:\s+([\d.]+)\s+(KB|MB|GB)", out)
        if m:
            return float(m.group(1)) * _FP_UNITS[m.group(2)], "footprint"
    except Exception:                                          # noqa: BLE001
        pass
    try:
        out = subprocess.run(["vmmap", "--summary", str(pid)], capture_output=True, text=True,
                             timeout=60).stdout
        m = re.search(r"Physical footprint:\s+([\d.]+)([KMG])", out)
        if m:
            return float(m.group(1)) * _FP_UNITS[m.group(2)], "vmmap"
    except Exception:                                          # noqa: BLE001
        pass
    return None, None


def ask(endpoint: str, query: str, timeout: float) -> dict:
    req = Request(f"{endpoint.rstrip('/')}/chat", data=json.dumps({"query": query}).encode(),
                  headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except (URLError, OSError) as e:
        return {"error": f"{type(e).__name__}: {e}", "wall_ms": (time.perf_counter() - t0) * 1000,
                "route": None, "model": None, "router_ms": None, "first_token_ms": None,
                "cost_usd": None, "counterfactual_usd": None, "cached": None}
    done = data.get("done") or {}
    route = data.get("route") or {}
    return {
        "error": data.get("error"),
        "wall_ms": (time.perf_counter() - t0) * 1000,
        "route": route.get("route"),
        "model": route.get("model"),
        "router_ms": route.get("router_ms"),
        "first_token_ms": done.get("first_token_ms"),
        "cost_usd": done.get("cost_usd"),
        "counterfactual_usd": done.get("counterfactual_usd"),
        "cached": done.get("cached"),
    }


def pct(xs: list[float], p: float) -> float:
    return float(np.percentile(np.asarray(xs, dtype=float), p)) if xs else float("nan")


def fmt_ms(x: float) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.0f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000")
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--demo-set", type=Path, default=Path("configs/demo_set.yaml"))
    ap.add_argument("--corpus", type=Path, default=Path("artifacts/routerbench/corpus.parquet"))
    ap.add_argument("--replay-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-seconds", type=float, default=30.0)
    ap.add_argument("--gap-seconds", type=float, default=0.2, help="pause between queries")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", type=Path, default=Path("notes/demo_perf.md"))
    args = ap.parse_args()

    import pandas as pd

    items: list[tuple[str, str]] = []                       # (phase, query)
    for it in yaml.safe_load(args.demo_set.read_text())["queries"]:
        items.append(("demo", it["query"]))
    corpus = pd.read_parquet(args.corpus, columns=["qid", "prompt", "split", "source"])
    test = corpus[(corpus["split"] == "test") & corpus["source"].isin(
        ("mmlu", "gsm8k", "arc-challenge", "winogrande"))]
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(test), size=min(args.replay_n, len(test)), replace=False)
    replay = [test.iloc[i]["prompt"] for i in sorted(pick)]
    print(f"soak: {sum(1 for p, _ in items if p == 'demo')} demo + {len(replay)} replay queries, "
          f"{args.minutes:g} minutes against {args.endpoint}")

    pid = server_pid(args.port)
    health = json.loads(urlopen(f"{args.endpoint}/health", timeout=10).read())
    print(f"server pid={pid} | {health['small']} + {health['large']} | tau*={health['tau']:.3f} "
          f"rule {health['rule']} | featurizer {health['bundle'].get('featurizer')}")

    def stats_now() -> dict | None:
        try:
            return json.loads(urlopen(f"{args.endpoint}/stats", timeout=10).read())
        except Exception:                                      # noqa: BLE001
            return None

    stats_start = stats_now()

    samples: list[dict] = []                                 # RSS + /stats snapshots
    stop = threading.Event()
    fp_tool: str | None = None

    def sampler() -> None:
        nonlocal fp_tool
        while not stop.is_set():
            fp_mb, tool = phys_footprint_mb(pid) if pid else (None, None)
            if tool:
                fp_tool = tool
            row = {"t": time.time(), "rss_kb": rss_kb(pid) if pid else None, "fp_mb": fp_mb}
            row["stats"] = stats_now()
            samples.append(row)
            stop.wait(args.sample_seconds)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    t_start = time.time()
    deadline = t_start + args.minutes * 60
    results: list[dict] = []
    order = [( "demo", q) for q in [x[1] for x in items if x[0] == "demo"]]
    order += [("replay", q) for q in replay]
    mix_rng = random.Random(args.seed)
    i = 0
    while time.time() < deadline:
        if i >= len(order):                                  # coverage done: cycle a shuffled mix
            pool = [x for _, x in items] + replay
            mix_rng.shuffle(pool)
            order += [("mix", q) for q in pool]
        phase, q = order[i]
        i += 1
        r = ask(args.endpoint, q, args.timeout)
        r.update({"phase": phase, "i": i, "t": time.time() - t_start})
        results.append(r)
        if i % 10 == 0 or r["error"]:
            n = len(results)
            fr = sum(1 for x in results if x.get("route") == "frontier")
            print(f"[{int(r['t'] // 60):3d}m {i:4d}] {phase:<6} route={r['route']} "
                  f"first={fmt_ms(r['first_token_ms'])}ms total={fmt_ms(r['wall_ms'])}ms "
                  f"frontier={fr}/{n}" + (f" ERROR={r['error']}" if r["error"] else ""))
        if time.time() + 5 < deadline:
            time.sleep(args.gap_seconds)
    stop.set()
    th.join(timeout=5)
    duration = time.time() - t_start

    # ---- aggregates ------------------------------------------------------------
    ok = [r for r in results if not r["error"]]
    errs = [r for r in results if r["error"]]
    local = [r for r in ok if r.get("route") == "local"]
    front = [r for r in ok if r.get("route") == "frontier"]
    ft_all = [r["first_token_ms"] for r in ok if r["first_token_ms"] is not None]
    ft_loc = [r["first_token_ms"] for r in local if r["first_token_ms"] is not None]
    ft_fro = [r["first_token_ms"] for r in front if r["first_token_ms"] is not None]
    wall = [r["wall_ms"] for r in ok]
    router = [r["router_ms"] for r in ok if r["router_ms"] is not None]
    # Cache hits vs live calls: done.cached says whether the answer the user saw came from the
    # answer cache (all k local samples cached for a local route, the frontier answer cached for
    # a frontier route). MCQ option scoring is always a live forward pass, so never a hit.
    hit_q = [r for r in ok if r.get("cached") is True]
    live_q = [r for r in ok if r.get("cached") is False]
    ft_hit = [r["first_token_ms"] for r in hit_q if r["first_token_ms"] is not None]
    ft_live = [r["first_token_ms"] for r in live_q if r["first_token_ms"] is not None]
    wall_hit = [r["wall_ms"] for r in hit_q]
    wall_live = [r["wall_ms"] for r in live_q]
    # Money from the queries that succeeded only: every failed call is excluded from both the
    # spend and the always-frontier counterfactual (a failed call cost nothing and produced no
    # answer, so it must not move the savings figure either way).
    spent_ok = float(sum(r["cost_usd"] or 0.0 for r in ok))
    all_front_ok = float(sum(r["counterfactual_usd"] or 0.0 for r in ok))
    saved_ok = all_front_ok - spent_ok
    saved_share_ok = 1.0 - spent_ok / all_front_ok if all_front_ok > 0 else float("nan")
    stats_end = stats_now() or next((s["stats"] for s in reversed(samples) if s["stats"]), None)
    last_stats = stats_end
    rss = [s["rss_kb"] for s in samples if s["rss_kb"]]
    cold = results[0] if results else {}
    demo_front = sum(1 for r in results if r["phase"] == "demo" and r["route"] == "frontier")

    try:
        cache_rows = subprocess.run(
            [sys.executable, "-c",
             "import sqlite3;c=sqlite3.connect('artifacts/cache/demo.sqlite');"
             "print(*c.execute('SELECT model,COUNT(*) FROM gens GROUP BY model').fetchall())"],
            capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:                                          # noqa: BLE001
        cache_rows = "?"

    mem_gb = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                text=True).stdout.strip()) / 2**30
    lines = [
        "# Demo soak test (T47)",
        "",
        f"*{time.strftime('%Y-%m-%d %H:%M', time.localtime(t_start))}, {duration / 60:.1f} minutes, "
        f"lane D on the M2 ({platform.machine()}, {mem_gb:.0f} GB).*",
        "",
        "```",
        f"python scripts/serve_demo.py            # tau*={health['tau']:.3f} rule {health['rule']}, "
        f"{health['bundle'].get('featurizer') or 'interim (fake)'}",
        f"python scripts/19_soak_demo.py --minutes {args.minutes:g}",
        "```",
        "",
        f"Models: **{health['small']}** (MLX 4-bit, this Mac) + **{health['large']}** "
        f"(frontier API), priced as {health['bundle'].get('pair', {}).get('large', 'gpt-4')} via the "
        "reference price table. Queries: every `configs/demo_set.yaml` entry, then "
        f"{len(replay)} replay-set test questions (seed {args.seed}), then a shuffled mix.",
        "",
        "## Traffic",
        "",
        f"| | |",
        f"|---|--:|",
        f"| Queries attempted | {len(results)} |",
        f"| Answered (used below) | {len(ok)} |",
        f"| Kept local | {len(local)} ({len(local) / max(1, len(ok)):.1%}) |",
        f"| Sent to frontier | {len(front)} ({len(front) / max(1, len(ok)):.1%}) |",
        f"| Served from cache | {len(hit_q)} ({len(hit_q) / max(1, len(ok)):.1%}) |",
        f"| Live calls | {len(live_q)} ({len(live_q) / max(1, len(ok)):.1%}) |",
        f"| Failed calls (excluded from spend, savings and latency) | {len(errs)} |",
        f"| Demo-set questions escalated | {demo_front} |",
        "",
        "## Spend and savings (failed calls excluded)",
        "",
        "Computed from the answered queries only: `cost_usd` is what serve.py charged for the "
        "route taken, `counterfactual_usd` is what the same traffic would have cost always-frontier "
        "(serve.py's money model). A failed call contributes to neither column, so it cannot "
        "inflate or deflate the saving.",
        "",
        "| | |",
        "|---|--:|",
        f"| Answered queries | {len(ok)} |",
        f"| Failed calls (excluded) | {len(errs)} |",
        f"| Spend | ${spent_ok:.4f} |",
        f"| Always-frontier counterfactual | ${all_front_ok:.4f} |",
        f"| Saved | ${saved_ok:.4f} (**{saved_share_ok:.1%}**) |",
        "",
        "## Latency (ms, first token = start of streaming to the user; answered queries only)",
        "",
        "| | p50 | p95 | p99 | max | n |",
        "|---|--:|--:|--:|--:|--:|",
        f"| First token, all | {fmt_ms(pct(ft_all, 50))} | {fmt_ms(pct(ft_all, 95))} | "
        f"{fmt_ms(pct(ft_all, 99))} | {fmt_ms(max(ft_all) if ft_all else float('nan'))} | {len(ft_all)} |",
        f"| First token, cache hits | {fmt_ms(pct(ft_hit, 50))} | {fmt_ms(pct(ft_hit, 95))} | "
        f"{fmt_ms(pct(ft_hit, 99))} | {fmt_ms(max(ft_hit) if ft_hit else float('nan'))} | {len(ft_hit)} |",
        f"| First token, live calls | {fmt_ms(pct(ft_live, 50))} | {fmt_ms(pct(ft_live, 95))} | "
        f"{fmt_ms(pct(ft_live, 99))} | {fmt_ms(max(ft_live) if ft_live else float('nan'))} | {len(ft_live)} |",
        f"| First token, local | {fmt_ms(pct(ft_loc, 50))} | {fmt_ms(pct(ft_loc, 95))} | "
        f"{fmt_ms(pct(ft_loc, 99))} | {fmt_ms(max(ft_loc) if ft_loc else float('nan'))} | {len(ft_loc)} |",
        f"| First token, frontier | {fmt_ms(pct(ft_fro, 50))} | {fmt_ms(pct(ft_fro, 95))} | "
        f"{fmt_ms(pct(ft_fro, 99))} | {fmt_ms(max(ft_fro) if ft_fro else float('nan'))} | {len(ft_fro)} |",
        f"| Whole query, cache hits | {fmt_ms(pct(wall_hit, 50))} | {fmt_ms(pct(wall_hit, 95))} | "
        f"{fmt_ms(pct(wall_hit, 99))} | {fmt_ms(max(wall_hit) if wall_hit else float('nan'))} | {len(wall_hit)} |",
        f"| Whole query, live calls | {fmt_ms(pct(wall_live, 50))} | {fmt_ms(pct(wall_live, 95))} | "
        f"{fmt_ms(pct(wall_live, 99))} | {fmt_ms(max(wall_live) if wall_live else float('nan'))} | {len(wall_live)} |",
        f"| Whole query, all | {fmt_ms(pct(wall, 50))} | {fmt_ms(pct(wall, 95))} | "
        f"{fmt_ms(pct(wall, 99))} | {fmt_ms(max(wall) if wall else float('nan'))} | {len(wall)} |",
        f"| Router decision (server) | {fmt_ms(pct(router, 50))} | {fmt_ms(pct(router, 95))} | "
        f"{fmt_ms(pct(router, 99))} | {fmt_ms(max(router) if router else float('nan'))} | {len(router)} |",
        "",
        "Cache hits and live calls are split by `done.cached`: a cache hit streams an answer that "
        "is already stored, a live call waits for the model. Failed calls have no first token and "
        "are not in these rows.",
        "",
        f"Cold start: the demo set was pre-warmed by `scripts/check_demo_set.py` (T45), so its "
        f"queries read local answers from the cache; the first query overall took "
        f"{fmt_ms(cold.get('wall_ms'))} ms (route {cold.get('route')}). The five slowest queries "
        "of the run, which is where model load, the first frontier calls and throttling show up:",
        "",
        "| # | phase | route | first token (ms) | total (ms) |",
        "|--:|---|---|--:|--:|",
    ]
    for r in sorted(results, key=lambda x: -(x["wall_ms"] or 0))[:5]:
        lines.append(f"| {r['i']} | {r['phase']} | {r['route']} | "
                     f"{fmt_ms(r['first_token_ms'])} | {fmt_ms(r['wall_ms'])} |")
    lines += [
        "",
        "## Memory (server process)",
        "",
        "`phys_footprint` (macOS `footprint -p <pid>`, falling back to `vmmap --summary`) is the "
        "number that counts MLX's IOAccelerator allocations; RSS alone misses them, so RSS by "
        "itself cannot support a no-leak claim. Growth in phys_footprint over the run is the "
        "leak signal.",
        "",
        "| | start | min | max | end |",
        "|---|--:|--:|--:|--:|",
    ]
    fps = [s["fp_mb"] for s in samples if s.get("fp_mb")]
    if rss and fps:
        lines += [
            f"| RSS (GB) | {rss[0] / 2**20:.2f} | {min(rss) / 2**20:.2f} | {max(rss) / 2**20:.2f} | "
            f"{rss[-1] / 2**20:.2f} |",
            f"| phys_footprint (GB) | {fps[0] / 1024:.2f} | {min(fps) / 1024:.2f} | {max(fps) / 1024:.2f} | "
            f"{fps[-1] / 1024:.2f} |",
        ]
    elif rss:
        lines += [
            f"| RSS (GB) | {rss[0] / 2**20:.2f} | {min(rss) / 2**20:.2f} | "
            f"{max(rss) / 2**20:.2f} | {rss[-1] / 2**20:.2f} |",
            "| phys_footprint (GB) | not sampled | not sampled | not sampled | not sampled |",
        ]
    else:
        lines.append("| – | – | – | – | – |")
    lines += [
        "",
        f"{len(samples)} samples every {args.sample_seconds:g} s over "
        f"{duration / 60:.1f} minutes (machine: {mem_gb:.0f} GB"
        + (f", phys_footprint from `{fp_tool}`" if fp_tool else ", phys_footprint NOT sampled") + ").",
        "",
        "## Errors and throttling",
        "",
    ]
    if errs:
        by_kind: dict[str, int] = {}
        for e in errs:
            k = str(e["error"])[:80]
            by_kind[k] = by_kind.get(k, 0) + 1
        lines += [f"- {n}× `{k}`" for k, n in sorted(by_kind.items(), key=lambda x: -x[1])]
        lines += ["",
                  f"These {len(errs)} calls are excluded from the spend, savings and latency "
                  "figures above: they produced no answer and charged nothing."]
    else:
        lines.append("- none: every query answered.")
    if stats_start and stats_end:
        delta = {k: stats_end.get(k, 0) - stats_start.get(k, 0)
                 for k in ("queries", "local", "frontier", "spent_usd", "all_frontier_usd")}
        delta["saved_share"] = (1 - delta["spent_usd"] / delta["all_frontier_usd"]
                                if delta["all_frontier_usd"] > 0 else float("nan"))
        lines += [
            "",
            "Server `/stats` (cumulative; the delta is what this soak added -- the server may have "
            "answered queries before the soak started):",
            "",
            "```",
            "start: " + json.dumps(stats_start),
            "end:   " + json.dumps(stats_end),
            "delta: " + json.dumps(delta),
            "```",
            "",
            f"The delta's saved_share is {delta['saved_share']:.1%} against the soak's own "
            f"{saved_share_ok:.1%}: the two differ whenever the server counted a request the soak "
            "saw as a failure, so the soak's own figure (answered queries only) is the one to quote.",
        ]
    elif last_stats:
        lines += [
            "",
            "Server `/stats` at the end:",
            "",
            "```",
            json.dumps(last_stats, indent=2),
            "```",
        ]
    lines += [
        "",
        "## Offline cache (T46)",
        "",
        f"`artifacts/cache/demo.sqlite` after the run: {cache_rows} "
        "(model rows). Frontier answers that completed are cached under the keys "
        "`serve_demo.py --offline` looks up; queries that hit an error were not cached and "
        "still need a live call.",
        "",
        "## Findings",
        "",
        f"- {len(results)} queries attempted in {duration / 60:.1f} minutes "
        f"({len(results) / (duration / 60):.0f}/min): {len(ok)} answered "
        f"({len(front)} through the frontier API, {len(hit_q)} served from cache, "
        f"{len(live_q)} live), {len(errs)} failed.",
        f"- First token p50 {fmt_ms(pct(ft_all, 50))} ms over all answered queries; split by "
        f"source: cache hits p50 {fmt_ms(pct(ft_hit, 50))} ms / p99 {fmt_ms(pct(ft_hit, 99))} ms "
        f"(n={len(ft_hit)}), live calls p50 {fmt_ms(pct(ft_live, 50))} ms / p99 "
        f"{fmt_ms(pct(ft_live, 99))} ms (n={len(ft_live)}). Local answers stream only after "
        "Qwen's k samples are done, so local latency is dominated by generation, not routing.",
        f"- Router decision p99 {fmt_ms(pct(router, 99))} ms (target < 15 ms).",
        (f"- phys_footprint (from `{fp_tool}`) {fps[0] / 1024:.2f} → {fps[-1] / 1024:.2f} GB, "
         f"peak {max(fps) / 1024:.2f} GB of {mem_gb:.0f} GB over {duration / 60:.1f} minutes; "
         f"RSS {rss[0] / 2**20:.2f} → {rss[-1] / 2**20:.2f} GB ({len(rss)} samples). The "
         "start/end/peak footprint are the numbers to quote for growth; RSS alone is not.")
         if fps and rss else
         ("- phys_footprint was not sampled this run (see Memory); RSS alone cannot support a "
          "growth claim." if rss else "- no memory samples."),
        (f"- {len(errs)} failed calls, all excluded from the spend, savings and latency figures "
         "above.") if errs else "- No errors and no exhausted retries.",
        "",
    ]
    args.out.write_text("\n".join(lines))
    tail = f"{len(results)} queries, {len(errs)} errors, savings {saved_share_ok:.1%}"
    if fps:
        tail += f", phys_footprint {fps[0] / 1024:.2f}->{fps[-1] / 1024:.2f} GB (max {max(fps) / 1024:.2f})"
    elif rss:
        tail += f", rss {rss[0] / 2**20:.2f}->{rss[-1] / 2**20:.2f} GB"
    print(f"\nwrote {args.out} | {tail}")


if __name__ == "__main__":
    main()
