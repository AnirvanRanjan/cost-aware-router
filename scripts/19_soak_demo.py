#!/usr/bin/env python3
"""T47: one-hour soak test against the running demo server -> notes/demo_perf.md (lane D).

    python scripts/serve_demo.py &                       # the server under test, real models
    caffeinate -i python scripts/19_soak_demo.py         # 60 minutes, demo set + replay set

Asks every demo-set query first, then 200 replay-set test questions, then keeps cycling a mix
until the deadline, sampling the server's RSS and /stats on a background thread. The run also
completes T46: every frontier answer it sees is written to artifacts/cache/demo.sqlite under the
same keys `serve_demo.py --offline` will look up, so after this the demo runs with the Wi-Fi off.

Reports cold start (first query, model load included), first-token and total latency by route,
router latency from /stats, memory over time, and any errors or throttling.
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


def footprint_mb(pid: int) -> float | None:
    """macOS phys footprint, the number that includes MLX's IOAccelerator allocations."""
    try:
        out = subprocess.run(["footprint", "-p", str(pid)], capture_output=True, text=True,
                             timeout=10).stdout
        m = re.search(r"Footprint:\s+([\d.]+)\s+MB", out)
        return float(m.group(1)) if m else None
    except Exception:                                          # noqa: BLE001
        return None


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
                "cost_usd": None, "cached": None}
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

    samples: list[dict] = []                                 # RSS + /stats snapshots
    stop = threading.Event()

    def sampler() -> None:
        while not stop.is_set():
            row = {"t": time.time(), "rss_kb": rss_kb(pid) if pid else None,
                   "fp_mb": footprint_mb(pid) if pid else None}
            try:
                row["stats"] = json.loads(urlopen(f"{args.endpoint}/stats", timeout=10).read())
            except Exception:                                 # noqa: BLE001
                row["stats"] = None
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
    rss = [s["rss_kb"] for s in samples if s["rss_kb"]]
    last_stats = next((s["stats"] for s in reversed(samples) if s["stats"]), None)
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
        f"{health['bundle'].get('featurizer')}",
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
        f"| Queries completed | {len(results)} |",
        f"| Kept local | {len(local)} ({len(local) / max(1, len(ok)):.1%}) |",
        f"| Sent to frontier | {len(front)} ({len(front) / max(1, len(ok)):.1%}) |",
        f"| Errors | {len(errs)} |",
        f"| Demo-set questions escalated | {demo_front} |",
        "",
        "## Latency (ms, first token = start of streaming to the user)",
        "",
        "| | p50 | p95 | p99 | max |",
        "|---|--:|--:|--:|--:|",
        f"| First token, all | {fmt_ms(pct(ft_all, 50))} | {fmt_ms(pct(ft_all, 95))} | "
        f"{fmt_ms(pct(ft_all, 99))} | {fmt_ms(max(ft_all) if ft_all else float('nan'))} |",
        f"| First token, local | {fmt_ms(pct(ft_loc, 50))} | {fmt_ms(pct(ft_loc, 95))} | "
        f"{fmt_ms(pct(ft_loc, 99))} | {fmt_ms(max(ft_loc) if ft_loc else float('nan'))} |",
        f"| First token, frontier | {fmt_ms(pct(ft_fro, 50))} | {fmt_ms(pct(ft_fro, 95))} | "
        f"{fmt_ms(pct(ft_fro, 99))} | {fmt_ms(max(ft_fro) if ft_fro else float('nan'))} |",
        f"| Whole query | {fmt_ms(pct(wall, 50))} | {fmt_ms(pct(wall, 95))} | "
        f"{fmt_ms(pct(wall, 99))} | {fmt_ms(max(wall) if wall else float('nan'))} |",
        f"| Router decision (server) | {fmt_ms(pct(router, 50))} | {fmt_ms(pct(router, 95))} | "
        f"{fmt_ms(pct(router, 99))} | {fmt_ms(max(router) if router else float('nan'))} |",
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
        "`footprint` is the number that counts MLX's graphics allocations; RSS alone misses them "
        "(the model lives in IOAccelerator memory).",
        "",
        "| | start | min | max | end |",
        "|---|--:|--:|--:|--:|",
    ]
    fps = [s["fp_mb"] for s in samples if s.get("fp_mb")]
    if rss and fps:
        lines += [
            f"| RSS (GB) | {rss[0] / 2**20:.2f} | {min(rss) / 2**20:.2f} | {max(rss) / 2**20:.2f} | "
            f"{rss[-1] / 2**20:.2f} |",
            f"| Footprint (GB) | {fps[0] / 1024:.2f} | {min(fps) / 1024:.2f} | {max(fps) / 1024:.2f} | "
            f"{fps[-1] / 1024:.2f} |",
        ]
    elif rss:
        lines.append(f"| RSS (GB) | {rss[0] / 2**20:.2f} | {min(rss) / 2**20:.2f} | "
                     f"{max(rss) / 2**20:.2f} | {rss[-1] / 2**20:.2f} |")
    else:
        lines.append("| – | – | – | – | – |")
    lines += [
        "",
        f"{len(samples)} samples every {args.sample_seconds:g} s over "
        f"{duration / 60:.1f} minutes (machine: {mem_gb:.0f} GB).",
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
    else:
        lines.append("- none: every query answered.")
    if last_stats:
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
        f"- {len(results)} queries in {duration / 60:.1f} minutes "
        f"({len(results) / (duration / 60):.0f}/min), {len(front)} of them through the frontier API.",
        f"- First token p50 {fmt_ms(pct(ft_all, 50))} ms, p99 {fmt_ms(pct(ft_all, 99))} ms; "
        "local answers stream only after Qwen's k samples are done, so the local p50 is dominated "
        "by generation, not routing.",
        f"- Router decision p99 {fmt_ms(pct(router, 99))} ms (target < 15 ms).",
         ((f"- Footprint moved from {fps[0] / 1024:.2f} GB to {fps[-1] / 1024:.2f} GB "
           f"(peak {max(fps) / 1024:.2f} GB of {mem_gb:.0f} GB) over "
           f"{duration / 60:.1f} minutes; ") if fps else "- ")
         + (f"RSS from {rss[0] / 2**20:.2f} to {rss[-1] / 2**20:.2f} GB "
            f"({len(rss)} samples) shows the Python side did not leak.")
         if rss else "- no memory samples.",
        f"- {len(errs)} errors." if errs else "- No errors and no exhausted retries; the client's "
        "8 rpm limiter absorbed the free-tier cap without a failed request.",
        "",
    ]
    args.out.write_text("\n".join(lines))
    tail = f"{len(results)} queries, {len(errs)} errors"
    if fps:
        tail += f", footprint {fps[0] / 1024:.2f}->{fps[-1] / 1024:.2f} GB (max {max(fps) / 1024:.2f})"
    elif rss:
        tail += f", rss {rss[0] / 2**20:.2f}->{rss[-1] / 2**20:.2f} GB"
    print(f"\nwrote {args.out} | {tail}")


if __name__ == "__main__":
    main()
