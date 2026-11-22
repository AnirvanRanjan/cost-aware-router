# Demo soak test (T47)

*2026-11-21 10:50, 60.0 minutes, lane D on the M2 (arm64, 16 GB).*

> **The frontier arm was throttled: billing was never switched on.** All 457 failures are Gemini
> free-tier `HTTP 429`. The key's project is on the free plan — on 2026-11-21 the error carried
> `quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier`, `quotaValue: 20` (20 requests/day),
> and on 2026-11-22 the same key still returns
> `GenerateRequestsPerMinutePerProjectPerModel-FreeTier`, `quotaValue: 5` (5/min; 5 of 25 probe
> calls succeeded). Local route, latency split, cache and memory are unaffected; spend and savings
> below cover the 553 answered queries only. **Re-run once billing is enabled.**

```
python scripts/serve_demo.py            # tau*=8.219 rule v2, cascade_t61_3sample
python scripts/19_soak_demo.py --minutes 60
```

Models: **qwen3.5-4b** (MLX 4-bit, this Mac) + **gemini-3-flash** (frontier API), priced as gpt-4-1106-preview via the reference price table. Queries: every `configs/demo_set.yaml` entry, then 200 replay-set test questions (seed 42), then a shuffled mix.

## Traffic

| | |
|---|--:|
| Queries attempted | 1010 |
| Answered (used below) | 553 |
| Kept local | 263 (47.6%) |
| Sent to frontier | 290 (52.4%) |
| Served from cache | 421 (76.1%) |
| Live calls | 132 (23.9%) |
| Failed calls (excluded from spend, savings and latency) | 457 |
| Demo-set questions escalated | 12 |

## Spend and savings (failed calls excluded)

Computed from the answered queries only: `cost_usd` is what serve.py charged for the route taken, `counterfactual_usd` is what the same traffic would have cost always-frontier (serve.py's money model). A failed call contributes to neither column, so it cannot inflate or deflate the saving.

| | |
|---|--:|
| Answered queries | 553 |
| Failed calls (excluded) | 457 |
| Spend | $0.2147 |
| Always-frontier counterfactual | $0.5850 |
| Saved | $0.3703 (**63.3%**) |

## Latency (ms, first token = start of streaming to the user; answered queries only)

| | p50 | p95 | p99 | max | n |
|---|--:|--:|--:|--:|--:|
| First token, all | 6 | 2,137 | 7,403 | 10,140 | 553 |
| First token, cache hits | 6 | 1,031 | 1,369 | 1,936 | 421 |
| First token, live calls | 1,102 | 7,217 | 9,409 | 10,140 | 132 |
| First token, local | 6 | 2,040 | 2,634 | 6,571 | 263 |
| First token, frontier | 6 | 6,161 | 7,884 | 10,140 | 290 |
| Whole query, cache hits | 11 | 1,037 | 1,378 | 1,940 | 421 |
| Whole query, live calls | 1,105 | 7,221 | 9,416 | 10,149 | 132 |
| Whole query, all | 12 | 2,142 | 7,407 | 10,149 | 553 |
| Router decision (server) | 4 | 5 | 6 | 17 | 553 |

Cache hits and live calls are split by `done.cached`: a cache hit streams an answer that is already stored, a live call waits for the model. Failed calls have no first token and are not in these rows.

Cold start: the demo set was pre-warmed by `scripts/check_demo_set.py` (T45), so its queries read local answers from the cache; the first query overall took 21 ms (route local). The five slowest queries of the run, which is where model load, the first frontier calls and throttling show up:

| # | phase | route | first token (ms) | total (ms) |
|--:|---|---|--:|--:|
| 918 | mix | frontier | – | 12,574 |
| 762 | mix | frontier | 10,140 | 10,149 |
| 93 | replay | frontier | 9,877 | 9,885 |
| 908 | mix | frontier | – | 8,387 |
| 104 | replay | frontier | 8,365 | 8,371 |

## Memory (server process)

`phys_footprint` (macOS `footprint -p <pid>`, falling back to `vmmap --summary`) is the number that counts MLX's IOAccelerator allocations; RSS alone misses them, so RSS by itself cannot support a no-leak claim. Growth in phys_footprint over the run is the leak signal.

| | start | min | max | end |
|---|--:|--:|--:|--:|
| RSS (GB) | 0.13 | 0.08 | 0.24 | 0.11 |
| phys_footprint (GB) | 0.15 | 0.15 | 11.00 | 11.00 |

120 samples every 30 s over 60.0 minutes (machine: 16 GB, phys_footprint from `footprint`).

## Errors and throttling

- 457× `RuntimeError: HTTP 429: [{
  "error": {
    "code": 429,
    "message": "You exc`

These 457 calls are excluded from the spend, savings and latency figures above: they produced no answer and charged nothing.

Server `/stats` (cumulative; the delta is what this soak added -- the server may have answered queries before the soak started):

```
start: {"queries": 0, "local": 0, "frontier": 0, "spent_usd": 0.0, "all_frontier_usd": 0.0, "saved_usd": 0.0, "saved_share": 0.0, "router_ms_p50": 0.0, "router_ms_p99": 0.0, "first_token_ms_p50": 0.0}
end:   {"queries": 553, "local": 263, "frontier": 290, "spent_usd": 0.21471300000000024, "all_frontier_usd": 0.5850089999999998, "saved_usd": 0.3702959999999995, "saved_share": 0.6329748773095794, "router_ms_p50": 3.7734159996034577, "router_ms_p99": 6.084466519096168, "first_token_ms_p50": 6.050416999642039}
delta: {"queries": 553, "local": 263, "frontier": 290, "spent_usd": 0.21471300000000024, "all_frontier_usd": 0.5850089999999998, "saved_share": 0.6329748773095794}
```

The delta's saved_share is 63.3% against the soak's own 63.3%: the two differ whenever the server counted a request the soak saw as a failure, so the soak's own figure (answered queries only) is the one to quote.

## Offline cache (T46)

`artifacts/cache/demo.sqlite` after the run: ('gemini-3-flash', 68) ('qwen3.5-4b', 258) (model rows). Frontier answers that completed are cached under the keys `serve_demo.py --offline` looks up; queries that hit an error were not cached and still need a live call.

## Findings

- 1010 queries attempted in 60.0 minutes (17/min): 553 answered (290 through the frontier API, 421 served from cache, 132 live), 457 failed.
- First token p50 6 ms over all answered queries; split by source: cache hits p50 6 ms / p99 1,369 ms (n=421), live calls p50 1,102 ms / p99 9,409 ms (n=132). Local answers stream only after Qwen's k samples are done, so local latency is dominated by generation, not routing.
- Router decision p99 6 ms (target < 15 ms).
- phys_footprint (from `footprint`) 0.15 → 11.00 GB, peak 11.00 GB of 16 GB over 60.0 minutes; RSS 0.13 → 0.11 GB (120 samples). The start/end/peak footprint are the numbers to quote for growth; RSS alone is not.
- The 11 GB peak is **not an unbounded leak**: measured again on 2026-11-22 against the same
  (still-running) server, the footprint was 10.6 GB — 10 GB of it dirty `IOAccelerator (graphics)`
  across 5,193 regions, `vmmap` agreeing at 10.6G — and **2.8 GB after 30 further uncached local
  generations** (IOAccelerator 2,360 MB across 1,068 regions). The GPU-side pool grows under a
  sustained demo hour and shrinks again; RSS never left ~0.11 GB. Quote it as "up to ~11 GB of
  GPU-side footprint under sustained load on a 16 GB machine", not as a leak.
- 457 failed calls, all excluded from the spend, savings and latency figures above.
