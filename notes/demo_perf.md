# Demo soak test (T47)

> **Superseded (T47 re-run pending).** This run used the scikit-learn 1.9.1 bundle
> (tau\* 9.657, not the certified 8.2187) and the Gemini free tier, so 432 of the 929
> queries failed with HTTP 429 and the `/stats` spend, savings ("73% saved") and
> first-token figures below mix failures and cache hits together; only RSS was sampled,
> which does not include MLX's GPU allocations, so the memory line is not evidence of
> no growth. Do not quote these numbers. `scripts/19_soak_demo.py` now samples
> phys_footprint, splits cache hits from live calls, and excludes failed calls from
> spend/savings; the re-run waits for Gemini billing to be switched on (T47).
>
> The numbers below are kept as the record of the 2026-11-19 run.

*2026-11-19 14:22, 60.0 minutes, lane D on the M2 (arm64, 16 GB).*

```
python scripts/serve_demo.py            # tau*=9.657 rule v2, cascade_t61_3sample
python scripts/19_soak_demo.py --minutes 60
```

Models: **qwen3.5-4b** (MLX 4-bit, this Mac) + **gemini-3-flash** (frontier API), priced as gpt-4-1106-preview via the reference price table. Queries: every `configs/demo_set.yaml` entry, then 200 replay-set test questions (seed 42), then a shuffled mix.

## Traffic

| | |
|---|--:|
| Queries completed | 929 |
| Kept local | 299 (60.2%) |
| Sent to frontier | 198 (39.8%) |
| Errors | 432 |
| Demo-set questions escalated | 14 |

## Latency (ms, first token = start of streaming to the user)

| | p50 | p95 | p99 | max |
|---|--:|--:|--:|--:|
| First token, all | 7 | 4,737 | 7,321 | 8,835 |
| First token, local | 799 | 2,478 | 5,572 | 8,835 |
| First token, frontier | 4 | 7,069 | 7,463 | 7,703 |
| Whole query | 11 | 4,739 | 7,324 | 8,837 |
| Router decision (server) | 3 | 5 | 6 | 17 |

Cold start: the demo set was pre-warmed by `scripts/check_demo_set.py` (T45), so its queries read local answers from the cache; the first query overall took 3,150 ms (route frontier). The five slowest queries of the run, which is where model load, the first frontier calls and throttling show up:

| # | phase | route | first token (ms) | total (ms) |
|--:|---|---|--:|--:|
| 118 | replay | frontier | – | 15,141 |
| 101 | replay | frontier | – | 15,141 |
| 110 | replay | frontier | – | 15,117 |
| 99 | replay | frontier | – | 14,825 |
| 86 | replay | frontier | – | 14,823 |

## Memory (server process)

`footprint` is the number that counts MLX's graphics allocations; RSS alone misses them (the model lives in IOAccelerator memory).

| | start | min | max | end |
|---|--:|--:|--:|--:|
| RSS (GB) | 0.11 | 0.06 | 0.13 | 0.11 |

119 samples every 30 s over 60.0 minutes (machine: 16 GB).

## Errors and throttling

- 432× `RuntimeError: HTTP 429: [{
  "error": {
    "code": 429,
    "message": "You exc`

Server `/stats` at the end:

```
{
  "queries": 990,
  "local": 595,
  "frontier": 395,
  "spent_usd": 0.2730030000000002,
  "all_frontier_usd": 1.012410000000002,
  "saved_usd": 0.7394070000000018,
  "saved_share": 0.7303434379352243,
  "router_ms_p50": 3.2200414570979774,
  "router_ms_p99": 6.283161968458444,
  "first_token_ms_p50": 7.454353966750205
}
```

## Offline cache (T46)

`artifacts/cache/demo.sqlite` after the run: ('gemini-3-flash', 48) ('qwen3.5-4b', 234) (model rows). Frontier answers that completed are cached under the keys `serve_demo.py --offline` looks up; the queries that hit HTTP 429 were not cached and still need a live call. Checked separately: all **24/24** demo-set questions answer with `--offline` after the run.

## Findings

- 929 queries in 60.0 minutes (15/min), 198 of them through the frontier API.
- First token p50 7 ms, p99 7,321 ms; local answers stream only after Qwen's k samples are done, so the local p50 is dominated by generation, not routing.
- Router decision p99 6 ms (target < 15 ms).
- RSS 0.11 → 0.11 GB over 119 samples (min 0.06, max 0.13 GB): no Python-side leak over the hour.
- 432 errors, all HTTP 429 from the Gemini free tier after its 20/day cap was reached; the run continued and recorded them instead of aborting.
