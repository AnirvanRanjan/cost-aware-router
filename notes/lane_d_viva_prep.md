# Lane D — viva prep (T57)

*Draft for Kartiken (lane D). Each answer is short enough to say out loud, with the file to open
if the examiner pushes. The five questions named in `TASKS.md` T57 are marked ★ — know those
cold even though three belong to other lanes.*

---

## The five T57 questions

### ★ Why PGR, and not plain accuracy?
**Say:** Accuracy alone rewards a router that sends everything to the frontier model. PGR measures
how much of the *available* improvement over the local model the router actually captures,
$\text{PGR} = (Q(\pi) - Q_S)/(Q_L - Q_S)$, and we require $\ge 0.95$: the router may keep at most
5% of the gap it could have closed. That is the constraint the threshold is certified under
(Chapter 5).
**If pressed:** PGR is what makes "cheap" honest — a router that saves money by degrading answers
scores low on PGR regardless of cost.
**Source:** `notes/chapter_C_draft.md` §1, `notes/threshold_report.md`.

### ★ How does $\tau$ avoid leakage?
**Say:** $\tau^*$ is chosen on the **calibration** split only, with a bound (WSR) that already
accounts for scanning many thresholds; the test split is then looked at exactly once. Sweeping
$\tau$ on test is the "leaky" row in Chapter 5's table — 14.3% savings but uncertified — versus
**41.3% [39.0, 43.5]** certified.
**If pressed:** the seed-free bound (`select_tau(method="wsr_avg")`) averages terminal e-values
over 50 permutations, so the number doesn't depend on one lucky seed.
**Source:** `notes/t60_certification_report.md`, `scripts/17_t60_seed_free_wsr.py`.

### ★ Why the cascade (answer first, then decide)?
**Say:** A query-only router guesses from the question; a cascade *reads the small model's actual
answer* — agreement across 3 samples, log-probability, chosen-option margin — before deciding.
That is the signal the certified `cascade_t61` router uses, and it is why the demo computes
signals before the threshold instead of after.
**If pressed:** it costs local generation even for queries that end up going to the frontier —
that is the trade: ~1 s of Qwen to avoid a wrong route.
**Source:** `router/signals.py`, `notes/d15_demo_router.md`.

### ★ bf16 vs 4-bit — does quantisation break the router?
**Say:** Measured, not assumed: first 100 GSM8K + 100 MMLU test questions, Mac 4-bit MLX vs H100
bf16 — GSM8K 0.840 vs 0.860/0.870, MMLU 0.820 vs 0.790, i.e. −2 to −3 and +3 points with a
**±7-point** interval at $n=100$. No measurable gap, so the demo runs the model the router was
trained on.
**If pressed:** the Mac run is greedy, the H100 sampled at $T{=}0.7$ — a 500-question rerun would
tighten the interval if you want it.
**Source:** `notes/m2_checks.md` (T59), chapter 6.2.

### ★ What if the free tier disappears mid-defence?
**Say:** Offline mode: `serve_demo.py --offline` swaps the frontier client for a same-named
stand-in, so cache keys don't change — cached frontier answers stream normally, uncached ones
return a clear error instead of hanging. The cache is already filled: **68 frontier + 258 local**
answers, and **all 24 demo-set questions answer offline** (24/24, 0 errors, re-checked
2026-11-22).
**If pressed:** right now the key is free tier anyway (20 requests/day on 2026-11-21, still
5/min on 2026-11-22), so the *live* path throttles — that is why billing is on the list, and why
the offline path is the safe default for the video.
**Source:** `notes/demo_perf.md`, `artifacts/cache/demo.sqlite` counts, chapter 6.7.

---

## Your own architecture questions

### Why one lock? Won't that serialize the demo?
Yes — deliberately. One Mac, one GPU: concurrent requests queue rather than contend for the
accelerator. Router decisions take milliseconds and stay off the lock where the bundle is
query-only. In the soak the router was never the bottleneck: **p50 4 ms, p99 6 ms, max 17 ms**
against a 15 ms target.
**Source:** `router/serve.py`, `notes/demo_perf.md`.

### Why does local latency look like two different numbers?
Because there are two paths. A **cache hit** streams an answer already stored: first token
**6 ms** p50. A **live** call waits for the model: **1,102 ms** p50, 9,409 ms p99. The soak now
reports them separately (`done.cached` decides the bucket) instead of blending them into one
misleading median.
**Source:** `scripts/19_soak_demo.py`, `notes/demo_perf.md`.

### Why is 63.3% not the headline number?
It is **observed on demo traffic, not a certified figure**: one hour, one machine, 76% of the
answered queries served from cache, no WSR bound behind it. The certified saving for this router
is **41.3% [39.0, 43.5]** on the held-out test split. Three different savings numbers exist on
purpose — 41.3% certified, 37.3% replay, 63.3% observed — and they measure different things.
**Source:** `notes/t60_certification_report.md`, `notes/replay_report_d15.md`,
`notes/demo_perf.md`.

### Why does memory say 11 GB when RSS says 0.11 GB?
RSS counts only RAM pages; MLX's buffers live in `IOAccelerator` (GPU) memory, which RSS never
sees. `phys_footprint` (macOS `footprint -p`, `vmmap --summary` fallback) is the honest number:
**0.15 → 11.00 GB** over the soak hour, 10 GB of it dirty IOAccelerator across 5,193 regions.
It is not an unbounded leak — the next day it read 10.6 GB and fell to **2.8 GB after 30 further
local generations** — but 11 GB of 16 GB is a real capacity note, and capping MLX's GPU cache is
the fix.
**Source:** `scripts/19_soak_demo.py`, `notes/demo_perf.md`.

### Why exclude failed calls from spend and savings?
A 429 produced no answer and charged nothing. Counting it on either side would move the saving
without any query having been served. The report now computes spend, savings and latency from the
**553 answered** queries only and lists the **457 failures** separately; it also prints `/stats`
start/end/delta so the two can be compared.
**Source:** `scripts/19_soak_demo.py`, `notes/demo_perf.md`.

### Why does the demo send a free-form question to the frontier labelled "not certified"?
The certified signals exist for multiple choice (option probabilities) and arithmetic (sampled
agreement). Anything else — a prompt with no digit, a chat question — still gets three brief
answers and textual agreement, but the page labels it **not a certified question type**, because
$\tau^*$ was never fitted to that feature vector. The replay found exactly one such query in 200.
**Source:** `router/serve.py`, `notes/replay_report_d15.md`.

### Why is $\tau^* = 8.219$ here and 9.657 in an older note?
scikit-learn. The release router and 8.2187 came from **1.8.0**; a **1.9.1** refit reproduces the
pipeline but certifies **48.3% [46.1, 50.5]** instead of **41.3% [39.0, 43.5]** on the same data.
`requirements.txt` pins `scikit-learn==1.8.0` for exactly that reason, and a fresh clone
reproduces 8.2187 and 41.3% end to end.
**Source:** `notes/d15_demo_router.md`, README "Reproduce the headline and the demo".

### Why is the MCQ path a forward pass and not a chat?
Because the router's features are `p_chosen` and `margin` over the option letters — the same
forward pass Track B recorded on the H100. Generating a chat answer would give text the features
were never trained on, and would be slower than scoring four letters.
**Source:** `router/signals.py`, chapter 6.4.

### What did *you* own, versus the rest of the team?
Edge deployment of Qwen3.5-4B (4-bit MLX, quantisation and speed measurements), the H100 run that
produced Track B's Qwen answers, the serving stack, cache and offline mode, the web page, the
demo-set check, the soak and its instrumentation, the server-path replay, and the fresh-clone
reproducibility check. Dataset and release: Lane A. Features and the `cascade_t61` router:
Lane B. Thresholds and the certified $\tau^*$: Lane C.
**Source:** the Author Declaration block in `thesis/06_system.md`.

---

## Numbers to keep straight

| Number | Meaning | Source |
|---|---|---|
| 41.3% [39.0, 43.5] | certified test-split saving | `notes/t60_certification_report.md` |
| 48.3% [46.1, 50.5] | what a 1.9.1 refit certifies instead | `notes/d15_demo_router.md` |
| 37.3% [29.8, 44.8] | 200-query replay saving | `notes/replay_report_d15.md` |
| 63.3% | observed demo hour (not certified) | `notes/demo_perf.md` |
| 8.2187 / 8.219 | $\tau^*$ under sklearn 1.8.0 | `artifacts/bundle/router.pkl` |
| 9.657 | $\tau^*$ from the 1.9.1 bundle (older notes) | `notes/d15_demo_router.md` |
| 24/24 | demo set on the pinned bundle | `notes/demo_set_check_m2.md` |
| 200/200 | server decisions matching offline | `notes/replay_report_d15.md` |
| 29.1 / 23.4 tok/s | MLX CLI / end-to-end generation | `notes/m2_checks.md` |
| 327.5 s | H100 run for 22,350 GSM8K + 16,130 MCQ | `artifacts/trackb/gpu_manifest.json` |
| 11 GB → 2.8 GB | peak footprint, then after 30 more generations | `notes/demo_perf.md` |
| 20/day, 5/min | free-tier quota on this key | `notes/demo_perf.md` banner |
