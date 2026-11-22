# Lane D — presentation slides (T56)

*Draft for Kartiken (lane D) to cut down and speak from. Every number carries the file it comes
from; keep the citation when you move a number onto a slide. Target: 10–12 minutes of speaking,
so ~12 slides — drop slides before you drop numbers.*

---

## Slide 1 — What I built: the demo that has to keep the certificate true

- The router is certified offline (Chapter 5); the demo must not break that meaning.
- Four requirements, all met (chapter 6.1):
  1. same decisions live as offline → same code path, same $\tau^*$;
  2. a real small model on the Mac → Qwen3.5-4B, the one the router was trained on;
  3. streaming answers → SSE, page never blank;
  4. offline mode → runs with the network off.
- **Show:** the browser at `127.0.0.1:8000` (Safari, checked 2026-11-18, `notes/m2_checks.md`).

## Slide 2 — The local arm: 4-bit Qwen3.5-4B on a 16 GB M2

- `mlx-community/Qwen3.5-4B-MLX-4bit`, **~3 GB** of a 16 GB machine (`configs/clients.yaml`).
- Does quantisation change the answers the router depends on? First 100 GSM8K + 100 MMLU test
  questions, Mac 4-bit vs H100 bf16 (`notes/m2_checks.md`, T59):

| | Mac 4-bit | H100 bf16 | Δ |
|---|--:|--:|--:|
| GSM8K | 0.840 | 0.860 (1 sample) / 0.870 (mean of 3) | −2 to −3 pts |
| MMLU | 0.820 | 0.790 | +3 pts |

- With $n=100$ per cell the interval is about **±7 points** → **no measurable quantisation gap**.

## Slide 3 — Speed, and why batch size stays 1

| Measurement (M2, 2026-11-15) | tok/s |
|---|--:|
| `mlx_lm.generate`, median of 3 | **29.1** |
| same, demo's own prompt, generation only | **29.5** (prompt 165) |
| `scripts/bench_batch.py --batches 1`, end to end | **23.4** |

- 110-token answer ≈ **5 s** end to end (`notes/m2_checks.md`).
- Batching is 1.4× faster at batch 4 (`notes/batch_speedup.json`) but the batch path does not
  record `mean_logprob`, which the cascade features are built from → **`max_batch: 1`**,
  trading 40% speed for the signal the certificate depends on.

## Slide 4 — The labels came from an H100, in the same shapes as serving

- 7,450 GSM8K × 3 samples + 16,130 MCQ, single **H100 80GB**, **327.5 s total**
  (234.4 s GSM8K, 34.0 s MCQ — `artifacts/trackb/gpu_manifest.json`).
- Same model one precision up (bf16, vLLM 0.32.0), same prompt builder, same sampling
  (3 × $T{=}0.7$, top-p 0.8, top-k 20), same MCQ forward pass → published as release
  **`trackb-6265adf7ffa4a602`** (23,580 rows).
- Why it matters: the Mac serves the distribution the router was trained on.

## Slide 5 — Serving: one process, one GPU, one lock

- `router/serve.py` + `scripts/serve_demo.py`; `Demo` has no HTTP in it, so the tests drive
  routing/streaming/totals directly.
- Endpoints: `/` · `/health` · `/route` · `/chat` · `/chat/stream` (SSE) · `/stats`.
- Cascade path: `status` → signals (MCQ one forward pass / 3 sampled answers) → `bundle.decide`
  with $\tau^*$ → stream local or frontier.
- One lock: concurrent requests queue instead of contending for the accelerator.

## Slide 6 — The cache is why the demo feels instant (and why offline works)

- SQLite, SHA-256 key over **(model, prompt, generation params, sample index)**.
- Write-through → a killed run resumes; a repeat query costs nothing.
- **Templates are part of the key** → editing a prompt template invalidates exactly the answers
  it affects.
- `--offline` swaps in a stand-in client of the **same name**, so keys don't change;
  `--fake` uses a separate file (bug found in T42: a fake run shared the real cache).

## Slide 7 — Money: list prices, not the free tier

- Frontier priced at the reference list rate (claude-sonnet-4.6: $3 / $15 per M) even though the
  demo calls Gemini, so "saved" is what the same traffic would have cost at list prices.
- Local answers count at **$0** (cost model M1, Chapter 5); the always-frontier counterfactual
  uses the shown local answer's length as its output estimate (`router/serve.py`).

## Slide 8 — The demo set: 24/24 on the certified bundle

- `python scripts/check_demo_set.py` → **24/24 as planned**, $\tau^* = 8.219$
  (`notes/demo_set_check_m2.md`, 2026-11-21).
- 9 easy (stay local) · **8 hard — hard because Qwen is unsure** (the case a router exists for) ·
  4 free-form · 3 chat.
- The router reads Qwen's *live* answers, so a route can move across $\tau^*$ between runs;
  spares live in `configs/demo_candidates.yaml`.

## Slide 9 — The server answers exactly like the evaluation (T44)

- 200 held-out test queries (seed 42) through `Demo._answer_local_first`, answers read from the
  release files, no live generation (`notes/replay_report_d15.md`):

| Check | Server path | Offline reference |
|---|--:|--:|
| Decision agreement | **200/200 (100%)** | 200/200 |
| Frontier escalation | 74.5% | 74.5% |
| Quality / PGR | 0.8600 / 1.111 | 0.8600 / 1.111 |
| Savings | **37.3%** ($0.1313 of $0.2093) | 37.3% [29.8, 44.8] |
| Router latency | p50 3.57 ms, p99 5.43 ms | < 15 ms target |

- 3 of 4,707 test queries can't be replayed: Track B stored option log-probabilities but no
  sampled answers for them.

## Slide 10 — One-hour soak (T47): what a defence-length session looks like

- 2026-11-21, `scripts/19_soak_demo.py`, `notes/demo_perf.md`:

| | |
|---|--:|
| Attempted → answered | 1,010 → **553** (local 263 / frontier 290) |
| Cache hits / live calls | 421 (76%) / 132 |
| Failed (free-tier HTTP 429) | 457 — **excluded** from spend and latency |
| First token | cache **6 ms** p50 · live **1,102 ms** p50, 9,409 ms p99 |
| Router decision | p50 4 ms, **p99 6 ms** (max 17) vs 15 ms target |
| Observed saving | **63.3%** — *observed on demo traffic, not a certified figure* |

- Certified number for the same router: **41.3% [39.0, 43.5]** on the held-out test split
  (`notes/t60_certification_report.md`).

## Slide 11 — Memory: RSS lies, so I sample physical footprint

- RSS stayed **0.08–0.24 GB** all hour — and missed everything MLX allocates.
- `phys_footprint` (`footprint -p`, `vmmap --summary` fallback): **0.15 → 11.00 GB**, of which
  **10 GB dirty `IOAccelerator` across 5,193 regions**.
- Not a leak: measured again the next day it was 10.6 GB and fell to **2.8 GB after 30 further
  local generations** (IOAccelerator 2,360 MB / 1,068 regions) — the GPU pool grows under
  sustained load and shrinks again. Still, 11 GB of 16 GB is a capacity note; the MLX cache cap
  is the fix.

## Slide 12 — Offline, and what is still open

- Cache after the soak: **68 frontier + 258 local** answers; **all 24 demo-set questions answer
  offline** (re-checked 2026-11-22: 24/24, 0 errors).
- Reproducibility (T55): fresh clone, Python 3.13.2, **scikit-learn 1.8.0** pinned →
  $\tau^*$ **8.21873510404553**, T41 100%, and `17_t60_seed_free_wsr.py` reproduces
  **41.3% [39.0, 43.5]** (README, "Reproduce the headline and the demo").
- Open: Gemini key is still free tier (20/day, then 5/min) → the throttling-free soak re-run and
  the live video wait on billing; everything else in the demo runs from cache.

---

### Numbers you must not mix up on stage

| Figure | What it is | Source |
|---|---|---|
| **41.3% [39.0, 43.5]** | certified test-split saving (WSR) | `notes/t60_certification_report.md` |
| **37.3%** | 200-query replay, stored answers, no live calls | `notes/replay_report_d15.md` |
| **63.3%** | one observed hour of demo traffic, 76% cache hits | `notes/demo_perf.md` |
| **24.1%** | unit-cost saving in the bundle's own `meta["test"]` | `artifacts/bundle/router.pkl` |
