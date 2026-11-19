# Chapter 6 — Local Inference, the GPU Run, Serving and Demo Performance

> **DRAFT for Kartiken (lane D, task T53).** Written from `notes/m2_checks.md`,
> `notes/runbook_trackb.md`, `gpu/README.md`, `artifacts/trackb/gpu_manifest.json`,
> `router/serve.py` and the T42/T45/T47 run logs. Every number names the script or note it comes
> from. Rewrite it in your own voice and check UPES's rules on declaring AI assistance before
> submission.

## 6.1 What the system has to do

Chapters 3 to 5 build a router that is certified offline: a threshold $\tau^*$ picked on a held-out
calibration split, a guarantee that survives one look at the test split. A demo has to keep that
meaning while everything else changes. The decisions must come from the same code path the
certification used, the small model must be the one the router was trained on, and the whole thing
must run on one laptop in front of an audience, with no network if necessary.

Four requirements follow:

1. **The same decisions, live.** The server must compute the router's features exactly as Track B
   did offline, or the certified $\tau^*$ would be applied to a distribution it never saw.
2. **A real small model on the Mac.** Not a stand-in: Qwen3.5-4B, the model whose answers the
   router's small-model head was trained on (D-07).
3. **Streaming answers.** A demo audience watches text appear; the page must never look stuck while
   the local model works.
4. **An offline mode.** The free tier of a frontier API can disappear during a defence (a viva
   question in `TASKS.md` T57); the demo must run from a cache with the network off.

The rest of this chapter describes how each was met: the local arm in 6.2, the GPU run that produced
the training labels in 6.3, the server in 6.4, the page in 6.5, measured performance in 6.6,
offline mode in 6.7, and what still limits the result in 6.8.

## 6.2 The local arm: Qwen3.5-4B in 4-bit on Apple Silicon

The demo's small model is **Qwen3.5-4B**, run through **MLX** (`mlx-lm`) on the team's 16 GB M2.
Three build decisions matter.

**4-bit weights.** The 4-bit build (`mlx-community/Qwen3.5-4B-MLX-4bit`, `configs/clients.yaml`)
needs about 3 GB, leaving room for the OS and the router's own data on a 16 GB machine. The question
this raises is whether quantisation changes the answers the router depends on. T59
(`notes/m2_checks.md`) answered it on the first 100 GSM8K and 100 MMLU test questions, scored on the
Mac (greedy, $\le$ 512 tokens) against the H100's bf16 answers from the release:

| | Mac: 4-bit MLX | H100: bf16 vLLM | Difference |
|---|--:|--:|--:|
| GSM8K accuracy | 0.840 | 0.860 (first sample) / 0.870 (mean of 3) | −2 to −3 points |
| MMLU accuracy | 0.820 | 0.790 | +3 points |

With 100 questions per cell the 95% interval is about $\pm 7$ points, so there is **no measurable
quantisation gap**: the demo runs, within noise, the model the router was trained on. The Mac run is
greedy where the H100 sampled at $T = 0.7$; a 500-question rerun would tighten the interval if a
viva question demands it.

**Speed.** Generation speed decides how long the audience waits. Re-measured on the M2 on
2026-11-15 (`notes/m2_checks.md`, `mlx-lm` 0.32.2):

| Measurement | Result |
|---|--:|
| `mlx_lm.generate` CLI, median of 3 | **29.1 tok/s** |
| CLI on the demo's own chat-templated prompt, generation only | **29.5 tok/s** (prompt processing 165 tok/s) |
| `scripts/bench_batch.py --batches 1`, same code path as the demo | **23.4 tok/s** end to end |

A 110-token answer therefore takes about 5 s end to end, not the ~13 s an earlier run suggested.
Cost model M2 keeps `tokens_per_sec: 28.5` for generation (`configs/models.yaml`) and uses ~23 tok/s
only when pricing whole-call wall time.

**Batch size stays 1.** Batched decoding is faster — 1.4× at batch 4 on the M2
(`notes/batch_speedup.json`) — but the batch path does not record `mean_logprob`, and the cascade
router's features are built from it (`router/signals.py`). The client therefore keeps `max_batch: 1`
and trades 40% speed for the signal the certification depends on.

## 6.3 The GPU run that produced the labels

The router's labels for the Qwen pair (Track B) are Qwen's own answers, so somebody has to answer
7,450 GSM8K questions three times and score 16,130 multiple-choice questions. The M1 managed about
11 tok/s — days of runs (D-13). The run was moved to a single **NVIDIA H100 80GB** on the UPES
cluster (`gpu/README.md`, decision D-13) and finished in **327.5 s total**: 234.4 s for 22,350
GSM8K answers and 34.0 s for the multiple choice
(`artifacts/trackb/gpu_manifest.json`).

The GPU run deliberately mirrors what the Mac will do, because the router learns from one and
serves the other:

- **Same model, one precision up:** Qwen3.5-4B in bf16, thinking mode off, `vLLM` 0.32.0
  (`gpu_manifest.json`).
- **Same prompts:** the GSM8K prompt is `router.generate.build_prompt`, byte-identical to the
  demo's, so an answer's tokens mean the same thing in training and serving.
- **Same sampling:** three answers per question at temperature 0.7, top-p 0.8, top-k 20, up to
  1,024 tokens — the settings `serve_demo.py` reuses from the bundle's metadata, so the agreement
  signal has the distribution it was certified on.
- **Same scoring for multiple choice:** one forward pass over the option letters, recorded as
  `p_chosen` and `margin` (`scripts/gpu_trackb_vllm.py` → `qwen_mcq_options.parquet`).

The job ran with `bash gpu/run_trackb.sh` (or `sbatch gpu/run_trackb.sbatch`), and its outputs were
published as release `trackb-6265adf7ffa4a602`, which every later script reads
(`scripts/verify_data.py` checks the fingerprints). T59 above is the closing check that the Mac's
4-bit build of the same model is close enough to the H100's bf16 build for the demo to run.

## 6.4 Serving: one process, one GPU, one lock

The server is `router/serve.py`, launched by `scripts/serve_demo.py`. It is deliberately small.

**Shape.** A `Demo` object holds the router bundle, the two model clients, the answer cache and the
running totals; `create_app` wraps it in FastAPI. The `Demo` class knows nothing about HTTP, so the
tests drive routing, streaming and totals directly without a server.

**Endpoints** (`router/serve.py`):

| Endpoint | Returns |
|---|---|
| `GET /` | the page (`router/static/index.html`) |
| `GET /health` | model names, $\tau^*$, rule, offline flag, bundle metadata |
| `POST /route` | the routing decision only, no generation |
| `POST /chat` | decision + full answer + cost + running totals |
| `GET /chat/stream` | server-sent events: `status`, `route`, `token`, `done` (or `error`) |
| `GET /stats` | queries, local vs frontier, spend, savings, router p50/p99 |

**The cascade path.** For the demo bundle (D-15, `cascade_t61_3sample`), every query follows the
order the certification used:

1. emit `status` — "qwen3.5-4b is answering on this Mac…" — so the page is never blank while the
   local model works;
2. compute the signals: **multiple choice** is one forward pass over the option letters with the
   question in RouterBench's MMLU format; **arithmetic** is three sampled answers with the Track B
   prompt and sampling, reduced to agreement and log-probability features; **anything else** gets
   three brief answers and textual agreement, and is labelled *not a certified question type* on
   the page;
3. `bundle.decide` applies the featurizer, the gradient-boosted trees and $\tau^*$ (rule v2), and
   the page draws $\hat s_S$, $\hat s_L$ and $\hat\Delta$ against the threshold;
4. if the decision is *local*, the answer that produced the signals is streamed back; if
   *frontier*, the query goes to the API and its answer is streamed.

**One lock.** Model calls are serialised with a single lock ("one Mac, one GPU"). Two concurrent
requests queue rather than contend for the accelerator; router decisions themselves take
milliseconds and stay off the lock where the bundle is query-only.

**The cache.** `router/generate.py` stores every generation in SQLite under a SHA-256 key of
(model name, final prompt text, generation parameters, sample index). Consequences that the demo
relies on:

- *Write-through:* each answer is committed as soon as it arrives, so a killed run resumes and a
  repeated query costs nothing — the demo's latency after the first answer is a cache lookup.
- *Templates are part of the key:* editing `PROMPT_TEMPLATES` invalidates exactly the answers it
  affects, so stale text can never be served under a new prompt.
- *Offline reuses the keys:* `--offline` replaces the frontier client with a stand-in of the **same
  name**, so cache keys do not change and a filled cache answers the same queries with the network
  off. `--fake` uses a separate file (`demo_fake.sqlite`) because a bug reported in T42 let a fake
  run share the real run's cache.

**Money.** The frontier arm is priced at the reference list rate (`PriceTable`, default
claude-sonnet-4.6) even though the demo calls Gemini's free tier, so "saved" is what the same
traffic would have cost at list prices; local answers count at $0 (cost model M1), and the
always-frontier counterfactual uses the local answer's length as its output-length estimate
(`router/serve.py` docstring).

## 6.5 The web page

The page is one file of vanilla HTML, CSS and JavaScript (`router/static/index.html`) — no build
step, nothing to break offline. It opens an `EventSource` on `/chat/stream` and reacts to four
events: `status` fills the decision line while Qwen is still answering (so the page is never blank
during the 3-sample arithmetic path); `route` draws the badge, the signal behind the decision and
the three bars against $\tau^*$; `token` appends text as it streams; `done` records first-token and
total time and refreshes the sidebar totals. The sidebar prices every query and keeps a running
"saved vs always-frontier" figure.

The page was checked in **Safari on the M2** (T43): it loads with `local: qwen3.5-4b · frontier:
gemini-3-flash · τ* = 9.657 (rule v2)` in the header, both routes were exercised end to end — a
multiple-choice question answered locally and an arithmetic question escalated to Gemini — and the
`status` message appears before the first token in both.

## 6.6 Demo performance (T42, T47)

**Router latency.** The routing decision itself is not the bottleneck: with the interim bundle the
server measured p99 **7.6 ms** against a 15 ms target (`notes/m2_checks.md`, T42). The D-15 bundle
adds option-scoring or three sampled answers *before* the decision, so its wait is dominated by
generation, not by the model's inference on the features.

**One-hour soak (T47).** `scripts/19_soak_demo.py` drives the running server for an hour: every
`configs/demo_set.yaml` query first, then 200 replay-set test questions (seed 42), then a shuffled
mix, while a background thread samples the process's memory and `/stats` every 30 s. The run also
fills T46's cache, because each frontier answer that arrives lands under the keys `--offline` will
look up. The run of 2026-11-19 (14:22–15:22 IST, `notes/demo_perf.md`) completed **929 queries**
(about 15/min): 299 answered locally and 198 by the frontier, with **432 HTTP 429 errors** once the
Gemini free tier's 20/day cap was reached. The driver recorded those errors and kept going rather
than aborting, so the run measures throttling as well as the happy path.

| Measured over the hour | p50 | p95 | p99 |
|---|--:|--:|--:|
| First token, all successful queries | 7 ms | 4,737 ms | 7,321 ms |
| First token, local route | 799 ms | 2,478 ms | 5,572 ms |
| Router decision (server) | 3 ms | 5 ms | 6 ms |

The router itself stays well under the 15 ms target (p99 6 ms, max 17 ms); a user's wait is the
local model's three samples, which stream only after Qwen has finished them (6.5). The first query
of the run took 3,150 ms, and the five slowest were all frontier calls that ended in 429 after
roughly 15 s. RSS stayed at **0.11 GB from start to end** (min 0.06, max 0.13 GB across 119 samples),
so there is no Python-side leak over a defence-length session. Server `/stats` at the end reported
990 queries, p99 router latency 6.3 ms, and 73% saved against always-frontier list prices.

**Demo set (T45).** `scripts/check_demo_set.py` routes every planned demo query through the same
code path as the server. On the M2 with the D-15 bundle the check passes **21/24**
(`notes/demo_set_check_m2.md`): two easy questions Qwen answers correctly (3/3 agreement) land just
above the refit's $\tau^*$, and one question the H100's Qwen got wrong the Mac's Qwen gets right
with 3/3 agreement, so the router keeps it local. All three differences come from live sampling and
the scikit-learn refit discussed in `notes/d15_demo_router.md`; the demo set's spare questions
(`configs/demo_candidates.yaml`) exist to swap them out on stage.

## 6.7 Offline mode (T46)

`python scripts/serve_demo.py --offline` starts the same server with the frontier client replaced by
a same-named stand-in: a cached frontier answer streams normally, and an uncached one returns a
clear "offline mode: this frontier answer is not cached" error instead of a silent hang. The cache
is filled by running the demo set and the replay set once against the live server — exactly what
the T47 soak does — so the fill is a side effect of a test the project already needed. After the
2026-11-19 soak the cache holds 48 frontier and 234 local answers, and **all 24 demo-set questions
answer offline** (checked through `/chat` against the offline server). Replay-set questions whose
frontier call was throttled were not cached, so an offline replay must skip those.

## 6.8 Limitations

- **The free tier is a dependency.** The demo calls Gemini's free tier; the T47 soak hit its 20/day
  cap and recorded 432 HTTP 429 errors within the hour. If it disappears or tightens mid-defence,
  the offline mode of 6.7 is the fallback, and the cache must have been filled beforehand.
- **The demo set moves with the sampler.** The router reads Qwen's live answers, so a question's
  route can differ from the check's (21/24 on this machine). The check is re-run before every demo,
  and the spares are verified on the demo machine before they are added.
- **The bundle is tied to its scikit-learn version.** A refit on another machine reproduces the
  pipeline but not the exact predictions (`notes/d15_demo_router.md`); the demo bundle is built
  once, on the M2, and shipped as a pickle.
- **The soak is one hour, not one day.** It covers the demo's traffic shape — a few hundred queries,
  mixed local and frontier — but not multi-day drift; memory over the hour is the leak check that
  matters for a defence-length session.

## Reproducing this chapter

```bash
# local arm (M2)
python scripts/bench_batch.py --client qwen3.5-4b --batches 1 2 4 8 --n 16   # speed
python scripts/14_score_mcq.py --sources mmlu --limit 50                      # 4-bit smoke

# labels (H100)
bash gpu/run_trackb.sh --limit 20     # smoke; then the full run, ~5.5 min
python scripts/10_build_trackb.py --responses artifacts/trackb/qwen_responses.parquet \
                                  --mcq artifacts/trackb/qwen_mcq_scores.parquet

# serving and demo
python scripts/11_build_bundle.py --mode cascade_t61 --samples 3             # the D-15 bundle
python scripts/check_demo_set.py --out-md notes/demo_set_check_m2.md          # 21/24 on the M2
python scripts/serve_demo.py                                                 # live, both routes
caffeinate -i python scripts/19_soak_demo.py --minutes 60                    # T46 fill + T47 soak
python scripts/serve_demo.py --offline                                       # same demo, no network
```
