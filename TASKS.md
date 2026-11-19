# Every task, from here to the defence

The project ends with **the router working live on a Mac**. This file lists every task with its
status. Updated 2026-11-01, after Track B (Qwen's answers) came back from the H100.

Owners: **A** Data (Ani) · **B** Features & router (HSID007) · **C** Evaluation & statistics
(Sarthak, SjScripter) · **D** Systems & demo (kartiken07, has the M2). **All** = whole team.
IDs are stable; reference them in branch names and PRs (e.g. `b/T36-qwen-head`).
Status: ✅ done · 🔶 in progress / needs a fix · ⬜ not started.

---

## Where we are

- **Data:** frozen (D-11). Track A = RouterBench (Mixtral, Yi-34B vs GPT-4); Track B = Qwen3.5-4B vs GPT-4,
  generated on a UPES H100 (D-13).
- **Headline so far (preview, `notes/trackb_results.md`):** a cascade router on **Qwen's own confidence**
  saves **47.5% [45.4, 49.7] with the guarantee** (PGR ≥ 0.95, τ\* from cal, test once), about **1.9× cheaper**.
  Question-only routing saves 9.3%. Test curve suggests ~78% is reachable if the bound gets tighter.
- **Demo:** server, web page and offline mode exist but have never run with the real models.

```bash
# Data (both tracks)
gh release download data-910a584c8279baa7 -R AnirvanRanjan/cost-aware-router -D artifacts/routerbench \
  -p corpus.parquet -p scores.parquet -p dedup_pairs.parquet -p manifest.json -p SHA256SUMS
gh release download data-910a584c8279baa7 -R AnirvanRanjan/cost-aware-router -D artifacts/features -p features.parquet
gh release download trackb-6265adf7ffa4a602 -R AnirvanRanjan/cost-aware-router -D artifacts/trackb
python scripts/verify_data.py        # must print "Same data as everyone else."
```

## This week

| Who | Do this now | Why it can't wait |
|---|---|---|
| **All** | Vote on **D-15** (demo uses the T61 3-answer router; `notes/d15_demo_router.md`). Everyone: read ch. 1, 7, 8 for anything wrong about your lane. | D-15 decides what the M2 runs |
| **A** · Ani | Push; run T45 run 3 on the Mac with the D-15 bundle (`python scripts/11_build_bundle.py --mode cascade_t61 --samples 3`, then `python scripts/check_demo_set.py --out-md notes/demo_set_check_run3.md`). Rewrite ch. 1–3, 7–8 in your own words. | The demo set must pass before the video |
| **B** · HSID | Review D-15: the demo now rebuilds your T61 `cascade_3sample` router (`--mode cascade_t61`, `router/signals.py`, tested against `16_trackb_preview.py`); your `--mode cascade` bundle stays as the D-14 comparison. Rewrite ch. 4 in your own words. | The demo router is yours |
| **C** · Sarthak | Seed-free bound is now `select_tau(method="wsr_avg")` in `router/threshold.py` (ported from your script 17, identical p-values): check it. **T28**: lock numbers from the release file and record the scikit-learn version (a refit certifies 48.3%, the release file 41.3%: `notes/d15_demo_router.md`). Move `chapter_C_draft.md` to `thesis/05_results.md`. | The headline number is his |
| **D** · Kartiken | Soak ✅, offline ✅, Safari ✅, sklearn 1.8.0 pinned on the M2 (tau* 8.219) ✅, demo set 24/24 on the M2 ✅. **Next:** turn on billing for the Gemini key and confirm; **T48** backup video; T55 fresh-clone check; ch. 6 in your own words. | The video is the critical path |

---

## Phase 0 · Foundations

| ID | Owner | Task | Status |
|---|---|---|---|
| T01 | All | Agree the model pairs (D-05 to D-08) | ✅ |
| T02 | All | Review and merge the audit and loader PRs | ✅ |
| T03 | A | Loader: RouterBench → corpus + scores, GSM8K re-graded | ✅ |
| T04 | A | Near-duplicate grouping, both passes; **splits frozen** (D-11) | ✅ |
| T05 | C | `estimate_spend.py`: measured vs estimated speeds labelled | ✅ |
| T06 | D | `CODEOWNERS` + PR template ✅; branch protection on `main` ✅ (2026-11-19: no force-pushes, no deletion) | ✅ |
| T07 | A | Gold answer letters for multiple choice | ✅ |

## Track A · Research result on RouterBench (Mixtral pair)

| ID | Owner | Task | Status |
|---|---|---|---|
| T10 | B | F0 heuristic features | ✅ |
| T11 | B | F1 TF-IDF features, fit on train only | ✅ |
| T12 | B | F2 `bge-base` embeddings (in the data release) | ✅ |
| T13 | B | Baselines: LogReg on F0/F1, XGBoost on F0+F2 | ✅ |
| T14 | B | Two-head MLP, 5 seeds, train predictions saved | ✅ |
| T15 | B | Isotonic calibration per head, reliability diagrams, ECE | ✅ |
| T16 | B | Ablation ladder (DESIGN §4.3) on frozen splits | ✅ |
| T20 | C | `eval.py`: curves, PGR, APGR, random and oracle | ✅ |
| T21 | C | `threshold.py`: exact sweep + concave envelope | ✅ |
| T22 | C | Bounds: Hoeffding, WSR (default), paired bootstrap | ✅ |
| T23 | C | Learn-then-Test τ\* on cal only | ✅ |
| T24 | C | Cost model M2 + Batch-API baseline | ✅ |
| T25 | C | `plot_pareto.py` panels A–D | ✅ |
| T26 | C | Leave-one-subject-out on MMLU, like-for-like (PR #17) | ✅ merged |
| T27 | C | Headroom sweep: savings vs gap across Mixtral, Yi-34B and Qwen (`notes/t27_t28_locked_numbers.md`) | ✅ |
| T28 | C | **Lock the numbers**: Qwen pair headline, Mixtral pair comparison, CIs, test once | 🔶 preview 41.3% [39.0, 43.5]; lock once predictions are in the release |

## Track B · The Qwen pair (headline result)

| ID | Owner | Task | Status |
|---|---|---|---|
| T30 | D | `generate.py`: MLX + OpenAI-compatible clients, cache, resumable | ✅ |
| T31 | D | Batched decoding speed-up: M1 1.17×, M2 1.4× at batch 4 (`notes/m2_checks.md`); stays at batch 1 for log-probs. Batch-1 speed re-checked: 29.1 tok/s (CLI), 23.4 end to end; the 8.4 did not reproduce | ✅ |
| T32 | B | `mcq_logprob` scorer (bf16 fix applied) | ✅ |
| T33 | D | Qwen on GSM8K — **done on the H100 instead** (7,450 × 3 samples, D-13) | ✅ |
| T34 | D | Qwen on MMLU / ARC / Winogrande — **done on the H100** (16,130 questions) | ✅ |
| T35 | A | Track B scores aligned with GPT-4; release `trackb-6265adf7ffa4a602` | ✅ |
| T36 | B | Router for the Qwen pair: retrain the small-model head on Qwen's labels; calibrate on cal (`notes/qwen_query_report.md`) | ✅ |
| T37 | C | τ\* for the Qwen pair (Learn-then-Test) and its Pareto curve, for the final router | 🔶 preview τ\* = 8.22 on `cascade_3sample` |
| **T59** | D | **4-bit vs bf16** on 200 test questions: no measurable gap (GSM8K −2, MMLU +3 points, ±7) — `notes/m2_checks.md` | ✅ |
| **T60** | C | **Tighter certification**: seed-free WSR (mean terminal e-value over 50 orderings) — `notes/t60_certification_report.md` | ✅ |
| **T61** | B | **Cascade router (D-12, D-14)**: official version of `scripts/16_trackb_preview.py`. Two variants: k = 3 (agreement + log-prob, research table) and 1 answer + confidence (demo). Regularise or reduce F2 before combining (it hurt in the preview). `notes/trackb_cascade_report.md` | ✅ (predictions uploaded to release `trackb-6265adf7ffa4a602`) |

## Demo · The router live on a Mac

| ID | Owner | Task | Status |
|---|---|---|---|
| T40 | D | Gemini key in `.env`; model id `gemini-3-flash-preview` confirmed | ✅ |
| T41 | B | Final router bundle (cascade, D-14), `router.predict`; same decisions as evaluation on 1,000 test questions. `router/bundle.py` is a stand-in to replace | ✅ bundle built (`artifacts/bundle/router.pkl`), 100% agreement on 1,000 test queries; `serve.py` cascade integration complete |
| T42 | D | `serve.py` running with Qwen (MLX) + Gemini on the M2: p99 7.6 ms with the interim bundle; re-run with B's final bundle (T41) | ✅ D-15 bundle serving (tau*=9.657, T41 agreement 1.0); router p99 6 ms, max 17 ms in the T47 soak (`notes/demo_perf.md`) |
| T43 | D | Web page with real models: both routes verified on the M2; check Safari too | ✅ both routes over SSE; Safari renders and populates (2026-11-18) |
| T44 | C | Replay mode: 200 test questions through the live router, matches offline (PR #17) | ✅ merged; re-run with T41 |
| T45 | A | Demo set checked against the final bundle (`scripts/check_demo_set.py`) | ✅ 24/24 on the M2 (sklearn 1.8.0, tau* 8.219, `notes/demo_set_check_m2.md`); 9 easy, 8 hard (escalated because Qwen is unsure), 4 free-form, 3 chat |
| T46 | D | Offline mode: fill the cache with frontier answers for the demo and replay sets | ✅ after the 2026-11-19 soak (48 frontier + 234 local rows): all 24 demo-set questions answer with `--offline`; replay queries throttled by the 20/day cap were not cached |
| T47 | D | One-hour soak test on the M2: memory, first-token latency, throttling → `notes/demo_perf.md` | ✅ 2026-11-19 14:22–15:22 IST: 929 queries, 432 HTTP 429s, router p99 6 ms, RSS 0.11→0.11 GB (`notes/demo_perf.md`) |
| T48 | All | Backup video of the full demo | ⬜ |
| T49 | All | Two timed dry runs, one on a projector | ⬜ |

## Report and defence

| ID | Owner | Task | Status |
|---|---|---|---|
| T50 | A | Chapter 3, Data (`thesis/03_data.md`) | 🔶 draft; review + own wording |
| T51 | B | Chapter 4, Features, router, ablation, cascade | ✅ draft in `thesis/04_router.md` |
| T52 | C | Chapter 5, Cost model, thresholds, guarantees, results (`notes/chapter_C_draft.md`) | 🔶 draft needs fixes |
| T53 | D | Chapter 6, Local inference, the GPU run, serving, demo performance | 🔶 draft in `thesis/06_system.md` with T47 numbers and the offline check; needs Kartiken's own wording |
| T54 | A | Ch. 1, 2, 7, 8 drafts ✅ (ch. 1 has the Qwen preview numbers); own wording; merge all chapters | 🔶 |
| T55 | D | Reproducibility: one command per figure, tested on a fresh clone. scikit-learn pinned to 1.8.0 in `requirements.txt`: the release router and tau* 8.2187 came from 1.8.0; a 1.9.1 refit certifies 48.3% vs 41.3% (`notes/d15_demo_router.md`) | 🔶 version pinned |
| T56 | All | Slides | ⬜ |
| T57 | All | Viva prep (why PGR, how τ avoids leakage, why the cascade, bf16 vs 4-bit, what if the free tier disappears) | ⬜ |
| T58 | A | Repo public (D-09), MIT licence (D-10) | ✅ |

---

## Order of work from here

```
Now          T41 + serve.py cascade (B, D) · predictions to release (B) · T28 lock (C) · tok/s (D) · own wording (A)
Next 1–2 wk  T37 → T41 → T42/T43 on the M2 → T45 re-check → T46
Then         T27, T28 (numbers locked) · T47 · chapters 4–6
Last         T48 video · T49 dry runs · T56 slides · T57 viva prep · T54 merge
```

**Critical path to the demo:** T61 ✅ → T41 + `serve.py` cascade (B, D) → T42/T43 on the M2 (D) → T45 (A) → T46 → T48.
