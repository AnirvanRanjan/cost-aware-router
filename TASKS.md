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
| **All** | Final numbers locked 2026-11-23 (41.3% M1 / 28.4% M2 / 29.5% one-answer / 2.1% question-only). Each lane: cut your slides to 3–4, finish viva notes, send dry-run dates. | Presentation is the critical path |
| **A** · Ani | Confirm the ch. 1 AI declaration; get the UPES template, page limit and deadline from the guide; assemble the combined deck; merge chapters into the template (T54). | The template blocks the final thesis |
| **B** · HSID | Ch. 4 merged. Declaration: say AI helped *draft and refine* (as ch. 5); own-words pass; slides to 3–4. | — |
| **C** · Sarthak | Ch. 5 merged with the locked numbers (M2 28.4%). Draft your 3–4 slides and viva notes; check the cost-model attribution with Ani (costs.py is lane A's code). | — |
| **D** · Kartiken | Backup video ✅ (offline). Own-words pass on ch. 6; re-record only if the new demo page (`a/frontend`) is merged; T47 stays free-tier throttled unless billing is enabled. | — |

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
| T28 | C | **Lock the numbers**: Qwen pair headline, Mixtral pair comparison, CIs, test once | ✅ locked 2026-11-23 from the release file: 41.3% [39.0, 43.5] M1, 28.4% [26.1, 30.6] M2 (all local answers billed), τ\* 8.2187 (`notes/t27_t28_locked_numbers.md`, ch. 5 Table 5.1) |

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
| T37 | C | τ\* for the Qwen pair (Learn-then-Test) and its Pareto curve, for the final router | ✅ τ\* = 8.2187 (seed-free WSR) |
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
| T44 | C | Replay mode: 200 test questions through the live router, matches offline (PR #17) | ✅ merged; re-run by D with the pinned T41 cascade bundle (2026-11-20): 200/200 decisions match offline, 37.3% savings, `notes/replay_report_d15.md` |
| T45 | A | Demo set checked against the final bundle (`scripts/check_demo_set.py`) | ✅ 24/24 on the M2 (sklearn 1.8.0, tau* 8.219, `notes/demo_set_check_m2.md`); 9 easy, 8 hard (escalated because Qwen is unsure), 4 free-form, 3 chat |
| T46 | D | Offline mode: fill the cache with frontier answers for the demo and replay sets | ✅ after the 2026-11-19 soak (48 frontier + 234 local rows): all 24 demo-set questions answer with `--offline`; replay queries throttled by the 20/day cap were not cached |
| T47 | D | One-hour soak test on the M2: memory, first-token latency, throttling → `notes/demo_perf.md` | 🔶 2026-11-21 re-run with the fixed script: 1010 queries, 553 answered, cache/live split, phys_footprint 0.15→11.00 GB, savings **63.3%** (answered only) — but the frontier arm was still free-tier capped (**457× HTTP 429**; `quotaId` FreeTier, 20/day on 11-21, 5/min on 11-22) so **billing is still off**; clean re-run pending (`notes/demo_perf.md`) |
| T48 | All | Backup video of the full demo | ✅ recorded offline on the M2 (2026-11-23); keep a copy in the shared drive and on a USB stick |
| T49 | All | Two timed dry runs, one on a projector | ⬜ |

## Report and defence

| ID | Owner | Task | Status |
|---|---|---|---|
| T50 | A | Chapter 3, Data (`thesis/03_data.md`) | ✅ own wording (2026-11-19) |
| T51 | B | Chapter 4, Features, router, ablation, cascade | ✅ merged; declaration wording to fix |
| T52 | C | Chapter 5, Cost model, thresholds, guarantees, results (`thesis/05_results.md`) | ✅ merged with locked numbers |
| T53 | D | Chapter 6, Local inference, the GPU run, serving, demo performance | 🔶 draft in `thesis/06_system.md`; facts updated 2026-11-21 (demo set 24/24, T44 replay, sklearn pin, soak marked PENDING T47 re-run, chapter commands re-run); needs Kartiken's own wording |
| T54 | A | Ch. 1, 2, 3, 7, 8 in own words ✅, final numbers ✅; merge all chapters into the UPES template | 🔶 waits for the template |
| T55 | D | Reproducibility: one command per figure, tested on a fresh clone. scikit-learn pinned to 1.8.0 in `requirements.txt`: the release router and tau* 8.2187 came from 1.8.0; a 1.9.1 refit certifies 48.3% vs 41.3% (`notes/d15_demo_router.md`) | ✅ fresh-clone repro verified 2026-11-21 (Python 3.13.2, sklearn 1.8.0): bundle tau* 8.21873510404553, T41 100%, `17_t60_seed_free_wsr` 41.3% [39.0, 43.5], demo set 24/24 — README "Reproduce the headline and the demo" |
| T56 | All | Slides | 🔶 lane D draft ready: `notes/lane_d_presentation_slides.md` (12 slides, every number sourced) — Kartiken to cut and speak |
| T57 | All | Viva prep (why PGR, how τ avoids leakage, why the cascade, bf16 vs 4-bit, what if the free tier disappears) | 🔶 lane D draft ready: `notes/lane_d_viva_prep.md` (all five T57 questions + serving/memory/number-confusion set); lane A: `notes/lane_a_viva_prep.md` |
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
