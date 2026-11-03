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
| **All** | Read `notes/trackb_results.md`; reply yes/no on **D-14** (demo uses one Qwen answer + its confidence). Review PR #18. | D-14 decides how the demo router is built |
| **A** · Ani | Merge #17 once its chapter is fixed, and #18. Rewrite chapters 1–3 in your own words; add the Track B numbers to ch. 1. Re-run T45 when B's bundle exists. | Everyone builds on `main` |
| **B** · HSID | **T36 + T61**: the official cascade router for the Qwen pair, then **T41** the demo bundle. Then T51 (chapter). | The demo, T37 and T28 all wait on it |
| **C** · Sarthak | Fix `chapter_C_draft.md` (see PR #17 review) so #17 can merge. Then **T60** (tighter certification) and **T37** on the Qwen pair. | The headline number is his |
| **D** · Kartiken | **T40** (Gemini key), **T59** (4-bit Mac vs GPU labels, 200 questions), **T42/T43** first real demo run on the M2. Runbook: `notes/runbook_trackb.md`. | The demo is the one thing only the M2 can do |

---

## Phase 0 · Foundations

| ID | Owner | Task | Status |
|---|---|---|---|
| T01 | All | Agree the model pairs (D-05 to D-08) | ✅ |
| T02 | All | Review and merge the audit and loader PRs | ✅ |
| T03 | A | Loader: RouterBench → corpus + scores, GSM8K re-graded | ✅ |
| T04 | A | Near-duplicate grouping, both passes; **splits frozen** (D-11) | ✅ |
| T05 | C | `estimate_spend.py`: measured vs estimated speeds labelled | ✅ |
| T06 | D | `CODEOWNERS` + PR template ✅; branch protection on `main` ⬜ (Ani can do it in repo settings) | 🔶 |
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
| T26 | C | Leave-one-subject-out on MMLU, like-for-like (PR #17) | 🔶 merge after the chapter fix |
| T27 | C | Headroom sweep: savings vs gap across Mixtral, Yi-34B and Qwen | ⬜ |
| T28 | C | **Lock the numbers**: Qwen pair headline, Mixtral pair comparison, CIs, test once | ⬜ needs T37 |

## Track B · The Qwen pair (headline result)

| ID | Owner | Task | Status |
|---|---|---|---|
| T30 | D | `generate.py`: MLX + OpenAI-compatible clients, cache, resumable | ✅ |
| T31 | D | Batched decoding speed-up: M1 1.17×, M2 1.4× at batch 4 (`notes/m2_checks.md`); stays at batch 1 for log-probs. Re-check the M2's batch-1 speed (8.4 vs 28.5 tok/s) | ✅ |
| T32 | B | `mcq_logprob` scorer (bf16 fix applied) | ✅ |
| T33 | D | Qwen on GSM8K — **done on the H100 instead** (7,450 × 3 samples, D-13) | ✅ |
| T34 | D | Qwen on MMLU / ARC / Winogrande — **done on the H100** (16,130 questions) | ✅ |
| T35 | A | Track B scores aligned with GPT-4; release `trackb-6265adf7ffa4a602` | ✅ |
| T36 | B | Router for the Qwen pair: retrain the small-model head on Qwen's labels; calibrate on cal | ⬜ |
| T37 | C | τ\* for the Qwen pair (Learn-then-Test) and its Pareto curve, for the final router | ⬜ needs T36/T61 |
| **T59** | D | **4-bit vs bf16** on 200 test questions: no measurable gap (GSM8K −2, MMLU +3 points, ±7) — `notes/m2_checks.md` | ✅ |
| **T60** | C | **Tighter certification**: the WSR bound certifies 67% frontier calls where the test curve needs ~40%. Try per-dataset τ\*, a coarser grid, and report how much each recovers (cal only!) | ⬜ new |
| **T61** | B | **Cascade router (D-12, D-14)**: official version of `scripts/16_trackb_preview.py`. Two variants: k = 3 (agreement + log-prob, research table) and 1 answer + confidence (demo). Regularise or reduce F2 before combining (it hurt in the preview) | ⬜ new |

## Demo · The router live on a Mac

| ID | Owner | Task | Status |
|---|---|---|---|
| T40 | D | Gemini key in `.env`; model id `gemini-3-flash-preview` confirmed | ✅ |
| T41 | B | Final router bundle (cascade, D-14), `router.predict`; same decisions as evaluation on 1,000 test questions. `router/bundle.py` is a stand-in to replace | ⬜ needs T61 |
| T42 | D | `serve.py` running with Qwen (MLX) + Gemini on the M2: p99 7.6 ms with the interim bundle; re-run with B's final bundle (T41) | 🔶 |
| T43 | D | Web page with real models: both routes verified on the M2; check Safari too | 🔶 |
| T44 | C | Replay mode: 200 test questions through the live router, matches offline (PR #17) | 🔶 merge |
| T45 | A | Demo set (25 questions) checked against the final bundle (`scripts/check_demo_set.py`) | 🔶 draft; re-check with T41 |
| T46 | D | Offline mode: fill the cache with frontier answers for the demo and replay sets | 🔶 code exists |
| T47 | D | One-hour soak test on the M2: memory, first-token latency, throttling → `notes/demo_perf.md` | ⬜ |
| T48 | All | Backup video of the full demo | ⬜ |
| T49 | All | Two timed dry runs, one on a projector | ⬜ |

## Report and defence

| ID | Owner | Task | Status |
|---|---|---|---|
| T50 | A | Chapter 3, Data (`thesis/03_data.md`) | 🔶 draft; review + own wording |
| T51 | B | Chapter 4, Features, router, ablation, cascade | ⬜ |
| T52 | C | Chapter 5, Cost model, thresholds, guarantees, results (`notes/chapter_C_draft.md`) | 🔶 draft needs fixes |
| T53 | D | Chapter 6, Local inference, the GPU run, serving, demo performance | ⬜ |
| T54 | A | Ch. 1 intro + ch. 2 related work ✅ drafts; ch. 7–8 discussion and conclusion; merge all chapters | 🔶 |
| T55 | D | Reproducibility: one command per figure, tested on a fresh clone | ⬜ |
| T56 | All | Slides | ⬜ |
| T57 | All | Viva prep (why PGR, how τ avoids leakage, why the cascade, bf16 vs 4-bit, what if the free tier disappears) | ⬜ |
| T58 | A | Repo public (D-09), MIT licence (D-10) | ✅ |

---

## Order of work from here

```
Now          T36/T61 (B) · T26/T44 merge, T60 (C) · T40, T59 (D) · ch. 1–3 polish (A)
Next 1–2 wk  T37 → T41 → T42/T43 on the M2 → T45 re-check → T46
Then         T27, T28 (numbers locked) · T47 · chapters 4–6
Last         T48 video · T49 dry runs · T56 slides · T57 viva prep · T54 merge
```

**Critical path to the demo:** T61 (B) → T37 (C) → T41 (B) → T42/T43 on the M2 (D) → T45 (A) → T46 → T48.
