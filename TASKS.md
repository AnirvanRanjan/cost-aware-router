# Every task, from here to the defence

The project ends with **the router working live on a Mac**. That one requirement reshapes the plan:
the demo can't use Mixtral or GPT-4 (neither runs on a laptop, and GPT-4 costs money), so it needs
its own model pair, and a router trained for that pair. This file lists every remaining task, in order.

Owners: **A** Data (Ani) · **B** Features & router · **C** Evaluation & statistics ·
**D** Systems & demo (kartiken07, has the M2). **All** = whole team. IDs are stable; reference them
in branch names and PRs (e.g. `b/T12-embeddings`).

---

## This week — start here

Done already: the code foundation, the RouterBench audit, the loader with re-graded GSM8K (T03),
near-duplicate grouping (T04), and kartiken07's measured M2 speeds. The data is a GitHub release:

```bash
gh release download data-d9ef5869b99739aa -R AnirvanRanjan/cost-aware-router -D artifacts/routerbench
python scripts/verify_data.py        # must print "Same data as everyone else."
```

**Everyone, first:** read `DECISIONS.md` (D-05 to D-08 are new) and approve or object on PR #4.
Then review the open PRs: #3 audit, #4 loader + decisions, the dedup PR, and this task plan.

| Who | Do this now | Why it can't wait |
|---|---|---|
| **A** · Ani | T58: ask the guide about the public repo. Write the Track B alignment code for T35 now (Qwen's answers joined to RouterBench's GPT-4 scores by qid), so it is ready when Qwen's runs land. Run the embedding dedup pass as soon as B hands over F2. | T35 is on the critical path to the headline result |
| **B** | **T32 first:** the `mcq_logprob` grader. Then T10–T12 (heuristics, TF-IDF, `bge-base` embeddings) on the real corpus, and hand the F2 file to A for the embedding dedup pass. | D needs T32 before Qwen can be scored on MMLU, ARC and Winogrande |
| **C** | T05 (two-line fix), then T20–T21: metrics and the exact threshold sweep, tested on synthetic data and then run on the Mixtral scores already in the release. | The first real Pareto curve (week 8) is built on these |
| **D** · kartiken07 | **T30 first:** `generate.py` with the MLX client and the cache. Then T31 (batched decoding), T06 (branch protection) and T40 (Gemini key on the M2). | T30 → T33 is the longest chain in the project: Qwen's overnight runs feed both the headline and the demo |

## The demo, decided up front

| Piece | Choice | Why |
|---|---|---|
| Machine | The 16 GB M2 | It ran Qwen3.5-4B at 28.5 tok/s, measured |
| Small model (local) | **Qwen3.5-4B, 4-bit, MLX** (~3 GB) | Measured speed; fits with room for everything else. Also the headline result (DECISIONS D-08) |
| Frontier model (API) | **Gemini 3 Flash, free tier**; GitHub Models GPT-4o-class as backup | $0; a live demo sends a few dozen queries, far under the free limits |
| Router | Same architecture as Track A, **with the small-model head retrained on Qwen's own answers** | A router learns *when its small model fails*. Mixtral's failures aren't Qwen's |
| Cost shown on screen | Frontier queries priced at Sonnet 4.6 list rate, local at $0 | Same cost model as the paper (DESIGN Rev A) |
| If the Wi-Fi dies | Cached frontier answers for the demo queries, plus a recorded video | Live demos fail at defences |

**Why the two-head design pays off here:** swapping the small model means retraining one head
(`ŝ_S`), not the whole router. The frontier head (`ŝ_L`) is trained on RouterBench's GPT-4 scores
and stays as is: Gemini 3 Flash is at least as strong, so it is a conservative stand-in.

**Memory budget on the M2 during the demo:** Qwen ~3 GB + embedding model ~0.5 GB + Python and the
web app ~1 GB. About 5 GB of 16, so it runs alongside a browser and a slide deck.

---

## Phase 0 · Close out week 2 (now)

| ID | Owner | Task | Done when |
|---|---|---|---|
| T01 | All | Agree the model pairs (DECISIONS D-05 to D-08): Qwen vs GPT-4 is the headline and the demo; Mixtral vs GPT-4 the development pair and the RouteLLM comparison; Yi-34B optional; Mistral-7B dropped; HellaSwag reported separately. | PR #4 approved by every lane |
| T02 | All | Review and merge PR #3 (audit), then PR #4 (loader). One reviewer per lane on #4, since it changes `schema.py` and `costs.py`. | Both merged into `main` |
| T03 | A | ✅ Done. Loader rerun with Mixtral + Yi-34B; data published as release `data-d9ef5869b99739aa`. | — |
| T04 | A | ✅ Lexical pass done (585 pairs, `notes/dedup_report.md`). Remaining: the embedding pass, once B delivers F2 (T12). | Report shows the embedding pass ran |
| T05 | C | Fix `estimate_spend.py` so measured and estimated speeds are labelled separately. | Output says which is which |
| T06 | D | Branch protection on `main`, PR template, `CODEOWNERS` so any `schema.py` change requests every lane. | A test PR shows the rules |

## Track A · The research result (RouterBench, no model runs)

| ID | Owner | Task | Needs | Done when |
|---|---|---|---|---|
| T10 | B | F0: ~25 heuristic features as a pure function | — | Hand-computed values match in tests |
| T11 | B | F1: TF-IDF, word 1–2 + char 3–5 grams, fit on train only | — | Fit-on-train enforced by a test |
| T12 | B | F2: `bge-base-en-v1.5` embeddings for all 34,778 prompts, cached to `features.parquet` | T03 | One file, minutes to compute on any Mac |
| T13 | B | Baselines: logistic regression on F0, F1; XGBoost on F0+F2 | T10–T12 | Validation Spearman ρ per feature set |
| T14 | B | Two-head MLP (ŝ_S, ŝ_L), soft-label loss + gap term, 5 seeds | T12 | `predictions.parquet` written |
| T15 | B | Isotonic calibration per head; reliability diagrams + ECE | T14 | Before/after figures in `notes/` |
| T16 | B | Ablation ladder: F0 → F1 → XGBoost → F2 → F2+F0 → cost-normalised rule v2 | T13–T15 | Every row of the DESIGN §4.3 table filled |
| T20 | C | `eval.py`: Q(τ), C(τ), PGR, APGR, random and oracle curves | — | Oracle ≥ router ≥ random on synthetic data |
| T21 | C | `threshold.py`: exact O(n log n) sweep + concave envelope | — | Matches brute force on 200 random cases |
| T22 | C | Lower bounds: Hoeffding, paired bootstrap | T21 | Tested on a low-headroom synthetic set |
| T23 | C | Learn-then-Test τ\* selection | T22 | τ\* chosen on `cal` only, by construction |
| T24 | C | Cost model M2 (hardware time) and a batch-API frontier baseline | — | Both baselines in every savings number |
| T25 | C | `plot_pareto.py` panels A–D, with random and oracle on every plot | T20–T23 | First real Pareto figure (week-8 gate) |
| T26 | C | Leave-one-subject-out evaluation on MMLU | T04, T25 | Grouped vs LOSO gap reported |
| T27 | C | Headroom sweep: savings vs gap, across Mixtral and Qwen (and Yi-34B if time allows) | T25, T37 | One figure, two or three pairs |
| T28 | C | **Lock the numbers**: CPT(95%) and savings on the untouched test split, with confidence intervals — **Qwen pair as the headline**, Mixtral pair as the comparison | all above | No number changes after this |

## Track B · The Mac model (the headline result and the demo)

| ID | Owner | Task | Needs | Done when |
|---|---|---|---|---|
| T30 | D | `generate.py`: one client interface; `MLXClient` (local) and `OpenAICompatClient` (Gemini / GitHub Models); content-hash cache; resumable | — | Re-running makes zero model calls |
| T31 | D | Batched MLX decoding; measure the real batch speed-up on the M2 | T30 | Speed-up committed to `models.yaml` |
| T32 | B | `mcq_logprob` grader: score A/B/C/D by the model's next-token probability, in one forward pass | — | Can't produce an "always A" failure |
| T33 | D | **Overnight run on the M2:** Qwen3.5-4B on all 7,450 GSM8K questions, k=1. About 18 h unbatched, ~5 h batched. Plugged in, lid open. | T30, T31 | Responses cached; run survives a restart |
| T34 | D | Qwen3.5-4B on MMLU, ARC, Winogrande via `mcq_logprob`. About 2 h total. | T32 | Scores for ~16,800 questions |
| T35 | A | Track B `scores.parquet`: Qwen's scores beside RouterBench's GPT-4 scores, same qids, same splits; headroom kill-check | T33, T34 | Gap per dataset reported |
| T36 | B | Retrain **only the small-model head** on Qwen's labels; calibrate on Qwen's `cal` split | T14, T35 | `router-qwen` bundle exported |
| T37 | C | τ\* for the Qwen pair (Learn-then-Test) and its Pareto curve | T23, T36 | The τ\* the demo will use |

## Demo · The router working live on the Mac

| ID | Owner | Task | Needs | Done when |
|---|---|---|---|---|
| T40 | D | Gemini and GitHub Models keys in `.env` on the M2; free-tier rate limits checked | — | One call to each succeeds |
| T41 | B | Router bundle: one file holding encoder name, heuristics version, head weights, calibrators, τ\*; `router.predict(text)` | T36, T37 | Same decisions as the evaluation code on 1,000 test queries |
| T42 | D | `serve.py` (FastAPI): `/route` (decision only), `/chat` (route, generate, stream), `/stats` (running totals). Qwen and the router load once at start-up. | T30, T41 | Router decision p99 < 15 ms on the M2 |
| T43 | D | Web page: query box → **routed to local or API**, predicted ŝ_S / ŝ_L against τ\*, the answer streaming in, latency, this query's cost, **money saved so far vs always-frontier** | T42 | Works in Safari and Chrome |
| T44 | C | **Replay mode**: push 200 held-out test queries through the live router, showing accuracy against always-frontier and cost saved as it runs. This is the Pareto point, live. | T42 | Matches the offline result for the same 200 |
| T45 | A | Demo set: 10 easy queries the router should keep local, 10 hard ones it should send out, 5 open chat prompts. Check every decision in advance. | T41 | Every planned query routes as expected |
| T46 | D | Offline mode: cached frontier answers for the demo and replay sets; `--offline` flag | T44, T45 | Full demo runs with Wi-Fi off |
| T47 | D | Soak test on the M2: one hour of queries; record memory peak, first-token latency, thermal throttling | T43 | Numbers in `notes/demo_perf.md` |
| T48 | All | Record the backup video of the full demo | T46 | Video on the shared drive |
| T49 | All | Two timed dry runs, one on a projector | T48 | Under the time limit, twice |

## Report and defence

| ID | Owner | Task | Done when |
|---|---|---|---|
| T50 | A | Chapter: datasets, the audit, re-grading, dedup and splits | Draft reviewed by one teammate |
| T51 | B | Chapter: features and router architecture, the ablation ladder | Draft reviewed |
| T52 | C | Chapter: cost model, threshold selection, guarantees, results | Draft reviewed |
| T53 | D | Chapter: local inference, serving, the demo, performance on the M2 | Draft reviewed |
| T54 | A | Introduction, related work (RouteLLM, RouterBench, FrugalGPT, Learn-then-Test), conclusion; merging the chapters | Full draft |
| T55 | D | Reproducibility: one command from raw files to every figure; tested on a fresh clone on another Mac | Someone outside the lane reproduces a figure |
| T56 | All | Slides | Reviewed by the guide |
| T57 | All | Viva prep: likely questions (why PGR, why not GPT-3.5 as the small model, how τ avoids leakage, what if the free tier disappears) | Each person answers every question once |
| T58 | A | Ask the guide about publishing the repo before evaluation; add a licence if it stays public | Answer recorded in `DECISIONS.md` |

---

## Order of work

```
Now        T01–T06                      close week 2
Weeks 3–5  T10–T13  T20–T24  T30–T32    everyone builds in parallel
Weeks 5–6  T33–T34                      Qwen runs overnight on the M2
Weeks 6–8  T14–T15  T25  T35–T36        first real Pareto, Qwen router
Weeks 9–11 T16  T26–T27  T37  T41–T42   results harden, demo backend
Weeks 12–13 T28  T43–T47                numbers locked, demo built
Weeks 14–15 T48–T57                     video, dry runs, report, slides
Week 16    defence
```

**The critical path to the demo** runs through D's machine: T30 → T31 → T33/T34 → T35 → T36 →
T37 → T41 → T42 → T43. Start T30 this week; the overnight run can't move until it's done.
