# Lane A — presentation slides (T56)

*Draft for Ani (lane A) to cut down and speak from, same format as lanes B and D. Every number carries
the file it comes from; keep the citation when a number moves onto a slide. Target: ~10 minutes, so
11 slides. Lane A opens the talk, so slide 1 is the problem for the whole team.*

---

## Slide 1 — The question

- Frontier models are accurate and expensive; a 4-bit 4B model on a laptop is free per question.
- Most questions don't need the frontier: Qwen3.5-4B gets **81.3%** of the 4,707 held-out test
  questions right, GPT-4 **88.7%** (`notes/t27_t28_locked_numbers.md`, T27).
- **Our question:** how much frontier spend can be cut while keeping ≥ 95% of the frontier's
  quality advantage — and can that be *guaranteed before* deployment, not just observed after?
- Quality kept = **PGR** = (Q − Q_S) / (Q_L − Q_S) (RouteLLM).

## Slide 2 — Who did what

| Lane | Person | Owns |
|---|---|---|
| A | Anirvan Ranjan | data and splits, cost models, threshold code, Track B generation, demo router (D-15) |
| B | Huwaid Siddiqui | features, routers, calibration, the T61 cascade routers |
| C | Sarthak Jakhmola | certification, bootstrap undercoverage, seed-free WSR, results |
| D | Kartiken Sharma | Qwen on the Mac, serving, demo, performance |

## Slide 3 — Read the responses, not the score column

RouterBench: 36,497 rows, 11 models; kept **34,778 questions in 7 datasets** (`notes/routerbench_audit.md`).

| What we found | Consequence |
|---|---|
| Mistral-7B answers "A" on **94%** of MMLU, **100%** of HellaSwag | dropped (D-05): a format failure, not a skill |
| GSM8K scores take 5 values; GPT-4 never scores 1.0; mean 0.66 vs ~0.9 real | re-graded |
| GSM8K items are its **train** split | gold answers from GSM8K train |

- Takeaway: a router can only learn what its labels say.

## Slide 4 — Re-grading GSM8K and recovering answer letters

- New grader (`router/graders.py`): GPT-4 **0.949**, Mixtral **0.593**; hand check **48/50 = 96%**
  against a 95% gate (`notes/gsm8k_grader_check.md`).
- Gold letters voted from every model that was right, **0 conflicts**: MMLU 95.5%, ARC 99.1%,
  Winogrande 100%, HellaSwag 97.6% (`notes/chapter_A_facts.md`).

## Slide 5 — Leakage: near-duplicates and frozen splits

- Lexical pass (MinHash): **585** pairs. Embedding pass with raw cosine chained all **895** MMLU
  moral-scenarios questions into one group (shared preamble) → unusable.
- Fix: subtract the dataset/subject mean first (**template-centred** cosine ≥ 0.90): **1,298** more pairs.
- **1,883** pairs total; grouped 60/20/20 splits, frozen: train 20,867 · cal 6,957 · test 6,954
  (`notes/dedup_report.md`, D-11).
- Why it matters: a duplicate across cal and test makes the certificate optimistic.

## Slide 6 — What a question costs: M1 and M2

- **M1** (API spend): local = $0; frontier = list price × tokens.
- **M2** (laptop as a cost): $1,199 MacBook over 3 years at 8 h/day, 20 W, $0.12/kWh →
  cost per second of generation (`router/costs.py`).
- Crossover speed: below it, the laptop costs more than the API. Qwen on the M2 runs at
  **28.5 tok/s**, well above it (`thesis/05_results.md` §5.2).
- Batch-API baseline (50% off) is the bar a real-time router has to beat.

## Slide 7 — Track B: our own labels, on the model we actually serve

- RouterBench has no Qwen and no log-probabilities → we generated our own on an H100:
  7,450 GSM8K × 3 samples + 16,130 MCQ in **327.5 s** (`artifacts/trackb/gpu_manifest.json`).
- Same prompts, same sampling (3 × T 0.7, top-p 0.8) and the same option-letter forward pass the
  Mac uses → release **`trackb-6265adf7ffa4a602`**.
- These are the confidence signals the cascade router reads.

## Slide 8 — Why the first demo failed (and what changed)

- D-14 demo router read Qwen's answer *text*. On the demo set it escalated **every** easy question
  Qwen got right (`notes/demo_set_check_run2.md`).
- D-15: serve the **same** router the thesis certifies (T61 `cascade_3sample`), with the signals
  computed live exactly as in Track B (`router/signals.py`, one implementation for both).

| Seed-free, test split | Frontier calls | PGR |
|---|--:|--:|
| D-14 answer-text bundle | 85.2% | 100.5% |
| **D-15 T61 signals, 3 answers** | **65.8%** | **100.6%** |

(`notes/d15_demo_router.md`)

## Slide 9 — A reproducibility trap: the scikit-learn version

- Same data, same code, refit on scikit-learn 1.9.1 → τ* **9.657** instead of **8.2187**.
- Both are valid certificates for their own model; only one is the thesis model.
- Fix: read `_sklearn_version` from the release pickle, pin **`scikit-learn==1.8.0`**, and the
  preflight refuses to start on anything else (`scripts/20_demo_preflight.py`).

## Slide 10 — Live: how the router decides

- Multiple choice: **one forward pass** → probability of each option letter.
- Arithmetic: **3 sampled answers** → do they agree?
- Anything else: answered the same way, but labelled **not certified**.
- Score = predicted quality lost by staying local ÷ frontier cost; above **τ* = 8.2187** → frontier.
- **Show:** the page — an easy question (3/3 agree → Mac), a hard one (option bars → Gemini).
- Preflight on the M2: **24/24** demo questions as planned; replay **200/200** decisions identical
  to offline (`notes/replay_report_d15.md`).

## Slide 11 — Result and handover

- Certified: **41.3% [39.0, 43.5]** less API spend at test PGR **102.8%** (Ch. 5, Table 5.1).
- Question-only routing: **2.1%**. Looking at the small model's own answers is what works.
- Hand over to lane B (how the router is built) → lane C (why the guarantee holds) → lane D (the Mac).
