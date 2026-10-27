# Chapter 1 — Introduction

> **DRAFT for Anirvan Ranjan (task T54).** Numbers marked **[TBD]** wait for the Qwen results (Track B)
> and the locked test-split numbers (T28). Rewrite in your own voice and check UPES's rules on
> declaring AI assistance.

## 1.1 The problem

Frontier language models answer more questions correctly than small ones, and cost far more to use.
At 2026 list prices a model such as Claude Sonnet 4.6 charges $3 per million input tokens and $15 per
million output tokens; a 4-billion-parameter open model such as Qwen3.5-4B runs on a 16 GB laptop for
the price of electricity. Yet most questions a user asks do not need the frontier model: on the
benchmarks we study, a 47-billion-parameter open model (Mixtral-8x7B) already answers 55–83% of
questions correctly, depending on the dataset, against 81–96% for GPT-4 (Chapter 3).

A **router** exploits this. For each incoming query it predicts whether the small model will be good
enough, answers locally when it will, and calls the frontier model only when it will not. If the
prediction is good, most of the frontier model's quality is kept for a fraction of its cost.

This project builds such a router between a local small model and a frontier API model, and asks:

> **How much of the frontier model's cost can be saved while keeping at least 95% of its quality
> advantage — and can that be guaranteed in advance, not just observed after the fact?**

"95% of its quality advantage" is measured as the *performance gap recovered* (PGR), the metric of
RouteLLM [@ong2025routellm]: if the small model scores Q_S, the frontier model Q_L and the router Q,
then PGR = (Q − Q_S) / (Q_L − Q_S). The original target was 2–5× lower cost at PGR ≥ 0.95.

## 1.2 Why this is harder than it looks

Three things make an honest answer difficult.

- **Benchmarks mislabel difficulty.** Public routing data contains scores that measure formatting
  habits rather than correctness, and near-copies of questions across splits. A router trained on it
  learns the artefacts (Chapter 3).
- **Picking the operating point is itself a source of optimism.** Every router has a threshold. Chosen
  by looking at the results, it reports a saving that will not hold on new queries. Most published
  routing results are curves, not pre-chosen thresholds.
- **"Free" local inference is not free.** A laptop has a purchase price and a power draw. Below a
  certain generation speed, running a small model locally costs more per query than calling a cheap
  API model (Chapter 5, cost model M2).

## 1.3 Approach

We use RouterBench [@hu2024routerbench], which stores GPT-4's answers to 36,497 benchmark questions, as
the frontier arm, so no API budget is needed. Our pipeline:

1. **Data** (Chapter 3): audit RouterBench, re-grade what is mis-scored, recover missing gold answers,
   group near-duplicates with lexical and template-centred embedding similarity, and freeze
   train / calibration / test splits.
2. **Router** (Chapter 4): a two-head network predicting the small and the frontier model's chance of
   being right, from text features of increasing richness (heuristics, TF-IDF, sentence embeddings),
   calibrated on a held-out split, with a decision rule that weighs predicted gain against predicted cost.
3. **Threshold with a guarantee** (Chapter 5): choose the routing threshold on the calibration split by
   Learn-then-Test [@angelopoulos2021ltt] with a betting-based confidence bound [@waudbysmith2024betting],
   so that PGR ≥ 0.95 holds on new queries with probability at least 95%; report the test split once.
4. **Local model and live system** (Chapter 6): run Qwen3.5-4B on the project's Mac with MLX, align its
   answers with GPT-4's on the same questions, and serve the router behind a web interface.

## 1.4 Contributions

1. **A re-audited routing corpus.** We show that RouterBench's GSM8K scores do not measure correctness
   and that one of its small models answers multiple choice at chance level; we re-grade, recover gold
   letters for 95–100% of multiple-choice questions, and find that raw embedding similarity cannot be
   used for de-duplication on templated benchmarks without centring (Chapter 3).
2. **Guaranteed operating points for routing.** To our knowledge, the first application of
   Learn-then-Test to LLM routing thresholds. We find that the bootstrap bound recommended in our own
   design undercovers on low-headroom data (15% failures where 5% were promised), while the betting
   bound is valid and nearly as tight (Chapter 5).
3. **What the router can and cannot see.** Query-only routers, from heuristics to sentence embeddings,
   recover 4–9% savings at a guaranteed PGR of 0.95 on the Mixtral pair. Adding the small model's
   agreement with a second opinion raises this to 26% (Chapter 5), consistent with cascade work
   [@yue2024cascades]. For the local model this becomes Qwen's agreement with itself **[TBD]**.
4. **A router that runs on a laptop**, with costs reported both as API spend and as the laptop's own
   running cost, and a live demonstration **[TBD: headline numbers for Qwen3.5-4B vs GPT-4]**.

## 1.5 Organisation

Chapter 2 reviews cascades, routers, routing benchmarks and distribution-free risk control.
Chapter 3 builds the corpus. Chapter 4 describes the features and the router. Chapter 5 covers the cost
model, threshold selection and results. Chapter 6 describes local inference and the demonstration
system. Chapter 7 discusses limitations and future work, and Chapter 8 concludes.

**Team contributions** (fill in): Data and corpus — Anirvan Ranjan; features and router — HSID007;
evaluation and statistics — [C]; systems and demo — kartiken07.
