# Chapter 1 — Introduction

> **Author Declaration & Attributions (Chapters 1, 2, 3, 7 and 8):** These chapters were written by
> Anirvan Ranjan (Lane A: data, corpus, evaluation pipeline and system integration). In line with UPES
> guidelines on generative AI: AI tools (Claude) were used to draft and revise text, to check every
> number against the repository's reports, and to help write parts of the code; the author directed the
> work, reviewed and rewrote the text, and is responsible for its content. Lane A's work comprises the
> RouterBench audit and re-grading, gold-letter recovery, template-centred de-duplication and frozen
> splits (Chapter 3), the threshold code (`router/threshold.py`), the cost models M1 and M2 with the crossover speed and batch baseline (`router/costs.py`), the
> Track B generation pipeline, and the D-15 demo router (`router/signals.py`, the `cascade_t61` bundle,
> the server's cascade path). Results from other lanes are credited in their chapters: features and
> routers to Lane B (Huwaid Siddiqui), certification and the seed-free bound to Lane C (Sarthak
> Jakhmola), and the on-device system and demo to Lane D (Kartiken Sharma).

## 1.1 The Problem: Frontier Capability at Frontier Cost

Frontier large language models (LLMs) solve complex reasoning tasks with remarkable accuracy, yet their commercial deployment incurs steep economic costs. At 2026 list prices, a flagship proprietary model such as Claude Sonnet 4.6 (the reference price used throughout this report) charges $3.00 per million input tokens and $15.00 per million output tokens. In contrast, small open-weight models such as Qwen3.5-4B execute locally on standard consumer hardware (e.g., an Apple Silicon laptop with 16 GB unified memory) for the negligible marginal cost of electricity.

Crucially, the majority of everyday user requests do not require the full parameter capacity of a frontier system. On standard academic evaluations, a medium-sized open model such as Mixtral-8x7B (47 billion parameters) successfully resolves 55% to 83% of benchmark queries depending on domain complexity, compared to 81% to 96% for GPT-4 (Chapter 3). Edge-tier architectures display even greater efficiency relative to their footprint: running locally under 4-bit quantization, Qwen3.5-4B achieves 69% to 92% accuracy across identical benchmarks, against 82% to 97% for GPT-4 (`notes/trackb_results.md`).

A **cost-aware model router** exploits this capability overlap. By evaluating each query prior to or during execution, the router assesses whether the local lightweight model can resolve the task satisfactorily. When local competence is predicted with high confidence, the query is serviced on-device at near-zero marginal expense; only ambiguous or intractable prompts are escalated to the proprietary cloud endpoint. If routing decisions are accurate, the composite system preserves the high accuracy of the frontier model while slashing inference expenditures.

This project designs, evaluates, and deploys an end-to-end cost-aware routing architecture arbitrating between a local edge model and a cloud frontier API, addressing the following fundamental research question:

> **What proportion of frontier API expenditure can be eliminated while guaranteeing that at least 95% of the frontier model's performance advantage is preserved—and can this performance guarantee be mathematically certified prior to deployment, rather than merely observed retrospectively on test data?**

We quantify this preservation threshold via the **Performance Gap Recovered** (PGR), formalized by RouteLLM [@ong2025routellm]. Letting $Q_S$ denote the accuracy of the small edge model, $Q_L$ the accuracy of the large frontier model, and $Q$ the aggregate accuracy of the hybrid router, PGR is defined as:

$$\text{PGR} = \frac{Q - Q_S}{Q_L - Q_S}$$

Our engineering objective is to achieve a 2× to 5× cost reduction at $\text{PGR} \ge 0.95$ under verifiable statistical guarantees.

---

## 1.2 Methodological Obstacles in Empirical Routing

While conceptually intuitive, establishing a dependable, guaranteed model router confronts three severe empirical hurdles:

1. **Benchmark Artefacts and Mislabeled Ground Truth.** Existing routing benchmarks frequently contain systematic labeling errors, broken answer extraction heuristics, and evaluation scripts that penalize formatting rather than reasoning capability. For instance, in raw RouterBench outputs [@hu2024routerbench], entire math benchmarks contain corrupted scores where correct outputs are marked incorrect, while multiple-choice benchmarks suffer from chance-level guessing due to prompt format mismatches. A router trained on uncurated data optimizes for benchmark quirks rather than intrinsic task difficulty.
2. **The Illusion of Test-Selected Operating Thresholds.** In published literature, routing thresholds are routinely selected post-hoc by sweeping the test partition to locate the optimal trade-off on an empirical Pareto frontier. This practice injects severe selection bias: a cutoff tuned on test evaluations invariably overstates cost savings and understates error rates on subsequent unseen traffic.
3. **The Hidden Overhead of Local Execution.** The premise that local edge inference is "free" ignores physical hardware realities. A local workstation incurs capital depreciation, battery drain, and thermal throttling. If local token generation is sluggish, the fully amortized cost per query can exceed that of low-cost cloud APIs (e.g., Gemini 3 Flash). Realistic economic accounting must incorporate both API token fees and localized hardware compute amortization.

---

## 1.3 System Overview and Pipeline

To circumvent proprietary API budget constraints during development, we ground our benchmark evaluations on RouterBench [@hu2024routerbench], which catalogs GPT-4 responses and accuracy metrics across 36,497 standardized questions. Our research methodology proceeds across four coordinated phases:

```
[ Raw Benchmark Prompts ]
           │
           ▼
┌──────────────────────────────────────────────┐
│ Phase 1: Corpus Audit & Deduplication (Ch 3) │
│ - Numeric regrading of GSM8K                │
│ - Consensus gold-letter recovery for MC      │
│ - Template-centered BGE semantic clustering  │
│ - Frozen 60/20/20 train/cal/test splits      │
└──────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│ Phase 2: Feature Extraction & Router (Ch 4)  │
│ - Lexical, syntactic, & semantic embeddings  │
│ - Two-head calibrated classifier (S vs L)    │
│ - Small-model cascade self-consistency       │
└──────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│ Phase 3: Certified Risk Control (Ch 5)       │
│ - Learn-then-Test (LTT) on calibration split │
│ - Seed-free betting martingale bounds (WSR)  │
│ - Single evaluation pass on frozen test set  │
└──────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│ Phase 4: Local Runtime & Serving (Ch 6)      │
│ - Qwen3.5-4B in 4-bit MLX on Apple Silicon   │
│ - Sub-10ms routing decision engine           │
│ - FastAPI backend with interactive UI        │
└──────────────────────────────────────────────┘
```

1. **Corpus Sanitization and Split Isolation (Chapter 3):** We audit the source data, reconstruct numeric grading parsers, recover missing ground-truth options, purge near-duplicate prompts via template-centered embedding projections, and lock a disjoint train/calibration/test partition.
2. **Predictive Modeling and Cascades (Chapter 4):** We train multi-head discriminators predicting per-query success probabilities from input representations (heuristics, TF-IDF, and dense sentence embeddings). Furthermore, we design a cascade architecture that inspects the local model's own output agreement and token probabilities.
3. **Distribution-Free Statistical Guarantees (Chapter 5):** Using the calibration split, we calibrate the escalation threshold via Learn-then-Test (LTT) [@angelopoulos2021ltt] paired with Ville's betting-based martingales [@waudbysmith2024betting]. This guarantees that $\text{PGR} \ge 0.95$ holds on incoming test queries with probability at least $1 - \delta = 0.95$.
4. **On-Device Deployment and Dual Cost Modeling (Chapter 6):** We deploy Qwen3.5-4B under 4-bit MLX quantization on an Apple MacBook, measuring inference latency, power draw, and throughput against dual economic metrics: API spend savings (Model M1) and total amortized cost of ownership (Model M2).

---

## 1.4 Summary of Contributions

The primary contributions of this thesis are:

1. **A Repaired, Leak-Free Routing Corpus:** We identify critical failure modes in RouterBench, demonstrating that raw GSM8K scores do not reflect mathematical correctness and that multiple-choice evaluations contain severe formatting collapse. We implement robust numeric extraction, recover gold targets for 95% to 100% of multiple-choice tasks, and show that semantic deduplication on templated prompts requires template-centered normalization to prevent spurious clustering.
2. **Finite-Sample Statistical Guarantees for Model Routing:** To our knowledge, we present the first application of Learn-then-Test to LLM routing thresholds. We demonstrate that standard empirical bootstrap bounds undercover on low-headroom tasks (failing 15% of the time against a nominal 5% error budget), whereas terminal Waudby-Smith–Ramdas (WSR) betting martingales provide strict finite-sample validity without sacrificing tightness.
3. **Information Limits of Query-Only Routing vs. Cascades:** We show that text-only features (heuristics, lexical shingles, dense embeddings) provide insufficient signal to reliably forecast edge model failure, yielding only 0.5% to 2.1% certified savings on the Qwen3.5-4B / GPT-4 pair (seed-free). Conversely, incorporating internal generation signals (three-sample consensus and top-choice confidence) unlocks **41.3% [39.0%, 43.5%] certified API savings** at $\text{PGR} \ge 0.95$ (test $\text{PGR} = 102.8\%$).
4. **Edge Deployment and Realistic Cost Benchmarking:** We package the complete pipeline into an on-device runtime on an Apple M2 MacBook. The router makes decisions in about 6 ms (99th percentile), executing Qwen3.5-4B in 4-bit MLX with no measurable accuracy loss relative to the 16-bit reference model. We show that when the laptop's purchase price and power are counted for all three local answers (Model M2), the system still cuts cost by 28.4%.

---

## 1.5 Results in Brief

On our primary evaluation benchmark comparing local Qwen3.5-4B against cloud GPT-4 (Table 1.1), fixing the routing cutoff on the calibration split via seed-free WSR martingales delivers **41.3% [39.0%, 43.5%] API expenditure reduction** while securing a realized test PGR of **102.8% [99.1%, 107.0%]**—representing a 1.7× cost efficiency improvement. Counting the laptop's purchase price and power for all three local answers (Cost Model M2), the net cost reduction is 28.4% [26.1%, 30.6%]. 

Importantly, if the decision threshold is selected retrospectively by scanning the empirical test curve, the apparent cost savings reach 78.9% (a 4.7× reduction). The difference between 41.3% and 78.9% illustrates the concrete statistical penalty required to guarantee safety in production.

**Table 1.1 — Summary of Headline Routing Results on Qwen3.5-4B vs. GPT-4 Test Set ($N=4,707$).**

| Routing Configuration | Selection Protocol | Statistical Guarantee | Realized Test PGR | Cost Savings (API M1) | Hardware-Inclusive Savings (M2) |
|---|---|---|--:|--:|--:|
| **Question-Only Router (F2)** | Calibrated (Seed-Free WSR) | $\mathbb{P}(\text{PGR} \ge 0.95) \ge 0.95$ | 98.6% | 2.1% [1.5%, 2.7%] | 2.0% |
| **Cascade Router (3-sample)** | Calibrated (Seed-Free WSR) | $\mathbb{P}(\text{PGR} \ge 0.95) \ge 0.95$ | **102.8%** | **41.3% [39.0%, 43.5%]** | **28.4%** |
| **Cascade Router (3-sample)** | Post-hoc Test Curve | None (Optimistic baseline) | 95.0% | 78.9% (4.7×) | 66.1% |
| **Cascade Router (1-sample, D-14 baseline)**| Calibrated (Seed-Free WSR) | $\mathbb{P}(\text{PGR} \ge 0.95) \ge 0.95$ | 98.4% | 29.5% [27.8%, 31.5%] | 24.7% |

*M2 charges the laptop for every local answer the router generates (three per arithmetic question for the 3-sample cascade). Charging only one answer per question, as in an earlier draft, gives 36.5%. Source: Chapter 5, Table 5.1.*

---

## 1.6 Thesis Structure

- **Chapter 2 (Related Work):** Examines model cascades, query routers, LLM benchmarks, and distribution-free risk control theory.
- **Chapter 3 (Data: Corpus, Labels, and Splits):** Details the audit of RouterBench, numeric regrading, gold-label recovery, template-centered deduplication, and split freezing.
- **Chapter 4 (Features and Router Architectures):** Details the lexical and semantic feature engineering, two-head classification networks, probability calibration, and generation-level cascade mechanisms.
- **Chapter 5 (Cost Models, Threshold Certification, and Results):** Formalizes dual cost accounting, compares Learn-then-Test concentration bounds, and presents certified routing benchmarks.
- **Chapter 6 (Edge Inference and Live System Demonstration):** Describes the MLX quantization pipeline, Apple Silicon hardware profiling, FastAPI service architecture, and live interface.
- **Chapter 7 (Discussion):** Analyzes the limitations of query-only routing, latency overheads of cascading, the economic trade-offs of local hardware versus batch cloud endpoints, and threats to validity.
- **Chapter 8 (Conclusion):** Summarizes findings and outlines future research directions in test-time calibration.

**Author Contributions:**
- **Data Engineering, Cost Models & System Integration (Lane A):** Anirvan Ranjan (auditing, GSM8K regrading, gold answer reconstruction, semantic deduplication and split governance, the cost models M1 and M2, the threshold code, Track B generation, and the D-15 demo router).
- **Feature Extraction & Routing Models (Lane B):** Huwaid Siddiqui (feature extractors, two-head classifiers, calibration, and cascade feature design).
- **Statistical Evaluation & Risk Control (Lane C):** Sarthak Jakhmola (LTT concentration bounds, bootstrap undercoverage, seed-free WSR implementation, and Pareto analyses).
- **Edge Deployment & Demonstration (Lane D):** Kartiken Sharma (MLX optimization, Apple Silicon profiling, FastAPI serving engine, and user interface).
