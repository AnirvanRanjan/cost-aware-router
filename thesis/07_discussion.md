# Chapter 7 — Discussion

## 7.1 Answering the Central Research Question

Chapter 1 posed a fundamental question for multi-model inference systems: *What proportion of frontier API expenditure can be eliminated while guaranteeing that at least 95% of the frontier model's performance advantage is preserved ($\text{PGR} \ge 0.95$), and can this saving be certified prior to deployment?*

On our primary edge-deployment benchmark—arbitrating between local Qwen3.5-4B and cloud GPT-4 across $N=4,707$ held-out test queries—the empirical answer bifurcates sharply across evaluation paradigms (Table 7.1):

**Table 7.1 — Final Performance and Cost Summary on the Headline Qwen3.5-4B vs. GPT-4 Test Set.**

| Model & Decision Mechanism | Threshold Selection Locus | Finite-Sample Guarantee | Test Performance Gap Recovered (PGR) | Realized API Spend Savings (M1) | Net Cost Reduction Ratio |
|---|---|---|--:|--:|:---:|
| **Cascade Router (3-sample)** | Calibrated via Seed-Free WSR | **Yes ($\mathbb{P} \ge 0.95$)** | **102.8% [99.1, 107.0]** | **41.3% [39.0, 43.5]** | **1.7×** |
| **Cascade Router (3-sample)** | Post-hoc Test Curve Sweep | None (Retrospective) | 95.0% | 78.9% | 4.7× |
| **Question-Only Router (F2)** | Calibrated via Seed-Free WSR | **Yes ($\mathbb{P} \ge 0.95$)** | 98.6% | 2.5% [1.8, 3.1] | ≈1.0× |
| **Question-Only Router (F2)** | Post-hoc Test Curve Sweep | None (Retrospective) | 95.0% | 21.1% | 1.3× |

These findings provide a clear resolution to our investigation:

1. **The Target Ratio is Achieved on Test Curves, but Constrained under Formal Guarantees:** The initial engineering aspiration of a 2× to 5× cost reduction at $\text{PGR} \ge 0.95$ is readily achieved if operating thresholds are scanned along the test curve (78.9% savings, or a 4.7× cost reduction). However, when operating cutoffs are locked *strictly on calibration data* under finite-sample concentration guarantees, the certified cost reduction settles at **41.3% [39.0%, 43.5%]** (a 1.7× reduction).
2. **The Discrepancy is the Price of Statistical Safety:** We emphasize that the gap between 78.9% and 41.3% does not represent a failure of the router, but rather the rigorous mathematical premium required to eliminate deployment risk. The vast majority of published routing literature [@ding2024hybrid; @ong2025routellm] reports uncalibrated test curves. Our findings demonstrate that deploying such uncertified operating points in clinical or legal environments exposes organizations to severe real-world failure rates.

---

## 7.2 The Structural Limitations of Question-Only Routing

A major negative finding of this thesis is the near-total failure of pre-execution query routers to deliver meaningful certified savings on high-capacity edge models. Across all evaluated input representations—shallow heuristics, lexical n-grams, dense BGE embeddings, and two-head neural classifiers—question-only routing delivered a modest 4% to 9% certified savings on the older Mixtral pair, and collapsed to an imperceptible **0.5% to 2.5% certified savings on Qwen3.5-4B** (`notes/qwen_query_report.md`).

Three structural mechanisms explain this phenomenon:

```
[ Traditional Router Assumption ]
Query Text ──► Semantic Difficulty ──► Predictable Model Failure
(Breaks down when prompt semantics do not correlate with reasoning traps)

[ Cascade Reality ]
Query Text ──► Model Execution ──► Output Divergence / Low Confidence ──► Robust Escalation
(Generation consistency directly exposes local failure)
```

### 7.2.1 Severe Headroom Compression
As lightweight language models rapidly advance, the capability delta ($\Delta$) separating them from frontier architectures contracts dramatically. While the headroom between Mixtral-8x7B and GPT-4 on our benchmark is 0.237, the gap between Qwen3.5-4B and GPT-4 contracts to merely **0.075** (Chapter 3). 

Under narrow headroom, preserving $\text{PGR} \ge 0.95$ leaves zero margin for classification error: escalating a single query that the small model could have answered correctly squanders cost efficiency, while failing to escalate a query that only the frontier model could resolve instantly breaches the risk tolerance. A query-only classifier cannot achieve the required AUC on text alone to navigate this knife-edge boundary.

### 7.2.2 The Semantic Opacity of Problem Difficulty
Task complexity is largely orthogonal to prompt semantics. In mathematical reasoning (GSM8K), two problems possessing identical token lengths, vocabulary distributions, and syntactic phrasing can differ fundamentally in computational depth (e.g., modular arithmetic vs. straightforward subtraction). 

Furthermore, our leave-one-subject-out (LOSO) cross-validation on MMLU revealed that the router's discriminative ability deteriorates significantly when confronted with unseen academic disciplines: the ROC-AUC dropped from **0.665 under pooled cross-validation to 0.617 out-of-fold** (`notes/loso_report.md`). A text classifier memorizes subject-level failure rates rather than learning a generalized concept of cognitive difficulty.

### 7.2.3 Synthetic Distribution Shift on Conversational Inputs
When we deployed our interim query router against an interactive validation suite representing standard user interactions, the model routed **8 out of 10 trivial queries to the cloud frontier API** (`notes/demo_set_check.md`). Prompts such as *"Is 17 a prime number?"* or *"Summarize the capital cities of Scandinavia"* possess high lexical density but simple reasoning demands. Because benchmark training distributions contain zero colloquial interactions, the classifier misinterprets brief conversational prompts as out-of-distribution risks, exposing the fundamental hazard of training static routing models on academic benchmark corpora.

---

## 7.3 Why Cascades Succeed, and Their Associated Latency Tax

The cascade router circumvents the limitations of query text by interrogating the lightweight model’s own generation dynamics. The local model reliably signals its own competence through internal consistency and token likelihoods.

### 7.3.1 Empirical Diagnostics of Self-Knowledge
On GSM8K, when all three stochastic generations from Qwen3.5-4B reach consensus (representing 87% of all test problems), the local model achieves an accuracy of **96.9%**, compared to only **60.5%** when outputs diverge (`notes/trackb_results.md`). 

Similarly, on multiple-choice benchmarks, when the top option probability exceeds $p \ge 0.90$ (49% of all test questions), local accuracy reaches **94.2%**, against **58.5%** on lower-probability completions. By training gradient-boosted decision trees over these internal indicators, the cascade router operates on direct measures of epistemic uncertainty rather than external textual proxies.

### 7.3.2 Cross-Domain Escalation Heterogeneity
Because the calibrated threshold $\tau^*$ is optimized across the pooled corpus, escalation behavior adapts dynamically to domain difficulty (`notes/trackb_cascade_report.md`):
- **GSM8K:** Retains **46.6%** local execution, exploiting strong self-consistency on structured math.
- **ARC-Challenge:** Retains **30.3%** local execution on science reasoning.
- **MMLU:** Retains **18.1%** local execution due to broad multidisciplinary variance.
- **Winogrande:** Escalates **98.4%** of prompts (retaining only 1.6% locally), reflecting Qwen’s baseline weakness on adversarial pronoun disambiguation.

### 7.3.3 The Real-World Overhead of Cascading
While cascades resolve the accuracy challenge, they impose two physical overheads that must be managed in production:

1. **Compounded Latency on Escalated Prompts:** Every escalated prompt must execute locally before invoking the cloud endpoint. For multiple-choice questions, the local pass requires only a single forward logit evaluation ($<50$ ms). However, generating a 110-token mathematical solution on an Apple Silicon M2 processor requires approximately **4.8 to 5.2 seconds** end-to-end (averaging 23 tokens/second during interactive decoding, `notes/m2_checks.md`). For the 75.9% of queries escalated at the certified operating point, the user incurs this localized latency penalty prior to cloud dispatch.
2. **Computational Energy and Token Spend:** Generating three stochastic candidate paths consumes three times the localized compute energy. In response to this latency and energy constraint, our architecture initially formulated a **single-sample cascade** (Decision D-14), which achieves **27.8% [25.9%, 29.6%] certified spend savings** while eliminating multi-sample decoding latency (`notes/trackb_cascade_report.md`). However, live validation on conversational inputs revealed that answer-text features were brittle under prompt formatting shifts, escalating 8 of 10 (run 1) and all 10 of 10 (run 2) easy questions that Qwen answered correctly (`notes/demo_set_check.md`, `notes/demo_set_check_run2.md`), while escalating 85.2% across the wider Track B test split (`notes/d15_demo_router.md`). Consequently, Decision D-15 unified the production demonstration with the certified 3-sample research router (`cascade_3sample`), which executes 3 brief samples on arithmetic and a single logit pass on multiple-choice, while retaining the 1-sample bundle as a latency-optimized comparative baseline.

---

## 7.4 The Statistical Price of a Guarantee

Table 7.2 compares the empirical properties of alternative threshold selection methodologies on the frozen calibration split.

**Table 7.2 — Comparison of Threshold Calibration Methodologies on `cascade_3sample` (Rule v2).**

| Threshold Selection Rule | Mathematical Guarantee Framework | Test PGR Realized | Realized API Spend Savings (M1) | Empirical Safety & Coverage Profile |
|---|---|--:|--:|---|
| **Naive Empirical Cutoff** | None (Sample mean on Cal) | 97.2% | 77.0% | Unsafe: Overfits sample variance; highly variable |
| **Paired Bootstrap Bound** | Asymptotic normal approximation | 100.0% | 65.8% | **Invalid:** Severe undercoverage (15% failure rate vs. 5% nominal) |
| **Hoeffding Inequality** | Finite-sample distribution-free | 100.3% | 4.6% | Valid but Impractical: Excessive bound looseness discards 90% of savings |
| **Seed-Free Terminal WSR** | **Finite-sample betting e-values** | **102.8%** | **41.3%** | **Valid & Tight:** Exact coverage; completely eliminates permutation variance |

### 7.4.1 The Hidden Danger of Empirical Bootstrap Bounds
Our original system design specified paired bootstrap resampling on the calibration set to establish confidence bounds. However, extensive Monte Carlo simulation on low-headroom tasks exposed that the bootstrap bound **failed to cover the true risk 15% of the time under a nominal 5% error budget ($\alpha=0.05$)** (`notes/chapter_C_facts.md`). Asymptotic approximations fail when empirical error distributions exhibit severe skewness and low sample headroom.

### 7.4.2 Resolving Permutation Variance via E-Value Mixtures
While single-sequence Waudby-Smith–Ramdas (WSR) martingales [@waudbysmith2024betting] guarantee mathematical validity unconditionally, their sequential nature induces substantial sensitivity to the random ordering of calibration data: across seeds 0 to 19, realized test savings oscillated widely between **40.8% and 74.9%** (`notes/t60_certification_report.md`). Selecting the most favorable seed would re-introduce the exact selection optimism the methodology seeks to eradicate.

We resolved this challenge by formulating a **seed-free terminal e-value averaging procedure**. By evaluating the terminal wealth across $M=50$ random permutations and computing their arithmetic mean:
$$\bar{E} = \frac{1}{M} \sum_{m=1}^M E^{(m)}$$
linearity of expectation guarantees that $\bar{E}$ remains an exact finite-sample e-value ($\mathbb{E}[\bar{E}] \le 1$ under $H_0$). Inverting this combined test via Markov's inequality yields a completely deterministic, permutation-invariant threshold certifying **41.3% savings** with guaranteed validity.

---

## 7.5 Real-World Economics: Edge Hardware vs. Cloud Batch APIs

Our primary cost model (M1) assumes that localized edge inference is economically free. We contextualize this assumption against commercial cloud alternatives and physical device constraints:

```
                  ┌───────────────────────────────┐
                  │ Cost Model Trade-off Analysis │
                  └───────────────┬───────────────┘
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
┌───────────────────────────────┐         ┌───────────────────────────────┐
│     Cloud Batch Endpoints     │         │    Amortized Edge Hardware    │
│  - 50% discount on API lists  │         │  - Device purchase ($1,299)   │
│  - Multi-hour queue latency   │         │  - 15W active power draw      │
│  - Router is 18.5% more costly│         │  - M2 savings settle at 36.5% │
└───────────────────────────────┘         └───────────────────────────────┘
```

1. **The Cloud Batch API Paradox:** Major cloud vendors provide asynchronous Batch APIs at a 50% discount relative to interactive endpoints. Against a continuous batch cloud pipeline, our certified real-time router is **18.5% more expensive** than simply submitting all queries to the batch cloud service (`notes/trackb_cascade_report.md`). Consequently, model routers provide economic utility strictly for interactive, real-time user-facing applications; non-urgent background tasks should invariably leverage asynchronous cloud batching.
2. **Full Hardware Amortization (Cost Model M2):** Factoring in device acquisition costs ($1,299 capital expenditure amortized over 3 years), battery degradation, and 15W active electrical consumption, the certified cost reduction shifts from **41.3% down to 36.5%** at our benchmark throughput of 28.5 tokens/second (Chapter 5).
3. **Break-Even Generation Velocity:** Against modern low-cost frontier models such as Gemini 3 Flash ($0.075 / $0.30 per million tokens), the break-even local generation speed is **12.1 tokens/second**. Because Apple Silicon M2 achieves 23 tokens/second end-to-end and 29 tokens/second during decoding (`notes/m2_checks.md`), localized edge execution remains demonstrably cheaper than cloud invocation even when benchmarked against hyper-optimized commercial APIs.

---

## 7.6 Threats to Validity

- **Exchangeability Violations:** Learn-then-Test guarantees assume that incoming test traffic is exchangeable with the calibration split. While guaranteed on held-out benchmark splits, this assumption degrades if production traffic undergoes substantial semantic or linguistic drift.
- **Static Cloud Baseline:** Frontier model performance is grounded in RouterBench’s cataloged GPT-4 completions (November 2023). Contemporary frontier models exhibit different error topographies, requiring recalibration on contemporary reference outputs.
- **Quantization and Platform Discrepancies:** Training labels were acquired via 16-bit bfloat models hosted on H100 GPU clusters, whereas live edge execution evaluates 4-bit MLX models on Apple Silicon. While our verification audit on 200 sampled questions revealed negligible divergence within standard error bounds ($\pm 2.5\%$, `notes/m2_checks.md`), subtle quantization shifts exist.
- **Gold-Letter Censoring:** Excluding the 649 hardest multiple-choice questions (where no historical model answered correctly) renders the headline evaluation slightly more accessible than the raw academic benchmarks.
- **Academic Benchmark Specialization:** Standardized benchmarks evaluate structured, single-turn prompts with objective ground truth; they do not capture conversational nuance, ambiguous creative writing, or ultra-long-context retrieval.

---

## 7.7 Future Research Directions

1. **Online Sequential Conformal Recalibration:** Replace static split calibration with adaptive sequential martingales that dynamically adjust $\tau^*$ online in response to non-stationary user traffic.
2. **Task-Conditional (Mondrian) Risk Control:** Implement group-conditional Learn-then-Test to certify distinct operational thresholds $\tau_k^*$ for mathematics, code, and multiple choice, preventing difficult tasks (Winogrande) from suppressing local execution on well-calibrated tasks (GSM8K).
3. **Single-Token Early-Exit Verifiers:** Train ultra-lightweight linear classifiers over the hidden activations of the first 5 generated tokens to forecast failure without completing full multi-sample rollouts.
4. **Hierarchical Multi-Tier Cascades:** Extend the two-tier paradigm into an ordered hierarchy (e.g., Qwen-4B $\rightarrow$ LLaMA-70B $\rightarrow$ Claude Sonnet $\rightarrow$ GPT-4), establishing certified thresholds at each escalation tier.
5. **Latency-Penalized Optimization:** Reformulate the routing objective to jointly optimize monetary cost and end-to-end wall-clock user latency.
