# Chapter 2 — Related Work

## 2.1 Overview and Taxonomy

Assigning simple inputs to lightweight models and reserving expensive classifiers for challenging queries is an established paradigm in statistical machine learning, tracing back to classical rejection options and classifier cascades [@viola2001cascade]. With the advent of modern large language models (LLMs), this architectural strategy has transitioned from an academic curiosity to an urgent economic necessity. The proliferation of open-weight models spanning from 1 billion to 70 billion parameters—operating alongside massive proprietary cloud systems—creates profound opportunities for cost-quality arbitrage.

Current literature addressing multi-model inference divides into three broad methodological categories:

1. **Model Cascades (Post-Execution Escalation):** Multi-stage architectures that invoke a lightweight model first, evaluate the quality or confidence of its generated response, and selectively escalate the query to a stronger model only when ambiguity is detected.
2. **Query Routers (Pre-Execution Decision Engines):** Predictive gating mechanisms that inspect the input query in isolation, forecasting the competence of candidate models before dispatching the request to a single selected endpoint.
3. **Routing Benchmarks and Evaluation Frameworks:** Empirical corpora, simulation suites, and Pareto metric definitions designed to compare routing algorithms under standardized operating assumptions.

Beyond routing architectures, our methodology draws directly from **distribution-free risk control**, a branch of statistical inference that provides finite-sample performance certificates without distributional assumptions. This chapter surveys each foundational domain and positions our contributions within the literature (Table 2.1).

---

## 2.2 Model Cascades: Post-Hoc Escalation on Uncertainty

Cascades prioritize latency-tolerant cost reduction by executing models sequentially. Because the decision to invoke the frontier model occurs *after* inspecting initial outputs, cascades exploit rich generation-level signals that remain invisible to pre-execution routers.

### 2.2.1 Systems-Level Cost Optimization: FrugalGPT
Chen et al. [@chen2024frugalgpt] framed LLM cost reduction as an end-to-end systems challenge, formulating **FrugalGPT**. Their framework combines three techniques: prompt adaptation (compressing prompt token lengths), LLM approximation (caching previous responses and substituting cheaper models), and sequential LLM cascades. In the cascade formulation, queries traverse an ordered pipeline of models ($M_1, M_2, \ldots, M_K$). A learned binary discriminator evaluates the output of model $M_k$ to determine whether to terminate inference or escalate to $M_{k+1}$. While FrugalGPT demonstrated that commercial pipelines could match GPT-4 accuracy with up to 98% cost savings, its decision boundaries depend on task-specific scoring heuristics, and every rejected candidate incurs cumulative inference latency and token costs.

### 2.2.2 Self-Consistency and Multi-Thought Agreement
The insight that stochastic generation agreement mirrors reasoning correctness was formalized by Wang et al. [@wang2023selfconsistency] via **self-consistency**. By sampling multiple diverse reasoning paths from a single model and extracting the marginal majority answer, self-consistency reliably identifies true-positive solutions on mathematical and symbolic tasks.

Building directly on this principle, Yue et al. [@yue2024cascades] proposed **Mixture-of-Thought (MoT) cascades**. Rather than relying on scalar confidence scores, MoT evaluates answer consistency across diverse reasoning formats (interleaving chain-of-thought and program-of-thought). If multiple sampled outputs from a smaller model (such as GPT-3.5) reach identical final answers, the query terminates locally; otherwise, it escalates to GPT-4. Across six mathematical benchmarks, MoT preserved GPT-4 competitive accuracy at approximately 40% of the cost. However, generating multiple reasoning chains imposes significant compute overhead on the edge device, necessitating a careful balance between sample count and local energy expenditure.

### 2.2.3 Self-Verification via Prompted POMDPs: AutoMix
Aggarwal et al. [@aggarwal2024automix] introduced **AutoMix**, which circumvents multi-sample decoding by utilizing *self-verification*. Under AutoMix, a smaller language model (such as LLaMA-2-13B or Vicuna-13B) generates an initial answer followed by a few-shot verification prompt assessing whether its response is grounded in the source context. A meta-verifier formulated as a partially observable Markov decision process (POMDP) integrates these noisy self-assessments to dictate escalation to PaLM-2 or GPT-4. While AutoMix demonstrates cost reductions exceeding 50%, its reliance on meta-prompting consumes additional input context tokens, and smaller edge models often exhibit pronounced self-confirmation bias, confidently validating their own hallucinations.

---

## 2.3 Query Routers: Pre-Execution Decision Engines

Unlike cascades, pre-execution query routers dispatch each query to exactly one model, incurring zero multi-model overhead or redundant token expenditure. The router operates as a classifier predicting whether the small model will succeed where the frontier model would.

```
                  ┌──────────────────────┐
                  │    Incoming Query    │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Pre-Execution Router │
                  │  (Bi-Encoder / MLP)  │
                  └──────┬────────┬──────┘
                         │        │
         Predicted Easy  │        │  Predicted Hard
         (High Local P)  │        │  (Low Local P)
                         │        │
                         ▼        ▼
           ┌────────────────┐  ┌────────────────┐
           │   Edge Model   │  │ Frontier Model │
           │  (Qwen3.5-4B)  │  │ (Cloud GPT-4)  │
           └────────────────┘  └────────────────┘
```

### 2.3.1 Hybrid LLM: Margin-Based Quality Routing
Ding et al. [@ding2024hybrid] introduced **Hybrid LLM**, a framework that predicts the expected quality difference between an open-source model (such as LLaMA-2-7B, LLaMA-2-13B, or Mistral-7B) and a proprietary frontier model. An encoder extracts dense representations from the raw query, feeding a regression head trained on the performance gap between models. At inference, a user-configured threshold $\tau$ routes queries to the local model whenever the predicted performance degradation remains below $\tau$. To account for stochastic variance in output generation, their training pipeline evaluates average scores across $k$ sampled responses per query. While Hybrid LLM reported up to a 40% reduction in frontier calls without aggregate quality degradation, operating thresholds were evaluated strictly along post-hoc test curves without finite-sample safety bounds.

### 2.3.2 RouteLLM: Routing from Preference Data
Ong et al. [@ong2025routellm] developed **RouteLLM**, framing router training around pairwise preference datasets derived from Chatbot Arena alongside synthetic comparisons scored by GPT-4. They evaluated four architectural designs: matrix factorization, similarity-weighted ranking over exemplar embeddings, fine-tuned BERT classifiers, and causal language model evaluators.

RouteLLM established standard formal metrics for the field:
- **Performance Gap Recovered (PGR):** The fraction of the performance delta between small model $S$ and large model $L$ achieved by router $R$:
  $$\text{PGR}(R) = \frac{Q(R) - Q(S)}{Q(L) - Q(S)}$$
- **Call-Percentage Threshold at $x\%$ (CPT($x\%$)):** The minimum escalation rate required to achieve $\text{PGR} \ge x/100$.
- **Area under the PGR Curve (APGR):** The integral of PGR over all possible escalation fractions, summarizing Pareto efficiency.

RouteLLM demonstrated that routing between Mixtral-8x7B and GPT-4 could yield over 2× cost savings on public benchmarks. However, two critical caveats govern their findings:
1. **Domain Augmentation Necessity:** In their domain-specific evaluations, preference data alone failed to generalize to structured reasoning tasks (such as MMLU or GSM8K); competitive routing required injecting labeled in-domain validation exemplars into the training set.
2. **Uncalibrated Test Thresholds:** RouteLLM reports evaluation curves swept over the test set retrospectively. The reported operating points lack finite-sample guarantees, leaving real-world deployments vulnerable to unexpected performance drops.

---

## 2.4 Benchmarks for LLM Routing: RouterBench

Systematic investigation of routing algorithms requires comprehensive multi-model response datasets. Hu et al. [@hu2024routerbench] introduced **RouterBench**, compiling 405,000 inference outcomes across 11 diverse language models evaluating 64 distinct benchmark tasks. RouterBench standardized the evaluation of routing efficiency by defining:
- **Average Improvement in Quality (AIQ):** The normalized area beneath a router’s convex hull in cost-accuracy space.
- **The Zero Router:** A baseline strategy that randomly routes queries to mimic the lower convex envelope of the candidate models, against which learned routers must demonstrate non-trivial gains.

While RouterBench provided a foundational resource for the research community, our deep forensic audit (Chapter 3) revealed critical structural shortcomings: formatting-induced evaluation collapses on mathematical benchmarks (GSM8K), chance-level performance on multiple-choice tasks for specific open-weight models, and label leakage across dataset splits. Consequently, raw benchmark corpora cannot be trusted without extensive sanitation.

---

## 2.5 Distribution-Free Risk Control and Conformal Inference

A critical vulnerability across existing cascade and routing literature is the absence of formal performance guarantees. When an operating cutoff $\tau$ is tuned on a finite validation set to satisfy $\text{PGR} \ge 0.95$, standard empirical estimates suffer from optimistic bias. Distribution-free risk control resolves this challenge by providing exact finite-sample bounds.

### 2.5.1 Risk-Controlling Prediction Sets (RCPS)
Bates et al. [@bates2021rcps] developed **Risk-Controlling Prediction Sets (RCPS)**, extending split conformal inference to control general statistical risk functions. Given an exchangeable calibration dataset and an upper tolerance level $\alpha$, RCPS constructs confidence bounds across a parameter grid, identifying the most aggressive parameter that rigorously guarantees $\mathbb{E}[\text{loss}] \le \alpha$ with probability at least $1 - \delta$. However, classical RCPS requires the underlying loss function to be monotonically non-increasing in the operating parameter.

### 2.5.2 Learn-then-Test (LTT)
Angelopoulos et al. [@angelopoulos2021ltt] formulated **Learn-then-Test (LTT)**, eliminating the monotonicity constraint by treating calibration as a multiple hypothesis testing problem. Each candidate threshold $\tau$ constitutes a null hypothesis:
$$H_0^\tau: \mathbb{E}[L(\tau)] > \alpha$$
By evaluating valid p-values $p(\tau)$ across candidate thresholds and controlling the family-wise error rate (FWER), LTT identifies a certified set of valid operating parameters. Under **fixed-sequence testing**—where thresholds are ordered monotonically from conservative to aggressive and evaluated sequentially until the first rejection failure—LTT controls FWER at level $\delta$ without incurring conservative Bonferroni penalties.

### 2.5.3 Betting-Based Concentration Inequalities (WSR)
The practical power of LTT depends entirely on the tightness of its underlying concentration bounds. While Hoeffding's inequality provides unconditional validity, its loose bounds discard substantial operational headroom. Waudby-Smith and Ramdas [@waudbysmith2024betting] introduced **betting-based confidence sequences (WSR)**. By casting mean estimation as a wealth process wherein a fictitious gambler places sequential bets against the hypothesized null mean:
$$K_t = \prod_{i=1}^t \left(1 - \lambda_i (X_i - \mu_0)\right)$$
Ville's maximal inequality guarantees that the probability of the wealth process ever exceeding $1/\alpha$ is bounded by $\alpha$. Betting martingales adapt dynamically to the empirical variance of the observations, providing finite-sample confidence intervals that approach the asymptotic efficiency of the empirical bootstrap while preserving exact mathematical validity.

---

## 2.6 Positioning of This Thesis

Table 2.1 situates this project within the broader landscape of model routing and cascade systems.

**Table 2.1 — Comprehensive Comparison with Closest Prior Art.**

| System / Paper | Decision Locus | Small Model Tier | Threshold Guarantee | Cost Accounting Metric | Verified Local Execution |
|---|---|---|---|---|---|
| **FrugalGPT** [@chen2024frugalgpt] | Cascade (Post-generation) | Cloud API models | None (Empirical sweep) | Pure API Token Price | No (Cloud only) |
| **MoT Cascades** [@yue2024cascades] | Cascade (Post-generation) | GPT-3.5 API | None (Empirical sweep) | Pure API Token Price | No (Cloud only) |
| **AutoMix** [@aggarwal2024automix] | Cascade (Verification prompt)| LLaMA-2-13B / Vicuna | None (Empirical sweep) | Normalized Compute Ratios| No (Server only) |
| **Hybrid LLM** [@ding2024hybrid] | Router (Query-only) | LLaMA-2-7B / Mistral-7B | None (Empirical sweep) | Frontier Call Percentage| No (Simulation) |
| **RouteLLM** [@ong2025routellm] | Router (Query-only) | Mixtral-8x7B (API) | None (Empirical sweep) | Pure API Token Price | No (API proxy) |
| **This Thesis** | **Router & Cascade (Hybrid)** | **Qwen3.5-4B (Edge)** | **Yes (LTT + Seed-Free WSR)** | **Dual: API (M1) & Hardware (M2)** | **Yes (4-bit MLX on Mac)** |

### Gaps Addressed by This Project:
1. **From Optimistic Curves to Provable Guarantees:** Prior routing literature reports uncertified Pareto frontiers. We incorporate Learn-then-Test and seed-free betting martingales to deliver the first mathematically guaranteed operating points for LLM routing.
2. **From Query Guesses to Cascade Confidence:** We demonstrate that text embeddings alone achieve negligible certified savings on high-capability small models, whereas generation-level consensus and probability metrics unlock 41.3% certified cost reductions.
3. **From Synthetic Token Math to Realistic Edge Economics:** While prior studies assume local inference is free or analyze abstract FLOP counts, we implement a production-grade 4-bit MLX engine on Apple Silicon, measuring real wall-clock latency, energy dissipation, and amortized hardware depreciation.
4. **Data Hygiene as an Experimental Prerequisite:** We show that standard benchmark datasets suffer from severe grading distortions and split contamination, providing a reproducible, leak-free corpus with fully verified gold targets.
