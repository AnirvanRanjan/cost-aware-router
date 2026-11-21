# Chapter 5 — Cost Model, Threshold Selection, Statistical Guarantees, and Empirical Results

> **Author Declaration:** This chapter was authored by Sarthak Jakhmola (Lane C: Calibration, Evaluation, and Risk Control). In accordance with University of Petroleum and Energy Studies (UPES) guidelines on academic integrity and generative artificial intelligence assistance, AI tools were utilized exclusively as an editorial and drafting assistant to refine syntactic structure, generate KaTeX formatting, and assemble tables from verified empirical logs (`scripts/17_t60_seed_free_wsr.py`, `scripts/08_threshold_report.py`, and `notes/trackb_cascade_report.md`). All theoretical formulations, mathematical proofs, experimental designs, and statistical evaluations were conceived, implemented, verified, and audited by the author.

---

## 5.1 Introduction and Formal Problem Formulation

The economic challenge of deploying Large Language Models (LLMs) stems from a stark asymmetry: high-capability frontier models (such as GPT-4 and Claude Sonnet 4.6) command premium API pricing, while compact open-weight models can execute on local hardware at near-zero marginal operational cost. A cost-aware router dynamically assigns an incoming prompt $q \in \mathcal{X}$ to either:
1. A local edge model $S$ incurring zero direct API fees (or minimal amortized device expense), or
2. A cloud frontier model $L$ characterized by state-of-the-art capability at significant per-token expenditure.

Let $s_S(q), s_L(q) \in [0, 1]$ represent the ground-truth accuracy scores of models $S$ and $L$ on prompt $q$, and let $c_S(q), c_L(q) \ge 0$ denote their respective monetary inference costs in USD. A routing policy $\pi: \mathcal{X} \to \{S, L\}$ maps each input to a selected model. The expected quality and cost under policy $\pi$ are defined as:
$$Q(\pi) = \mathbb{E}[s_{\pi(q)}(q)], \quad C(\pi) = \mathbb{E}[c_{\pi(q)}(q)]$$

Rather than pursuing unconstrained scalar optimization ($\max Q - \lambda C$), enterprise production deployments demand hard performance preservation constraints. Specifically, the system must retain a target fraction $\rho \in (0, 1)$ (conventionally $\rho = 0.95$) of the total quality gain achievable by the frontier model over the local baseline:
$$\text{PGR}(\pi) = \frac{Q(\pi) - Q_S}{Q_L - Q_S} \ge \rho$$
where $Q_S = \mathbb{E}[s_S(q)]$ and $Q_L = \mathbb{E}[s_L(q)]$ denote the baseline capabilities of the local and frontier systems.

### 5.1.1 Routing Decision Rules

We consider threshold-based decision rules parameterized by cutoff $\tau \in \mathbb{R}$. A predictive router provides estimated qualities $\hat{s}_S(q), \hat{s}_L(q) \in [0, 1]$ (or predicted capability gap $\hat{\Delta}(q) = \hat{s}_L(q) - \hat{s}_S(q)$) and predicted frontier call cost $\hat{c}_L(q)$:

1. **Rule v1 (Quality Gap Thresholding):**
   Escalate to frontier model $L$ whenever the anticipated performance gain exceeds cutoff $\tau$:
   $$\pi_{v1}(q; \tau) = \begin{cases} L & \text{if } \hat{\Delta}(q) > \tau \\ S & \text{otherwise} \end{cases}$$

2. **Rule v2 (Cost-Normalized Return Thresholding):**
   Queries exhibit wide variation in context length and generation volume. Escalating a lengthy, expensive prompt yields lower marginal utility than escalating a concise, low-cost query with identical quality delta. Rule v2 weights the predicted advantage by expected expenditure:
   $$\pi_{v2}(q; \tau) = \begin{cases} L & \text{if } \text{Gain}(q) = \frac{\hat{\Delta}(q)}{\hat{c}_L(q)} > \tau \\ S & \text{otherwise} \end{cases}$$
   where $\hat{c}_L(q) = (p_{\text{in}} \cdot \hat{n}_{\text{in}} + p_{\text{out}} \cdot \hat{n}_{\text{out}}) \times 10^{-6}$.

---

## 5.2 Dual Cost Accounting: Models M1 and M2 and the Batch Baseline

Published routing benchmarks frequently treat on-device inference as entirely free while evaluating cloud endpoints solely on on-demand token prices. To ground our analysis in realistic systems economics, we formulate two complementary cost regimes in `router/costs.py`.

### 5.2.1 Cost Model M1: Direct Cloud API Expenditure

Model M1 reflects the perspective of an engineering team operating pre-allocated or sunk local hardware resources. Local computation incurs no marginal billing:
$$c_S^{(M1)}(q) = 0$$
$$c_L^{(M1)}(q) = \frac{p_{\text{in}} \cdot n_{\text{in}}(q) + p_{\text{out}} \cdot n_{\text{out}}(q)}{10^6}$$
Under our standard frontier reference pricing (`claude-sonnet-4.6` rates mapped per DESIGN Rev A / `configs/models.yaml`), input tokens cost $p_{\text{in}} = \$3.00$ per million tokens and output tokens cost $p_{\text{out}} = \$15.00$ per million tokens. Counterfactual costs across all evaluation rows are tracked via `PriceTable`.

### 5.2.2 Cost Model M2: Amortized Local Hardware and Energy Costs

Model M2 incorporates capital asset depreciation and electrical energy consumption during local model execution. We instantiate this model for an Apple M2 MacBook Air (16 GB unified RAM), strictly separating explicit assumptions from empirical hardware benchmarks:

**Hardware Assumptions (`configs/models.yaml`):**
- Hardware purchase price: $P_{\text{hw}} = \$1,199$
- Expected operational lifecycle: $Y = 3$ years (straight-line amortisation)
- Active daily utilization: $H = 8$ hours/day
- Power draw assumption: $W = 20$ watts (assumed active package power during local decoding)
- Commercial electricity tariff: $R_{\text{kwh}} = \$0.12/\text{kWh}$

$$\text{DepreciationPerHour} = \frac{P_{\text{hw}}}{Y \times 365 \times H} = \frac{1199}{3 \times 365 \times 8} \approx \$0.1369/\text{hour}$$
$$\text{ElectricityPerHour} = \frac{W}{1000} \times R_{\text{kwh}} = \frac{20}{1000} \times 0.12 = \$0.0024/\text{hour}$$
$$\text{TotalOperationalCostPerHour} \approx \$0.1393/\text{hour} \quad (\approx \$0.139/\text{hour})$$

**Empirical Throughput Measurements (`scripts/estimate_spend.py`):**
- Autoregressive decoding rate $v$:
  - `qwen3.5-4b`: **28.5 tok/s (measured)** on Apple M2 (batch size 1, 4-bit MLX)
  - `llama-3.2-3b`: **41.6 tok/s (measured)** on Apple M2 (batch size 1, 4-bit MLX)
  - Hardware estimates for comparison: `phi-4-mini` (37 tok/s), `gemma-3-4b` (32 tok/s), `qwen3.5-9b` (13 tok/s)
- Generating $n_{\text{out}}$ tokens on-device incurs an amortized expense:
$$c_S^{(M2)}(q) = \frac{n_{\text{out}}(q)}{v \times 3600} \times \text{TotalOperationalCostPerHour}$$
For a representative GSM8K problem (120 input tokens, 250 output tokens), local generation with Qwen3.5-4B costs $\$0.351$ per 1,000 queries ($\$0.244$ for Llama 3.2 3B).

#### Economic Crossover Decoding Speed ($v_{\text{cross}}$)
Local edge inference is financially superior to cloud invocation only when $c_S^{(M2)}(q) < c_L^{(M1)}(q)$. Equating both cost expressions reveals the break-even throughput threshold:
$$v_{\text{cross}} = \frac{n_{\text{out}} \times \text{TotalOperationalCostPerHour}}{3600 \times c_L^{(M1)}(q)}$$
At our M2 operating rate ($\$0.139/\text{hour}$), on-device execution economically beats:
- `claude-haiku-4.5` (Batch pricing) above **7.1 tok/s**
- `gemini-3-flash` (List pricing) above **12.1 tok/s**
- `gpt-5-nano` (Batch pricing) above **102.5 tok/s**

Because the Apple M2 generates at 28.5 tok/s for Qwen3.5-4B (and 41.6 tok/s for Llama 3.2 3B), local edge serving cleanly outperforms lightweight cloud endpoints like Claude Haiku and Gemini 3 Flash in unit economics under Cost Model M2.

### 5.2.3 The Cloud Batch API Baseline

Commercial cloud providers offer asynchronous Batch endpoints providing a standard 50% discount against interactive rates:
$$C_{\text{batch}}(q) = 0.5 \times C_{\text{API}}(q)$$
For non-interactive or offline workloads, an interactive router that saves less than 50% total spend is economically dominated by routing all queries directly to the cloud Batch API. Our reporting benchmarks system savings against both standard on-demand and 50% discounted batch baselines.

---

## 5.3 Exact Breakpoint Sweep and Concave Pareto Envelope

Rather than relying on coarse heuristic grids that introduce discretization errors, `router/threshold.py` implements an exact $O(n \log n)$ prefix-sum algorithm (`sweep()`):
1. Given $n$ calibration instances, compute sorting permutation $\sigma$ such that:
   $$\text{score}_{\sigma(1)} \ge \text{score}_{\sigma(2)} \ge \dots \ge \text{score}_{\sigma(n)}$$
2. Evaluate prefix sums of routed quality, costs, and decisions across all unique sample transitions $k \in \{0, 1, \dots, n\}$.
3. Compute the upper concave hull over the resulting $(\alpha, \text{PGR})$ operating points via `upper_envelope()`.

The concave envelope establishes the supremum performance reachable by any randomized policy interpolating between two hull thresholds $\tau_a$ and $\tau_b$. The Cost-Per-Target metric $\text{CPT}(\rho)$ identifies the minimal frontier escalation frequency $\alpha$ necessary to satisfy $\text{PGR} \ge \rho$ on this envelope.

---

## 5.4 Distribution-Free Calibration via Learn-then-Test (LTT)

Selecting threshold $\tau$ post-hoc on test data introduces selection bias and produces overly optimistic savings estimates. To guarantee that deployed routers do not degrade user experience below $\rho = 0.95$ with user-specified failure probability $\delta = 0.05$, we apply the **Learn-then-Test (LTT)** risk control framework [@angelopoulos2021ltt].

### 5.4.1 The Paired Margin Statistic $e_i(\tau)$

Preserving performance $\text{PGR}(\tau) \ge \rho$ is mathematically equivalent to:
$$\mathbb{E}[s_{\pi(q)}(q)] \ge (1 - \rho) Q_S + \rho Q_L$$
For every sample $i$ in calibration split $\mathcal{D}_{\text{cal}}$, we define the paired margin random variable:
$$e_i(\tau) = s_{\text{route}(i)} - \left[(1 - \rho) s_{S,i} + \rho s_{L,i}\right]$$
- If escalated to frontier $L$: $e_i(\tau) = (1 - \rho)(s_{L,i} - s_{S,i})$
- If retained on local $S$: $e_i(\tau) = \rho (s_{S,i} - s_{L,i})$

Computing differences paired on the exact same question eliminates cross-query difficulty variance, providing variance reduction without requiring parametric assumptions.

### 5.4.2 Sequential Hypothesis Testing Protocol

1. An ordered grid of candidate thresholds is constructed exclusively on training data:
   $$\tau_1 < \tau_2 < \dots < \tau_K$$
   The sequence begins at the most conservative threshold ($\tau_1 = -\infty$, which routes all queries to the frontier model $L$, trivially satisfying the target PGR) and ascends toward progressively larger thresholds that route fewer queries to $L$.
2. For each candidate threshold in ascending order, we formulate the null hypothesis:
   $$H_{0, k}: \mathbb{E}[e(\tau_k)] < 0 \quad (\text{i.e. } \text{PGR}(\tau_k) < \rho)$$
3. We test sequentially using valid finite-sample p-values $p_k$. Testing terminates at the very first failure index $k_{\text{fail}} = \min \{k : p_k > \delta\}$.
4. The calibrated threshold is selected as the last threshold that successfully rejected the null hypothesis: $\tau^* = \tau_{k_{\text{fail}} - 1}$.

**Family-Wise Error Rate (FWER) Control:**  
Because the sequence of hypotheses is evaluated in a fixed order starting from the safe all-frontier baseline and halts at the first non-rejection, testing achieves rigorous FWER control:
$$\mathbb{P}(\text{Type I Error}) \le \delta$$
without requiring Bonferroni or Holm-Šidák multi-testing penalties.

---

## 5.5 Comparison of Concentration Bounds and the Seed-Free Martingale Solution

### 5.5.1 The Bootstrap Undercoverage Trap

Standard empirical practice relies on the percentile bootstrap under Central Limit Theorem assumptions. However, in LLM routing:
1. **Bounded Support with Zero Point Mass**: When models agree ($s_{S,i} = s_{L,i}$), the margin is identically zero ($e_i = 0$). In benchmarks where both models answer correctly or both fail on 70% to 80% of items, the margin distribution spikes heavily at 0.
2. **Small Headroom Regimes**: When $Q_L - Q_S \le 0.07$, the true expectation $\mathbb{E}[e]$ sits near the null boundary ($0$).
3. **Severe Empirical Undercoverage**: Monte Carlo simulations on held-out calibration sets ($n_{\text{cal}} = 2,000$) demonstrate that nominal 95% bootstrap lower confidence bounds fail to cover the true parameter up to **15.2% of the time** (more than 3× the nominal risk $\delta = 0.05$).

### 5.5.2 Hoeffding's Inequality (Conservative Baseline)

Hoeffding's inequality provides valid finite-sample p-values for bounded variables $e_i \in [-a, b]$ with range $R = a + b$:
$$p_{\text{Hoeffding}} = \exp\left(-\frac{2 n (\bar{e})^2}{R^2}\right)$$
While distribution-free, Hoeffding's bound cannot leverage empirical variance, producing excessively conservative bounds that discard up to 90% of achievable cost savings.

### 5.5.3 WSR Betting Martingales and the Permutation Variance Problem

To achieve sharp, variance-adaptive bounds with exact finite-sample validity, we implement the betting martingale bound of Waudby-Smith & Ramdas [@waudbysmith2024betting]. By scaling margins into normalized losses $l_i \in [0, 1]$, we construct the sequential wealth process:
$$K_t = \prod_{i=1}^t \left(1 - \nu_i (l_i - 0.5)\right)$$
where $\nu_i \in [0, 1)$ are predictable bet fractions tuned via sample variance. By **Ville's maximal inequality for supermartingales**:
$$\mathbb{P}\left(\exists t \ge 1 : K_t \ge \frac{1}{\delta}\right) \le \delta$$

**The Permutation Variance Dilemma:**  
While Ville's inequality guarantees validity unconditionally for any fixed sequence, the sequential betting path depends heavily on the arbitrary random ordering of the calibration data. In our empirical investigations across permutation seeds 0 to 19 on the standard training grid ($n=1,000$), realized test savings fluctuated wildly:
- Across seeds 0 to 19, certified savings ranged from **40.8% to 74.9%** (test escalation rates spanning 41.2% to 77.7%).
- Selecting an operating cutoff or grid density solely because a specific seed happened to yield high savings constitutes hidden test leakage and breaks the finite-sample certificate.

### 5.5.4 The Seed-Free Solution: Averaging Terminal E-Values

To eliminate permutation noise rigorously without sacrificing validity:
1. **Terminal Wealth as an E-Value**: For any fixed permutation $m$, the terminal martingale wealth at sample size $n$:
   $$E^{(m)} = K_n^{(m)} = \prod_{i=1}^n \left(1 - \nu_i (l_i - 0.5)\right)$$
   satisfies $\mathbb{E}[E^{(m)}] \le 1$ under the null hypothesis $H_0: \mathbb{E}[e(\tau)] < 0$. Hence, $E^{(m)}$ is an exact e-value.
2. **Mixture of E-Values**: By linearity of expectation, the arithmetic mean of terminal wealth across $M=50$ independent permutations:
   $$\bar{E} = \frac{1}{M} \sum_{m=1}^M E^{(m)}$$
   is also an exact, valid e-value ($\mathbb{E}[\bar{E}] \le 1$ under $H_0$).
3. **Markov P-Value**: Applying Markov's inequality:
   $$p = \min\left(1, \frac{1}{\bar{E}}\right)$$
   yields an unconditionally valid, finite-sample p-value free of single-sequence seed variance. This estimator evaluates the complete calibration sample without taking the running maximum over $t$, completely resolving the 40.8%–74.9% permutation oscillation.

This algorithm is formalized as `select_tau(method="wsr_avg")` in `router/threshold.py` and validated against `scripts/17_t60_seed_free_wsr.py`.

---

## 5.6 Empirical Benchmark Results

### 5.6.1 Headline Results on Qwen3.5-4B vs. GPT-4 (`cascade_3sample`)

Table 5.1 presents the locked benchmark results on the frozen Track B test set ($N=4,707$ queries across MMLU, GSM8K, ARC-Challenge, and Winogrande) evaluated on the official release file `cascade_predictions.parquet` (release `trackb-6265adf7ffa4a602`).

**Table 5.1 — Locked Headline Results for Qwen3.5-4B vs. GPT-4 on Track B Test Split.**

| Router Model | Decision Rule | Calibration Method | Cal $\tau^*$ | Test Escalation $\alpha_{\text{test}}$ [95% CI] | Realized Test PGR [95% CI] | Certified API Savings (M1) [95% CI] | Amortized Hardware Savings (M2) |
|---|---|---|--:|--:|:---:|:---:|:---:|
| **Question-Only (`query_f2`)** | Rule v2 | Seed-Free WSR | 9.8934 | 98.1% [97.7, 98.5] | 98.6% [96.1, 101.0] | **2.1% [1.5, 2.7]** | **2.0%** |
| **Cascade 1-Sample (Sample-0 Quality)** | Rule v2 | Seed-Free WSR | 13.6310 | 67.9% [66.5, 69.2] | 98.4% [95.2, 101.6] | **29.5% [27.8, 31.5]** | **24.7%** |
| Cascade 1-Sample (Majority Quality) | Rule v2 | Seed-Free WSR | 13.5077 | 68.7% [67.3, 70.0] | 97.7% [94.4, 101.1] | 27.8% [26.0, 29.7] | 23.0% |
| **Cascade 3-Sample (`cascade_3sample`)** | **Rule v2** | **Seed-Free WSR ($M=50$)** | **8.2187** | **75.9% [74.7, 77.1]** | **102.8% [99.1, 107.0]** | **41.3% [39.0, 43.5]** | **36.5% [34.3, 38.7]** |
| Cascade 3-Sample (`cascade_3sample`) | Rule v2 | Naive on cal | 4.1200 | 42.9% [41.5, 44.3] | 97.2% [91.9, 103.2] | 77.0% [75.4, 78.7] | 73.3% |
| Cascade 3-Sample (`cascade_3sample`) | Rule v2 | Post-hoc Test Curve | — | 39.5% | 95.0% | 78.9% (uncertified) | 71.2% |

*Accounting Notes on Model M2 and Single-Sample Evaluation:*
1. **Model M2 Accounting Conventions**: In a question-only router (`query_f2`), local inference runs exclusively on queries assigned to model $S$; charging local execution only when queries are kept on-device yields **2.0% M2 savings** (matching M1 spend closely). In cascade routing, local inference is evaluated prior to escalation. For `cascade_3sample`, the headline **36.5% M2 savings** assesses local execution charging one full generation per question (the multi-choice logit pass plus a primary autoregressive solution). If all three sampled generation rollouts on GSM8K are fully billed at local hardware amortisation rates, M2 savings adjust to **28.4%** (direct M1 API spend savings remain unchanged at 41.3%).
2. **1-Sample Cascade Calibration**: When recalibrated on true single-sample responses (where the user receives sample 0 directly), seed-free WSR yields $\tau^* = 13.6310$, providing **29.5% [27.8%, 31.5%] certified API spend savings** ($\alpha_{\text{test}} = 67.9\%$, PGR = 98.4%, M2 = 24.7%). Evaluated against the majority-of-3 label as a proxy, the cutoff is $\tau^* = 13.5077$ (27.8% savings). We report the 29.5% figure as the true user-facing metric for single-sample deployments.

### 5.6.2 Key Empirical Takeaways

1. **Certified Savings vs. Post-hoc Observations:** Scanning the empirical test curve retrospectively suggests apparent cost savings of **78.9%** (a 4.7× cost reduction). However, statistical risk control via seed-free WSR establishes the certified, production-safe savings at **41.3% [39.0%, 43.5%]** (test $\text{PGR} = 102.8\%$). The difference represents the essential safety premium required to guarantee performance without data leakage.
2. **Failure of Question-Only Routing under Narrow Headroom:** On query text features alone (`query_f2`), the seed-free calibrated router saves only **2.1% [1.5%, 2.7%]** certified spend, requiring a 98.1% escalation rate. Because the capability gap between Qwen3.5-4B and GPT-4 is narrow ($\Delta = 0.0748$), prompt syntax alone cannot reliably identify failure modes.
3. **Generation Diagnostics Drive Cascade Superiority:** In contrast to query embeddings, inspecting Qwen's own generation consensus (3-sample agreement on GSM8K and chosen option probabilities on MCQ) unlocks **41.3% certified spend reduction**.
4. **Reproducibility and Software Environment Dependency:** Rebuilding gradient-boosted trees on differing scikit-learn versions alters tree leaf splits: the release model fitted with `scikit-learn==1.8.0` certifies at $\tau^* = 8.2187$ (**41.3% savings**), whereas refitting on `scikit-learn==1.9.1` yields $\tau^* = 9.657$ (certifying at 48.3% [46.1%, 50.5%]). Both are valid certificates for their respective fitted models; this thesis locks and quotes the official release file (`trackb-6265adf7ffa4a602`) with `scikit-learn==1.8.0` pinned in `requirements.txt`.

---

## 5.7 Cross-Model Headroom Sweep (Task T27)

To evaluate router behavior across varying capability deltas, Table 5.2 compares three candidate small models against `gpt-4-1106-preview` evaluated on the **exact same held-out Track B question set** ($N=4,707$ test questions).

**Table 5.2 — Cross-Model Headroom Sweep on Track B Common Test Questions ($N=4,707$).**

| Small Model ($S$) | Large Model ($L$) | $Q_S$ | $Q_L$ | Headroom $\Delta$ | Oracle CPT(95%) | Oracle Savings (uncertified) | Router Architecture | Router CPT(95%) | Router Savings (uncertified) |
|---|---|--:|--:|--:|--:|--:|---|--:|--:|
| `mixtral-8x7b-chat` | `gpt-4-1106-preview` | 0.6501 | 0.8874 | **0.2373** | 22.6% | 79.9% | Question-Only (`F2`) | 95.2% | **12.9%** |
| `Yi-34B-Chat` | `gpt-4-1106-preview` | 0.6820 | 0.8874 | **0.2054** | 19.5% | 85.4% | Question-Only (`F2`) | 96.4% | **12.0%** |
| `qwen3.5-4b` | `gpt-4-1106-preview` | 0.8126 | 0.8874 | **0.0748** | 7.1% | 98.5% | Question-Only (`query_f2`) | 85.9% | **21.1%** |
| `qwen3.5-4b` | `gpt-4-1106-preview` | 0.8126 | 0.8874 | **0.0748** | 7.1% | 98.5% | Cascade (`cascade_3sample`) | 39.5% | **78.9%** |

*Note: All savings figures above reflect test-curve operating points and are explicitly uncertified.*

### Observations:
1. **Headroom Compression:** Moving from Mixtral ($\Delta = 0.2373$) to Yi-34B ($\Delta = 0.2054$) and Qwen3.5-4B ($\Delta = 0.0748$) contracts the capability gap threefold.
2. **Universal Limits of Question-Only Classifiers:** Across all three pairs, question-only routing yields modest uncertified savings ($12.0\%$ to $21.1\%$), escalating $85.9\%–96.4\%$ of prompts to attain $\text{PGR} \ge 0.95$.
3. **Cascade Leverage:** Generation-level inspection on the Qwen pair boosts uncertified curve savings from $21.1\%$ to $78.9\%$ at CPT $39.5\%$, confirming that output verification substantially outperforms prompt embeddings.

---

## 5.8 Leave-One-Subject-Out (LOSO) Domain Generalization (Task T26)

On MMLU (2,808 test rows across 57 academic subjects), subject diversity tests whether the router memorizes subject-specific vocabulary or captures generalized task difficulty. We evaluate the development pair (`mixtral-8x7b-chat` vs. `gpt-4-1106-preview`) across 10 held-out subject folds (`scripts/13_loso_mmlu.py`):

**Table 5.3 — Leave-One-Subject-Out Cross-Validation on MMLU (Mixtral Pair, 10 Folds).**

| Evaluation Scheme | Comparison Basis | Test Rows | ROC-AUC | CPT(95%) | Savings @ PGR .95 | APGR |
|---|---|--:|:--:|:--:|:--:|:--:|
| **Pooled Baseline (Fold Mean)** | Same folds averaged | 2,808 | 0.6177 | 0.8690 | 14.6% | 0.5865 |
| **LOSO Retrained (Fold Mean)** | Same folds averaged | 2,808 | 0.5896 | 0.9130 | 10.3% | 0.5633 |
| **Pooled Baseline (All Test Rows)** | All test rows scored | 2,808 | 0.6652 | 0.9000 | 11.7% | 0.6331 |
| **LOSO Out-of-Fold (All Test Rows)** | All test rows scored | 2,808 | 0.6171 | 0.9100 | 10.1% | 0.5845 |

**Findings:**
- Out-of-fold generalization incurs an average ROC-AUC drop of $0.028$ to $0.048$ (savings reducing from 14.6% to 10.3% across fold averages).
- While aggregate out-of-fold savings remain at 10.1%, subject-level variation is wide: savings across individual held-out folds span from **29.7% on the best fold (Fold 5)** down to **4.0% on the most challenging fold (Fold 6)**. This variation illustrates that text representations remain sensitive to domain shifts across specialized disciplines.

---

## 5.9 Interactive Replay and Latency Profiling (Task T44)

To verify serving pipeline integrity between offline experimental code and real-time execution, `scripts/15_replay_demo.py` processed 200 held-out test queries through the serving engine path:

1. **Interim Development Replay (Mixtral Pair)**: Replaying through the development bundle (`artifacts/bundle/router.pkl`, calibrated cutoff $\tau^* = 0.0791$) confirmed 100.0% decision agreement with offline references, an escalation frequency $\alpha = 92.0\%$, and a CPU routing overhead of median 14.3 ms (p99 29.7 ms).
2. **Production Demonstration Replay (D-15 Qwen Cascade Bundle, PR #23 / `notes/replay_report_d15.md`)**: Replaying 200 held-out Track B test queries through the live server path (`router/serve.py`, `Demo._answer_local_first`) with the certified D-15 bundle (`cascade_t61_3sample`, $\tau^* = 8.2187$):
   - **Decision Agreement**: **200/200 (100.0% exact match)** against offline reference evaluation.
   - **Frontier Escalation Rate ($\alpha_{\text{sample}}$)**: **74.5%** (51 local, 149 frontier calls).
   - **Spend Savings**: **37.3% [29.8%, 44.8%]** on this 200-query slice ($0.0780 saved of $0.2093 counterfactual).
   - **Router Decision Latency**: **Median 3.57 ms, 99th percentile 5.43 ms** (200 decisions executed in 0.8s wall-clock time).

These results confirm that the calibrated decision boundaries, cost models, and feature extractors operate identically in interactive serving.
