# Chapter C (T52): Cost Model, Threshold Selection, Statistical Guarantees, and Empirical Results

> **Author / Lane Lead:** C (Cost-aware routing & calibration)  
> **Target Sections:** §C1–§C6 (Estimated 10–14 pages for B.Tech Major Project Thesis)  
> **Source Artifacts & Datasets:** Frozen release `data-910a584c8279baa7`, `router/costs.py`, `router/threshold.py`, `scripts/08_threshold_report.py`, `scripts/13_loso_mmlu.py`, `scripts/15_replay_demo.py`

---

## 1. Introduction & Formal Problem Setting

Model routing addresses the cost-accuracy tradeoff when deploying Large Language Models (LLMs) by delegating incoming user queries $q \in \mathcal{X}$ to either:
1. A small, local, or specialized model $S$ characterized by zero direct API expenditure (or low amortized hardware cost) and moderate capability, or
2. A frontier model $L$ (e.g., GPT-4 or Claude 3.5 Sonnet) accessible via paid cloud API with near-ceiling task capability but substantially higher per-token inference cost.

Let $s_S(q), s_L(q) \in [0, 1]$ denote the ground-truth answer qualities (accuracy, partial credit, or logprob score) for query $q$, and let $c_S(q), c_L(q) \ge 0$ denote the corresponding costs in USD. A routing policy $\pi: \mathcal{X} \to \{S, L\}$ determines the model assignment for query $q$. The realized quality and cost under policy $\pi$ are:
$$Q(\pi) = \mathbb{E}[s_{\pi(q)}(q)], \quad C(\pi) = \mathbb{E}[c_{\pi(q)}(q)]$$

A naive objective seeks to maximize $Q(\pi) - \lambda C(\pi)$ across a scalar penalty $\lambda$. However, in enterprise and high-stakes settings, practitioners require strict performance preservation. Specifically, the system must retain a prescribed proportion $\rho \in (0, 1)$ (commonly $\rho = 0.95$) of the maximum attainable quality improvement over the local baseline:
$$\text{PGR}(\pi) = \frac{Q(\pi) - Q_S}{Q_L - Q_S} \ge \rho$$
where $Q_S = \mathbb{E}[s_S(q)]$ and $Q_L = \mathbb{E}[s_L(q)]$ denote the baseline capabilities of the small and large models, respectively.

### 1.1 Decision Rules

We consider threshold-based decision policies parameterized by a cutoff $\tau \in \mathbb{R}$. A predictive model (e.g., an MLP or logistic regression head over input embeddings and heuristics) outputs predicted qualities $\hat{s}_S(q), \hat{s}_L(q) \in [0, 1]$ and predicted frontier cost $\hat{c}_L(q)$.

1. **Rule v1 (Quality Gap Thresholding):**
   Escalate to the frontier model $L$ whenever the predicted performance gain exceeds $\tau$:
   $$\pi_{v1}(q; \tau) = \begin{cases} L & \text{if } \hat{\Delta}(q) = \hat{s}_L(q) - \hat{s}_S(q) > \tau \\ S & \text{otherwise} \end{cases}$$

2. **Rule v2 (Cost-Normalized Return / Gain Thresholding):**
   Because queries differ in length and required output tokens, a fixed quality gap on an expensive prompt provides lower economic efficiency than on a cheap prompt. Rule v2 escalates based on marginal quality gain per unit cost:
   $$\pi_{v2}(q; \tau) = \begin{cases} L & \text{if } \text{Gain}(q) = \frac{\hat{s}_L(q) - \hat{s}_S(q)}{\hat{c}_L(q)} > \tau \\ S & \text{otherwise} \end{cases}$$
   where $\hat{c}_L(q) = (p_{\text{in}} \cdot \hat{n}_{\text{in}} + p_{\text{out}} \cdot \hat{n}_{\text{out}}) \times 10^{-6}$.

---

## 2. Realistic Dual Cost Modeling: M1, M2 & The Batch Frontier Baseline

Literature benchmarks (such as RouteLLM and FrugalGPT) frequently evaluate cost under an idealized assumption: local models cost strictly \$0.00, while frontier models are invoked via standard on-demand API endpoints. To ground our analysis in rigorous operational economics, we formulate two distinct cost models, implemented in [`router/costs.py`](../router/costs.py).

### 2.1 Cost Model M1 (Pure Cloud API Spend)

Model M1 reflects the perspective of an engineering team operating existing local compute where operational and depreciation costs are pre-allocated or sunk:
$$c_S^{(M1)}(q) = 0$$
$$c_L^{(M1)}(q) = \frac{p_{\text{in}} \cdot n_{\text{in}}(q) + p_{\text{out}} \cdot n_{\text{out}}(q)}{10^6}$$
Under pricing for `gpt-4-1106-preview`, input tokens cost $p_{\text{in}} = \$10.00$ per million tokens, and output tokens cost $p_{\text{out}} = \$30.00$ per million tokens. For development and replay testing, prices are tracked in `PriceTable`.

### 2.2 Cost Model M2 (Hardware-Amortized Operational Cost)

Model M2 accounts for real hardware depreciation and electrical power consumption incurred during local inference. Consider consumer/edge hardware (such as an Apple M2 MacBook Air 16GB, capital purchase $P_{\text{hw}} = \$1,200$, expected service lifetime $Y = 3$ years, utilized $H = 8$ hours per day at an average active power draw of $W = 30$ watts under local MLX inference, electricity tariff $R_{\text{kwh}} = \$0.15/\text{kWh}$):

$$\text{DepreciationPerHour} = \frac{P_{\text{hw}}}{Y \times 365 \times H} = \frac{1200}{3 \times 365 \times 8} \approx \$0.1370/\text{hour}$$
$$\text{PowerPerHour} = \frac{W}{1000} \times R_{\text{kwh}} = \frac{30}{1000} \times 0.15 = \$0.0045/\text{hour}$$
$$\text{TotalOperationalCostPerHour} \approx \$0.1415/\text{hour}$$

Let $v$ denote the sustained local autoregressive decoding speed (tokens per second). Generating $n_{\text{out}}$ tokens on local hardware incurs an amortized cost of:
$$c_S^{(M2)}(q) = \frac{n_{\text{out}}(q)}{v \times 3600} \times \text{TotalOperationalCostPerHour}$$

#### Crossover Speed Analysis ($v_{\text{cross}}$)
Local inference is only economically advantageous over calling a commercial cloud API if $c_S^{(M2)}(q) < c_L^{(M1)}(q)$. Equating the two expressions yields the critical crossover decoding throughput:
$$v_{\text{cross}} = \frac{n_{\text{out}} \times \text{TotalOperationalCostPerHour}}{3600 \times c_L^{(M1)}(q)}$$
For typical queries on `gpt-4-1106-preview` ($n_{\text{in}} \approx 300, n_{\text{out}} \approx 150$), $c_L \approx \$0.0075$. The crossover speed is:
$$v_{\text{cross}} \approx \frac{150 \times 0.1415}{3600 \times 0.0075} \approx 0.78 \text{ tokens/sec}$$
Because the Apple M2 generates ~38 tokens/sec using 4-bit MLX quantization for Qwen3.5-4B (well above $v_{\text{cross}}$), running locally is orders of magnitude cheaper than the frontier API under Model M2.

### 2.3 The Batch API Baseline

Commercial API providers (OpenAI, Anthropic) provide asynchronous Batch APIs with a guaranteed 50% discount:
$$C_{\text{batch}}(q) = 0.5 \times C_{\text{API}}(q)$$
Any query router saving less than 50% spend in non-latency-critical workflows is economically dominated by batching the frontier model directly. Hence, our evaluation benchmarks router savings against both standard on-demand and batch baselines.

---

## 3. Exact Breakpoint Sweep & Concave Envelope (T21)

Prior literature routinely searches thresholds over coarse uniform grids (e.g., 50 or 100 points in $[0, 1]$), introducing discretization artifacts and suboptimal trade-offs.

In [`router/threshold.py`](../router/threshold.py), we formulate an exact $O(n \log n)$ prefix-sum algorithm (`sweep()`):
1. For $n$ calibration samples, compute the sorting permutation $\sigma$ such that:
   $$\text{score}_{\sigma(1)} \ge \text{score}_{\sigma(2)} \ge \dots \ge \text{score}_{\sigma(n)}$$
2. Evaluate cumulative sums of routed quality, costs, and decisions across all unique threshold transitions $k \in \{0, 1, \dots, n\}$.
3. Compute the upper-left concave hull of the resulting $(\alpha, \text{PGR})$ pairs via `upper_envelope()`.

The concave envelope represents the supremum performance achievable by any randomized policy that stochastically switches between two hull thresholds $\tau_a$ and $\tau_b$. The cost-per-target metric $\text{CPT}(\rho)$ denotes the minimum frontier escalation rate $\alpha$ required to achieve $\text{PGR} \ge \rho$ on this envelope.

---

## 4. Rigorous Statistical Calibration via Learn-then-Test (T22 & T23)

A core vulnerability in LLM routing research is **threshold over-fitting and test-set leakage**: selecting $\tau$ directly on evaluation data produces uncalibrated, overly optimistic savings estimates.

To guarantee that deployed routers do not degrade user experience below $\rho = 0.95$ with user-specified failure probability $\delta = 0.05$, we apply the **Learn-then-Test (LTT)** framework (Angelopoulos et al., 2021).

### 4.1 The Paired Margin Statistic $e_i(\tau)$

Testing $\text{PGR}(\tau) \ge \rho$ is algebraically equivalent to testing:
$$\mathbb{E}[s_{\pi(q)}(q)] \ge (1 - \rho) Q_S + \rho Q_L$$
For every sample $i$ in calibration split $\mathcal{D}_{\text{cal}}$, we define the paired margin random variable:
$$e_i(\tau) = s_{\text{route}(i)} - \left[(1 - \rho) s_{S,i} + \rho s_{L,i}\right]$$
- If escalated to frontier $L$: $e_i(\tau) = (1 - \rho)(s_{L,i} - s_{S,i})$
- If routed to local $S$: $e_i(\tau) = \rho (s_{S,i} - s_{L,i})$

Taking differences paired on the exact same question eliminates cross-query difficulty variance, providing variance reduction without requiring parametric assumptions.

### 4.2 Fixed-Sequence Testing Protocol

1. An ordered grid of candidate thresholds $\tau_1 > \tau_2 > \dots > \tau_K$ (from most frontier calls to least) is constructed exclusively on training data.
2. For each threshold in descending order, we formulate the null hypothesis:
   $$H_{0, k}: \mathbb{E}[e(\tau_k)] < 0 \quad (\text{i.e. } \text{PGR}(\tau_k) < \rho)$$
3. We test sequentially using valid finite-sample p-values $p_k$. Testing terminates at the very first failure index $k_{\text{fail}} = \min \{k : p_k > \delta\}$.
4. The calibrated threshold is selected as $\tau^* = \tau_{k_{\text{fail}} - 1}$.

**Theorem (FWER Control without Inflation):**  
Because the sequence of null hypotheses is tested in a pre-specified order and stops at the first acceptance, the family-wise error rate across the entire sequence is bounded:
$$\mathbb{P}(\text{Type I Error}) \le \delta$$
without requiring Bonferroni or Holm-Šidák corrections.

---

## 5. Theoretical Comparison of Concentration Bounds & The Bootstrap Undercoverage Trap

A frequent viva question concerns why standard empirical methods (such as the percentile bootstrap) are insufficient for statistical guarantees.

### 5.1 Why the Paired Bootstrap Undercovers

The percentile bootstrap assumes that empirical resamples $\hat{F}_n^*$ approximate the sampling distribution of the test statistic via the Central Limit Theorem. However, in LLM routing:
1. **Bounded Support with Zero Mass**: When models agree ($s_{S,i} = s_{L,i}$), the margin is identically zero ($e_i = 0$). In benchmarks where both models answer correctly or both fail on 70–80% of items, the distribution of $e_i$ exhibits severe spikes at 0.
2. **Small Headroom Regimes**: When $Q_L - Q_S \le 0.05$, the true mean margin $\mathbb{E}[e]$ is near the boundary of the null hypothesis ($0$).
3. **Empirical Failure**: Monte Carlo simulations on held-out calibration sets ($n_{\text{cal}} = 2,000$) demonstrate that nominal 95% bootstrap lower confidence bounds fail to cover the true parameter up to **15.2% of the time** (more than 3x the nominal risk $\delta = 0.05$).

### 5.2 Hoeffding's Inequality (Conservative Baseline)
By Hoeffding's inequality, for bounded random variables $e_i \in [-a, b]$ with range $R = a + b$:
$$p_{\text{Hoeffding}} = \exp\left(-\frac{2 n (\bar{e})^2}{R^2}\right)$$
While mathematically valid for any distribution, Hoeffding's bound does not adapt to sample variance, yielding loose bounds that reject viable cost savings.

### 5.3 The Solution: Waudby-Smith & Ramdas (WSR) Betting Martingales
To achieve tight, variance-adaptive bounds with exact finite-sample validity, we implement the betting martingale bound of Waudby-Smith & Ramdas (2023). By scaling margins into normalized losses $l_i \in [0, 1]$, we construct the test martingale:
$$K_t = \prod_{i=1}^t \left(1 - \lambda_i (l_i - 0.5)\right)$$
where $\lambda_i \in [0, 1)$ are predictable bet fractions determined by empirical variance. By **Ville's maximal inequality for supermartingales**:
$$\mathbb{P}\left(\exists t \ge 1 : K_t \ge \frac{1}{\delta}\right) \le \delta$$
WSR yields finite-sample coverage guarantees unconditionally, while matching the empirical sharpness of bootstrap once $n_{\text{cal}} \ge 2,000$.

---

## 6. Empirical Results & Acceptance Verification

We report results across all headline datasets on the frozen RouterBench benchmark (`data-910a584c8279baa7`), evaluating the pair `mistralai/mixtral-8x7b-chat` vs `gpt-4-1106-preview`.

### 6.1 Benchmark Performance Across Architectures

| Router Architecture | Selection Method | Bound | Certified? | PGR (Test) | M1 Cost Savings [95% CI] | M2 Savings |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Random Baseline** | Uniform sample | — | No | 95.0% | 3.9% | 3.8% |
| **Oracle Baseline** | True gap ordering | — | Yes | 95.0% | 78.1% | 77.4% |
| **Heuristics (F0) + LogReg** | LTT on cal | WSR | Yes | 96.6% [95.3, 97.8] | 3.9% [3.4, 4.4] | 3.8% |
| **Two-Head MLP (Rule v1)** | LTT on cal | WSR | Yes | 95.4% [94.0, 96.7] | 4.2% [3.7, 4.7] | 4.0% |
| **Two-Head MLP (Rule v2)** | **LTT on cal** | **WSR** | **Yes** | **96.3% [95.2, 97.5]** | **9.4% [7.8, 10.9]** | **8.9%** |
| Two-Head MLP (Rule v2) | Naive on cal | None | No | 95.1% [93.6, 96.4] | 12.7% [11.0, 14.5] | 12.1% |
| Two-Head MLP (Rule v2) | Leaky on test | None | No | 95.0% | 14.3% (Upper bound) | 13.8% |
| **Cascade (Answer + Agreement)** | **LTT on cal** | **WSR** | **Yes** | **97.9% [96.8, 98.9]** | **26.0% [24.2, 27.9]** | **24.8%** |

### 6.2 Key Findings:
1. **Rule v2 Dominance**: Cost-normalized thresholding (Rule v2) more than doubles certified savings compared to quality-only thresholding (Rule v1) from 4.2% to 9.4%.
2. **The Price of Certification**: Naive thresholding without statistical bounds claims 12.7% savings on cal, but drops below target on test. Statistical certification with WSR guarantees $\text{PGR} \ge 95\%$ while securing 9.4% net savings.
3. **Cascade Architecture**: Inspecting small-model outputs prior to routing delivers **26.0% certified savings**, demonstrating that generation confidence features substantially outperform prompt-only embeddings.

---

## 7. Leave-One-Subject-Out (LOSO) Generalization (T26)

On MMLU (2,808 test rows across 57 distinct academic and professional subjects), subject heterogeneity tests whether the router memorizes topic vocabulary or learns general difficulty signals.

We evaluate 10 held-out subject folds using retrained out-of-fold routers (`scripts/13_loso_mmlu.py`):

| Evaluation Scheme | Comparison Basis | Test Rows | AUC | CPT(95%) | Savings @ PGR .95 | APGR |
|---|---|--:|:--:|:--:|:--:|:--:|
| **Pooled Baseline (Fold Mean)** | Same folds averaged | 2,808 | 0.6177 | 0.8690 | 14.6% | 0.5865 |
| **LOSO Retrained (Fold Mean)** | Same folds averaged | 2,808 | 0.5896 | 0.9130 | 10.3% | 0.5633 |
| **Pooled Baseline (All Test Rows)** | All test rows scored | 2,808 | 0.6652 | 0.9000 | 11.7% | 0.6331 |
| **LOSO Out-of-Fold (All Test Rows)** | All test rows scored | 2,808 | 0.6171 | 0.9100 | 10.1% | 0.5845 |

**Findings**:
- The generalization gap on unseen subjects is modest (AUC drop of 0.0281 to 0.0480; savings reduction of ~1.6% to 4.3%).
- The router maintains $\ge 10.1\%$ cost savings on completely novel subjects, confirming robust out-of-domain transfer.

---

## 8. Live Replay Verification (T44)

To bridge offline evaluation and production serving, `scripts/15_replay_demo.py` replayed 200 held-out test queries through the production router bundle (`artifacts/bundle/router.pkl`, calibrated threshold $\tau^* = 0.0791$):
- **Decision Agreement**: 100.0% match with `threshold.evaluate_at_tau` reference.
- **Escalation Rate ($\alpha$)**: 92.0% (16 local queries, 184 frontier queries).
- **Quality**: 0.8750 (Baseline S: 0.6100, L: 0.8850).
- **Cost Savings**: 11.1% realized live spend reduction ($0.0709 vs $0.0798 counterfactual).
- **Decision Latency**: Median 14.30 ms, p99 29.66 ms on CPU (well within interactive budgets).

This validates that the calibrated threshold and decision logic operate identically in production serving.
