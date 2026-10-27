# Chapter C (T52): Cost Model, Threshold Selection, Statistical Guarantees & Results

Everything Chapter C needs for your thesis/defence, with mathematical derivations, code mappings, empirical results, and explanations for key viva questions.

---

## 1. Chapter Outline & Core Concepts

1. **The Routing Problem & Decision Rules**
   - Routing formulation: selecting between small model $S$ and frontier model $L$.
   - Decision rule v1: $\hat{\Delta} = \hat{s}_L - \hat{s}_S > \tau$.
   - Decision rule v2: $\text{Gain}(q) = \frac{\hat{\Delta}(q)}{\hat{c}_L(q)} > \tau$ (cost-normalized gain).
2. **Dual Cost Models: M1, M2 & The Batch Frontier Baseline**
   - **Cost Model M1 (Pure API Spend)**: Local model calls cost \$0. Frontier API calls priced at list token rates ($p_{\text{in}}, p_{\text{out}}$).
   - **Cost Model M2 (Hardware-Amortized Operational Cost)**: Local inference priced by hardware depreciation + power consumption per hour divided by decode token speed.
   - **Batch API Baseline**: Always-frontier at 50% discount (asynchronous non-real-time baseline) to ensure fair comparison.
   - **Crossover Speed**: The local token generation speed below which running locally on hardware costs *more* than calling the cloud API.
3. **Exact Breakpoint Sweep & Concave Envelope (T21)**
   - $O(n \log n)$ sweep over all distinct candidate thresholds without grid search.
   - Upper-left concave envelope: the theoretical Pareto frontier achievable via randomized thresholding.
4. **Statistical Guarantees & Threshold Calibration via Learn-then-Test (T22 & T23)**
   - Defining the paired margin statistic $e_i(\tau) = s_{\text{route}(i)} - [(1 - \rho) s_{S,i} + \rho s_{L,i}]$.
   - Fixed-sequence hypothesis testing on calibration split $\mathcal{D}_{\text{cal}}$ without Bonferroni inflation.
   - Comparing statistical bounds: Hoeffding vs. Empirical Bernstein vs. Paired Bootstrap vs. **WSR (Waudby-Smith & Ramdas)**.
   - **Critical Viva Question: Why the paired bootstrap undercovers in small headroom regimes.**
5. **Empirical Results & Findings (RouterBench & Cascade)**
   - Official certified savings on test: Two-Head MLP + Rule v2 yields **9.4% [7.8, 10.9]** savings at PGR 96.3% [95.2, 97.5].
   - Why naive threshold selection (14.3% / 12.7%) is an optimistic upper bound with test-split leakage.
   - Cascade exploration: why routing on small-model answers + agreement jumps certified savings from 9.4% to **26.0%**.

---

## 2. Mathematical Formulations & Cost Models

### 2.1 The Two Cost Models (M1 and M2)
In [`router/costs.py`](file:///c:/Users/HP/Desktop/major%20project/cost-aware-router/router/costs.py):

#### Cost Model M1 (API Spend):
$$C_{M1}(q) = \begin{cases} 0 & \text{if routed to local model } S \\ \frac{p_{\text{in}} \cdot n_{\text{in}} + p_{\text{out}} \cdot n_{\text{out}}}{10^6} & \text{if routed to frontier API } L \end{cases}$$

#### Cost Model M2 (Hardware Time):
For a machine with initial cost $P_{\text{hw}}$, lifetime $Y$ years, usage $H$ hours/day, power $W$ watts, and electricity tariff $R_{\text{kwh}}$ (\$/kWh):
$$\text{CostPerHour} = \frac{P_{\text{hw}}}{Y \times 365 \times H} + \left(\frac{W}{1000} \times R_{\text{kwh}}\right)$$
For Apple M2 (MacBook Air 16GB, \$1,200, 3-year life, 8h/day, 30W active, \$0.15/kWh):
$$\text{CostPerHour} \approx \$0.1370 + \$0.0045 = \$0.1415/\text{hour}$$
Generating $n_{\text{out}}$ tokens at speed $v$ tokens/sec costs:
$$C_{\text{local}}(n_{\text{out}}) = \frac{n_{\text{out}}}{v \times 3600} \times \text{CostPerHour}$$

#### Crossover Speed:
Local inference is only cheaper than frontier API if:
$$v > v_{\text{cross}} = \frac{n_{\text{out}} \times \text{CostPerHour}}{3600 \times C_{\text{API}}(n_{\text{in}}, n_{\text{out}})}$$

#### The Batch API Baseline:
Frontier providers offer Batch APIs at a 50% discount:
$$C_{\text{batch}}(q) = 0.5 \times C_{\text{API}}(q)$$
Any query router saving less than 50% under M1 is economically dominated by simply running the frontier model via Batch API if latency allows.

---

## 3. Statistical Guarantees & Threshold Selection

### 3.1 The Performance Guarantee Ratio (PGR)
Given router quality $Q(\tau) = \mathbb{E}[s_{\text{route}(q)}]$, small model quality $Q_S = \mathbb{E}[s_S]$, and large model quality $Q_L = \mathbb{E}[s_L]$:
$$\text{PGR}(\tau) = \frac{Q(\tau) - Q_S}{Q_L - Q_S}$$
Target requirement: $\text{PGR}(\tau) \ge \rho = 0.95$ with confidence $1 - \delta = 0.95$.

### 3.2 The Paired Margin Statistic $e_i(\tau)$
Rearranging $\text{PGR}(\tau) \ge \rho$:
$$Q(\tau) \ge Q_S + \rho (Q_L - Q_S) = (1 - \rho) Q_S + \rho Q_L$$
For query $i$, define the paired margin:
$$e_i(\tau) = s_{\text{route}(i)} - \left[(1 - \rho) s_{S,i} + \rho s_{L,i}\right]$$
- If query $i$ is routed to $L$: $e_i(\tau) = (1 - \rho)(s_{L,i} - s_{S,i})$.
- If query $i$ is routed to $S$: $e_i(\tau) = \rho (s_{S,i} - s_{L,i})$.

Because both models are evaluated on the exact same prompt, taking the difference cancels query-level variance, making $\text{Var}(e_i)$ drastically smaller than $\text{Var}(s_{\text{route}})$.

### 3.3 Learn-then-Test (LTT) Fixed Sequence Testing
1. Discretize candidate thresholds into an ordered sequence $\tau_1 > \tau_2 > \dots > \tau_K$ (from most frontier calls to least) on training data.
2. For each threshold in sequence, test the null hypothesis:
   $$H_{0, k}: \mathbb{E}[e(\tau_k)] < 0 \quad (\text{i.e. } \text{PGR}(\tau_k) < \rho)$$
3. Compute valid p-value $p_k$. Reject $H_{0, k}$ if $p_k \le \delta$.
4. **Stop at the first failure**: the last rejected threshold is the certified $\tau^*$. Because testing stops at the first failure, Family-Wise Error Rate (FWER) $\le \delta$ is preserved without any Bonferroni correction.

---

## 4. Why Bootstrap Undercovers (Viva & Theory Defense)

### The Core Problem:
- **Asymptotic vs. Finite-Sample**: The paired bootstrap assumes that the empirical distribution $\hat{F}_n$ has converged to the true distribution $F$, and the empirical quantile approximates the true lower tail through the Central Limit Theorem.
- **Extreme Boundary & Low Headroom**: When $Q_L - Q_S$ is small (e.g. 4–5%), $e_i(\tau)$ is heavily skewed with point mass at zero (when both models are correct or both fail).
- **Failure Mode**: In empirical simulation on calibration sets ($n_{\text{cal}} = 2,000$) with small headroom ($s_S \approx 0.80, s_L \approx 0.84$), the paired bootstrap nominal 95% confidence bound violated the true coverage by failing on fresh test draws **15% of the time** (3x higher than $\delta = 0.05$).

### The Solution: Waudby-Smith & Ramdas (WSR) Betting Martingale
- In contrast to bootstrap, WSR (2023) constructs a test martingale:
  $$K_t = \prod_{i=1}^t (1 - \lambda_i (l_i - 1/2))$$
  where $l_i \in [0, 1]$ is the normalized loss and $\lambda_i$ are predictable bet sizes.
- By **Ville's inequality**, $\mathbb{P}(\exists t: K_t \ge 1/\delta) \le \delta$ holds unconditionally for *any* finite sample size $n$ without asymptotic normality assumptions.
- Hence, WSR guarantees rigorous finite-sample coverage while matching the tightness of bootstrap once $n_{\text{cal}} \ge 2,000$.

---

## 5. Summary of Empirical Results

| Model / Strategy | Selection Method | Bound | Certified? | PGR (Test) | M1 Savings [95% CI] | M2 Savings |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Random baseline** | Uniform sample | — | No | 95.0% | 3.9% | 3.8% |
| **Oracle baseline** | Ground truth gap | — | Yes | 95.0% | 78.1% | 77.4% |
| **F0 Heuristics + LogReg** | LTT on cal | WSR | Yes | 96.6% [95.3, 97.8] | 3.9% [3.4, 4.4] | 3.8% |
| **Two-Head MLP (Rule v1)** | LTT on cal | WSR | Yes | 95.4% [94.0, 96.7] | 4.2% [3.7, 4.7] | 4.0% |
| **Two-Head MLP (Rule v2)** | **LTT on cal** | **WSR** | **Yes** | **96.3% [95.2, 97.5]** | **9.4% [7.8, 10.9]** | **8.9%** |
| Two-Head MLP (Rule v2) | Naive on cal | None | No | 95.1% [93.6, 96.4] | 12.7% [11.0, 14.5] | 12.1% |
| Two-Head MLP (Rule v2) | Leaky on test | None | No | 95.0% | 14.3% (Upper bound) | 13.8% |
| **Cascade (Answer + Agreement)** | **LTT on cal** | **WSR** | **Yes** | **97.9% [96.8, 98.9]** | **26.0% [24.2, 27.9]** | **24.8%** |

---

## 6. Key Files and Code References
- [`router/costs.py`](file:///c:/Users/HP/Desktop/major%20project/cost-aware-router/router/costs.py): Implementation of M1 and M2 cost models, `PriceTable`, and `pair_costs()`.
- [`router/threshold.py`](file:///c:/Users/HP/Desktop/major%20project/cost-aware-router/router/threshold.py): Implementation of `sweep()`, `upper_envelope()`, `wsr_pvalues()`, `select_tau()`, and `evaluate_at_tau()`.
- [`scripts/08_threshold_report.py`](file:///c:/Users/HP/Desktop/major%20project/cost-aware-router/scripts/08_threshold_report.py): Production generator for calibration and test guarantee tables.
- [`notes/threshold_report.md`](file:///c:/Users/HP/Desktop/major%20project/cost-aware-router/notes/threshold_report.md): Benchmark results across routers, rules, and bounds.
- [`notes/cascade_explore.md`](file:///c:/Users/HP/Desktop/major%20project/cost-aware-router/notes/cascade_explore.md): Exploration showing self-consistency / second-opinion escalation boost.
