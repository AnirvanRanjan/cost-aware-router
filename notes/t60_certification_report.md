# T60: Seed-Free Certification via Terminal WSR E-Value Averaging

*Deliverable for Task T60 (Lane C: Calibration & Statistics)*  
*Code artifact: `scripts/17_t60_seed_free_wsr.py`*  
*Data: Track B (`artifacts/trackb/`), frozen splits, headline benchmarks (MMLU, GSM8K, ARC-Challenge, Winogrande)*

---

## 1. Problem: Seed Variance in Single-Permutation WSR Martingales

In fixed-sequence Learn-then-Test (LTT), the standard WSR betting martingale bound relies on a random query permutation to construct sequential bets:
$$K_t = \prod_{i=1}^t \left(1 - \nu_i (l_i - 0.5)\right)$$

While Ville's maximal inequality guarantees finite-sample validity unconditionally for any fixed permutation, the resulting calibrated threshold $\tau^*$ is sensitive to the permutation order.

On the standard train grid ($n=1,000$), sweeping permutation seeds $0$ through $19$ reveals significant seed-dependent fluctuation on the held-out test split:
- **Test savings range across seeds 0–19**: **$40.8\%$ to $74.9\%$**
- Lowest seed ($s=2, 7, 8, 15$): $\tau^* = 8.1093 \implies 40.8\%$ savings
- Highest seed ($s=17$): $\tau^* = 16.7222 \implies 74.9\%$ savings
- Seed 0: $\tau^* = 9.1861 \implies 53.9\%$ savings

This demonstrates that choosing a coarser grid ($n=200$) solely based on its calibration performance risks capturing seed variance rather than genuine systematic headroom.

---

## 2. Solution: Seed-Free WSR via Averaging Terminal E-Values

To eliminate permutation noise rigorously without violating statistical guarantees:

1. **Terminal Wealth as an E-Value**: For any fixed permutation $m$, the terminal martingale wealth at $t=n$:
   $$E^{(m)} = K_n^{(m)} = \prod_{i=1}^n \left(1 - \nu_i (l_i - 0.5)\right)$$
   satisfies $\mathbb{E}[E^{(m)}] \le 1$ under the null hypothesis $H_0: \mathbb{E}[e(\tau)] < 0$. Hence, $E^{(m)}$ is an exact, non-negative e-value.

2. **Arithmetic Mean of E-Values**: By linearity of expectation, the mixture of e-values across $M$ fixed permutations:
   $$\bar{E} = \frac{1}{M} \sum_{m=1}^M E^{(m)}$$
   is also a valid e-value ($\mathbb{E}[\bar{E}] \le 1$ under $H_0$).

3. **Markov P-Value**: By Markov's inequality:
   $$p = \min\left(1, \frac{1}{\bar{E}}\right)$$
   is an unconditionally valid finite-sample p-value. It evaluates the entire calibration sample without dependence on any single query ordering or stopping at an intermediate maximum over $t$.

---

## 3. Empirical Results (M=50 Permutations)

Running `scripts/17_t60_seed_free_wsr.py` with $M=50$ permutations on `cascade_3sample` across the standard train grid ($n=1,000$):

| Method | Permutations ($M$) | Cal $\tau^*$ | Cal Escalation | Test Escalation [95% CI] | Test PGR [95% CI] | Certified Test Savings [95% CI] |
|---|:---:|--:|--:|:---:|:---:|:---:|
| **Single-Permutation WSR (Seed 0)** | 1 | 9.1861 | 61.3% | 61.2% [59.9, 62.7] | 100.9% [96.6, 105.6] | 53.9% [51.8, 56.0] |
| **Seed Variance Range (Seeds 0–19)**| 1 | 8.11–16.72 | 41.2%–77.7% | 40.8%–77.5% | 96.1%–104.2% | 40.8%–74.9% |
| **Seed-Free Terminal WSR ($M=50$, `cascade_3sample`)** | **50** | **8.2187** | **76.1%** | **75.9% [74.7, 77.1]** | **102.8% [99.1, 107.0]** | **41.3% [39.0, 43.5]** |

---

## 4. Key Takeaways

1. **Permutation Invariance**: Averaging terminal e-values across $M=50$ permutations completely eliminates the 40.8%–74.9% seed fluctuation.
2. **Conservative Rigor**: Because it does not take the running maximum over $t$, the pure terminal wealth bound is robust against early drawdown artifacts and yields **41.3% [39.0%, 43.5%] certified spend reduction** with exact finite-sample validity.
3. **Consistent Unified Numbers**: This seed-free result (41.3% certified spend savings, 75.9% escalation) aligns with the official `cascade_3sample` model from `cascade_predictions.parquet` and is quoted consistently in the T28 preview.
3. **Reproducibility**: The standalone runner `scripts/17_t60_seed_free_wsr.py` reproduces the evaluation and bounds in one command.
