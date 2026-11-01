# Headroom Sweep (T27) and Locked Benchmark Numbers (T28)

*Deliverables for Lane C (Evaluation, Calibration & Statistics)*  
*Artifacts: `artifacts/routerbench/`, `artifacts/trackb/`, `router/threshold.py`, `router/costs.py`*

---

## 1. Task T27: Headroom Sweep Across Model Pairs

We evaluate headroom ($\Delta = Q_L - Q_S$) and routing capability across all three small candidate models against `gpt-4-1106-preview` on the headline benchmarks (MMLU, GSM8K, ARC-Challenge, Winogrande):

| Model Pair | Small Model ($S$) | Large Model ($L$) | $Q_S$ | $Q_L$ | Headroom $\Delta$ | Oracle CPT(95%) | Oracle Savings @ PGR .95 | Random CPT(95%) |
|---|---|---|--:|--:|--:|--:|--:|--:|
| **Mixtral Pair (Track A Development)** | `mistralai/mixtral-8x7b-chat` | `gpt-4-1106-preview` | 0.6316 | 0.8621 | **0.2305** | 21.9% | 80.1% | 96.1% |
| **Yi Pair (Track A Secondary)** | `zero-one-ai/Yi-34B-Chat` | `gpt-4-1106-preview` | 0.6625 | 0.8621 | **0.1996** | 18.9% | 85.5% | 96.1% |
| **Qwen Pair (Track B Headline / Demo)**| `qwen3.5-4b` | `gpt-4-1106-preview` | 0.8126 | 0.8874 | **0.0748** | 7.1% | 98.5% | 96.4% |

### Observations:
1. **Headroom Compression**: As small model quality advances from Mixtral (63.2%) and Yi-34B (66.3%) to Qwen3.5-4B (81.3%), the performance headroom drops from 23.0% to 7.5%.
2. **Oracle Savings Escalation**: In low-headroom regimes, the theoretical oracle needs to escalate only 7.1% of queries to achieve PGR $\ge 0.95$, unlocking up to 98.5% theoretical spend reduction.
3. **Certification Challenge**: Conversely, statistical certification becomes strictly more conservative as headroom shrinks, because the paired margin $e_i$ is concentrated near zero.

---

## 2. Task T37: Calibrated Threshold $\tau^*$ and Pareto Curve for Final Qwen Cascade Router

Applying Rule v2 (cost-normalized gain $\frac{\hat{\Delta}}{\hat{c}_L}$) on Qwen cascade confidence features (`cascade_conf`):
- **Calibrated Operating Threshold**: $\tau^* = 9.1861$ (calibrated on `cal` via fixed-sequence Learn-then-Test with WSR betting martingales at $\delta = 0.05, \rho = 0.95$).
- **Test-split empirical Pareto characteristics**:
  - Empirical $\text{CPT}(95\%)$: **39.5%**
  - Concave Envelope $\text{CPT}(95\%)$: **38.9%**
  - Area under the PGR Curve ($\text{APGR}$): **0.8267**
  - Uncertified Savings at Empirical $\text{CPT}(95\%)$: **78.9%**

---

## 3. Task T28: Locked Official Numbers for Thesis & Defense

*Evaluated on the frozen, untouched test split ($N=4,707$ queries). Every number below is frozen with paired-bootstrap 95% confidence intervals.*

### 3.1 Headline Result: Qwen Cascade Router (`cascade_conf`, Rule v2, WSR)

| Metric | Point Estimate | 95% Confidence Interval | Notes / Operational Meaning |
|---|--:|:---:|---|
| **Calibrated Threshold $\tau^*$** | **9.1861** | — | Selected strictly on calibration split |
| **Calibration Escalation Rate ($\alpha_{\text{cal}}$)**| **61.3%** | — | Guaranteed under WSR martingale test |
| **Test Escalation Rate ($\alpha_{\text{test}}$)** | **61.2%** | [59.9%, 62.7%] | Actual frontier calls executed |
| **Realized Test PGR** | **100.9%** | [96.6%, 105.6%] | **Exceeds target 95% with high confidence** |
| **Model M1 Spend Savings** | **53.9%** | **[51.8%, 56.0%]** | **Official Headline Savings Number** |
| **Savings vs. Batch-API Frontier** | **+7.8%** | — | Beats 50%-discounted Batch API |
| **Model M2 Amortized Savings** | **51.2%** | **[49.2%, 53.2%]** | Full hardware amortisation on Apple M2 |

### 3.2 Comparison Result: Mixtral Development Pair (Two-Head MLP, Rule v2, WSR)

| Metric | Point Estimate | 95% Confidence Interval |
|---|--:|:---:|
| **Test Escalation Rate ($\alpha_{\text{test}}$)** | 96.2% | [95.8%, 96.6%] |
| **Realized Test PGR** | 96.3% | [95.2%, 97.5%] |
| **Model M1 Spend Savings** | **9.4%** | [7.8%, 10.9%] |
| **Model M2 Amortized Savings** | **8.9%** | — |
