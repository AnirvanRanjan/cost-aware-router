# Headroom Sweep (T27) and Preview Benchmark Numbers (T37 / T28)

*Deliverables for Lane C (Evaluation, Calibration & Statistics)*  
*Artifacts: `artifacts/routerbench/`, `artifacts/trackb/`, `router/threshold.py`, `router/costs.py`*

---

## 1. Task T27: Headroom Sweep Across Model Pairs on Track B Question Set

To provide a strictly fair like-with-like comparison, all three candidate small models are evaluated against `gpt-4-1106-preview` on the **exact same set of held-out test questions** (the headline Track B test set, $N=4,707$ questions across MMLU, GSM8K, ARC-Challenge, and Winogrande).

For each pair, we report both the theoretical Oracle reference and the realized performance of an F2-based query router (using cost-normalized Rule v2):

| Model Pair | Small Model ($S$) | Large Model ($L$) | $Q_S$ | $Q_L$ | Headroom $\Delta$ | Oracle CPT(95%) | Oracle Savings @ .95 | Router CPT(95%) | Router Savings @ .95 |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|
| **Mixtral Pair** | `mistralai/mixtral-8x7b-chat` | `gpt-4-1106-preview` | 0.6501 | 0.8874 | **0.2373** | 22.6% | 79.9% | 95.2% | **12.9%** |
| **Yi-34B Pair** | `zero-one-ai/Yi-34B-Chat` | `gpt-4-1106-preview` | 0.6820 | 0.8874 | **0.2054** | 19.5% | 85.4% | 96.4% | **12.0%** |
| **Qwen Pair (Cascade Preview)** | `qwen3.5-4b` | `gpt-4-1106-preview` | 0.8126 | 0.8874 | **0.0748** | 7.1% | 98.5% | 39.5% | **78.9%** |

### Observations:
1. **Headroom Compression**: Moving from Mixtral ($\Delta = 0.2373$) to Yi-34B ($\Delta = 0.2054$) and Qwen3.5-4B ($\Delta = 0.0748$) significantly reduces the capability gap.
2. **Query Router Limits on Large Headrooms**: On query features alone (F2 embeddings), both Mixtral and Yi-34B routers require escalating $>95\%$ of queries to maintain PGR $\ge 0.95$, saving only $12\%–13\%$.
3. **Cascade Leverage**: On the Qwen pair, inspecting small-model generation confidence allows the cascade router to achieve $78.9\%$ empirical savings on the test curve at CPT $39.5\%$.

---

## 2. Task T37: Calibrated Threshold $\tau^*$ and Pareto Curve (PREVIEW)

*PREVIEW ONLY: Pending HSID's final router bundle from Lane B (T61 / D-12 / D-14).*

Applying Rule v2 on Qwen preview confidence signals (`cascade_conf`):
- **Calibrated Operating Threshold**: $\tau^* = 8.8700$ (seed-free WSR terminal averaging, $M=50$).
- **Test-split empirical Pareto characteristics**:
  - Empirical $\text{CPT}(95\%)$: **39.5%**
  - Concave Envelope $\text{CPT}(95\%)$: **38.9%**
  - Area under the PGR Curve ($\text{APGR}$): **0.8267**
  - Uncertified Savings at Empirical $\text{CPT}(95\%)$: **78.9%**

---

## 3. Task T28: Preview Headline Numbers (PREVIEW)

*PREVIEW ONLY: Final numbers will be locked once T61 lands.*

Evaluated on the frozen test split ($N=4,707$ queries) using seed-free WSR terminal averaging ($M=50$):

### 3.1 Qwen Cascade Router (`cascade_conf`, Rule v2, Seed-Free WSR) — PREVIEW

| Metric | Preview Value | 95% Confidence Interval |
|---|--:|:---:|
| **Calibrated Threshold $\tau^*$** | **8.8700** | — |
| **Test Escalation Rate ($\alpha_{\text{test}}$)** | **66.6%** | [65.2%, 67.9%] |
| **Realized Test PGR** | **101.5%** | [97.5%, 105.8%] |
| **Model M1 Spend Savings** | **48.3%** | **[45.9%, 50.6%]** |
| **Model M2 Amortized Savings** | **45.9%** | [43.6%, 48.2%] |

### 3.2 Mixtral Development Pair (Two-Head MLP, Rule v2, WSR)

| Metric | Point Estimate | 95% Confidence Interval |
|---|--:|:---:|
| **Test Escalation Rate ($\alpha_{\text{test}}$)** | 96.2% | [95.8%, 96.6%] |
| **Realized Test PGR** | 96.3% | [95.2%, 97.5%] |
| **Model M1 Spend Savings** | **9.4%** | [7.8%, 10.9%] |
| **Model M2 Amortized Savings** | **8.9%** | — |
