# Headroom Sweep (T27) and Preview Benchmark Numbers (T37 / T28)

*Deliverables for Lane C (Evaluation, Calibration & Statistics)*  
*Artifacts: `artifacts/routerbench/`, `artifacts/trackb/`, `router/threshold.py`, `router/costs.py`*

---

## 1. Task T27: Headroom Sweep Across Model Pairs on Track B Question Set

To provide a strictly fair like-with-like comparison, all three candidate small models are evaluated against `gpt-4-1106-preview` on the **exact same set of held-out test questions** (the headline Track B test set, $N=4,707$ questions across MMLU, GSM8K, ARC-Challenge, and Winogrande).

For each pair, we report both the theoretical Oracle reference and the realized performance of an F2-based query router (using cost-normalized Rule v2):

| Model Pair | Small Model ($S$) | Large Model ($L$) | $Q_S$ | $Q_L$ | Headroom $\Delta$ | Oracle CPT(95%) | Oracle Savings (uncertified) | Router Type | Router CPT(95%) | Router Savings (uncertified) |
|---|---|---|--:|--:|--:|--:|--:|---|--:|--:|
| **Mixtral Pair** | `mistralai/mixtral-8x7b-chat` | `gpt-4-1106-preview` | 0.6501 | 0.8874 | **0.2373** | 22.6% | 79.9% | Question-only (`F2`) | 95.2% | **12.9%** |
| **Yi-34B Pair** | `zero-one-ai/Yi-34B-Chat` | `gpt-4-1106-preview` | 0.6820 | 0.8874 | **0.2054** | 19.5% | 85.4% | Question-only (`F2`) | 96.4% | **12.0%** |
| **Qwen Pair** | `qwen3.5-4b` | `gpt-4-1106-preview` | 0.8126 | 0.8874 | **0.0748** | 7.1% | 98.5% | Question-only (`query_f2`) | 85.9% | **21.1%** |
| **Qwen Pair (Cascade)** | `qwen3.5-4b` | `gpt-4-1106-preview` | 0.8126 | 0.8874 | **0.0748** | 7.1% | 98.5% | Cascade (`cascade_3sample`) | 39.5% | **78.9%** |

*Note: All savings figures above are test-picked curve points and are explicitly uncertified (requiring frozen $\tau^*$ calibration to achieve formal finite-sample statistical guarantees).*

### Observations:
1. **Headroom Compression**: Moving from Mixtral ($\Delta = 0.2373$) to Yi-34B ($\Delta = 0.2054$) and Qwen3.5-4B ($\Delta = 0.0748$) significantly reduces the capability gap.
2. **Query Router Limits across all pairs**: On question features alone (F2 embeddings), query routers on all three pairs achieve modest uncertified savings ($12.0\%$ to $21.1\%$), requiring escalation rates of $85.9\%–96.4\%$ to reach PGR $\ge 0.95$.
3. **Cascade Leverage**: On the Qwen pair, inspecting small-model generation signals (`cascade_3sample`) allows the cascade router to achieve $78.9\%$ uncertified test-curve savings at CPT $39.5\%$, compared to $21.1\%$ for the question-only router.

---

## 2. Task T37: Calibrated Threshold $\tau^*$ and Pareto Curve (PREVIEW)

*PREVIEW ONLY: Evaluated on HSID's `cascade_predictions.parquet` (`cascade_3sample`, T61 / D-12 / D-14).*

Applying Rule v2 on `cascade_3sample`:
- **Calibrated Operating Threshold**: $\tau^* = 8.2187$ (seed-free WSR terminal averaging, $M=50$).
- **Test-split empirical Pareto characteristics**:
  - Empirical $\text{CPT}(95\%)$: **39.5%**
  - Concave Envelope $\text{CPT}(95\%)$: **38.9%**
  - Area under the PGR Curve ($\text{APGR}$): **0.8267**
  - Uncertified Savings at Empirical $\text{CPT}(95\%)$: **78.9%**

---

## 3. Task T28: Preview Headline Numbers (PREVIEW)

*PREVIEW ONLY: Evaluated on HSID's `cascade_predictions.parquet` (`cascade_3sample`).*

Evaluated on the frozen test split ($N=4,707$ queries) using seed-free WSR terminal averaging ($M=50$):

### 3.1 Qwen Cascade Router (`cascade_3sample`, Rule v2, Seed-Free WSR) — PREVIEW

| Metric | Preview Value | 95% Confidence Interval |
|---|--:|:---:|
| **Calibrated Threshold $\tau^*$** | **8.2187** | — |
| **Test Escalation Rate ($\alpha_{\text{test}}$)** | **75.9%** | [74.7%, 77.1%] |
| **Realized Test PGR** | **102.8%** | [99.1%, 107.0%] |
| **Model M1 Spend Savings** | **41.3%** | **[39.0%, 43.5%]** |
| **Model M2 Amortized Savings** | **36.5%** | [34.3%, 38.7%] |

### 3.2 Mixtral Development Pair (Two-Head MLP, Rule v2, WSR)

| Metric | Point Estimate | 95% Confidence Interval |
|---|--:|:---:|
| **Test Escalation Rate ($\alpha_{\text{test}}$)** | 96.2% | [95.8%, 96.6%] |
| **Realized Test PGR** | 96.3% | [95.2%, 97.5%] |
| **Model M1 Spend Savings** | **9.4%** | [7.8%, 10.9%] |
| **Model M2 Amortized Savings** | **8.9%** | — |
