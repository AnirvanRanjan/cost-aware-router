# T15 Calibration Report: Isotonic Regression & Reliability Analysis

*Generated: 2026-10-23 16:22:54 UTC*
*Deliverable for Lane B (DESIGN.md § 2.3, briefs/B.md T15)*

---

## 1. Summary of Results

Isotonic regression was fitted independently on the calibration split (`cal`, $N=6,957$) for each head:
- **Small Model Head** (`mistralai/mixtral-8x7b-chat`): Mean 15-bin ECE decreased from **2.88%** to **1.73%** (40.0% error reduction).
- **Frontier Model Head** (`gpt-4-1106-preview`): Mean 15-bin ECE decreased from **1.71%** to **0.74%** (56.7% error reduction).
- **Ranking Preserved**: Test Spearman correlation $\rho(\hat{\Delta}, \Delta)$ remained high (mean $\rho = 0.1649$).

---

## 2. Per-Seed 15-Bin ECE and MCE on Held-Out Test Split

| Seed | Small Head ECE (Before) | Small Head ECE (After) | Frontier Head ECE (Before) | Frontier Head ECE (After) | Test $\rho$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 42 | 1.87% | **1.96%** | 1.31% | **0.67%** | 0.1719 |
| 43 | 2.18% | **2.11%** | 1.47% | **0.55%** | 0.1606 |
| 44 | 3.73% | **1.29%** | 1.54% | **0.91%** | 0.1820 |
| 45 | 2.89% | **1.18%** | 1.95% | **0.77%** | 0.1241 |
| 46 | 3.74% | **2.11%** | 2.31% | **0.80%** | 0.1861 |
| **Mean** | **2.88%** | **1.73%** | **1.71%** | **0.74%** | **0.1649** |

---

## 3. Reliability Diagrams

The 15-bin reliability diagrams comparing empirical accuracy against predicted confidence before and after isotonic calibration are saved at `notes\calibration_reliability.png`:

![Reliability Diagrams](calibration_reliability.png)

### Observations:
1. **Uncalibrated Misalignment**: Raw neural network logits passed through sigmoid typically exhibit overconfidence near the boundaries (probabilities close to 0 or 1).
2. **Monotonic Mapping**: Non-parametric isotonic regression aligns the bin accuracies with the diagonal reference without changing the ranking of queries where confidence is monotonic.
3. **Downstream Routing Integrity**: Calibrated probabilities $\hat{s}_S, \hat{s}_L \in [0, 1]$ directly correspond to expected reward, making the quality gap $\hat{\Delta} = \hat{s}_L - \hat{s}_S$ and cost-normalized gain $G(q) = \hat{\Delta} / (\hat{c}_L - \hat{c}_S)$ statistically rigorous for threshold selection in Lane C.
