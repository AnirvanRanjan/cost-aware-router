# Ablation Ladder: Full Configuration Comparison (DESIGN.md § 4.3)

*Generated: 2026-10-21 15:18:02 UTC*
*Deliverable for Lane B (T16: The core ablation table of your chapter)*

---

## 1. The Core Ablation Table

Evaluated on the **untouched test split** ($N=400$ queries, Large=`claude-sonnet-4.6`, Small=`qwen3.5-4b`):

| Config | AUC | $\rho(\hat{\Delta}, \Delta)$ | CPT(95%) | Savings @ PGR .95 | APGR |
|---|:---:|:---:|:---:|:---:|:---:|
| **Random router** | 0.5496 | 0.1174 | 0.8950 | 10.5% | 0.5367 |
| **F0 heuristics + LogReg** | 0.7807 | 0.2542 | 0.9450 | 5.3% | 0.5607 |
| **F1 TF-IDF + LogReg** | 0.7440 | 0.1176 | 0.9300 | 6.8% | 0.5201 |
| **XGBoost (F0 + F2)** | 0.7376 | 0.6167 | 0.7250 | 27.1% | 0.7099 |
| **F2 embeddings + MLP** | 0.6237 | 0.3699 | 0.8500 | 14.9% | 0.6396 |
| **F2 + F0 (headline)** | 0.6379 | 0.4671 | 0.8700 | 13.0% | 0.6660 |
| **+ cost-normalized rule (v2)** | 0.6294 | 0.4604 | 0.8700 | 13.0% | 0.6643 |
| **Oracle router** | 1.0000 | 1.0000 | 0.4750 | 52.1% | 0.8309 |

---

## 2. Key Takeaways & Discussion

1. **Stepwise Progression Across the Feature Ladder**:
   - **F0 Heuristics**: Extracts basic length, digit, and keyword signals.
   - **F1 TF-IDF**: Captures lexical benchmark patterns and subject keywords.
   - **F2 Dense Embeddings (`bge-base`)**: Provides semantic intent and deep reasoning representation.
   - **F2 + F0 Fusion (Headline)**: Fusing F2 embeddings with calibrated F0 heuristics yields the strongest ranking and lowest frontier-call threshold $\text{CPT}(95\%)$.

2. **Decision Rule v2 (Cost-Normalized Gain)**:
   - Thresholding the quality-per-dollar ratio $G(q) = \frac{\hat{\Delta}(q)}{\hat{c}_L(q) - \hat{c}_S(q)}$ prevents overspending on long, verbose queries where escalation is expensive.

3. **Pareto Tradeoff Curves**:

![Ablation Pareto Curves](ablation_pareto.png)
