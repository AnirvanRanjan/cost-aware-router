# Subject Evaluation & Leave-One-Subject-Out on mmlu (T26)

*Generated 2026-10-28 15:49 UTC by `scripts/13_loso_mmlu.py`.*

Pair: S = `mistralai/mixtral-8x7b-chat`, L = `gpt-4-1106-preview`.

## 2. True Leave-One-Subject-Out (LOSO) Evaluation (Retrained on F2 / LogReg)

Subjects grouped into 10 held-out folds. For each fold, the router is retrained with that fold's train rows removed, then evaluated on the fold's test rows.

### Fold-by-Fold Results

| Fold | Subjects Sample | Subjects Count | Test n | LOSO AUC | Pooled Fold AUC | LOSO Savings@PGR .95 | Pooled Fold Savings |
|---|---|--:|--:|:--:|:--:|:--:|:--:|
| 0 | high-school-psychology, moral-disputes, medical-genetics (+3 more) | 6 | 303 | 0.5954 | 0.5906 | 8.5% | 13.3% |
| 1 | security-studies, college-biology, public-relations (+3 more) | 6 | 341 | 0.5566 | 0.5762 | 8.5% | 5.9% |
| 2 | high-school-biology, us-foreign-policy, prehistory (+3 more) | 6 | 251 | 0.6159 | 0.6469 | 9.7% | 9.2% |
| 3 | high-school-macroeconomics, high-school-mathematics, clinical-knowledge (+3 more) | 6 | 277 | 0.5681 | 0.5830 | 10.8% | 9.4% |
| 4 | college-computer-science, high-school-statistics, high-school-microeconomics (+3 more) | 6 | 204 | 0.6273 | 0.6277 | 9.3% | 26.8% |
| 5 | global-facts, high-school-world-history, elementary-mathematics (+3 more) | 6 | 207 | 0.6543 | 0.6832 | 29.7% | 32.8% |
| 6 | high-school-physics, international-law, human-aging (+3 more) | 6 | 250 | 0.5878 | 0.5639 | 4.0% | 6.2% |
| 7 | professional-accounting, high-school-computer-science, high-school-government-and-politics (+2 more) | 5 | 266 | 0.5772 | 0.6162 | 5.1% | 5.3% |
| 8 | world-religions, professional-medicine, machine-learning (+2 more) | 5 | 288 | 0.5496 | 0.7013 | 11.5% | 25.2% |
| 9 | marketing, professional-law, college-medicine (+2 more) | 5 | 421 | 0.5636 | 0.5875 | 5.5% | 12.1% |

### Like-With-Like Comparisons

To compare like with like, we examine both: (1) averaging metrics across the exact same folds, and (2) scoring predictions across all test rows together.

| Evaluation Scheme | Comparison Basis | Test n | AUC | CPT(95%) | Savings@PGR .95 | APGR |
|---|---|--:|:--:|:--:|:--:|:--:|
| **Pooled Baseline (Fold Mean)** | Same folds averaged | 2,808 | 0.6177 | 0.8690 | 14.6% | 0.5865 |
| **LOSO Retrained (Fold Mean)** | Same folds averaged | 2,808 | 0.5896 | 0.9130 | 10.3% | 0.5633 |
| **Pooled Baseline (All Test Rows)** | All test rows scored | 2,808 | 0.6652 | 0.9000 | 11.7% | 0.6331 |
| **LOSO Out-of-Fold (All Test Rows)** | All test rows scored | 2,808 | 0.6171 | 0.9100 | 10.1% | 0.5845 |

**Like-with-like Generalization Gaps:**
- **Fold-level average gap (Pooled Fold Mean - LOSO Fold Mean):** AUC +0.0281, Savings +4.4%, CPT -0.0440.
- **All test rows gap (Pooled Retrained - LOSO Out-of-Fold):** AUC +0.0480, Savings +1.6%, CPT -0.0100.
- Worst fold: **Fold 6** (high-school-physics, international-law, human-aging (+3 more), savings 4.0%).
