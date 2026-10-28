# Subject Evaluation & Leave-One-Subject-Out on mmlu (T26)

*Generated 2026-10-28 14:49 UTC by `scripts/13_loso_mmlu.py`.*

Pair: S = `qwen3.5-4b`, L = `claude-sonnet-4.6`.

## 1. Within-Subject Evaluation (Existing Router Predictions)

Scores existing test-split predictions per subject. Note: These routers were trained on all subjects, so this measures within-subject ranking performance rather than generalization to unseen subjects.

### interim (v1)

| mmlu subject | n | AUC | CPT(95%) | Savings@PGR .95 | APGR |
|---|--:|:--:|:--:|:--:|:--:|
| anatomy | 7 | 0.8333 | 0.6450 | 28.6% | 0.6620 |
| astronomy | 18 | 0.9412 | 0.6950 | 27.8% | 0.6823 |
| econometrics | 14 | 1.0000 | 0.8250 | 14.3% | 0.6016 |
| formal_logic | 18 | 0.9688 | 0.5850 | 38.9% | 0.7515 |
| high_school_physics | 15 | 0.8571 | 0.9700 | 0.0% | 0.6839 |
| jurisprudence | 10 | 0.5000 | 0.9500 | 0.0% | 0.6128 |
| machine_learning | 17 | 0.2500 | 0.9150 | 5.9% | 0.5592 |
| moral_disputes | 13 | 0.7222 | 0.4250 | 53.8% | 0.8075 |
| nutrition | 16 | 0.6000 | 0.8450 | 12.5% | 0.7319 |
| philosophy | 14 | 0.5000 | 0.8950 | 7.1% | 0.5999 |
| virology | 8 | 0.5000 | 0.9400 | 0.0% | 0.6369 |
| world_religions | 13 | 1.0000 | 0.8100 | 15.4% | 0.6163 |
| **Subject mean** | 163 | — | 0.7917 | 17.0% | 0.6622 |
| **Grouped (pooled)** | 163 | — | 0.8200 | 17.8% | 0.6648 |

**Gaps (pooled - subject mean):** CPT +0.0283, savings +0.8%, APGR +0.0027. Worst subject: **high_school_physics** (savings 0.0%).

### interim (v2)

| mmlu subject | n | AUC | CPT(95%) | Savings@PGR .95 | APGR |
|---|--:|:--:|:--:|:--:|:--:|
| anatomy | 7 | 0.8333 | 0.6450 | 28.6% | 0.6620 |
| astronomy | 18 | 0.9412 | 0.6950 | 27.8% | 0.6823 |
| econometrics | 14 | 1.0000 | 0.8250 | 14.3% | 0.6016 |
| formal_logic | 18 | 0.9688 | 0.5850 | 38.9% | 0.7515 |
| high_school_physics | 15 | 0.8571 | 0.9700 | 0.0% | 0.6839 |
| jurisprudence | 10 | 0.5000 | 0.9500 | 0.0% | 0.6128 |
| machine_learning | 17 | 0.2500 | 0.9150 | 5.9% | 0.5592 |
| moral_disputes | 13 | 0.7222 | 0.4250 | 53.8% | 0.8075 |
| nutrition | 16 | 0.6000 | 0.8450 | 12.5% | 0.7319 |
| philosophy | 14 | 0.5000 | 0.8950 | 7.1% | 0.5999 |
| virology | 8 | 0.5000 | 0.9400 | 0.0% | 0.6369 |
| world_religions | 13 | 1.0000 | 0.8100 | 15.4% | 0.6163 |
| **Subject mean** | 163 | — | 0.7917 | 17.0% | 0.6622 |
| **Grouped (pooled)** | 163 | — | 0.8200 | 17.8% | 0.6648 |

**Gaps (pooled - subject mean):** CPT +0.0283, savings +0.8%, APGR +0.0027. Worst subject: **high_school_physics** (savings 0.0%).

## 2. True Leave-One-Subject-Out (LOSO) Evaluation (Retrained on F2 / LogReg)

Subjects grouped into 10 held-out folds. For each fold, the router is retrained with that fold's train rows removed, then evaluated on the fold's test rows.

### Fold-by-Fold Results

| Fold | Subjects Sample | Subjects Count | Test n | LOSO AUC | Pooled Fold AUC | LOSO Savings@PGR .95 | Pooled Fold Savings |
|---|---|--:|--:|:--:|:--:|:--:|:--:|
| 0 | anatomy, astronomy | 2 | 25 | 0.6522 | 0.6739 | 14.8% | 14.9% |
| 1 | moral_disputes, nutrition | 2 | 29 | 0.6417 | 0.6583 | 6.6% | 3.3% |
| 2 | machine_learning | 1 | 17 | 0.8125 | 0.7500 | 0.0% | 0.0% |
| 3 | philosophy | 1 | 14 | 0.5000 | 0.5000 | 12.4% | 6.4% |
| 4 | world_religions | 1 | 13 | 0.3333 | 0.2500 | 6.8% | 6.8% |
| 5 | formal_logic | 1 | 18 | 0.5625 | 0.5312 | 4.5% | 4.5% |
| 6 | jurisprudence | 1 | 10 | 0.5000 | 0.5000 | 0.0% | 0.0% |
| 7 | econometrics | 1 | 14 | 0.4615 | 0.7692 | 5.7% | 5.7% |
| 8 | high_school_physics | 1 | 15 | 0.7857 | 0.7857 | 0.0% | 0.0% |
| 9 | virology | 1 | 8 | 0.5000 | 0.5000 | 9.6% | 0.0% |

### Like-With-Like Comparisons

To compare like with like, we examine both: (1) averaging metrics across the exact same folds, and (2) scoring predictions across all test rows together.

| Evaluation Scheme | Comparison Basis | Test n | AUC | CPT(95%) | Savings@PGR .95 | APGR |
|---|---|--:|:--:|:--:|:--:|:--:|
| **Pooled Baseline (Fold Mean)** | Same folds averaged | 163 | 0.5918 | 0.9205 | 4.2% | 0.5694 |
| **LOSO Retrained (Fold Mean)** | Same folds averaged | 163 | 0.5749 | 0.8975 | 6.0% | 0.5634 |
| **Pooled Baseline (All Test Rows)** | All test rows scored | 163 | 0.6795 | 0.9200 | 7.2% | 0.5843 |
| **LOSO Out-of-Fold (All Test Rows)** | All test rows scored | 163 | 0.6267 | 0.9200 | 7.1% | 0.5679 |

**Like-with-like Generalization Gaps:**
- **Fold-level average gap (Pooled Fold Mean - LOSO Fold Mean):** AUC +0.0169, Savings -1.9%, CPT +0.0230.
- **All test rows gap (Pooled Retrained - LOSO Out-of-Fold):** AUC +0.0528, Savings +0.1%, CPT +0.0000.
- Worst fold: **Fold 2** (machine_learning, savings 0.0%).
