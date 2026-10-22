# Threshold selection and guarantees (T22-T24)

*Generated 2026-10-22 12:46 UTC by `scripts/08_threshold_report.py` from `artifacts/predictions/baselines.parquet`.*

Pair: S = `mistralai/mixtral-8x7b-chat`, L = `gpt-4-1106-preview`. Target PGR ≥ 0.95, δ = 0.05. tau* chosen on **cal** only; every test number below is one application of a frozen tau*.

> **INTERIM.** F0/F1 routers only (no embeddings yet) and provisional splits (the embedding dedup pass has not run). The pipeline is final; the numbers are not. Re-run after T04 + F2.

References on the headline test set: random CPT(95%) = 94.8%, APGR 0.492; oracle CPT(95%) = 22.1%, APGR 0.984.

## Headline: tau* from cal, applied once to test (MMLU, GSM8K, ARC, Winogrande)

| Router | Rule | Bound | Certified | α cal | α test | PGR test [95% CI] | Savings M1 [CI] | vs Batch | Savings M2 |
|---|---|---|:-:|--:|--:|--:|--:|--:|--:|
| f0_logreg | v1 | wsr | yes | 91.2% | 91.8% | 96.2% [94.8, 97.5] | 3.7% [3.3, 4.2] | -92.5% | 3.6% |
| f0_logreg | v1 | bootstrap | yes | 91.4% | 92.1% | 96.4% [95.0, 97.7] | 3.7% [3.2, 4.1] | -92.6% | 3.6% |
| f0_logreg | v1 | hoeffding | yes | 94.9% | 95.6% | 98.1% [97.2, 99.2] | 2.8% [2.4, 3.3] | -94.3% | 2.7% |
| f0_logreg | v1 | naive | yes | 89.7% | 90.2% | 95.7% [94.2, 97.2] | 4.1% [3.6, 4.6] | -91.8% | 4.0% |
| f0_logreg | v2 | wsr | yes | 93.4% | 92.9% | 97.5% [96.4, 98.7] | 1.1% [1.0, 1.3] | -97.7% | 1.1% |
| f0_logreg | v2 | bootstrap | yes | 92.5% | 92.1% | 97.2% [96.1, 98.5] | 1.3% [1.2, 1.4] | -97.4% | 1.2% |
| f0_logreg | v2 | hoeffding | yes | 95.5% | 95.1% | 98.1% [97.2, 99.2] | 0.8% [0.7, 0.9] | -98.5% | 0.7% |
| f0_logreg | v2 | naive | yes | 90.0% | 89.7% | 95.7% [94.2, 97.2] | 1.7% [1.6, 1.9] | -96.6% | 1.6% |
| f0f1_twohead | v1 | wsr | yes | 90.4% | 91.1% | 95.5% [94.0, 96.9] | 2.9% [2.6, 3.3] | -94.1% | 2.8% |
| f0f1_twohead | v1 | bootstrap | yes | 89.8% | 90.7% | 95.4% [93.9, 96.8] | 3.0% [2.7, 3.4] | -93.9% | 2.9% |
| f0f1_twohead | v1 | hoeffding | yes | 95.3% | 95.5% | 98.0% [96.8, 99.1] | 1.5% [1.3, 1.8] | -96.9% | 1.5% |
| f0f1_twohead | v1 | naive | yes | 86.5% | 87.4% | 93.6% [91.8, 95.2] | 4.1% [3.7, 4.6] | -91.8% | 4.0% |
| f0f1_twohead | v2 | wsr | yes | 91.3% | 91.7% | 95.9% [94.3, 97.3] | 3.9% [3.5, 4.5] | -92.2% | 3.8% |
| f0f1_twohead | v2 | bootstrap | yes | 89.6% | 90.1% | 95.2% [93.4, 96.9] | 4.6% [4.1, 5.2] | -90.9% | 4.4% |
| f0f1_twohead | v2 | hoeffding | yes | 94.5% | 94.9% | 98.3% [97.2, 99.5] | 2.5% [2.1, 2.9] | -95.0% | 2.4% |
| f0f1_twohead | v2 | naive | yes | 87.4% | 87.7% | 93.7% [91.9, 95.2] | 5.5% [5.0, 6.2] | -89.0% | 5.3% |
| f1_logreg | v1 | wsr | yes | 91.7% | 91.7% | 96.7% [95.3, 98.0] | 2.9% [2.5, 3.3] | -94.2% | 2.8% |
| f1_logreg | v1 | bootstrap | yes | 90.8% | 90.9% | 96.3% [94.8, 97.7] | 3.1% [2.8, 3.5] | -93.8% | 3.0% |
| f1_logreg | v1 | hoeffding | yes | 95.9% | 96.1% | 98.6% [97.8, 99.3] | 1.5% [1.2, 1.7] | -97.1% | 1.4% |
| f1_logreg | v1 | naive | yes | 87.7% | 88.0% | 94.7% [92.9, 96.3] | 4.3% [3.9, 4.9] | -91.3% | 4.2% |
| f1_logreg | v2 | wsr | yes | 93.9% | 93.9% | 97.4% [96.3, 98.5] | 2.2% [1.7, 2.8] | -95.5% | 2.1% |
| f1_logreg | v2 | bootstrap | yes | 92.6% | 92.6% | 96.2% [95.0, 97.3] | 2.8% [2.2, 3.5] | -94.4% | 2.7% |
| f1_logreg | v2 | hoeffding | yes | 96.6% | 96.2% | 98.9% [98.2, 99.6] | 1.3% [0.9, 1.6] | -97.5% | 1.2% |
| f1_logreg | v2 | naive | yes | 91.0% | 90.8% | 95.4% [94.0, 96.7] | 3.4% [2.8, 4.2] | -93.1% | 3.3% |

## Test-split curves (no tau selection; for comparing routers)

| Router | Rule | CPT(95%) | Envelope CPT(95%) | APGR | Savings at CPT |
|---|---|--:|--:|--:|--:|
| f0_logreg | v1 | 89.1% | 88.5% | 0.616 | 4.3% |
| f0_logreg | v2 | 88.9% | 88.3% | 0.596 | 1.9% |
| f0f1_twohead | v1 | 89.5% | 88.0% | 0.622 | 3.4% |
| f0f1_twohead | v2 | 89.8% | 88.6% | 0.622 | 4.7% |
| f1_logreg | v1 | 88.5% | 88.1% | 0.616 | 4.2% |
| f1_logreg | v2 | 89.3% | 88.7% | 0.600 | 4.0% |

## Per source at the best certified tau* (f0f1_twohead, v2, wsr)

| Source | n | Q_S | Q_L | α | PGR | Savings M1 |
|---|--:|--:|--:|--:|--:|--:|
| arc-challenge | 294 | 0.854 | 0.976 | 90.8% | 83.3% | 8.0% |
| gsm8k | 1,490 | 0.603 | 0.956 | 100.0% | 100.0% | 0.0% |
| hellaswag | 2,008 | 0.403 | 0.831 | 98.3% | 99.0% | 0.8% |
| mbpp | 85 | 0.494 | 0.694 | 71.8% | 88.2% | 26.8% |
| mmlu | 2,808 | 0.637 | 0.815 | 86.6% | 92.0% | 19.0% |
| mt-bench | 16 | 0.669 | 0.944 | 100.0% | 100.0% | 0.0% |
| winogrande | 253 | 0.545 | 0.806 | 100.0% | 100.0% | 0.0% |

HellaSwag (reported separately, D-06) and the small sets (MBPP, MT-Bench) are not in the headline pool. A single tau* is chosen on the pooled headline sources, so per-source PGR varies around the pooled 95%.

## How to read this

- **Certified = no** means the bound could not certify any threshold that sends less than everything to L at δ = 0.05: the honest answer is then always-frontier.
- **naive** picks the cheapest threshold whose *point estimate* meets the target on cal. It is shown to measure the optimism gap; it carries no guarantee.
- **wsr** is finite-sample valid (Waudby-Smith & Ramdas; the bound RCPS uses). **bootstrap** is DESIGN.md's paired bootstrap, asymptotic only. **hoeffding** is valid and loose.
- **vs Batch**: savings against always-frontier at Batch-API prices (50% off), the fair baseline DESIGN.md §3.2 asks for. **M2** prices local generation as hardware time (assumptions in configs/models.yaml).
