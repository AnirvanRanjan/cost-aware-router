# Threshold selection and guarantees (T22-T24)

*Generated 2026-10-22 16:11 UTC by `scripts/08_threshold_report.py` from `artifacts/predictions/all_interim.parquet`.*

Pair: S = `mistralai/mixtral-8x7b-chat`, L = `gpt-4-1106-preview`. Target PGR ≥ 0.95, δ = 0.05. tau* chosen on **cal** only; every test number below is one application of a frozen tau*.

> **INTERIM.** F0/F1 routers only (no embeddings yet) and provisional splits (the embedding dedup pass has not run). The pipeline is final; the numbers are not. Re-run after T04 + F2.

References on the headline test set: random CPT(95%) = 96.1%, APGR 0.493; oracle CPT(95%) = 21.9%, APGR 0.985.

## Headline: tau* from cal, applied once to test (MMLU, GSM8K, ARC, Winogrande)

| Router | Rule | Bound | Certified | α cal | α test | PGR test [95% CI] | Savings M1 [CI] | vs Batch | Savings M2 |
|---|---|---|:-:|--:|--:|--:|--:|--:|--:|
| f0_logreg | v1 | wsr | yes | 91.4% | 91.0% | 96.6% [95.3, 97.8] | 3.9% [3.4, 4.4] | -92.2% | 3.8% |
| f0_logreg | v1 | bootstrap | yes | 90.2% | 89.9% | 95.9% [94.5, 97.2] | 4.1% [3.7, 4.6] | -91.7% | 4.0% |
| f0_logreg | v1 | hoeffding | yes | 95.4% | 94.6% | 98.0% [97.1, 99.0] | 3.0% [2.5, 3.4] | -94.1% | 2.9% |
| f0_logreg | v1 | naive | yes | 87.2% | 86.6% | 95.3% [93.7, 96.7] | 4.9% [4.4, 5.4] | -90.3% | 4.7% |
| f0_logreg | v2 | wsr | yes | 92.3% | 92.0% | 97.0% [95.8, 98.0] | 1.3% [1.2, 1.5] | -97.3% | 1.3% |
| f0_logreg | v2 | bootstrap | yes | 89.5% | 89.5% | 95.8% [94.3, 97.0] | 1.8% [1.6, 1.9] | -96.5% | 1.7% |
| f0_logreg | v2 | hoeffding | yes | 95.2% | 94.7% | 98.6% [97.7, 99.4] | 0.9% [0.7, 1.0] | -98.3% | 0.8% |
| f0_logreg | v2 | naive | yes | 88.5% | 88.8% | 95.4% [93.9, 96.7] | 1.9% [1.7, 2.1] | -96.2% | 1.8% |
| f0f1_twohead | v1 | wsr | yes | 93.0% | 93.2% | 96.8% [95.4, 98.2] | 2.3% [2.0, 2.7] | -95.3% | 2.3% |
| f0f1_twohead | v1 | bootstrap | yes | 93.6% | 93.6% | 96.6% [95.3, 98.0] | 2.2% [1.9, 2.5] | -95.6% | 2.1% |
| f0f1_twohead | v1 | hoeffding | yes | 96.8% | 96.4% | 97.9% [96.8, 99.0] | 1.3% [1.1, 1.6] | -97.4% | 1.3% |
| f0f1_twohead | v1 | naive | yes | 91.5% | 91.5% | 96.0% [94.5, 97.6] | 2.9% [2.6, 3.3] | -94.2% | 2.8% |
| f0f1_twohead | v2 | wsr | yes | 94.9% | 93.8% | 95.7% [94.2, 97.1] | 3.7% [3.2, 4.1] | -92.7% | 3.5% |
| f0f1_twohead | v2 | bootstrap | yes | 93.6% | 92.5% | 95.3% [93.9, 96.9] | 4.3% [3.8, 4.8] | -91.4% | 4.1% |
| f0f1_twohead | v2 | hoeffding | yes | 96.8% | 96.3% | 98.0% [96.9, 99.1] | 2.3% [1.9, 2.7] | -95.4% | 2.2% |
| f0f1_twohead | v2 | naive | yes | 90.4% | 89.8% | 94.3% [92.6, 96.1] | 5.3% [4.7, 5.8] | -89.5% | 5.1% |
| f1_logreg | v1 | wsr | yes | 91.7% | 91.7% | 96.9% [95.5, 98.1] | 2.7% [2.3, 3.0] | -94.7% | 2.6% |
| f1_logreg | v1 | bootstrap | yes | 90.5% | 90.6% | 96.4% [95.1, 97.6] | 3.0% [2.6, 3.4] | -94.0% | 2.9% |
| f1_logreg | v1 | hoeffding | yes | 95.0% | 95.6% | 98.3% [97.4, 99.1] | 1.4% [1.2, 1.7] | -97.2% | 1.4% |
| f1_logreg | v1 | naive | yes | 87.4% | 87.2% | 95.4% [93.8, 96.9] | 4.2% [3.7, 4.7] | -91.7% | 4.0% |
| f1_logreg | v2 | wsr | yes | 91.6% | 91.7% | 96.5% [95.1, 97.8] | 3.2% [2.6, 3.8] | -93.7% | 3.0% |
| f1_logreg | v2 | bootstrap | yes | 90.4% | 90.3% | 95.8% [94.3, 97.2] | 4.0% [3.3, 4.7] | -92.0% | 3.8% |
| f1_logreg | v2 | hoeffding | yes | 95.4% | 95.5% | 97.9% [97.0, 98.8] | 1.9% [1.5, 2.4] | -96.2% | 1.8% |
| f1_logreg | v2 | naive | yes | 88.9% | 88.7% | 94.9% [93.3, 96.5] | 4.6% [3.8, 5.3] | -90.9% | 4.3% |
| two_head_mlp † | v1 | wsr | yes | 89.6% | 89.5% | 95.4% [94.0, 96.7] | 4.2% [3.7, 4.7] | -91.7% | 4.0% |
| two_head_mlp † | v1 | bootstrap | yes | 87.2% | 86.7% | 94.1% [92.5, 95.6] | 4.9% [4.5, 5.5] | -90.1% | 4.8% |
| two_head_mlp † | v1 | hoeffding | yes | 95.6% | 96.1% | 98.1% [97.2, 99.0] | 1.6% [1.3, 1.9] | -96.8% | 1.5% |
| two_head_mlp † | v1 | naive | yes | 84.1% | 83.6% | 93.4% [91.7, 95.0] | 5.7% [5.3, 6.3] | -88.5% | 5.6% |
| two_head_mlp † | v2 | wsr | yes | 96.2% | 96.2% | 96.3% [95.2, 97.5] | 9.4% [7.8, 10.9] | -81.3% | 8.9% |
| two_head_mlp † | v2 | bootstrap | yes | 95.9% | 95.6% | 95.6% [94.3, 96.8] | 11.0% [9.4, 12.7] | -78.0% | 10.5% |
| two_head_mlp † | v2 | hoeffding | yes | 97.1% | 97.5% | 98.4% [97.6, 99.2] | 5.6% [4.3, 6.8] | -88.8% | 5.3% |
| two_head_mlp † | v2 | naive | yes | 95.1% | 95.0% | 95.1% [93.6, 96.4] | 12.7% [11.0, 14.5] | -74.6% | 12.1% |

† No train-split predictions in the file, so the threshold grid came from cal itself. Learn-then-Test is then only approximately valid; write train predictions to fix it.

## Test-split curves (no tau selection; for comparing routers)

| Router | Rule | CPT(95%) | Envelope CPT(95%) | APGR | Savings at CPT |
|---|---|--:|--:|--:|--:|
| f0_logreg | v1 | 86.2% | 85.8% | 0.629 | 5.0% |
| f0_logreg | v2 | 87.5% | 87.0% | 0.612 | 2.2% |
| f0f1_twohead | v1 | 89.7% | 89.2% | 0.618 | 3.5% |
| f0f1_twohead | v2 | 91.3% | 88.6% | 0.618 | 4.7% |
| f1_logreg | v1 | 86.6% | 86.2% | 0.613 | 4.4% |
| f1_logreg | v2 | 88.1% | 87.9% | 0.596 | 4.8% |
| two_head_mlp | v1 | 89.2% | 86.7% | 0.642 | 4.2% |
| two_head_mlp | v2 | 95.0% | 93.9% | 0.463 | 12.7% |

## Per source at the best certified tau* (two_head_mlp, v2, wsr)

| Source | n | Q_S | Q_L | α | PGR | Savings M1 |
|---|--:|--:|--:|--:|--:|--:|
| arc-challenge | 294 | 0.844 | 0.976 | 100.0% | 100.0% | 0.0% |
| gsm8k | 1,490 | 0.592 | 0.950 | 93.8% | 94.6% | 9.7% |
| hellaswag | 2,008 | 0.417 | 0.848 | 100.0% | 100.0% | 0.0% |
| mbpp | 85 | 0.565 | 0.729 | 30.6% | 28.6% | 79.2% |
| mmlu | 2,808 | 0.638 | 0.810 | 96.8% | 97.5% | 8.9% |
| mt-bench | 16 | 0.669 | 0.944 | 18.8% | 56.8% | 95.0% |
| winogrande | 253 | 0.545 | 0.791 | 100.0% | 100.0% | 0.0% |

HellaSwag (reported separately, D-06) and the small sets (MBPP, MT-Bench) are not in the headline pool. A single tau* is chosen on the pooled headline sources, so per-source PGR varies around the pooled 95%.

## How to read this

- **Certified = no** means the bound could not certify any threshold that sends less than everything to L at δ = 0.05: the honest answer is then always-frontier.
- **naive** picks the cheapest threshold whose *point estimate* meets the target on cal. It is shown to measure the optimism gap; it carries no guarantee.
- **wsr** is finite-sample valid (Waudby-Smith & Ramdas; the bound RCPS uses). **bootstrap** is DESIGN.md's paired bootstrap, asymptotic only. **hoeffding** is valid and loose.
- **vs Batch**: savings against always-frontier at Batch-API prices (50% off), the fair baseline DESIGN.md §3.2 asks for. **M2** prices local generation as hardware time (assumptions in configs/models.yaml).
