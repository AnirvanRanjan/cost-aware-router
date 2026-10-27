# Threshold selection and guarantees (T22-T24)

*Generated 2026-10-22 19:19 UTC by `scripts/08_threshold_report.py` from `artifacts/predictions/cascade_compare.parquet`.*

Pair: S = `mistralai/mixtral-8x7b-chat`, L = `gpt-4-1106-preview`. Target PGR ≥ 0.95, δ = 0.05. tau* chosen on **cal** only; every test number below is one application of a frozen tau*.

> **INTERIM.** F0/F1 routers only (no embeddings yet) and provisional splits (the embedding dedup pass has not run). The pipeline is final; the numbers are not. Re-run after T04 + F2.

References on the headline test set: random CPT(95%) = 96.1%, APGR 0.493; oracle CPT(95%) = 21.9%, APGR 0.985.

## Headline: tau* from cal, applied once to test (MMLU, GSM8K, ARC, Winogrande)

| Router | Rule | Bound | Certified | α cal | α test | PGR test [95% CI] | Savings M1 [CI] | vs Batch | Savings M2 |
|---|---|---|:-:|--:|--:|--:|--:|--:|--:|
| cascade_answer | v1 | wsr | yes | 95.2% | 95.2% | 97.2% [96.1, 98.3] | 8.4% [7.2, 9.7] | -83.1% | 7.9% |
| cascade_answer | v1 | bootstrap | yes | 95.2% | 95.2% | 97.2% [96.1, 98.3] | 8.4% [7.2, 9.7] | -83.1% | 7.9% |
| cascade_answer | v1 | hoeffding | yes | 96.9% | 96.8% | 98.4% [97.5, 99.2] | 7.9% [6.6, 9.2] | -84.2% | 7.4% |
| cascade_answer | v1 | naive | yes | 95.2% | 95.2% | 97.2% [96.1, 98.3] | 8.4% [7.2, 9.7] | -83.1% | 7.9% |
| cascade_answer | v2 | wsr | yes | 93.8% | 93.1% | 95.5% [93.9, 97.0] | 6.8% [6.1, 7.6] | -86.4% | 6.6% |
| cascade_answer | v2 | bootstrap | yes | 93.6% | 93.0% | 95.5% [93.9, 97.1] | 7.0% [6.1, 7.8] | -86.1% | 6.7% |
| cascade_answer | v2 | hoeffding | yes | 97.0% | 96.8% | 97.4% [96.4, 98.4] | 3.9% [3.2, 4.5] | -92.3% | 3.7% |
| cascade_answer | v2 | naive | yes | 91.1% | 90.5% | 94.8% [93.0, 96.6] | 8.7% [7.8, 9.6] | -82.7% | 8.3% |
| cascade_answer+2nd | v1 | wsr | yes | 77.4% | 79.0% | 97.9% [96.8, 98.9] | 26.0% [24.2, 27.9] | -47.9% | 24.5% |
| cascade_answer+2nd | v1 | bootstrap | yes | 77.4% | 79.0% | 97.9% [96.8, 98.9] | 26.0% [24.2, 27.9] | -47.9% | 24.5% |
| cascade_answer+2nd | v1 | hoeffding | yes | 92.2% | 93.0% | 99.6% [99.1, 100.1] | 16.9% [15.1, 18.6] | -66.3% | 15.7% |
| cascade_answer+2nd | v1 | naive | yes | 66.7% | 67.9% | 95.0% [93.5, 96.6] | 35.1% [33.0, 37.1] | -29.8% | 33.1% |
| cascade_answer+2nd | v2 | wsr | yes | 76.4% | 77.2% | 95.9% [94.4, 97.3] | 27.0% [25.0, 28.8] | -46.0% | 25.5% |
| cascade_answer+2nd | v2 | bootstrap | yes | 74.1% | 75.0% | 95.6% [94.1, 97.1] | 28.1% [26.2, 30.0] | -43.8% | 26.5% |
| cascade_answer+2nd | v2 | hoeffding | yes | 93.5% | 92.9% | 99.4% [98.8, 100.0] | 13.4% [11.9, 14.8] | -73.2% | 12.6% |
| cascade_answer+2nd | v2 | naive | yes | 65.4% | 66.7% | 94.3% [92.5, 96.0] | 32.4% [30.4, 34.2] | -35.3% | 30.6% |
| cascade_query+answer+2nd | v1 | wsr | yes | 82.0% | 82.3% | 97.1% [95.9, 98.2] | 19.9% [18.2, 21.5] | -60.2% | 18.6% |
| cascade_query+answer+2nd | v1 | bootstrap | yes | 80.2% | 80.7% | 97.0% [95.8, 98.2] | 21.1% [19.3, 22.8] | -57.9% | 19.7% |
| cascade_query+answer+2nd | v1 | hoeffding | yes | 93.2% | 93.3% | 98.9% [98.2, 99.5] | 13.7% [12.2, 15.2] | -72.7% | 12.7% |
| cascade_query+answer+2nd | v1 | naive | yes | 71.5% | 72.1% | 95.5% [93.9, 97.0] | 26.5% [24.6, 28.3] | -47.0% | 24.9% |
| cascade_query+answer+2nd | v2 | wsr | yes | 86.5% | 86.9% | 96.4% [95.1, 97.6] | 10.7% [9.6, 11.8] | -78.6% | 10.2% |
| cascade_query+answer+2nd | v2 | bootstrap | yes | 83.1% | 83.2% | 95.5% [94.1, 96.8] | 15.0% [13.6, 16.4] | -70.1% | 14.2% |
| cascade_query+answer+2nd | v2 | hoeffding | yes | 94.6% | 93.9% | 98.4% [97.4, 99.4] | 5.3% [4.6, 6.0] | -89.4% | 5.1% |
| cascade_query+answer+2nd | v2 | naive | yes | 75.0% | 75.6% | 94.7% [93.1, 96.2] | 23.1% [21.4, 24.8] | -53.8% | 21.8% |
| f0_logreg | v1 | wsr | yes | 91.4% | 91.0% | 96.6% [95.3, 97.8] | 3.9% [3.4, 4.4] | -92.2% | 3.8% |
| f0_logreg | v1 | bootstrap | yes | 90.2% | 89.9% | 95.9% [94.5, 97.2] | 4.1% [3.7, 4.6] | -91.7% | 4.0% |
| f0_logreg | v1 | hoeffding | yes | 95.4% | 94.6% | 98.0% [97.1, 99.0] | 3.0% [2.5, 3.4] | -94.1% | 2.9% |
| f0_logreg | v1 | naive | yes | 87.2% | 86.6% | 95.3% [93.7, 96.7] | 4.9% [4.4, 5.4] | -90.3% | 4.7% |
| f0_logreg | v2 | wsr | yes | 92.3% | 92.0% | 97.0% [95.8, 98.0] | 1.3% [1.2, 1.5] | -97.3% | 1.3% |
| f0_logreg | v2 | bootstrap | yes | 89.5% | 89.5% | 95.8% [94.3, 97.0] | 1.8% [1.6, 1.9] | -96.5% | 1.7% |
| f0_logreg | v2 | hoeffding | yes | 95.2% | 94.7% | 98.6% [97.7, 99.4] | 0.9% [0.7, 1.0] | -98.3% | 0.8% |
| f0_logreg | v2 | naive | yes | 88.5% | 88.8% | 95.4% [93.9, 96.7] | 1.9% [1.7, 2.1] | -96.2% | 1.8% |
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
| cascade_answer | v1 | 95.2% | 91.4% | 0.639 | 8.4% |
| cascade_answer | v2 | 91.7% | 90.8% | 0.626 | 7.9% |
| cascade_answer+2nd | v1 | 68.2% | 67.9% | 0.764 | 34.4% |
| cascade_answer+2nd | v2 | 71.7% | 69.7% | 0.759 | 29.7% |
| cascade_query+answer+2nd | v1 | 69.2% | 68.7% | 0.764 | 28.4% |
| cascade_query+answer+2nd | v2 | 81.0% | 74.8% | 0.757 | 17.9% |
| f0_logreg | v1 | 86.2% | 85.8% | 0.629 | 5.0% |
| f0_logreg | v2 | 87.5% | 87.0% | 0.612 | 2.2% |
| two_head_mlp | v1 | 89.2% | 86.7% | 0.642 | 4.2% |
| two_head_mlp | v2 | 95.0% | 93.9% | 0.463 | 12.7% |

## Per source at the best certified tau* (cascade_answer+2nd, v2, wsr)

| Source | n | Q_S | Q_L | α | PGR | Savings M1 |
|---|--:|--:|--:|--:|--:|--:|
| arc-challenge | 294 | 0.844 | 0.976 | 61.2% | 94.9% | 44.7% |
| gsm8k | 1,490 | 0.592 | 0.950 | 73.2% | 98.9% | 25.2% |
| mmlu | 2,808 | 0.638 | 0.810 | 78.8% | 92.1% | 33.7% |
| winogrande | 253 | 0.545 | 0.791 | 100.0% | 100.0% | 0.0% |

HellaSwag (reported separately, D-06) and the small sets (MBPP, MT-Bench) are not in the headline pool. A single tau* is chosen on the pooled headline sources, so per-source PGR varies around the pooled 95%.

## How to read this

- **Certified = no** means the bound could not certify any threshold that sends less than everything to L at δ = 0.05: the honest answer is then always-frontier.
- **naive** picks the cheapest threshold whose *point estimate* meets the target on cal. It is shown to measure the optimism gap; it carries no guarantee.
- **wsr** is finite-sample valid (Waudby-Smith & Ramdas; the bound RCPS uses). **bootstrap** is DESIGN.md's paired bootstrap, asymptotic only. **hoeffding** is valid and loose.
- **vs Batch**: savings against always-frontier at Batch-API prices (50% off), the fair baseline DESIGN.md §3.2 asks for. **M2** prices local generation as hardware time (assumptions in configs/models.yaml).
