# Threshold selection and guarantees (T22-T24)

*Generated 2026-11-02 14:09 UTC by `scripts/08_threshold_report.py` from `artifacts\trackb\cascade_predictions.parquet`.*

Pair: S = `qwen3.5-4b`, L = `gpt-4-1106-preview`. Target PGR ≥ 0.95, δ = 0.05. tau* chosen on **cal** only; every test number below is one application of a frozen tau*.

References on the headline test set: random CPT(95%) = 96.4%, APGR 0.504; oracle CPT(95%) = 7.1%, APGR 1.438.

## Headline: tau* from cal, applied once to test (MMLU, GSM8K, ARC, Winogrande)

| Router | Rule | Bound | Certified | α cal | α test | PGR test [95% CI] | Savings M1 [CI] | vs Batch | Savings M2 |
|---|---|---|:-:|--:|--:|--:|--:|--:|--:|
| cascade_1sample | v1 | wsr | yes | 77.6% | 78.2% | 99.1% [96.6, 101.7] | 7.4% [6.6, 8.2] | -85.1% | 7.1% |
| cascade_1sample | v1 | bootstrap | yes | 77.6% | 78.2% | 99.1% [96.6, 101.7] | 7.4% [6.6, 8.2] | -85.1% | 7.1% |
| cascade_1sample | v1 | hoeffding | yes | 99.2% | 99.1% | 100.3% [98.8, 101.9] | 1.4% [0.8, 2.0] | -97.2% | 1.3% |
| cascade_1sample | v1 | naive | yes | 59.6% | 60.8% | 98.0% [94.2, 102.1] | 40.8% [38.7, 42.9] | -18.4% | 38.9% |
| cascade_1sample | v2 | wsr | yes | 68.1% | 68.7% | 97.7% [94.5, 100.9] | 27.8% [25.9, 29.6] | -44.5% | 26.5% |
| cascade_1sample | v2 | bootstrap | yes | 65.1% | 66.1% | 98.6% [95.3, 102.0] | 33.1% [31.1, 35.0] | -33.9% | 31.6% |
| cascade_1sample | v2 | hoeffding | yes | 99.2% | 99.1% | 100.3% [98.8, 101.9] | 1.4% [0.9, 2.0] | -97.1% | 1.4% |
| cascade_1sample | v2 | naive | yes | 58.2% | 59.4% | 97.7% [93.8, 102.2] | 43.6% [41.4, 45.6] | -12.8% | 41.5% |
| cascade_3sample | v1 | wsr | yes | 87.3% | 86.7% | 103.1% [99.7, 107.1] | 36.4% [34.5, 38.7] | -27.2% | 34.4% |
| cascade_3sample | v1 | bootstrap | yes | 87.3% | 86.7% | 103.1% [99.7, 107.1] | 36.4% [34.5, 38.7] | -27.2% | 34.4% |
| cascade_3sample | v1 | hoeffding | yes | 98.9% | 98.8% | 100.3% [98.8, 101.8] | 3.9% [3.0, 5.0] | -92.2% | 3.6% |
| cascade_3sample | v1 | naive | yes | 41.8% | 41.8% | 95.2% [89.6, 101.4] | 77.4% [75.8, 79.1] | 54.8% | 73.6% |
| cascade_3sample | v2 | wsr | yes | 77.6% | 77.2% | 102.8% [99.2, 107.0] | 40.8% [38.7, 43.0] | -18.5% | 38.6% |
| cascade_3sample | v2 | bootstrap | yes | 55.0% | 54.9% | 100.0% [95.7, 105.1] | 65.8% [64.0, 67.9] | 31.6% | 62.6% |
| cascade_3sample | v2 | hoeffding | yes | 98.7% | 98.5% | 100.3% [98.8, 101.8] | 4.6% [3.5, 5.7] | -90.9% | 4.3% |
| cascade_3sample | v2 | naive | yes | 42.7% | 42.9% | 97.2% [91.9, 103.2] | 77.0% [75.4, 78.7] | 54.1% | 73.3% |
| cascade_3sample_f2_shrunk | v1 | wsr | yes | 98.2% | 98.1% | 97.2% [94.4, 99.7] | 3.4% [2.5, 4.3] | -93.2% | 3.2% |
| cascade_3sample_f2_shrunk | v1 | bootstrap | yes | 87.7% | 86.8% | 99.4% [95.7, 103.8] | 33.8% [31.7, 36.1] | -32.4% | 31.9% |
| cascade_3sample_f2_shrunk | v1 | hoeffding | yes | 99.7% | 99.5% | 98.9% [96.8, 100.8] | 0.1% [0.1, 0.2] | -99.7% | 0.1% |
| cascade_3sample_f2_shrunk | v1 | naive | yes | 47.8% | 47.5% | 96.0% [90.9, 101.5] | 73.9% [72.2, 75.7] | 47.9% | 70.4% |
| cascade_3sample_f2_shrunk | v2 | wsr | yes | 98.3% | 98.1% | 97.2% [94.4, 99.7] | 3.2% [2.4, 4.2] | -93.6% | 3.0% |
| cascade_3sample_f2_shrunk | v2 | bootstrap | yes | 85.6% | 85.2% | 99.7% [95.9, 104.3] | 34.3% [32.2, 36.5] | -31.4% | 32.5% |
| cascade_3sample_f2_shrunk | v2 | hoeffding | yes | 99.7% | 99.5% | 99.1% [97.2, 101.0] | 0.1% [0.1, 0.2] | -99.8% | 0.1% |
| cascade_3sample_f2_shrunk | v2 | naive | yes | 42.9% | 43.1% | 92.6% [87.2, 99.1] | 75.6% [73.9, 77.5] | 51.3% | 72.0% |
| cascade_conf+f2_raw | v1 | wsr | yes | 98.4% | 98.2% | 98.0% [95.1, 100.6] | 1.1% [0.7, 1.6] | -97.7% | 1.1% |
| cascade_conf+f2_raw | v1 | bootstrap | yes | 98.1% | 97.9% | 97.7% [94.8, 100.3] | 1.3% [0.9, 1.8] | -97.3% | 1.3% |
| cascade_conf+f2_raw | v1 | hoeffding | yes | 99.5% | 99.4% | 99.4% [97.3, 101.2] | 0.1% [0.1, 0.2] | -99.7% | 0.1% |
| cascade_conf+f2_raw | v1 | naive | yes | 95.2% | 95.2% | 96.9% [93.7, 100.0] | 3.7% [3.0, 4.5] | -92.6% | 3.6% |
| cascade_conf+f2_raw | v2 | wsr | yes | 98.4% | 98.2% | 98.0% [95.1, 100.6] | 1.2% [0.7, 1.6] | -97.7% | 1.1% |
| cascade_conf+f2_raw | v2 | bootstrap | yes | 97.8% | 97.8% | 98.0% [95.1, 100.6] | 1.4% [0.9, 1.9] | -97.2% | 1.3% |
| cascade_conf+f2_raw | v2 | hoeffding | yes | 99.6% | 99.5% | 99.7% [97.7, 101.5] | 0.1% [0.1, 0.1] | -99.8% | 0.1% |
| cascade_conf+f2_raw | v2 | naive | yes | 95.5% | 95.6% | 97.2% [94.0, 100.3] | 3.4% [2.7, 4.1] | -93.3% | 3.2% |
| query_f2 | v1 | wsr | yes | 98.2% | 97.9% | 98.6% [96.1, 101.1] | 2.4% [1.8, 3.1] | -95.2% | 2.3% |
| query_f2 | v1 | bootstrap | yes | 97.4% | 97.5% | 98.9% [96.2, 101.8] | 2.8% [2.1, 3.6] | -94.3% | 2.7% |
| query_f2 | v1 | hoeffding | yes | 99.2% | 98.7% | 98.6% [96.4, 100.8] | 1.3% [0.9, 1.8] | -97.4% | 1.3% |
| query_f2 | v1 | naive | yes | 95.7% | 95.9% | 99.1% [96.0, 102.4] | 4.9% [4.0, 5.9] | -90.2% | 4.6% |
| query_f2 | v2 | wsr | yes | 98.1% | 97.7% | 98.6% [96.0, 101.3] | 2.5% [1.8, 3.1] | -95.1% | 2.3% |
| query_f2 | v2 | bootstrap | yes | 97.9% | 97.6% | 98.6% [96.0, 101.3] | 2.6% [1.9, 3.3] | -94.8% | 2.5% |
| query_f2 | v2 | hoeffding | yes | 99.2% | 98.7% | 98.6% [96.4, 100.8] | 1.3% [0.9, 1.8] | -97.3% | 1.3% |
| query_f2 | v2 | naive | yes | 96.5% | 96.2% | 98.3% [95.1, 101.6] | 3.9% [3.1, 4.6] | -92.3% | 3.7% |

## Test-split curves (no tau selection; for comparing routers)

| Router | Rule | CPT(95%) | Envelope CPT(95%) | APGR | Savings at CPT |
|---|---|--:|--:|--:|--:|
| cascade_1sample | v1 | 53.8% | 51.7% | 0.790 | 57.2% |
| cascade_1sample | v2 | 54.9% | 51.7% | 0.775 | 53.8% |
| cascade_3sample | v1 | 39.0% | 38.2% | 0.836 | 78.7% |
| cascade_3sample | v2 | 39.5% | 38.9% | 0.827 | 78.9% |
| cascade_3sample_f2_shrunk | v1 | 45.4% | 44.5% | 0.825 | 74.6% |
| cascade_3sample_f2_shrunk | v2 | 45.7% | 44.9% | 0.822 | 74.9% |
| cascade_conf+f2_raw | v1 | 47.5% | 43.7% | 0.825 | 70.7% |
| cascade_conf+f2_raw | v2 | 46.0% | 44.9% | 0.821 | 71.6% |
| query_f2 | v1 | 83.7% | 81.1% | 0.681 | 28.3% |
| query_f2 | v2 | 85.9% | 84.7% | 0.654 | 21.1% |

## Per source at the best certified tau* (cascade_3sample, v2, wsr)

| Source | n | Q_S | Q_L | α | PGR | Savings M1 |
|---|--:|--:|--:|--:|--:|--:|
| arc-challenge | 292 | 0.942 | 0.983 | 77.1% | 91.7% | 30.3% |
| gsm8k | 1,490 | 0.923 | 0.950 | 58.6% | 125.0% | 46.6% |
| mmlu | 2,672 | 0.748 | 0.851 | 85.6% | 100.4% | 18.1% |
| winogrande | 253 | 0.696 | 0.791 | 98.4% | 100.0% | 1.6% |

HellaSwag (reported separately, D-06) and the small sets (MBPP, MT-Bench) are not in the headline pool. A single tau* is chosen on the pooled headline sources, so per-source PGR varies around the pooled 95%.

## How to read this

- **Certified = no** means the bound could not certify any threshold that sends less than everything to L at δ = 0.05: the honest answer is then always-frontier.
- **naive** picks the cheapest threshold whose *point estimate* meets the target on cal. It is shown to measure the optimism gap; it carries no guarantee.
- **wsr** is finite-sample valid (Waudby-Smith & Ramdas; the bound RCPS uses). **bootstrap** is DESIGN.md's paired bootstrap, asymptotic only. **hoeffding** is valid and loose.
- **vs Batch**: savings against always-frontier at Batch-API prices (50% off), the fair baseline DESIGN.md §3.2 asks for. **M2** prices local generation as hardware time (assumptions in configs/models.yaml).
