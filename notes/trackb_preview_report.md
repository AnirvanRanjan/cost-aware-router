# Threshold selection and guarantees (T22-T24)

*Generated 2026-11-01 17:32 UTC by `scripts/08_threshold_report.py` from `artifacts/trackb/preview_predictions.parquet`.*

Pair: S = `qwen3.5-4b`, L = `gpt-4-1106-preview`. Target PGR ≥ 0.95, δ = 0.05. tau* chosen on **cal** only; every test number below is one application of a frozen tau*.

> **INTERIM.** F0/F1 routers only (no embeddings yet) and provisional splits (the embedding dedup pass has not run). The pipeline is final; the numbers are not. Re-run after T04 + F2.

References on the headline test set: random CPT(95%) = 96.4%, APGR 0.504; oracle CPT(95%) = 7.1%, APGR 1.438.

## Headline: tau* from cal, applied once to test (MMLU, GSM8K, ARC, Winogrande)

| Router | Rule | Bound | Certified | α cal | α test | PGR test [95% CI] | Savings M1 [CI] | vs Batch | Savings M2 |
|---|---|---|:-:|--:|--:|--:|--:|--:|--:|
| cascade_conf | v1 | wsr | yes | 87.6% | 87.3% | 102.8% [99.7, 106.5] | 35.6% [33.6, 37.8] | -28.8% | 33.7% |
| cascade_conf | v1 | bootstrap | yes | 51.4% | 51.5% | 100.0% [95.7, 104.7] | 72.6% [70.9, 74.4] | 45.2% | 69.0% |
| cascade_conf | v1 | hoeffding | yes | 98.9% | 99.0% | 100.0% [98.9, 101.1] | 3.3% [2.4, 4.3] | -93.4% | 3.1% |
| cascade_conf | v1 | naive | yes | 41.6% | 42.1% | 96.3% [90.7, 102.5] | 77.0% [75.4, 78.7] | 54.1% | 73.3% |
| cascade_conf | v2 | wsr | yes | 67.0% | 66.6% | 100.6% [96.9, 104.7] | 47.5% [45.4, 49.7] | -5.0% | 45.1% |
| cascade_conf | v2 | bootstrap | yes | 48.6% | 48.9% | 99.7% [95.3, 104.4] | 73.6% [71.9, 75.4] | 47.1% | 70.0% |
| cascade_conf | v2 | hoeffding | yes | 98.9% | 99.0% | 100.0% [98.9, 101.1] | 3.2% [2.4, 4.2] | -93.6% | 3.0% |
| cascade_conf | v2 | naive | yes | 42.9% | 43.4% | 96.3% [91.2, 102.6] | 76.8% [75.1, 78.4] | 53.5% | 73.0% |
| cascade_conf+f2 | v1 | wsr | yes | 98.4% | 98.2% | 99.1% [96.7, 101.5] | 0.9% [0.6, 1.3] | -98.1% | 0.9% |
| cascade_conf+f2 | v1 | bootstrap | yes | 97.2% | 97.0% | 97.4% [94.3, 100.5] | 2.3% [1.7, 3.0] | -95.5% | 2.2% |
| cascade_conf+f2 | v1 | hoeffding | yes | 99.4% | 99.2% | 98.9% [96.8, 100.9] | 0.2% [0.1, 0.4] | -99.5% | 0.2% |
| cascade_conf+f2 | v1 | naive | yes | 93.7% | 93.6% | 96.9% [93.2, 100.6] | 6.0% [5.1, 7.1] | -87.9% | 5.8% |
| cascade_conf+f2 | v2 | wsr | yes | 97.9% | 97.7% | 99.1% [96.4, 101.9] | 1.5% [1.1, 2.1] | -96.9% | 1.5% |
| cascade_conf+f2 | v2 | bootstrap | yes | 97.7% | 97.4% | 98.6% [95.7, 101.5] | 1.9% [1.3, 2.6] | -96.2% | 1.8% |
| cascade_conf+f2 | v2 | hoeffding | yes | 99.4% | 99.1% | 99.1% [97.0, 101.3] | 0.2% [0.1, 0.3] | -99.5% | 0.2% |
| cascade_conf+f2 | v2 | naive | yes | 90.7% | 90.9% | 96.3% [92.3, 100.6] | 10.5% [9.2, 11.9] | -79.0% | 10.0% |
| query_f2 | v1 | wsr | yes | 94.1% | 93.5% | 93.2% [89.1, 96.9] | 9.3% [8.0, 10.7] | -81.3% | 8.8% |
| query_f2 | v1 | bootstrap | yes | 93.1% | 92.6% | 92.3% [88.1, 96.3] | 10.7% [9.2, 12.1] | -78.7% | 10.1% |
| query_f2 | v1 | hoeffding | yes | 99.2% | 98.9% | 98.9% [96.8, 100.9] | 1.2% [0.8, 1.7] | -97.6% | 1.1% |
| query_f2 | v1 | naive | yes | 88.6% | 88.3% | 90.1% [85.9, 94.6] | 18.4% [16.5, 20.0] | -63.2% | 17.5% |
| query_f2 | v2 | wsr | yes | 94.4% | 93.9% | 92.6% [88.7, 96.5] | 7.8% [6.7, 9.0] | -84.4% | 7.4% |
| query_f2 | v2 | bootstrap | yes | 93.8% | 93.4% | 92.0% [88.1, 95.9] | 8.8% [7.5, 10.1] | -82.5% | 8.3% |
| query_f2 | v2 | hoeffding | yes | 99.2% | 98.8% | 98.9% [96.8, 100.9] | 1.2% [0.8, 1.8] | -97.6% | 1.2% |
| query_f2 | v2 | naive | yes | 90.5% | 89.7% | 91.5% [87.0, 96.4] | 13.9% [12.4, 15.4] | -72.1% | 13.2% |

## Test-split curves (no tau selection; for comparing routers)

| Router | Rule | CPT(95%) | Envelope CPT(95%) | APGR | Savings at CPT |
|---|---|--:|--:|--:|--:|
| cascade_conf | v1 | 40.8% | 39.3% | 0.832 | 77.8% |
| cascade_conf | v2 | 38.8% | 38.6% | 0.823 | 79.0% |
| cascade_conf+f2 | v1 | 50.1% | 47.9% | 0.811 | 67.5% |
| cascade_conf+f2 | v2 | 52.5% | 49.4% | 0.807 | 64.8% |
| query_f2 | v1 | 96.5% | 89.4% | 0.649 | 4.2% |
| query_f2 | v2 | 97.0% | 90.9% | 0.629 | 3.3% |

## Per source at the best certified tau* (cascade_conf, v2, wsr)

| Source | n | Q_S | Q_L | α | PGR | Savings M1 |
|---|--:|--:|--:|--:|--:|--:|
| arc-challenge | 292 | 0.942 | 0.983 | 49.7% | 91.7% | 55.9% |
| gsm8k | 1,490 | 0.923 | 0.950 | 51.4% | 117.5% | 53.2% |
| mmlu | 2,672 | 0.748 | 0.851 | 73.8% | 98.6% | 24.6% |
| winogrande | 253 | 0.696 | 0.791 | 100.0% | 100.0% | 0.0% |

HellaSwag (reported separately, D-06) and the small sets (MBPP, MT-Bench) are not in the headline pool. A single tau* is chosen on the pooled headline sources, so per-source PGR varies around the pooled 95%.

## How to read this

- **Certified = no** means the bound could not certify any threshold that sends less than everything to L at δ = 0.05: the honest answer is then always-frontier.
- **naive** picks the cheapest threshold whose *point estimate* meets the target on cal. It is shown to measure the optimism gap; it carries no guarantee.
- **wsr** is finite-sample valid (Waudby-Smith & Ramdas; the bound RCPS uses). **bootstrap** is DESIGN.md's paired bootstrap, asymptotic only. **hoeffding** is valid and loose.
- **vs Batch**: savings against always-frontier at Batch-API prices (50% off), the fair baseline DESIGN.md §3.2 asks for. **M2** prices local generation as hardware time (assumptions in configs/models.yaml).
