# Threshold selection and guarantees (T22-T24)

*Generated 2026-11-02 14:06 UTC by `scripts/08_threshold_report.py` from `artifacts\trackb\qwen_query_predictions.parquet`.*

Pair: S = `qwen3.5-4b`, L = `gpt-4-1106-preview`. Target PGR ≥ 0.95, δ = 0.05. tau* chosen on **cal** only; every test number below is one application of a frozen tau*.

References on the headline test set: random CPT(95%) = 96.4%, APGR 0.504; oracle CPT(95%) = 7.1%, APGR 1.438.

## Headline: tau* from cal, applied once to test (MMLU, GSM8K, ARC, Winogrande)

| Router | Rule | Bound | Certified | α cal | α test | PGR test [95% CI] | Savings M1 [CI] | vs Batch | Savings M2 |
|---|---|---|:-:|--:|--:|--:|--:|--:|--:|
| two_head_mlp | v1 | wsr | yes | 97.6% | 97.8% | 99.1% [96.6, 101.6] | 0.5% [0.4, 0.7] | -98.9% | 0.5% |
| two_head_mlp | v1 | bootstrap | yes | 97.6% | 97.8% | 99.1% [96.6, 101.6] | 0.5% [0.4, 0.7] | -98.9% | 0.5% |
| two_head_mlp | v1 | hoeffding | yes | 99.1% | 98.9% | 100.3% [98.6, 102.1] | 0.2% [0.1, 0.4] | -99.5% | 0.2% |
| two_head_mlp | v1 | naive | yes | 96.2% | 96.2% | 97.7% [94.7, 101.2] | 1.0% [0.7, 1.2] | -98.1% | 0.9% |
| two_head_mlp | v2 | wsr | yes | 97.7% | 97.9% | 99.1% [96.6, 101.6] | 0.5% [0.3, 0.7] | -99.0% | 0.5% |
| two_head_mlp | v2 | bootstrap | yes | 97.7% | 97.9% | 99.1% [96.6, 101.6] | 0.5% [0.3, 0.7] | -99.0% | 0.5% |
| two_head_mlp | v2 | hoeffding | yes | 99.4% | 99.1% | 100.3% [98.8, 101.9] | 0.2% [0.1, 0.4] | -99.6% | 0.2% |
| two_head_mlp | v2 | naive | yes | 96.3% | 96.2% | 98.0% [95.0, 101.4] | 1.0% [0.7, 1.3] | -98.1% | 0.9% |

## Test-split curves (no tau selection; for comparing routers)

| Router | Rule | CPT(95%) | Envelope CPT(95%) | APGR | Savings at CPT |
|---|---|--:|--:|--:|--:|
| two_head_mlp | v1 | 91.9% | 88.9% | 0.643 | 2.7% |
| two_head_mlp | v2 | 91.9% | 89.3% | 0.631 | 2.7% |

## Per source at the best certified tau* (two_head_mlp, v1, wsr)

| Source | n | Q_S | Q_L | α | PGR | Savings M1 |
|---|--:|--:|--:|--:|--:|--:|
| arc-challenge | 292 | 0.942 | 0.983 | 95.9% | 100.0% | 4.5% |
| gsm8k | 1,490 | 0.923 | 0.950 | 99.9% | 100.0% | 0.1% |
| mmlu | 2,672 | 0.748 | 0.851 | 96.6% | 98.9% | 2.0% |
| winogrande | 253 | 0.696 | 0.791 | 100.0% | 100.0% | 0.0% |

HellaSwag (reported separately, D-06) and the small sets (MBPP, MT-Bench) are not in the headline pool. A single tau* is chosen on the pooled headline sources, so per-source PGR varies around the pooled 95%.

## How to read this

- **Certified = no** means the bound could not certify any threshold that sends less than everything to L at δ = 0.05: the honest answer is then always-frontier.
- **naive** picks the cheapest threshold whose *point estimate* meets the target on cal. It is shown to measure the optimism gap; it carries no guarantee.
- **wsr** is finite-sample valid (Waudby-Smith & Ramdas; the bound RCPS uses). **bootstrap** is DESIGN.md's paired bootstrap, asymptotic only. **hoeffding** is valid and loose.
- **vs Batch**: savings against always-frontier at Batch-API prices (50% off), the fair baseline DESIGN.md §3.2 asks for. **M2** prices local generation as hardware time (assumptions in configs/models.yaml).
