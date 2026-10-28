# Replay Mode Report: Live Demo Pareto Point (T44)

*Generated 2026-10-28 15:10 UTC by `scripts/15_replay_demo.py`.*

## Summary

- **Held-out queries tested:** 200
- **Pair evaluated:** `qwen3.5-4b` vs `claude-sonnet-4.6`
- **Local queries (S):** 0 (0.0%)
- **Frontier queries (L):** 200 (100.0%)
- **Quality from stored scores:** 0.8587 (Small model baseline: 0.5725, Large model: 0.8587)
- **Actual spend:** $0.0415
- **Always-frontier counterfactual:** $0.0415
- **Money saved:** $0.0000 (**0.0%**)
- **Router decision latency (p50):** 9.60 ms
- **Router decision latency (p99):** 11.88 ms

## Acceptance Verification against `threshold.evaluate_at_tau`

Verified on 200 held-out test queries using real bundle threshold tau=-inf:

| Check / Metric | Live Replay | `threshold.evaluate_at_tau` Reference | Status |
|---|---:|---:|:---:|
| Decision Agreement | 100.0% | 100.0% | Pass |
| Frontier Escalation Rate (alpha) | 100.0% | 100.0% | Pass |
| Cost Savings vs Always-Frontier | 0.0% | 0.0% | Pass |
| Quality (Stored Scores) | 0.8587 | 0.8587 | Pass |
| Decision Latency p99 | 11.88 ms | < 15.0 ms | Pass |

**Overall Acceptance Result:** **Pass** (All decisions and savings match `evaluate_at_tau`).
