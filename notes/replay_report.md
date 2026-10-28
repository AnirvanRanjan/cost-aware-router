# Replay Mode Report: Live Demo Pareto Point (T44)

*Generated 2026-10-28 16:07 UTC by `scripts/15_replay_demo.py`.*

## Summary

- **Held-out queries tested:** 200
- **Pair evaluated:** `mistralai/mixtral-8x7b-chat` vs `gpt-4-1106-preview`
- **Local queries (S):** 16 (8.0%)
- **Frontier queries (L):** 184 (92.0%)
- **Quality from stored scores:** 0.8750 (Small model baseline: 0.6100, Large model: 0.8850)
- **Actual spend:** $0.0709
- **Always-frontier counterfactual:** $0.0798
- **Money saved:** $0.0089 (**11.1%**)
- **Router decision latency (p50):** 14.30 ms
- **Router decision latency (p99):** 29.66 ms

## Acceptance Verification against `threshold.evaluate_at_tau`

Verified on 200 held-out test queries using real bundle threshold tau=0.0791:

| Check / Metric | Live Replay | `threshold.evaluate_at_tau` Reference | Status |
|---|---:|---:|:---:|
| Decision Agreement | 100.0% | 100.0% | Pass |
| Frontier Escalation Rate (alpha) | 92.0% | 92.0% | Pass |
| Cost Savings vs Always-Frontier | 11.1% | 11.1% | Pass |
| Quality (Stored Scores) | 0.8750 | 0.8750 | Pass |
| Decision Latency p99 | 29.66 ms | < 15.0 ms (M2) / < 50.0 ms (CPU) | Pass |

**Overall Acceptance Result:** **Pass** (All decisions and savings match `evaluate_at_tau`).
